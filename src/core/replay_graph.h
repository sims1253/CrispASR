// src/core/replay_graph.h — allocate-once / compute-many graph primitive.
//
// Header-only, matching the other core/* helpers (attention.h, ffn.h, …): the
// build callback is a generic callable so it inlines straight into each caller,
// producing the exact same ggml op sequence as the original inline code.
//
// The recurring performance anti-pattern across the audio-LLM / encoder
// backends is: build a fresh ggml_cgraph, ggml_backend_sched_reset +
// ggml_backend_sched_alloc_graph, bind inputs, compute, read outputs — every
// single call. For graphs that are shape-stable across calls (a parakeet
// encoder fed a fixed T_mel, an autoregressive single-token decode step, …)
// the rebuild + realloc is pure overhead, and on CUDA it also defeats
// ggml-cuda's CUDA-graph capture (the transient cgraph carries uid 0, so the
// reuse fast-path in ggml-cuda.cu never matches).
//
// ReplayGraph fixes both:
//
//   * It builds the graph ONCE in a no_alloc=true metadata context, allocates
//     it on a PRIVATE ggml_gallocr_t (one per graph — the lesson from
//     parakeet issue #208: a shared gallocr's realloc invalidates other live
//     graphs' device pointers), and assigns it a STABLE nonzero uid. On CUDA
//     that uid engages ggml-cuda's captured-graph reuse fast-path
//     (cgraph->uid != 0 && cgraph->uid == graph->uid); on CPU / Metal / Vulkan
//     the stable uid is harmless and you still win by skipping the per-call
//     rebuild + realloc. The design degrades gracefully: a big win on CUDA, a
//     smaller-but-real win everywhere else, never a regression.
//
//   * On replay it only pushes the (small) host inputs and recomputes — no
//     sched_reset, no sched_alloc_graph, no rebuild. Inputs are bound by name
//     exactly the way the model code already does
//     (ggml_graph_get_tensor + ggml_backend_tensor_set), so wiring it in is a
//     mechanical replacement of the reset/alloc/bind/compute/read cycle.
//
// Two allocation paths, decided once per graph (mirrors the proven
// granite_speech bucketed-decode dispatch, src/granite_speech.cpp):
//
//   * raw-gallocr (default, every backend): allocate once on the single
//     primary backend via a private gallocr; reuse across replays. Safe as of
//     the pinned ggml "alloc-once/compute-many safe" commit (890278a), which
//     re-applies src rewires on repeat computes. Works on CUDA (capture
//     engages via the stable uid), Metal, Vulkan, CPU.
//
//   * sched fallback: if any op in the graph is not supported by the primary
//     backend (e.g. a weight pinned on the CPU backend under a partial GPU
//     offload), the graph spans two backends and must go through a
//     ggml_backend_sched_t. The sched path keeps reset+alloc per replay
//     (ggml's capture bookkeeping rejects the documented reuse-shortcut with
//     CUDA error: invalid argument — see docs/contributing.md §210) but
//     still skips the graph rebuild.
//
// Hard constraint: this primitive calls NO CUDA / Metal / Vulkan API. It is
// pure ggml-backend. Nothing here may be CUDA-only or regress non-CUDA
// platforms. It also does not change numerics — the graph the model builds is
// identical; ReplayGraph only persists + replays it instead of rebuilding.
//
// Usage (parakeet encoder sketch):
//
//   // once, keyed on the shape signature (here T_mel):
//   if (!ctx->enc_replay || ctx->enc_replay_T != T_mel) {
//       ctx->enc_replay = std::make_unique<core_replay::ReplayGraph>(
//           ctx->backend, ctx->backend_cpu, 8192,
//           [&](ggml_context* c) { return parakeet_build_graph_encoder(ctx, c, T_mel); });
//       ctx->enc_replay_T = T_mel;
//   }
//   ggml_cgraph* gf = ctx->enc_replay->graph();
//   core_replay::set_named(gf, "mel", mel, bytes);          // bind inputs
//   ...
//   if (!ctx->enc_replay->replay()) { /* error */ }
//   core_replay::read_named(gf, "enc_out", result.data(), bytes);
//
// Every model integration is env-gated (default OFF) so the existing
// rebuild-every-call path stays the default until each model is verified
// byte-exact against its baseline transcript.

#pragma once

#include "crispasr_env.h"

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

// ggml-impl.h is ggml's internal header (under ggml/src/); the crispasr-core
// target exposes that path privately, and this header is only included by TUs
// that link crispasr-core. It is the only place both the cgraph `uid` field
// and ggml_graph_next_uid() are declared (ggml.h leaves the cgraph opaque).
// The stable nonzero uid lets ggml-cuda skip its per-replay O(n_nodes) memcmp
// and reuse a captured CUDA graph (ggml-cuda.cu:2623).
#include "ggml-impl.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

namespace core_replay {

namespace detail {

inline bool replay_diag_on() {
    static const bool on = crispasr_env::present("CRISPASR_REPLAY_DIAG");
    return on;
}

// CRISPASR_REPLAY_GALLOCR=1 forces the raw-gallocr path, =0 forces sched.
// out_override is set to -1 (unset), 0 (force sched), or 1 (force gallocr).
inline bool graph_fits_on_backend(ggml_backend_t backend, ggml_cgraph* gf, int& out_override) {
    out_override = -1;
    if (const char* s = std::getenv("CRISPASR_REPLAY_GALLOCR"))
        out_override = std::atoi(s);
    if (!backend)
        return false;
    for (int i = 0; i < ggml_graph_n_nodes(gf); ++i) {
        if (!ggml_backend_supports_op(backend, ggml_graph_node(gf, i)))
            return false;
    }
    return true;
}

} // namespace detail

// A graph built once and replayed many times. Owns its own persistent ggml
// context (metadata arena) + private gallocr / sched, so it never interferes
// with the caller's ctx->sched or other cached graphs. Move-only (owns device
// resources).
class ReplayGraph {
public:
    // Build the graph via `build`, which receives a fresh no_alloc=true ggml
    // context (capacity `max_nodes`) and returns a fully-built ggml_cgraph
    // (already ggml_build_forward_expand'd over its output, exactly the way the
    // model's existing *_build_*_graph functions work). `backend` is the
    // primary compute backend (GPU or CPU); `cpu_backend` is the CPU fallback
    // used only if an op needs offload (may equal `backend` for a CPU-only
    // build). `build` is called exactly once, at construction.
    template <typename BuildFn>
    ReplayGraph(ggml_backend_t backend, ggml_backend_t cpu_backend, int max_nodes, BuildFn&& build)
        : backend_(backend), cpu_backend_(cpu_backend ? cpu_backend : backend), max_nodes_(max_nodes) {
        construct(std::forward<BuildFn>(build));
    }

    ~ReplayGraph() { cleanup(); }

    ReplayGraph(const ReplayGraph&) = delete;
    ReplayGraph& operator=(const ReplayGraph&) = delete;
    ReplayGraph(ReplayGraph&& o) noexcept { steal_from(o); }
    ReplayGraph& operator=(ReplayGraph&& o) noexcept {
        if (this != &o) {
            cleanup();
            steal_from(o);
        }
        return *this;
    }

    // The persistent cgraph. Bind named inputs on it between replays with
    // ggml_graph_get_tensor + ggml_backend_tensor_set (same as the model code
    // does today). Returns nullptr if construction failed.
    ggml_cgraph* graph() const { return gf_; }

    // Recompute the graph on the primary backend. Returns true on success.
    // Callers push inputs + read outputs around this via the helpers below.
    bool replay() {
        if (!gf_)
            return false;
        if (use_gallocr_) {
            // No reset / re-alloc: the graph keeps its buffers from construction.
            return ggml_backend_graph_compute(backend_, gf_) == GGML_STATUS_SUCCESS;
        }
        if (!sched_)
            return false;
        ggml_backend_sched_reset(sched_);
        if (!ggml_backend_sched_alloc_graph(sched_, gf_))
            return false;
        return ggml_backend_sched_graph_compute(sched_, gf_) == GGML_STATUS_SUCCESS;
    }

    // Which allocation path this graph settled on (diagnostics / env-gating).
    bool uses_gallocr() const { return use_gallocr_; }

private:
    template <typename BuildFn> void construct(BuildFn&& build) {
        if (!backend_)
            return;

        // Persistent metadata arena (outlives every replay). Sized like the
        // model's compute_meta: tensor slots + cgraph node slots. no_alloc=true
        // so tensor ->data stays NULL until the gallocr/sched assigns buffers.
        const size_t node_cap = (size_t)(max_nodes_ > 0 ? max_nodes_ : 8192);
        meta_.assign(ggml_tensor_overhead() * node_cap + ggml_graph_overhead_custom(node_cap, false), 0);
        ggml_init_params ip = {meta_.size(), meta_.data(), /*no_alloc=*/true};
        ctx_ = ggml_init(ip);
        if (!ctx_)
            return;

        // Build the graph. The caller's lambda builds + forward-expands the
        // cgraph in ctx_ (the same way the model's *_build_*_graph functions
        // already do) and returns it. ReplayGraph assigns the stable uid and
        // allocates it on its private gallocr / sched.
        gf_ = build(ctx_);
        if (!gf_) {
            ggml_free(ctx_);
            ctx_ = nullptr;
            return;
        }
        // Stable nonzero uid: engages ggml-cuda's captured-graph reuse fast-path
        // (cgraph->uid != 0 && cgraph->uid == graph->uid) and is a no-op on
        // other backends. Without this the transient cgraph carries uid 0 and
        // capture never reuses, so the per-call rebuild/alloc overhead stays.
        gf_->uid = ggml_graph_next_uid();

        allocate();

        if (detail::replay_diag_on()) {
            std::fprintf(stderr, "[replay] backend=%s max_nodes=%zu n_nodes=%d uid=%llu path=%s\n",
                         ggml_backend_name(backend_), node_cap, gf_->n_nodes, (unsigned long long)gf_->uid,
                         use_gallocr_ ? "gallocr" : "sched");
        }
    }

    void allocate() {
        int override = -1;
        const bool fits = detail::graph_fits_on_backend(backend_, gf_, override);
        use_gallocr_ = (override >= 0) ? (override != 0) : fits;
        if (override >= 0 && !use_gallocr_ && !cpu_backend_) {
            // sched fallback requested but there's no second backend to offload
            // to — the sched would be single-backend anyway, so keep gallocr.
            use_gallocr_ = true;
        }

        if (use_gallocr_) {
            // PRIVATE gallocr per graph (issue #208: a shared gallocr's realloc
            // invalidates other live graphs' device pointers). Allocated once;
            // gallocr_alloc_graph does not mutate the graph, so the same object
            // is re-fed every replay.
            galloc_ = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend_));
            if (!galloc_ || !ggml_gallocr_alloc_graph(galloc_, gf_)) {
                // Allocation failed (rare). Fall back to sched rather than leave
                // the graph unusable.
                if (galloc_) {
                    ggml_gallocr_free(galloc_);
                    galloc_ = nullptr;
                }
                use_gallocr_ = false;
            }
        }
        if (!use_gallocr_) {
            // Sched fallback: 2-backend {primary, cpu} so an op the primary
            // can't run can offload. Kept reset+alloc per replay (ggml's
            // capture bookkeeping rejects the reuse-shortcut — §210), but the
            // graph itself is still the persisted one, so the rebuild is skipped.
            if (cpu_backend_ && cpu_backend_ != backend_) {
                ggml_backend_t backends[2] = {backend_, cpu_backend_};
                ggml_backend_buffer_type_t bufs[2] = {ggml_backend_get_default_buffer_type(backend_),
                                                      ggml_backend_get_default_buffer_type(cpu_backend_)};
                sched_ = ggml_backend_sched_new(backends, bufs, 2, (size_t)max_nodes_, /*parallel=*/false,
                                                /*op_offload=*/true);
            } else {
                ggml_backend_t backends[1] = {backend_};
                sched_ = ggml_backend_sched_new(backends, nullptr, 1, (size_t)max_nodes_, /*parallel=*/false,
                                                /*op_offload=*/true);
            }
        }
    }

    void cleanup() noexcept {
        // Best-effort free; ggml_*_free are safe on null. Backends NOT owned.
        if (sched_)
            ggml_backend_sched_free(sched_);
        if (galloc_)
            ggml_gallocr_free(galloc_);
        if (ctx_)
            ggml_free(ctx_);
        sched_ = nullptr;
        galloc_ = nullptr;
        ctx_ = nullptr;
        gf_ = nullptr;
    }

    void steal_from(ReplayGraph& o) noexcept {
        backend_ = o.backend_;
        cpu_backend_ = o.cpu_backend_;
        max_nodes_ = o.max_nodes_;
        ctx_ = o.ctx_;
        gf_ = o.gf_;
        meta_ = std::move(o.meta_);
        use_gallocr_ = o.use_gallocr_;
        galloc_ = o.galloc_;
        sched_ = o.sched_;
        o.ctx_ = nullptr;
        o.gf_ = nullptr;
        o.galloc_ = nullptr;
        o.sched_ = nullptr;
    }

    ggml_backend_t backend_ = nullptr;     // primary (not owned)
    ggml_backend_t cpu_backend_ = nullptr; // CPU fallback (not owned)
    int max_nodes_ = 0;

    ggml_context* ctx_ = nullptr; // persistent metadata arena (owned)
    ggml_cgraph* gf_ = nullptr;   // the graph (lives in ctx_)
    std::vector<uint8_t> meta_;   // backs ctx_'s memory (owned)

    bool use_gallocr_ = true;              // gallocr fast path vs sched fallback
    ggml_gallocr_t galloc_ = nullptr;      // private; allocated once (owned)
    ggml_backend_sched_t sched_ = nullptr; // lazy; only if an op needs offload (owned)
};

// ---- name-based input/output helpers (the existing model binding idiom) ----

// Bind `nbytes` from `data` into the graph tensor named `name`. Returns false
// if the named tensor is absent (treated as a hard error by callers that
// expect it, a no-op by callers passing an optional input).
inline bool set_named(ggml_cgraph* gf, const char* name, const void* data, size_t nbytes) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name);
    if (!t)
        return false;
    ggml_backend_tensor_set(t, data, 0, nbytes);
    return true;
}

// Read `nbytes` from the graph tensor named `name` into `out`.
inline bool read_named(ggml_cgraph* gf, const char* name, void* out, size_t nbytes) {
    ggml_tensor* t = ggml_graph_get_tensor(gf, name);
    if (!t)
        return false;
    ggml_backend_tensor_get(t, out, 0, nbytes);
    return true;
}

} // namespace core_replay
