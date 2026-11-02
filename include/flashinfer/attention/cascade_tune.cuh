/*
 * Cascade draft kernel tuning knobs (Fast Draft). Defaults are the A6000
 * tuning (tests/tune_cascade_a6000.py, bs 1-32); ctas_per_sm and min_chunk_kv
 * are host-side planner knobs and can also be passed to cascade_plan. Override at JIT time with
 *   FLASHINFER_CASCADE_TUNE="<cta_tile_q1>,<ctas_per_sm>,<min_chunk_kv>"
 * which passes -DFD_CASCADE_* and gives the module its own cache entry.
 *   cta_tile_q1  : query tile of the shared-prefix runner (CTA_TILE_Q_1)
 *   ctas_per_sm  : CTAs per SM the scheduler plans for
 *   min_chunk_kv : at least ceil(max_kv / min_chunk_kv) KV chunks per level
 *                  (i.e. a cap on the KV chunk length)
 */
#ifndef FLASHINFER_ATTENTION_CASCADE_TUNE_CUH_
#define FLASHINFER_ATTENTION_CASCADE_TUNE_CUH_

#ifndef FD_CASCADE_CTA_TILE_Q1
#define FD_CASCADE_CTA_TILE_Q1 64
#endif
#ifndef FD_CASCADE_CTAS_PER_SM
#define FD_CASCADE_CTAS_PER_SM 0  // 0: chosen at plan time from the prefix work count
#endif
#ifndef FD_CASCADE_MIN_CHUNK_KV
#define FD_CASCADE_MIN_CHUNK_KV 16384
#endif

#endif  // FLASHINFER_ATTENTION_CASCADE_TUNE_CUH_
