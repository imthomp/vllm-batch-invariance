# PR #65931: end-to-end runs built from the branch's code

"main" = vLLM CI wheel at the PR's merge base (0.33.0rc1.dev850+g2d9d3093e, cu130), unmodified.
"this PR" (tuned) = the same wheel's `vllm` package with the branch diff applied (`git diff <merge-base> pr65931-narrow -- vllm/`, one Python file).
"off" = batch invariance off (reference). `vllm bench serve`, fresh server per run, 3 rounds alternating arm order.

- oss20_v2, oss120_v2: H100 80GB. h200_oss20: H200. (The first attempt's port collision is described in PROTOCOL.md; those runs are not here.)
- bench_<workload>_<arm>_r<round>.log/.json: client output; server_<arm>_r<round>.log: server.
- unit_branch_h200.out: the branch's unit tests on the branch build.
- VL_diag_largem_h100_*.out: kernel timings of the tuned configs vs the default at M = 1024..16384.
