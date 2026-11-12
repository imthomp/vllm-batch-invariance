# Better default (PROTOCOL.md "Better default")

- generic.json, generic_sel_*.out: selection of one generic configuration on 28 training shapes and the held-out kernel check (0 rows differ on the 10 held-out shapes).
- decode_*gen_*, cross_q25gen_generic.*, generic_eval_*.out: FIRST vLLM evaluation, VOID. vLLM enables breakable CUDA graphs (runtime tile lookup) only when a model's Linear shapes are in the tuned table, so for these untuned shapes the compiled graph froze the configuration lookup and the generic default was not applied per batch size. Kept for the record; see PROTOCOL.md (amendment 2026-11-11). The `generic.*` keys in stats.json from this run (score_generic.py, tags q25gen/g2gen) describe the void run.
- The rerun with breakable CUDA graphs and a stock+breakable control arm is in results/generic2 once scored.
- Diagnostics: src/diag_generic_compile.py (the hook applies under torch.compile), slurm/diag_*.local, H200 check in the research log.
