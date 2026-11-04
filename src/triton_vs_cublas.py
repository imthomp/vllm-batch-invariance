"""Does vLLM's Triton batch-invariant matmul produce the same bits as cuBLAS (flags as in batch-invariant mode)?"""
import torch
import torch.nn.functional as F
import vllm.model_executor.determinism.batch_invariant as bi

torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = (False, False)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = (False, False)
torch.backends.cuda.preferred_blas_library(backend="cublaslt")
torch.backends.cuda.matmul.fp32_precision = "ieee"
bi.resolve_tuned_matmul_configs()
g = torch.Generator("cuda").manual_seed(0)
for dtype in (torch.bfloat16, torch.float16):
    for n, k in [(4608, 3584), (3584, 3584), (37888, 3584), (3584, 18944), (152064, 3584), (6144, 4096), (151936, 4096), (2048, 1536)]:
        w = torch.randn(n, k, device="cuda", dtype=dtype, generator=g) * k ** -0.5
        for m in (1, 7, 32, 129, 1024):
            x = torch.randn(m, k, device="cuda", dtype=dtype, generator=g)
            t = bi.linear_batch_invariant(x, w)
            c = F.linear(x, w)
            print(str(dtype)[6:], n, k, m, "identical" if torch.equal(t, c) else f"DIFFER frac={(t != c).float().mean().item():.4f}", flush=True)
