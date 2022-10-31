# Measuring performance of torch.matmul on a CPU and a GPU

import torch
import time

print(f"Time measurement on CPU")
for n in [10, 50, 100, 500, 1000, 5000, 10000]:
    a = torch.ones(n,n, dtype=torch.float, device='cpu');
    b = torch.ones(n,n, dtype=torch.float, device='cpu');
    tic = time.perf_counter()
    c = torch.matmul(a, b);
    toc = time.perf_counter()
    print(f"Spent {(toc-tic)/(n*n*n):.2e} seconds for multiplying {n}x{n} matrices")

print(f"Time measurement on GPU")
for n in [10, 50, 100, 500, 1000, 5000, 10000]:
    a = torch.ones(n,n, dtype=torch.float, device='cuda');
    b = torch.ones(n,n, dtype=torch.float, device='cuda');
    tic = time.perf_counter()
    c = torch.matmul(a, b);
    toc = time.perf_counter()
    print(f"Spent {(toc-tic)/(n*n*n):.2e} seconds for multiplying {n}x{n} matrices using {torch.cuda.memory_allocated()} bytes of memory")
