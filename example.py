import ast
import inspect
import triton
import triton.language as tl
import torch
from tritonia import IntJitFunction, PseudoTensorValue

import numpy as np


@IntJitFunction
def bmm2x2_kernel(x_ptr, y_ptr, z_ptr, N_BLOCK: tl.constexpr, N: tl.constexpr, K: tl.constexpr, M: tl.constexpr):
    pid = tl.program_id(0)
    offsets = tl.arange(0, N_BLOCK) + pid * N_BLOCK
    x_stride = N * K
    y_stride = K * M
    z_stride = N * M
    
    x_values = np.array([tl.load(x_ptr + offsets * x_stride + i) for i in range(x_stride)]).reshape((N,K))
    y_values = np.array([tl.load(y_ptr + offsets * y_stride + i) for i in range(y_stride)]).reshape((K,M))
    reduced = np.abs(x_values @ y_values).flatten()

    [tl.store(z_ptr + offsets * z_stride + i, reduced[i]) for i in range(z_stride)]


if __name__=="__main__":
    torch.set_default_device('cuda:0')
    N_BLOCK = 32
    N_TOTAL = 128
    N, K, M = 3, 4, 3
    x = torch.randn((N_TOTAL, N, K), dtype = torch.float32)
    y = torch.randn((N_TOTAL, K, M), dtype = torch.float32)
    z  =torch.empty((N_TOTAL, N, M), dtype = torch.float32)

    ref_value = torch.abs(torch.bmm(x, y))
    grid = (triton.cdiv(N_TOTAL, N_BLOCK), )
    bmm2x2_kernel[grid](x, y, z, N_BLOCK, N, K, M)

    if torch.allclose(z, ref_value, 1e-4, 1e-4):
        print('[√] test passed')

 

