import triton
import triton.language as tl
import torch
from tritonia import ijit

import numpy as np

@ijit
def bmm2x2_kernel(x_ptr, y_ptr, N: tl.constexpr):
    offsets = tl.arange(0, N)
    x_values = np.array([tl.load(x_ptr + offsets * 4 + i) for i in range(4)]).reshape((2,2))
    y_values = np.array([tl.load(y_ptr + offsets * 4 + i) for i in range(4)]).reshape((2,2))
    x_reduced = (y_values @ x_values).flatten()

    [tl.store(x_ptr + offsets * 4 + i, x_reduced[i]) for i in range(4)]

if __name__=="__main__":
    torch.set_default_device('cuda:0')
    N = 32
    x = torch.randn((N, 2, 2), dtype = torch.float32)
    y = torch.randn((N, 2, 2), dtype = torch.float32)
    ref_value = torch.bmm(y, x)
    bmm2x2_kernel[(1, )](x, y, N)

    if torch.allclose(x, ref_value, 1e-4, 1e-4):
        print('[√] test passed')

 

