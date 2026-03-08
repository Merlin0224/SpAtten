import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = tl.store(output_ptr + offsets, x + y, mask=mask)

def test():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='cuda')
    y = torch.rand(size, device='cuda')
    output = torch.empty_like(x)
    grid = lambda meta: (triton.cdiv(size, meta['BLOCK_SIZE']),)
    add_kernel[grid](x, y, output, size, BLOCK_SIZE=1024)
    print(f"Triton add result matches torch: {torch.allclose(output, x + y)}")

if __name__ == "__main__":
    test()