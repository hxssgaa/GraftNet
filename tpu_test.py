import torch
import torch_xla
import torch_xla.core.xla_model as xm
import time

if __name__ == '__main__':
    print(xm.get_xla_supported_devices())
    t = torch.randn(8192, 8192, device=xm.xla_device())
    s = time.time()
    for _ in range(10):
        t += 2
    print(time.time() - s)
    print(t.device)