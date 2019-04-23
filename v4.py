import ctypes
import time

import torch
import torch.multiprocessing as mp


NUM_ITERATIONS = 10
SIZE = 100_000_000


class GPUProcess(mp.Process):

    def __init__(self, gpu_idx: int) -> None:
        super().__init__(daemon=True)
        self.gpu_idx = gpu_idx
        self.master_endpoint, self.worker_endpoint = mp.Pipe()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", index=self.gpu_idx)

    def run(self) -> None:
        torch.set_num_threads(1)
        torch.cuda.set_device(self.device)
        torch.empty((), dtype=torch.long, device=self.device)

        t0 = time.monotonic_ns()
        t: torch.Tensor = self.worker_endpoint.recv()
        t1 = time.monotonic_ns()

        ptr = ctypes.c_void_p(t.data_ptr())
        size = ctypes.c_size_t(t.numel() * t.element_size())
        flags = ctypes.c_uint(0)
        res = torch.cuda.cudart().cudaHostRegister(ptr, size, flags)
        torch.cuda.check_error(res)
        assert t.is_pinned()

        t2 = time.monotonic_ns()
        t = t[self.gpu_idx]
        gpu_t = torch.empty(t.shape, dtype=torch.long, device=self.device)
        t3 = time.monotonic_ns()
        for _ in range(NUM_ITERATIONS):
            gpu_t.copy_(t)
            gpu_t += 1
            t.copy_(gpu_t)
        t4 = time.monotonic_ns()
        print(f"[{self.gpu_idx}] receive shared memory {t1 - t0:,}")
        print(f"[{self.gpu_idx}] register memory {t2 - t1:,}")
        print(f"[{self.gpu_idx}] allocate GPU memory {t3 - t2:,}")
        print(f"[{self.gpu_idx}] compute {t4 - t3:,}")

        res = torch.cuda.cudart().cudaHostUnregister(ptr, size, flags)
        torch.cuda.check_error(res)

def main():
    process0: GPUProcess = GPUProcess(0)
    process0.start()
    process1: GPUProcess = GPUProcess(1)
    process1.start()

    t0 = time.monotonic_ns()
    s = torch.LongStorage._new_shared(2 * SIZE)
    t1 = time.monotonic_ns()
    t = torch.LongTensor(s).view((2, SIZE))
    process0.master_endpoint.send(t)
    process1.master_endpoint.send(t)
    process0.join()
    process1.join()
    print(t)
    print(f"allocate CPU memory {t1 - t0:,}")


if __name__ == "__main__":
    main()