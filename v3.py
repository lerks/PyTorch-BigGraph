import ctypes
import mmap
import multiprocessing as mp
import operator
import os
import re
import stat
import time
from contextlib import contextmanager
from functools import reduce

import numpy as np
from pycuda.driver import init, register_host_memory, Device, mem_host_register_flags
from pycuda.gpuarray import GPUArray


rtld = ctypes.CDLL(None, use_errno=True)
_shm_open = rtld.shm_open
_shm_unlink = rtld.shm_unlink

name_format = re.compile(b"/[^/]{1,253}")


@contextmanager
def shm_attach(name: bytes, *flags: int) -> int:
    if not isinstance(name, bytes):
        raise TypeError(f"Expected bytes, got {name:r}")
    if not name_format.fullmatch(name):
        raise ValueError(f"Expected /somename, got {name:r}")
    c_name = ctypes.create_string_buffer(name)

    result = _shm_open(
        c_name,
        ctypes.c_int(reduce(operator.or_, flags, os.O_RDWR)),
        ctypes.c_ushort(stat.S_IRUSR | stat.S_IWUSR),
    )

    if result == -1:
        errno = ctypes.get_errno()
        raise OSError(errno, os.strerror(errno), name)

    try:
        yield result
    finally:
        os.close(result)


@contextmanager
def shm_create(name: bytes, *flags: int) -> int:
    if not isinstance(name, bytes):
        raise TypeError(f"Expected bytes, got {name:r}")
    if not name_format.fullmatch(name):
        raise ValueError(f"Expected /somename, got {name:r}")
    c_name = ctypes.create_string_buffer(name)

    with shm_attach(name, os.O_CREAT, os.O_EXCL, *flags) as fd:
        try:
            yield fd
        finally:
            result = _shm_unlink(c_name)

            if result == -1:
                errno = ctypes.get_errno()
                raise OSError(errno, os.strerror(errno), name)


def round_up_to_multiple(value: int, factor: int) -> int:
    return ((value - 1) // factor + 1) * factor


NUM_ITERATIONS = 10
SIZE = 100_000_000
# SIZE = round_up_to_multiple(SIZE, mmap.PAGESIZE)


class GPUProcess(mp.Process):

    def __init__(self, gpu_idx: int) -> None:
        super().__init__(daemon=True)
        self.gpu_idx = gpu_idx
        self.master_endpoint, self.worker_endpoint = mp.Pipe()

    def run(self) -> None:
        init()
        d = Device(self.gpu_idx)
        d.make_context()
        self.worker_endpoint.send(None)
        name: bytes = self.worker_endpoint.recv()
        t0 = time.monotonic_ns()
        with shm_attach(name) as fd:
            os.ftruncate(fd, 2 * SIZE * 8)
            with mmap.mmap(fd, 2 * SIZE * 8) as m:
                a = np.frombuffer(m, dtype=np.int64)
                a = a.reshape((2, SIZE))
                # We can do the slicing here or later, i.e., pin the whole memory or just half
                # In one case pinning is faster in the other copying is slower
                t1 = time.monotonic_ns()
                a = register_host_memory(a)  # mem_host_register_flags.PORTABLE
                t2 = time.monotonic_ns()
                a = a[self.gpu_idx]
                gpu_a = GPUArray((SIZE,), a.dtype)
                t3 = time.monotonic_ns()
                for _ in range(NUM_ITERATIONS):
                    gpu_a.set(a)
                    gpu_a += 1
                    gpu_a.get(a)
                t4 = time.monotonic_ns()
                print(f"[{self.gpu_idx}] allocate CPU memory {t1 - t0:,}")
                print(f"[{self.gpu_idx}] register CPU memory {t2 - t1:,}")
                print(f"[{self.gpu_idx}] allocate GPU memory {t3 - t2:,}")
                print(f"[{self.gpu_idx}] compute {t4 - t3:,}")


def main():
    process0: GPUProcess = GPUProcess(0)
    process0.start()
    process1: GPUProcess = GPUProcess(1)
    process1.start()

    name = b"/foobarbaz"

    t0 = time.monotonic_ns()
    with shm_create(name) as fd:
        os.ftruncate(fd, 2 * SIZE * 8)
        with mmap.mmap(fd, 0) as m:
            a = np.frombuffer(m, dtype=np.int64)
            a = a.reshape((2, SIZE))
            t1 = time.monotonic_ns()
            process0.master_endpoint.send(name)
            process1.master_endpoint.send(name)
            process0.join()
            process1.join()
            print(a)
            print(f"allocate CPU memory {t1 - t0:,}")


if __name__ == "__main__":
    main()
