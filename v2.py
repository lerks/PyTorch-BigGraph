import ctypes
import mmap
import operator
import os
import re
import stat
import time
from contextlib import contextmanager
from functools import reduce

import numpy as np
from pycuda.driver import init, register_host_memory, Device
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
SIZE = 1_000_000_000
# SIZE = round_up_to_multiple(SIZE, mmap.PAGESIZE)


def main():
    init()
    d = Device(0)
    ctx = d.make_context()

    name = b"/foobarbaz"

    t0 = time.monotonic_ns()
    with shm_create(name) as fd:
        os.ftruncate(fd, 1 * SIZE * 8)
        with mmap.mmap(fd, 0) as m:
            a = np.frombuffer(m, dtype=np.int64)
            a = a.reshape((1, SIZE))
            t1 = time.monotonic_ns()
            a = register_host_memory(a)
            t2 = time.monotonic_ns()
            gpu_a = GPUArray(a.shape, a.dtype)
            t3 = time.monotonic_ns()
            for _ in range(NUM_ITERATIONS):
                gpu_a.set(a)
                gpu_a += 1
                gpu_a.get(a)
            t4 = time.monotonic_ns()
            print(a)

    ctx.pop()

    print(f"allocate CPU memory {t1 - t0:,}")
    print(f"register CPU memory {t2 - t1:,}")
    print(f"allocate GPU memory {t3 - t2:,}")
    print(f"compute {t4 - t3:,}")


if __name__ == "__main__":
    main()
