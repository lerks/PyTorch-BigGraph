import time

import torch
import torch.multiprocessing as mp


class GPUProcess(mp.Process):

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.master_endpoint, self.worker_endpoint = mp.Pipe()

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", index=0)

    def run(self) -> None:
        torch.set_num_threads(1)
        torch.cuda.set_device(self.device)
        t: torch.Tensor = self.worker_endpoint.recv()
        while True:
            time.sleep(5)
            t += 1


def main():
    process: GPUProcess = GPUProcess()
    process.start()

    t = torch.tensor([0], device="cuda:0")
    process.master_endpoint.send(t)
    while True:
        time.sleep(1)
        print(t)


if __name__ == "__main__":
    main()