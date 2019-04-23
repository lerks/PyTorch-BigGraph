#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE.txt file in the root directory of this source tree.

import argparse
import ctypes
import os.path
import sys
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import partial
from itertools import chain
from multiprocessing.connection import wait as mp_wait
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Set,
)

import gc
import numpy as np
import torch
import torch.distributed as td
import torch.multiprocessing as mp
from torch import nn
from torch.optim import Adagrad, Optimizer

from torchbiggraph.batching import (
    AbstractBatchProcessor,
    process_in_batches,
)
from torchbiggraph.bucket_scheduling import (
    AbstractBucketScheduler,
    DistributedBucketScheduler,
    LockServer,
    SingleMachineBucketScheduler,
)
from torchbiggraph.config import (
    ConfigSchema,
    LossFunction,
    RelationSchema,
    parse_config,
)
from torchbiggraph.distributed import (
    ProcessRanks,
    init_process_group,
    start_server,
)
from torchbiggraph.edgelist import EdgeList
from torchbiggraph.entitylist import EntityList
from torchbiggraph.fileio import (
    CheckpointManager,
    ConfigMetadataProvider,
    EdgeReader,
    MetadataProvider,
    PartitionClient,
    maybe_old_entity_path,
)
from torchbiggraph.losses import (
    AbstractLoss,
    LogisticLoss,
    RankingLoss,
    SoftmaxLoss,
)
from torchbiggraph.model import (
    MultiRelationEmbedder,
    make_model,
    override_model,
)
from torchbiggraph.parameter_sharing import ParameterServer, ParameterSharer
from torchbiggraph.row_adagrad import RowAdagrad
from torchbiggraph.stats import Stats
from torchbiggraph.types import (
    Bucket,
    EntityName,
    FloatTensorType,
    GPURank,
    LongTensorType,
    ModuleStateDict,
    OptimizerStateDict,
    Partition,
    Rank,
    Side,
    SubPartition,
)
from torchbiggraph.util import (
    DummyOptimizer,
    fast_approx_rand,
    get_partitioned_types,
    log,
    round_up_to_nearest_multiple,
    split_almost_equally,
    vlog,
)


class Trainer(AbstractBatchProcessor):

    loss_fn: AbstractLoss

    def __init__(
        self,
        global_optimizer: Optimizer,
        loss_fn: LossFunction,
        margin: float,
        relations: List[RelationSchema],
    ) -> None:
        super().__init__()
        self.global_optimizer = global_optimizer
        self.unpartitioned_entity_optimizers: Dict[EntityName, Optimizer] = {}
        self.partitioned_entity_optimizers: Dict[Tuple[EntityName, Partition, SubPartition], Optimizer] = {}

        if loss_fn is LossFunction.LOGISTIC:
            self.loss_fn = LogisticLoss()
        elif loss_fn is LossFunction.RANKING:
            self.loss_fn = RankingLoss(margin)
        elif loss_fn is LossFunction.SOFTMAX:
            self.loss_fn = SoftmaxLoss()
        else:
            raise NotImplementedError("Unknown loss function: %s" % loss_fn)

        self.relations = relations

    def process_one_batch(
        self,
        model: MultiRelationEmbedder,
        batch_edges: EdgeList,
    ) -> Stats:
        model.zero_grad()

        scores = model(batch_edges)

        lhs_loss = self.loss_fn(scores.lhs_pos, scores.lhs_neg)
        rhs_loss = self.loss_fn(scores.rhs_pos, scores.rhs_neg)
        relation = self.relations[batch_edges.get_relation_type_as_scalar()
                                  if batch_edges.has_scalar_relation_type()
                                  else 0]
        loss = relation.weight * (lhs_loss + rhs_loss)

        stats = Stats(
            loss=float(loss),
            violators_lhs=int((scores.lhs_neg > scores.lhs_pos.unsqueeze(1)).sum()),
            violators_rhs=int((scores.rhs_neg > scores.rhs_pos.unsqueeze(1)).sum()),
            count=len(batch_edges))

        loss.backward()

        self.global_optimizer.step(closure=None)
        for optimizer in self.unpartitioned_entity_optimizers.values():
            optimizer.step(closure=None)
        for optimizer in self.partitioned_entity_optimizers.values():
            optimizer.step(closure=None)

        return stats


def init_embs(
    entity: EntityName,
    entity_count: int,
    dim: int,
    scale: float,
) -> Tuple[FloatTensorType, None]:
    """Initialize embeddings of size entity_count x dim.
    """
    # FIXME: Use multi-threaded instead of fast_approx_rand
    vlog("Initializing %s" % entity)
    return fast_approx_rand(entity_count * dim).view(entity_count, dim).mul_(scale), None


RANK_ZERO = Rank(0)


class AbstractSynchronizer(ABC):

    @abstractmethod
    def barrier(self) -> None:
        pass


class DummySynchronizer(AbstractSynchronizer):

    def barrier(self):
        pass


class DistributedSynchronizer(AbstractSynchronizer):

    def __init__(self, group: 'td.ProcessGroup') -> None:
        self.group = group

    def barrier(self):
        td.barrier(group=self.group)


class IterationManager(MetadataProvider):

    def __init__(
        self,
        num_epochs: int,
        edge_paths: List[str],
        num_edge_chunks: int,
        *,
        iteration_idx: int = 0,
    ) -> None:
        self.num_epochs = num_epochs
        self.edge_paths = edge_paths
        self.num_edge_chunks = num_edge_chunks
        self.iteration_idx = iteration_idx

    @property
    def epoch_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks // self.num_edge_paths

    @property
    def num_edge_paths(self) -> int:
        return len(self.edge_paths)

    @property
    def edge_path_idx(self) -> int:
        return self.iteration_idx // self.num_edge_chunks % self.num_edge_paths

    @property
    def edge_path(self) -> str:
        return self.edge_paths[self.edge_path_idx]

    @property
    def edge_chunk_idx(self) -> int:
        return self.iteration_idx % self.num_edge_chunks

    def remaining_iterations(self) -> Iterable[Tuple[int, int, int]]:
        while self.epoch_idx < self.num_epochs:
            yield self.epoch_idx, self.edge_path_idx, self.edge_chunk_idx
            self.iteration_idx += 1

    def get_checkpoint_metadata(self) -> Dict[str, Any]:
        return {
            "iteration/num_epochs": self.num_epochs,
            "iteration/epoch_idx": self.epoch_idx,
            "iteration/num_edge_paths": self.num_edge_paths,
            "iteration/edge_path_idx": self.edge_path_idx,
            "iteration/edge_path": self.edge_path,
            "iteration/num_edge_chunks": self.num_edge_chunks,
            "iteration/edge_chunk_idx": self.edge_chunk_idx,
        }


EmbeddingHolder = Dict[Tuple[EntityName, Partition], Tuple[LongTensorType, FloatTensorType, RowAdagrad]]
SubEmbeddingHolder = Dict[Tuple[EntityName, Partition, SubPartition], Tuple[nn.Parameter, RowAdagrad]]


class SubprocessArgs(NamedTuple):
    lhs_partitioned_types: Set[str]
    rhs_partitioned_types: Set[str]
    lhs_part: Partition
    rhs_part: Partition
    lhs_subpart: SubPartition
    rhs_subpart: SubPartition
    model: MultiRelationEmbedder
    trainer: Trainer
    holder: EmbeddingHolder
    subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice]
    rel: LongTensorType
    lhs_subparts: LongTensorType
    rhs_subparts: LongTensorType
    lhs_offsets: LongTensorType
    rhs_offsets: LongTensorType
    batch_size: int
    lr: float


class SubprocessReturn(NamedTuple):
    gpu_idx: GPURank
    stats: Stats


class GPUProcess(mp.Process):

    def __init__(self, gpu_idx: GPURank) -> None:
        super().__init__(daemon=True, name=f"GPU worker #{gpu_idx}")
        self.gpu_idx = gpu_idx
        self.master_endpoint, self.worker_endpoint = mp.Pipe()
        self.pinned_ptrs_and_sizes: Set[Tuple[int, int]] = set()

    @property
    def my_device(self) -> torch.device:
        return torch.device("cuda", index=self.gpu_idx)

    def run(self) -> None:
        torch.set_num_threads(1)
        torch.cuda.set_device(self.my_device)
        while True:
            try:
                job: SubprocessArgs = self.worker_endpoint.recv()
            except EOFError:
                break
            if job is None:
                break

            # print(f"GPU #{self.gpu_idx}, before job has {torch.cuda.memory_allocated(self.my_device):,} bytes allocated")
            stats = self.do_one_job(
                lhs_partitioned_types=job.lhs_partitioned_types,
                rhs_partitioned_types=job.rhs_partitioned_types,
                lhs_part=job.lhs_part,
                rhs_part=job.rhs_part,
                lhs_subpart=job.lhs_subpart,
                rhs_subpart=job.rhs_subpart,
                model=job.model,
                trainer=job.trainer,
                holder=job.holder,
                subpart_slices=job.subpart_slices,
                rel=job.rel,
                lhs_subparts=job.lhs_subparts,
                rhs_subparts=job.rhs_subparts,
                lhs_offsets=job.lhs_offsets,
                rhs_offsets=job.rhs_offsets,
                batch_size=job.batch_size,
                lr=job.lr,
            )
            # del job
            # torch.cuda.synchronize(self.my_device)
            # gc.collect()
            # torch.cuda.empty_cache()
            # torch.cuda.synchronize(self.my_device)
            # print(f"GPU #{self.gpu_idx}, after job has {torch.cuda.memory_allocated(self.my_device):,} bytes allocated")

            self.worker_endpoint.send(SubprocessReturn(gpu_idx=self.gpu_idx, stats=stats))

    def do_one_job(
        self,
        lhs_partitioned_types: Set[str],
        rhs_partitioned_types: Set[str],
        lhs_part: Partition,
        rhs_part: Partition,
        lhs_subpart: SubPartition,
        rhs_subpart: SubPartition,
        model: MultiRelationEmbedder,
        trainer: Trainer,
        holder: EmbeddingHolder,
        subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice],
        rel: LongTensorType,
        lhs_subparts: LongTensorType,
        rhs_subparts: LongTensorType,
        lhs_offsets: LongTensorType,
        rhs_offsets: LongTensorType,
        batch_size: int,
        lr: float,
    ) -> Stats:
        these_ptrs_and_sizes: Set[Tuple[int, int]] = set()
        for _, embeddings, _ in holder.values():
            these_ptrs_and_sizes.add((
                embeddings.data_ptr(),
                embeddings.numel() * embeddings.element_size(),
            ))

        for pyptr, pysize in self.pinned_ptrs_and_sizes - these_ptrs_and_sizes:
            cptr = ctypes.c_void_p(pyptr)
            csize = ctypes.c_size_t(pysize)
            cflags = ctypes.c_uint(0)
            res = torch.cuda.cudart().cudaHostUnregister(cptr, csize, cflags)
            torch.cuda.check_error(res)

        for pyptr, pysize in these_ptrs_and_sizes - self.pinned_ptrs_and_sizes:
            cptr = ctypes.c_void_p(pyptr)
            csize = ctypes.c_size_t(pysize)
            cflags = ctypes.c_uint(0)
            res = torch.cuda.cudart().cudaHostRegister(cptr, csize, cflags)
            torch.cuda.check_error(res)

        for _, embeddings, _ in holder.values():
            assert embeddings.is_pinned()

        self.pinned_ptrs_and_sizes = these_ptrs_and_sizes

        sub_holder: SubEmbeddingHolder = {}

        occurrences: Dict[Tuple[EntityName, Partition, SubPartition], Set[Side]] = defaultdict(set)
        for entity_name in lhs_partitioned_types:
            occurrences[entity_name, lhs_part, lhs_subpart].add(Side.LHS)
        for entity_name in rhs_partitioned_types:
            occurrences[entity_name, rhs_part, rhs_subpart].add(Side.RHS)

        for entity_name, part, subpart in occurrences.keys():
            _, embeddings, optimizer = holder[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            # TODO have two permanent storages on GPU and move stuff in and out from them
            # print(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * embeddings.shape[1] * 4:,} bytes", flush=True)
            gpu_embeddings = torch.empty(
                (subpart_slice.stop - subpart_slice.start, embeddings.shape[1]),
                dtype=torch.float32,
                device=self.my_device,
            )
            gpu_embeddings.copy_(embeddings[subpart_slice])
            gpu_embeddings = nn.Parameter(gpu_embeddings)
            gpu_optimizer = RowAdagrad([gpu_embeddings], lr=lr)
            cpu_state, = optimizer.state.values()
            gpu_state, = gpu_optimizer.state.values()
            # print(f"GPU #{self.gpu_idx} allocating {(subpart_slice.stop - subpart_slice.start) * 4:,} bytes", flush=True)
            gpu_state["sum"] = cpu_state["sum"][subpart_slice].to(self.my_device)

            sub_holder[entity_name, part, subpart] = (gpu_embeddings, gpu_optimizer)

        for (entity_name, part, subpart), (gpu_embeddings, gpu_optimizer) in sub_holder.items():
            for side in occurrences[entity_name, part, subpart]:
                model.set_embeddings(entity_name, gpu_embeddings, side)
                trainer.partitioned_entity_optimizers[entity_name, part, subpart] = gpu_optimizer

        lhs_mask = lhs_subparts.eq(lhs_subpart)
        rhs_mask = rhs_subparts.eq(rhs_subpart)
        this_indices = (lhs_mask & rhs_mask).nonzero().flatten()
        np.random.shuffle(this_indices.numpy())
        # print(f"GPU #{self.gpu_idx} allocating {this_indices.shape[0] * 3 * 8:,} bytes", flush=True)
        gpu_edges = EdgeList(
            EntityList.from_tensor(lhs_offsets[this_indices]),
            EntityList.from_tensor(rhs_offsets[this_indices]),
            rel[this_indices],
        ).to(self.my_device)
        print(f"GPU #{self.gpu_idx} got {this_indices.shape[0]} edges", flush=True)

        stats = process_in_batches(
            batch_size=batch_size,
            model=model,
            batch_processor=trainer,
            edges=gpu_edges,
        )

        for (entity_name, part, subpart), (gpu_embeddings, gpu_optimizer) in sub_holder.items():
            _, embeddings, optimizer = holder[entity_name, part]
            subpart_slice = subpart_slices[entity_name, part, subpart]

            embeddings[subpart_slice].copy_(gpu_embeddings.detach())
            del gpu_embeddings
            cpu_state, = optimizer.state.values()
            gpu_state, = gpu_optimizer.state.values()
            cpu_state["sum"][subpart_slice].copy_(gpu_state["sum"])
            del gpu_state["sum"]

        return stats


class GPUProcessPool:

    def __init__(self, num_gpus: int):
        self.processes: List[GPUProcess] = [
            GPUProcess(gpu_idx) for gpu_idx in range(num_gpus)]
        for p in self.processes:
            p.start()

    @property
    def num_gpus(self):
        return len(self.processes)

    def schedule(self, gpu_idx: GPURank, args: SubprocessArgs) -> None:
        self.processes[gpu_idx].master_endpoint.send(args)

    def wait(self) -> Generator[Tuple[GPURank, SubprocessReturn], None, None]:
        all_objects = [p.sentinel for p in self.processes] + [p.master_endpoint for p in self.processes]
        ready_objects = mp_wait(all_objects)
        for obj in ready_objects:
            for p in self.processes:
                if obj is p.sentinel:
                    raise RuntimeError(
                        f"GPU worker #{p.gpu_idx} (PID: {p.pid}) terminated "
                        f"unexpectedly with exit code {p.exitcode}")  # @nocommit exitcode is still None at this time
                if obj is p.master_endpoint:
                    res = p.master_endpoint.recv()
                    yield p.gpu_idx, res

    def join(self):
        for p in self.processes:
            p.master_endpoint.send(None)  # This shouldn't be necessary @nocommit
            p.master_endpoint.close()
            p.join()


class NothingToAcquire(Exception):
    pass


class Locker:

    def __init__(self, num_lhs: int, num_rhs: int) -> None:
        self.num_lhs = num_lhs
        self.num_rhs = num_rhs
        self.locks: Dict[Rank, Tuple[SubPartition, SubPartition]] = {}
        self.prev_locked: Dict[Rank, Tuple[SubPartition, SubPartition]] = {}
        self.locked: Dict[SubPartition, Dict[Side, Rank]] = defaultdict(dict)
        self.done: Set[Tuple[SubPartition, SubPartition]] = set()

    def acquire(self, rank: Rank) -> Tuple[SubPartition, SubPartition]:
        if rank in self.locks:
            raise ValueError(f"Rank {rank} is already locking {self.locks[rank]}")
        to_do = {
            (lhs_subpart, rhs_subpart)
            for lhs_subpart in range(self.num_lhs)
            for rhs_subpart in range(self.num_rhs)
            if (lhs_subpart, rhs_subpart) not in self.done
            and len(self.locked[lhs_subpart]) == 0
            and len(self.locked[rhs_subpart]) == 0
        }
        if not to_do:
            raise NothingToAcquire()
        prev_b = set(self.prev_locked.get(rank, []))
        lhs_subpart, rhs_subpart = max(to_do, key=lambda sub_b: len(prev_b.intersection(sub_b)))
        self.locks[rank] = (lhs_subpart, rhs_subpart)
        self.locked[lhs_subpart][Side.LHS] = rank
        self.locked[rhs_subpart][Side.RHS] = rank
        return lhs_subpart, rhs_subpart

    def release(self, rank: Rank) -> None:
        try:
            lhs_subpart, rhs_subpart = self.locks[rank]
        except KeyError:
            raise ValueError(f"Rank {rank} isn't locking anything") from None
        del self.locks[rank]
        del self.locked[lhs_subpart][Side.LHS]
        del self.locked[rhs_subpart][Side.RHS]
        self.done.add((lhs_subpart, rhs_subpart))
        self.prev_locked[rank] = (lhs_subpart, rhs_subpart)


def train_and_report_stats(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    trainer: Optional[AbstractBatchProcessor] = None,
    rank: Rank = RANK_ZERO,
) -> Generator[Tuple[int, Optional[Stats], Stats, Optional[Stats]], None, None]:
    """Each epoch/pass, for each partition pair, loads in embeddings and edgelist
    from disk, runs HOGWILD training on them, and writes partitions back to disk.
    """

    if config.verbose > 0:
        import pprint
        pprint.PrettyPrinter().pprint(config.to_dict())

    log("Loading entity counts...")
    if maybe_old_entity_path(config.entity_path):
        log("WARNING: It may be that your entity path contains files using the "
            "old format. See D14241362 for how to update them.")
    entity_counts: Dict[str, List[int]] = {}
    for entity, econf in config.entities.items():
        entity_counts[entity] = []
        for part in range(econf.num_partitions):
            with open(os.path.join(
                config.entity_path, "entity_count_%s_%d.txt" % (entity, part)
            ), "rt") as tf:
                entity_counts[entity].append(int(tf.read().strip()))

    # Figure out how many lhs and rhs partitions we need
    nparts_lhs, lhs_partitioned_types = get_partitioned_types(config, Side.LHS)
    nparts_rhs, rhs_partitioned_types = get_partitioned_types(config, Side.RHS)
    lhs_partitioned_types.update(config.entities.keys())
    rhs_partitioned_types.update(config.entities.keys())
    vlog("nparts %d %d types %s %s" %
         (nparts_lhs, nparts_rhs, lhs_partitioned_types, rhs_partitioned_types))
    total_buckets = nparts_lhs * nparts_rhs
    num_subparts = config.num_sub_partitions

    sync: AbstractSynchronizer
    bucket_scheduler: AbstractBucketScheduler
    parameter_sharer: Optional[ParameterSharer]
    partition_client: Optional[PartitionClient]
    if config.num_machines > 1:
        if not 0 <= rank < config.num_machines:
            raise RuntimeError("Invalid rank for trainer")
        if not td.is_available():
            raise RuntimeError("The installed PyTorch version doesn't provide "
                               "distributed training capabilities.")
        ranks = ProcessRanks.from_num_invocations(
            config.num_machines, config.num_partition_servers)

        if rank == RANK_ZERO:
            log("Setup lock server...")
            start_server(
                LockServer(
                    num_clients=len(ranks.trainers),
                    nparts_lhs=nparts_lhs,
                    nparts_rhs=nparts_rhs,
                    lock_lhs=len(lhs_partitioned_types) > 0,
                    lock_rhs=len(rhs_partitioned_types) > 0,
                    init_tree=config.distributed_tree_init_order,
                ),
                server_rank=ranks.lock_server,
                world_size=ranks.world_size,
                init_method=config.distributed_init_method,
                groups=[ranks.trainers],
            )

        bucket_scheduler = DistributedBucketScheduler(
            server_rank=ranks.lock_server,
            client_rank=ranks.trainers[rank],
        )

        log("Setup param server...")
        start_server(
            ParameterServer(num_clients=len(ranks.trainers)),
            server_rank=ranks.parameter_servers[rank],
            init_method=config.distributed_init_method,
            world_size=ranks.world_size,
            groups=[ranks.trainers],
        )

        parameter_sharer = ParameterSharer(
            client_rank=ranks.parameter_clients[rank],
            all_server_ranks=ranks.parameter_servers,
            init_method=config.distributed_init_method,
            world_size=ranks.world_size,
            groups=[ranks.trainers],
        )

        if config.num_partition_servers == -1:
            start_server(
                ParameterServer(num_clients=len(ranks.trainers)),
                server_rank=ranks.partition_servers[rank],
                world_size=ranks.world_size,
                init_method=config.distributed_init_method,
                groups=[ranks.trainers],
            )

        if len(ranks.partition_servers) > 0:
            partition_client = PartitionClient(ranks.partition_servers)
        else:
            partition_client = None

        groups = init_process_group(
            rank=ranks.trainers[rank],
            world_size=ranks.world_size,
            init_method=config.distributed_init_method,
            groups=[ranks.trainers],
        )
        trainer_group, = groups
        sync = DistributedSynchronizer(trainer_group)
        dlog = log

    else:
        sync = DummySynchronizer()
        bucket_scheduler = SingleMachineBucketScheduler(
            nparts_lhs, nparts_rhs, config.bucket_order)
        parameter_sharer = None
        partition_client = None
        dlog = lambda msg: None

    # fork early for HOGWILD threads
    log("Creating workers...")
    torch.set_num_threads(1)
    pool = GPUProcessPool(config.num_gpus)

    holder: EmbeddingHolder = {}

    def make_optimizer(params: Iterable[nn.Parameter], is_emb: bool) -> Optimizer:
        params = list(params)
        if len(params) == 0:
            optimizer = DummyOptimizer()
        elif is_emb:
            optimizer = RowAdagrad(params, lr=config.lr)
        else:
            if config.relation_lr is not None:
                lr = config.relation_lr
            else:
                lr = config.lr
            optimizer = Adagrad(params, lr=lr)
        optimizer.share_memory()
        return optimizer

    # background_io is only supported in single-machine mode
    background_io = config.background_io and config.num_machines == 1

    checkpoint_manager = CheckpointManager(
        config.checkpoint_path,
        background=background_io,
        rank=rank,
        num_machines=config.num_machines,
        partition_client=partition_client,
    )
    checkpoint_manager.register_metadata_provider(ConfigMetadataProvider(config))
    checkpoint_manager.write_config(config)

    iteration_manager = IterationManager(
        config.num_epochs, config.edge_paths, config.num_edge_chunks,
        iteration_idx=checkpoint_manager.checkpoint_version)
    checkpoint_manager.register_metadata_provider(iteration_manager)

    if config.init_path is not None:
        loadpath_manager = CheckpointManager(config.init_path)
    else:
        loadpath_manager = None

    def load_embeddings(
        entity: EntityName,
        part: Partition,
        strict: bool = False,
        force_dirty: bool = False,
    ) -> Tuple[nn.Parameter, Optional[OptimizerStateDict]]:
        if strict:
            embs, optim_state = checkpoint_manager.read(entity, part,
                                                        force_dirty=force_dirty)
        else:
            # Strict is only false during the first iteration, because in that
            # case the checkpoint may not contain any data (unless a previous
            # run was resumed) so we fall back on initial values.
            embs, optim_state = checkpoint_manager.maybe_read(entity, part,
                                                              force_dirty=force_dirty)
            if embs is None and loadpath_manager is not None:
                embs, optim_state = loadpath_manager.maybe_read(entity, part)
            if embs is None:
                embs, optim_state = init_embs(entity, entity_counts[entity][part],
                                              config.dimension, config.init_scale)
        assert embs.is_shared()
        return embs, optim_state

    log("Initializing global model...")

    if model is None:
        model = make_model(config)
    model.share_memory()
    for param in model.parameters(recurse=True):
        param.grad: torch.Tensor = param.new_zeros(param.shape)
        param.grad.share_memory_()
    if trainer is None:
        trainer = Trainer(
            global_optimizer=make_optimizer(model.parameters(), False),
            loss_fn=config.loss_fn,
            margin=config.margin,
            relations=config.relations,
        )

    state_dict, optim_state = checkpoint_manager.maybe_read_model()

    if state_dict is None and loadpath_manager is not None:
        state_dict, optim_state = loadpath_manager.maybe_read_model()
    if state_dict is not None:
        model.load_state_dict(state_dict, strict=False)
    if optim_state is not None:
        trainer.global_optimizer.load_state_dict(optim_state)

    # vlog("Loading unpartitioned entities...")
    # for entity, econfig in config.entities.items():
    #     if econfig.num_partitions == 1:
    #         embs, optim_state = load_embeddings(entity, Partition(0))
    #         embs = nn.Parameter(embs)
    #         model.set_embeddings(entity, embs, Side.LHS)
    #         model.set_embeddings(entity, embs, Side.RHS)
    #         optimizer = make_optimizer([embs], True)
    #         if optim_state is not None:
    #             optimizer.load_state_dict(optim_state)
    #         trainer.unpartitioned_entity_optimizers[entity] = optimizer

    # start communicating shared parameters with the parameter server
    if parameter_sharer is not None:
        parameter_sharer.share_model_params(model)

    strict = False

    def swap_partitioned_embeddings(
        old_b: Optional[Bucket],
        new_b: Optional[Bucket],
    ):
        # 0. given the old and new buckets, construct data structures to keep
        #    track of old and new embedding (entity, part) tuples

        io_bytes = 0
        log("Swapping partitioned embeddings %s %s" % (old_b, new_b))

        old_parts: Set[Tuple[EntityName, Partition]] = set()
        if old_b is not None:
            old_parts.update((e, old_b.lhs) for e in lhs_partitioned_types)
            old_parts.update((e, old_b.rhs) for e in rhs_partitioned_types)
        new_parts: Set[Tuple[EntityName, Partition]] = set()
        if new_b is not None:
            new_parts.update((e, new_b.lhs) for e in lhs_partitioned_types)
            new_parts.update((e, new_b.rhs) for e in rhs_partitioned_types)

        to_checkpoint = old_parts - new_parts
        to_load = new_parts - old_parts

        # 1. checkpoint embeddings that will not be used in the next pair
        #
        log("Writing partitioned embeddings")
        for entity, part in to_checkpoint:
            vlog(f"Checkpointing ({entity} {part})")

            perm, permed_embs_p, optimizer = holder[entity, part]
            optim_state = OptimizerStateDict(optimizer.state_dict())
            del holder[entity, part]

            permed_embs = permed_embs_p.detach()

            # @nocommit do this in-place
            rev_perm = torch.argsort(perm)
            embs = permed_embs.detach()[rev_perm]

            checkpoint_manager.write(entity, part, embs, optim_state)
            io_bytes += embs.numel() * embs.element_size()  # ignore optim state

            # these variables are holding large objects; let them be freed
            del embs
            del optimizer
            del optim_state

        if old_b is not None:  # there are previous embeddings to checkpoint
            bucket_scheduler.release_bucket(old_b)

        # 3. load new embeddings into the model/optimizer, either from disk
        #    or the temporary dictionary
        #
        log("Loading entities")
        for entity, part in to_load:
            vlog(f"Loading ({entity}, {part})")

            perm = torch.randperm(entity_counts[entity][part])
            force_dirty = bucket_scheduler.check_and_set_dirty(entity, part)
            embs, optim_state = load_embeddings(
                entity, part, strict=strict, force_dirty=force_dirty)

            # @nocommit do this in-place
            storage = torch.FloatStorage._new_shared(embs.numel())
            permed_embs = torch.FloatTensor(storage).view(embs.shape)
            torch.index_select(embs, 0, perm, out=permed_embs)

            permed_embs_p = nn.Parameter(permed_embs)

            optimizer = make_optimizer([embs], True)
            if optim_state is not None:
                vlog("Setting optim state")
                optimizer.load_state_dict(optim_state)
            io_bytes += embs.numel() * embs.element_size()  # ignore optim state

            holder[entity, part] = (perm, permed_embs_p, optimizer)

        return io_bytes

    # Start of the main training loop.
    for epoch_idx, edge_path_idx, edge_chunk_idx \
            in iteration_manager.remaining_iterations():
        log("Starting epoch %d / %d edge path %d / %d edge chunk %d / %d" %
            (epoch_idx + 1, iteration_manager.num_epochs,
             edge_path_idx + 1, iteration_manager.num_edge_paths,
             edge_chunk_idx + 1, iteration_manager.num_edge_chunks))
        edge_reader = EdgeReader(iteration_manager.edge_path)
        log("edge_path= %s" % iteration_manager.edge_path)

        sync.barrier()
        dlog("Lock client new epoch...")
        bucket_scheduler.new_pass(is_first=iteration_manager.iteration_idx == 0)
        sync.barrier()

        remaining = total_buckets
        cur_b = None
        while remaining > 0:
            old_b = cur_b
            io_time = 0.
            io_bytes = 0
            cur_b, remaining = bucket_scheduler.acquire_bucket()
            print('still in queue: %d' % remaining, file=sys.stderr)
            if cur_b is None:
                if old_b is not None:
                    # if you couldn't get a new pair, release the lock
                    # to prevent a deadlock!
                    tic = time.time()
                    io_bytes += swap_partitioned_embeddings(old_b, None)
                    io_time += time.time() - tic
                time.sleep(1)  # don't hammer td
                continue

            def log_status(msg, always=False):
                f = log if always else vlog
                f("%s: %s" % (cur_b, msg))

            tic = time.time()

            io_bytes += swap_partitioned_embeddings(old_b, cur_b)

            current_index = \
                (iteration_manager.iteration_idx + 1) * total_buckets - remaining

            next_b = bucket_scheduler.peek()
            if next_b is not None and background_io:
                # Ensure the previous bucket finished writing to disk.
                checkpoint_manager.wait_for_marker(current_index - 1)

                log_status("Prefetching")
                for entity in lhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_b.lhs)
                for entity in rhs_partitioned_types:
                    checkpoint_manager.prefetch(entity, next_b.rhs)

                checkpoint_manager.record_marker(current_index)

            log_status("Loading edges")
            edges = edge_reader.read(
                cur_b.lhs, cur_b.rhs, edge_chunk_idx, config.num_edge_chunks)
            num_edges = len(edges)
            # this might be off in the case of tensorlist or extra edge fields
            io_bytes += edges.lhs.tensor.numel() * edges.lhs.tensor.element_size()
            io_bytes += edges.rhs.tensor.numel() * edges.rhs.tensor.element_size()
            io_bytes += edges.rel.numel() * edges.rel.element_size()

            print("Done reading")

            offset_to_subpart_map: Dict[Tuple[EntityName, Partition], LongTensorType] = {}
            subpart_slices: Dict[Tuple[EntityName, Partition, SubPartition], slice] = {}
            offset_to_suboffset_map: Dict[Tuple[EntityName, Partition], LongTensorType] = {}
            for (entity_name, part), (perm, embeddings, optimizer) in holder.items():
                num_entities, _ = embeddings.shape
                state, = optimizer.state.values()
                state["sum"] = state["sum"][perm]
                offset_to_subpart_map[entity_name, part] = torch.empty((num_entities,), dtype=torch.long)
                offset_to_suboffset_map[entity_name, part] = torch.empty((num_entities,), dtype=torch.long)
                for subpart, subpart_slice in enumerate(split_almost_equally(num_entities, num_parts=num_subparts)):
                    offset_to_subpart_map[entity_name, part][perm[subpart_slice]] = subpart
                    subpart_slices[entity_name, part, subpart] = subpart_slice
                    offset_to_suboffset_map[entity_name, part][perm[subpart_slice]] = \
                        torch.arange(subpart_slice.stop - subpart_slice.start)

            print("Done subpartitioning entities")
            print(subpart_slices)
            print([(x.min(), x.max()) for x in offset_to_subpart_map.values()])
            print([(x.min(), x.max()) for x in offset_to_suboffset_map.values()])

            # FIXME only supports non-featurized and non-unpartitioned entity types
            # FIXME consider masked_scatter and masked_fill and masked_select

            lhs_subparts_storage = torch.LongStorage._new_shared(num_edges)
            lhs_subparts = torch.LongTensor(lhs_subparts_storage).view((num_edges,))
            rhs_subparts_storage = torch.LongStorage._new_shared(num_edges)
            rhs_subparts = torch.LongTensor(rhs_subparts_storage).view((num_edges,))
            lhs_offsets_storage = torch.LongStorage._new_shared(num_edges)
            lhs_offsets = torch.LongTensor(lhs_offsets_storage).view((num_edges,))
            rhs_offsets_storage = torch.LongStorage._new_shared(num_edges)
            rhs_offsets = torch.LongTensor(rhs_offsets_storage).view((num_edges,))
            if config.dynamic_relations:
                rel_config, = config.relations
                lhs_subparts[:] = offset_to_subpart_map[rel_config.lhs, cur_b.lhs][edges.lhs.tensor]
                lhs_offsets[:] = offset_to_suboffset_map[rel_config.lhs, cur_b.lhs][edges.lhs.tensor]
                rhs_subparts[:] = offset_to_subpart_map[rel_config.rhs, cur_b.rhs][edges.rhs.tensor]
                rhs_offsets[:] = offset_to_suboffset_map[rel_config.rhs, cur_b.rhs][edges.rhs.tensor]
            else:
                for rel_idx, rel_config in enumerate(config.relations):
                    this_indices = edges.rel.eq(rel_idx).nonzero().flatten()
                    this_edges = edges[this_indices]
                    lhs_subparts[this_indices] = offset_to_subpart_map[rel_config.lhs, cur_b.lhs][this_edges.lhs.tensor]
                    lhs_offsets[this_indices] = offset_to_suboffset_map[rel_config.lhs, cur_b.lhs][this_edges.lhs.tensor]
                    rhs_subparts[this_indices] = offset_to_subpart_map[rel_config.rhs, cur_b.rhs][this_edges.rhs.tensor]
                    rhs_offsets[this_indices] = offset_to_suboffset_map[rel_config.rhs, cur_b.rhs][this_edges.rhs.tensor]

            rel = edges.rel
            del edges

            print("Done mapping edges to subpartitions")

            io_time += time.time() - tic
            tic = time.time()

            locker = Locker(num_lhs=num_subparts, num_rhs=num_subparts)

            def schedule(gpu_idx: GPURank, lhs_subpart: SubPartition, rhs_subpart: SubPartition) -> None:
                print(f"GPU #{gpu_idx} gets {lhs_subpart}, {rhs_subpart}")
                pool.schedule(gpu_idx, SubprocessArgs(
                    lhs_partitioned_types=lhs_partitioned_types,
                    rhs_partitioned_types=rhs_partitioned_types,
                    lhs_part=cur_b.lhs,
                    rhs_part=cur_b.rhs,
                    lhs_subpart=lhs_subpart,
                    rhs_subpart=rhs_subpart,
                    trainer=trainer,
                    model=model,
                    holder=holder,
                    subpart_slices=subpart_slices,
                    rel=rel,
                    lhs_subparts=lhs_subparts,
                    rhs_subparts=rhs_subparts,
                    lhs_offsets=lhs_offsets,
                    rhs_offsets=rhs_offsets,
                    batch_size=config.batch_size,
                    lr=config.lr,
                ))

            busy_gpus: Set[int] = set()
            for gpu_idx in range(pool.num_gpus):
                try:
                    lhs_subpart, rhs_subpart = locker.acquire(gpu_idx)
                except NothingToAcquire:
                    print(f"{num_subparts} sub-partitions aren't enough to "
                          f"fully utilize all {pool.num_gpus} GPUs: GPU "
                          f"#{gpu_idx} and later will be idle")
                    break
                else:
                    schedule(gpu_idx, lhs_subpart, rhs_subpart)
                    busy_gpus.add(gpu_idx)

            all_stats: List[Stats] = []
            while busy_gpus:
                for gpu_idx, result in pool.wait():
                    assert gpu_idx == result.gpu_idx
                    all_stats.append(result.stats)
                    locker.release(gpu_idx)
                    busy_gpus.remove(gpu_idx)
                for gpu_idx in range(config.num_gpus):
                    if gpu_idx not in busy_gpus:
                        try:
                            lhs_subpart, rhs_subpart = locker.acquire(gpu_idx)
                        except NothingToAcquire:
                            pass
                        else:
                            schedule(gpu_idx, lhs_subpart, rhs_subpart)
                            busy_gpus.add(gpu_idx)
            assert len(locker.done) == num_subparts * num_subparts

            stats = Stats.sum(all_stats).average()
            compute_time = time.time() - tic

            log_status(
                "bucket %d / %d : Processed %d edges in %.2f s "
                "( %.2g M/sec ); io: %.2f s ( %.2f MB/sec )" %
                (total_buckets - remaining, total_buckets,
                 num_edges, compute_time, num_edges / compute_time / 1e6,
                 io_time, io_bytes / io_time / 1e6),
                always=True)
            log_status("%s" % stats, always=True)

            # Add train/eval metrics to queue
            yield current_index, None, stats, None

        swap_partitioned_embeddings(cur_b, None)

        # Distributed Processing: all machines can leave the barrier now.
        sync.barrier()

        # Write metadata: for multiple machines, write from rank-0
        log("Finished epoch %d path %d pass %d; checkpointing global state."
            % (epoch_idx + 1, edge_path_idx + 1, edge_chunk_idx + 1))
        log("My rank: %d" % rank)
        if rank == 0:
            # for entity, econfig in config.entities.items():
            #     if econfig.num_partitions == 1:
            #         embs = model.get_embeddings(entity, Side.LHS)
            #         optimizer = trainer.unpartitioned_entity_optimizers[entity]
            #
            #         checkpoint_manager.write(
            #             entity, Partition(0),
            #             embs.detach(), OptimizerStateDict(optimizer.state_dict()))

            sanitized_state_dict: ModuleStateDict = {}
            for k, v in ModuleStateDict(model.state_dict()).items():
                if k.startswith('lhs_embs') or k.startswith('rhs_embs'):
                    # skipping state that's an entity embedding
                    continue
                sanitized_state_dict[k] = v

            log("Writing metadata...")
            checkpoint_manager.write_model(
                sanitized_state_dict,
                OptimizerStateDict(trainer.global_optimizer.state_dict()),
            )

        log("Writing the checkpoint...")
        checkpoint_manager.write_new_version(config)

        dlog("Waiting for other workers to write their parts of the checkpoint: rank %d" % rank)
        sync.barrier()
        dlog("All parts of the checkpoint have been written")

        log("Switching to new checkpoint version...")
        checkpoint_manager.switch_to_new_version()

        dlog("Waiting for other workers to switch to the new checkpoint version: rank %d" % rank)
        sync.barrier()
        dlog("All workers have switched to the new checkpoint version")

        # After all the machines have finished committing
        # checkpoints, we remove the old checkpoints.
        checkpoint_manager.remove_old_version(config)

        # now we're sure that all partition files exist,
        # so be strict about loading them
        strict = True

    # quiescence
    pool.join()

    sync.barrier()

    checkpoint_manager.close()
    if loadpath_manager is not None:
        loadpath_manager.close()

    # FIXME join distributed workers (not really necessary)

    log("Exiting")


def train(
    config: ConfigSchema,
    model: Optional[MultiRelationEmbedder] = None,
    trainer: Optional[AbstractBatchProcessor] = None,
    rank: Rank = RANK_ZERO,
) -> None:
    # Create and run the generator until exhaustion.
    for _ in train_and_report_stats(config, model, trainer, rank):
        pass


def main():
    config_help = '\n\nConfig parameters:\n\n' + '\n'.join(ConfigSchema.help())
    parser = argparse.ArgumentParser(
        epilog=config_help,
        # Needed to preserve line wraps in epilog.
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('config', help="Path to config file")
    parser.add_argument('-p', '--param', action='append', nargs='*')
    parser.add_argument('--rank', type=int, default=0,
                        help="For multi-machine, this machine's rank")
    opt = parser.parse_args()

    if opt.param is not None:
        overrides = chain.from_iterable(opt.param)  # flatten
    else:
        overrides = None
    config = parse_config(opt.config, overrides)

    train(config, rank=Rank(opt.rank))


if __name__ == '__main__':
    main()
