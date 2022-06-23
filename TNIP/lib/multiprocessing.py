# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

"""Multiprocessing helpers."""

import torch
import torch.distributed as dist


def run(
    local_rank,
    num_proc,
    func,
    init_method,
    shard_id,
    num_shards,
    backend,
    args,
    output_queue=None,
):
    """
    Runs a function from a child process.
    Args:
        local_rank (int): rank of the current process on the current machine.
        num_proc (int): number of processes per machine.
        func (function): function to execute on each of the process.
        init_method (string): method to initialize the distributed training.
            TCP initialization: equiring a network address reachable from all
            processes followed by the port.
            Shared file-system initialization: makes use of a file system that
            is shared and visible from all machines. The URL should start with
            file:// and contain a path to a non-existent file on a shared file
            system.
        shard_id (int): the rank of the current machine.
        num_shards (int): number of overall machines for the distributed
            training job.
        backend (string): three distributed backends ('nccl', 'gloo', 'mpi') are
            supports, each with different capabilities. Details can be found
            here:
            https://pytorch.org/docs/stable/distributed.html
        cfg (CfgNode): configs. Details can be found in
            slowfast/config/defaults.py
        output_queue (queue): can optionally be used to return values from the
            master process.
    """
    # Initialize the process group.
    world_size = num_proc * num_shards
    rank = shard_id * num_proc + local_rank

    try:
        torch.distributed.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
    except Exception as e:
        raise e

    args.world_size = world_size
    args.rank = rank

    torch.cuda.set_device(local_rank)
    ret = func(args)
    if output_queue is not None and local_rank == 0:
        output_queue.put(ret)

def launch_job(func, args, init_method="tcp://localhost:9999", daemon=False, SHARD_ID=0, NUM_SHARDS=1, DIST_BACKEND="nccl"):
    """
    Run 'func' on one or more GPUs, specified in cfg
    Args:
        cfg (CfgNode): configs. Details can be found in
            lib/config/defaults.py
        init_method (str): initialization method to launch the job with multiple
            devices.
        func (function): job to run on GPU(s)
        daemon (bool): The spawned processes’ daemon flag. If set to True,
            daemonic processes will be created
    """
    args.num_gpus = len(args.gpu.split(","))

    if args.num_gpus > 1:
        torch.multiprocessing.spawn(
            run,
            nprocs=args.num_gpus,
            args=(
                args.num_gpus,
                func,
                init_method,
                SHARD_ID,
                NUM_SHARDS,
                DIST_BACKEND,
                args,
            ),
            daemon=daemon,
        )
    else:
        args.world_size = 1
        args.rank = 1
        func(args)


def all_gather(tensors):
    """
    All gathers the provided tensors from all processes across machines.
    Args:
        tensors (list): tensors to perform all gather across all processes in
        all machines.
    """

    gather_list = []
    output_tensor = []
    world_size = dist.get_world_size()
    for tensor in tensors:
        tensor_placeholder = [
            torch.ones_like(tensor) for _ in range(world_size)
        ]
        dist.all_gather(tensor_placeholder, tensor, async_op=False)
        gather_list.append(tensor_placeholder)
    for gathered_tensor in gather_list:
        output_tensor.append(torch.cat(gathered_tensor, dim=0))
    return output_tensor