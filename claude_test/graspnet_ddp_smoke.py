# Synthetic multi-GPU smoke test for the GraspNet baseline (fork).
# Purpose: decide DataParallel vs DDP by exercising the pointnet2/knn
# custom CUDA ops under each parallel mode WITHOUT the form-gated dataset.
# Feeds random point clouds through GraspNet(is_training=False) and runs a
# forward + dummy backward on 1 GPU, on DataParallel, and on DDP.
# Lifetime: throwaway diagnostic; results recorded in claude_test/README.md.

import os
import sys
import argparse
import traceback

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel, DistributedDataParallel

REPO = "/workspace/tools/graspnet"
REPO_PARENT = os.path.dirname(REPO)

# The fork mixes import styles: models/* use package-relative imports
# (..pointnet2), while pointnet2/* use flat absolute imports (import
# pointnet2_utils). So we import the repo AS the `graspnet` package (parent on
# path) and also expose the pointnet2/utils dirs for the flat imports. Done at
# module top level so spawned DDP workers inherit it on re-import.
for _p in (REPO_PARENT, os.path.join(REPO, "pointnet2"), os.path.join(REPO, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# The fork's graspnet/__init__.py eagerly imports graspnet_baseline, which
# pulls open3d + graspnetAPI (inference/demo deps we do not need for training).
# Register `graspnet` as a namespace package pointing at the repo so the heavy
# __init__ is skipped; submodule __init__ files (models/utils) are empty.
import types  # noqa: E402

if "graspnet" not in sys.modules:
    _pkg = types.ModuleType("graspnet")
    _pkg.__path__ = [REPO]
    sys.modules["graspnet"] = _pkg


def build_inputs(batch_size, n_point, device):
    # GraspNet stage 1 only needs the raw point cloud (B, N, 3).
    point_clouds = torch.randn(batch_size, n_point, 3, device=device)
    return {"point_clouds": point_clouds}


def make_model():
    from graspnet.models.graspnet import GraspNet

    # Inference path is self-contained from point_clouds and still runs the
    # pointnet2 (FPS/ball-query) and knn (CloudCrop) custom ops.
    return GraspNet(is_training=False)


def dummy_loss(end_points):
    # Sum a differentiable head output so we can exercise backward.
    return end_points["grasp_score_pred"].float().sum()


def single_gpu(n_point):
    torch.cuda.set_device(0)
    net = make_model().cuda()
    ep = build_inputs(1, n_point, "cuda:0")
    out = net(ep)
    loss = dummy_loss(out)
    loss.backward()
    torch.cuda.synchronize()
    mem = torch.cuda.max_memory_allocated() / 1024**3
    return {"keys": len(out), "loss": float(loss), "peak_gb": round(mem, 2)}


def data_parallel(n_point, world):
    net = DataParallel(make_model().cuda(), device_ids=list(range(world)))
    # DataParallel splits dim 0 across GPUs, so batch must be >= world.
    ep = build_inputs(world, n_point, "cuda:0")
    out = net(ep)
    loss = dummy_loss(out)
    loss.backward()
    torch.cuda.synchronize()
    return {"world": world, "loss": float(loss)}


def ddp_worker(rank, world, n_point, ret):
    try:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        dist.init_process_group("nccl", rank=rank, world_size=world)
        torch.cuda.set_device(rank)
        net = make_model().to(rank)
        net = DistributedDataParallel(
            net, device_ids=[rank], find_unused_parameters=True
        )
        ep = build_inputs(1, n_point, f"cuda:{rank}")
        out = net(ep)
        loss = dummy_loss(out)
        loss.backward()
        torch.cuda.synchronize()
        if rank == 0:
            ret["ok"] = True
            ret["loss"] = float(loss)
        dist.destroy_process_group()
    except Exception as exc:  # noqa: BLE001
        if rank == 0:
            ret["ok"] = False
            ret["err"] = repr(exc)
            ret["tb"] = traceback.format_exc()


def stage(name, fn):
    print(f"\n===== {name} =====")
    try:
        result = fn()
        print(f"PASS {name}: {result}")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"FAIL {name}: {exc!r}")
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_point", type=int, default=20000)
    args = parser.parse_args()

    world = torch.cuda.device_count()
    print(f"torch {torch.__version__} | GPUs visible: {world}")
    print(f"arch_list: {torch.cuda.get_arch_list()}")

    ok_single = stage("single-GPU forward+backward", lambda: single_gpu(args.num_point))

    ok_dp = False
    if world > 1:
        ok_dp = stage(
            "DataParallel forward+backward",
            lambda: data_parallel(args.num_point, world),
        )
    else:
        print("\n(skip DataParallel: <2 GPUs)")

    ok_ddp = False
    if world > 1:
        print(f"\n===== DDP forward+backward (world={world}) =====")
        manager = mp.Manager()
        ret = manager.dict()
        mp.spawn(
            ddp_worker,
            args=(world, args.num_point, ret),
            nprocs=world,
            join=True,
        )
        ok_ddp = bool(ret.get("ok"))
        if ok_ddp:
            print(f"PASS DDP: loss={ret.get('loss')}")
        else:
            print(f"FAIL DDP: {ret.get('err')}")
            print(ret.get("tb", ""))
    else:
        print("\n(skip DDP: <2 GPUs)")

    print("\n===== VERDICT =====")
    print(f"single-GPU : {'OK' if ok_single else 'FAIL'}")
    print(f"DataParallel: {'OK' if ok_dp else 'FAIL'}")
    print(f"DDP        : {'OK' if ok_ddp else 'FAIL'}")


if __name__ == "__main__":
    main()
