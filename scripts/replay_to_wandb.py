"""Replay GraspNet training scalars from tfevents into a Weights & Biases run.

The training already happened; its scalars live in the ``train/`` and
``test/`` tfevents. This re-logs them to a wandb run so the curves can be
viewed in wandb. Defaults to offline mode (no upload, no account needed).

To get the curves onto wandb.ai, prefer an ONLINE direct run after
``wandb login``:
    WANDB_MODE=online python scripts/replay_to_wandb.py
NOTE: ``wandb sync`` of an OFFLINE run can fail with a wandb backend error
(HTTP 403 "storage.objects.delete access denied" while overwriting
wandb-metadata.json). Running online from the start avoids that path and
uploaded cleanly (verified). Use online mode to populate the dashboard.
"""

import os
import glob
import argparse
from collections import defaultdict

import wandb
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


def load_all_scalars(events_dir, prefix):
    """Return {f'{prefix}{tag}': [(step, value), ...]} for a tfevents dir."""
    out = {}
    files = sorted(glob.glob(os.path.join(events_dir, "events.*")))
    if not files:
        return out
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    for tag in acc.Tags().get("scalars", []):
        out[f"{prefix}{tag}"] = [(p.step, p.value) for p in acc.Scalars(tag)]
    return out


def main():
    """Build the wandb run by replaying the tfevents scalars."""
    parser = argparse.ArgumentParser(description="Replay tfevents to wandb")
    parser.add_argument(
        "--logdir", default="/workspace/tools/runs/full_scratch_realsense"
    )
    parser.add_argument("--project", default="graspnet-full-scratch")
    parser.add_argument("--name", default="full_scratch_realsense")
    parser.add_argument("--train-samples", type=int, default=25600)
    args = parser.parse_args()

    run = wandb.init(
        project=args.project,
        name=args.name,
        config={
            "mode": "scratch",
            "camera": "realsense",
            "batch_size": 4,
            "num_point": 20000,
            "max_epoch": 18,
            "learning_rate": 0.001,
            "lr_decay_steps": "8,12,16",
            "train_scenes": 100,
            "source": "replayed from tfevents",
        },
    )

    # Make epoch the natural x-axis for every metric.
    wandb.define_metric("epoch")
    wandb.define_metric("train/*", step_metric="epoch")
    wandb.define_metric("eval/*", step_metric="epoch")

    series = {}
    series.update(load_all_scalars(os.path.join(args.logdir, "train"), "train/"))
    series.update(load_all_scalars(os.path.join(args.logdir, "test"), "eval/"))

    # Merge points by global step so each wandb.log call is monotonic.
    merged = defaultdict(dict)
    for name, points in series.items():
        for step, value in points:
            merged[step][name] = value

    for step in sorted(merged):
        data = dict(merged[step])
        data["epoch"] = step / args.train_samples
        wandb.log(data, step=step)

    print(f"replayed {len(series)} scalar series, {len(merged)} steps")
    print(f"wandb run dir: {run.dir}")
    wandb.finish()


if __name__ == "__main__":
    main()
