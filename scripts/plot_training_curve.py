"""Plot GraspNet training curves from TensorBoard event files.

Reads the ``train/`` and ``test/`` tfevents written during training and
saves a PNG with (1) overall train vs eval loss and (2) the main train
sub-losses, both on an epoch x-axis. Useful for exporting/reporting the
run without launching TensorBoard.
"""

import os
import glob
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import (  # noqa: E402
    EventAccumulator,
)


def load_scalar(events_dir, tag):
    """Return (epochs, values) for ``tag`` in the tfevents at ``events_dir``.

    The x-axis is converted from global step to epoch units using
    ``train_samples`` (samples per epoch). Returns empty lists if the
    directory or tag is absent.
    """
    files = sorted(glob.glob(os.path.join(events_dir, "events.*")))
    if not files:
        return [], []
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    if tag not in acc.Tags().get("scalars", []):
        return [], []
    points = acc.Scalars(tag)
    return [p.step for p in points], [p.value for p in points]


def to_epoch(steps, train_samples):
    """Convert global steps to fractional epochs."""
    return [s / train_samples for s in steps]


def main():
    """Build and save the training-curve figure."""
    parser = argparse.ArgumentParser(description="Plot GraspNet training curves")
    parser.add_argument(
        "--logdir", default="/workspace/tools/runs/full_scratch_realsense"
    )
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--train-samples",
        type=int,
        default=25600,
        help="Samples per epoch (scenes x 256); used for the epoch axis.",
    )
    args = parser.parse_args()
    out = args.out or os.path.join(args.logdir, "training_curve.png")

    train_dir = os.path.join(args.logdir, "train")
    test_dir = os.path.join(args.logdir, "test")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Panel 1: overall loss, train vs eval.
    ts, tv = load_scalar(train_dir, "loss/overall_loss")
    es, ev = load_scalar(test_dir, "loss/overall_loss")
    if ts:
        ax1.plot(
            to_epoch(ts, args.train_samples),
            tv,
            label="train",
            color="tab:blue",
            linewidth=0.8,
            alpha=0.8,
        )
    if es:
        ax1.plot(
            to_epoch(es, args.train_samples),
            ev,
            label="eval (test_seen)",
            color="tab:red",
            marker="o",
            linewidth=1.5,
        )
    ax1.set_title("Overall loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Panel 2: main train sub-losses.
    sub_tags = [
        ("loss/stage1_objectness_loss", "objectness"),
        ("loss/stage1_view_loss", "view"),
        ("loss/stage2_grasp_score_loss", "grasp_score"),
        ("loss/stage2_grasp_angle_class_loss", "angle_class"),
        ("loss/stage2_grasp_width_loss", "width"),
        ("loss/stage2_grasp_tolerance_loss", "tolerance"),
    ]
    for tag, name in sub_tags:
        s, v = load_scalar(train_dir, tag)
        if s:
            ax2.plot(
                to_epoch(s, args.train_samples), v, label=name, linewidth=0.9, alpha=0.8
            )
    ax2.set_title("Train sub-losses")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    fig.suptitle("GraspNet full scratch training (18 epochs)")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved figure: {out}")
    if ts and es:
        print(f"train overall: {tv[0]:.3f} -> {tv[-1]:.3f}")
        print(f"eval  overall: {ev[0]:.3f} -> {ev[-1]:.3f}")


if __name__ == "__main__":
    main()
