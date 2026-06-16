"""Report-grade GraspNet train-loss figure from tfevents.

Plots ``train/loss/overall_loss`` vs epoch with large, readable fonts and
clear x/y/title labels, annotates the before/after loss values, and saves
both a vector PDF (infinitely sharp for print) and a 300-dpi PNG.
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

# Font sizes (larger than matplotlib defaults for readability in a report).
TITLE_FS = 22
LABEL_FS = 18
TICK_FS = 14
ANNOT_FS = 18


def load_scalar(events_dir, tag):
    """Return (steps, values) for ``tag`` from the tfevents in a directory."""
    files = sorted(glob.glob(os.path.join(events_dir, "events.*")))
    acc = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    acc.Reload()
    points = acc.Scalars(tag)
    return [p.step for p in points], [p.value for p in points]


def main():
    """Build and save the report-grade train-loss figure."""
    parser = argparse.ArgumentParser(description="Report train-loss figure")
    parser.add_argument(
        "--logdir", default="/workspace/tools/runs/full_scratch_realsense"
    )
    parser.add_argument("--train-samples", type=int, default=25600)
    parser.add_argument(
        "--out",
        default="/workspace/tools/runs/full_scratch_realsense/train_loss_report",
    )
    args = parser.parse_args()

    steps, vals = load_scalar(os.path.join(args.logdir, "train"), "loss/overall_loss")
    epochs = [s / args.train_samples for s in steps]
    y_start, y_end = vals[0], vals[-1]
    x_start, x_end = epochs[0], epochs[-1]

    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.plot(epochs, vals, color="tab:blue", linewidth=1.4)

    ax.set_title(
        "GraspNet training loss (scratch, 18 epochs)",
        fontsize=TITLE_FS,
        fontweight="bold",
        pad=14,
    )
    ax.set_xlabel("Epoch", fontsize=LABEL_FS, labelpad=10)
    ax.set_ylabel("Training loss (overall)", fontsize=LABEL_FS, labelpad=10)
    ax.tick_params(axis="both", labelsize=TICK_FS)
    ax.grid(True, alpha=0.3)

    # Mark + annotate the before/after loss at the same large font size.
    ax.scatter([x_start, x_end], [y_start, y_end], color="tab:red", zorder=5, s=60)
    ax.annotate(
        f"before: {y_start:.2f}",
        xy=(x_start, y_start),
        xytext=(x_start + 1.5, y_start - 0.25),
        fontsize=ANNOT_FS,
        color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red"),
    )
    ax.annotate(
        f"after: {y_end:.2f}",
        xy=(x_end, y_end),
        xytext=(x_end - 5.0, y_end + 0.30),
        fontsize=ANNOT_FS,
        color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red"),
    )

    fig.tight_layout()
    fig.savefig(args.out + ".pdf")  # vector, infinitely sharp
    fig.savefig(args.out + ".png", dpi=300)  # high-dpi raster
    print(f"saved: {args.out}.pdf and {args.out}.png")
    print(
        f"before={y_start:.3f} (epoch {x_start:.2f}), "
        f"after={y_end:.3f} (epoch {x_end:.2f})"
    )


if __name__ == "__main__":
    main()
