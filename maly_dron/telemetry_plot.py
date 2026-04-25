#!/usr/bin/env python3
"""Plot all telemetry columns from a drone CSV log and save as PNGs."""

import sys
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- config ---
GROUPS = {
    "attitude_deg": ["roll_deg", "pitch_deg", "yaw_deg"],
    "angular_rate_dps": ["rollspeed_dps", "pitchspeed_dps", "yawspeed_dps"],
    "rc_channels": [
        "ch1_roll",
        "ch2_pitch",
        "ch3_throttle",
        "ch4_yaw",
        "ch5",
        "ch6",
        "ch7",
        "ch8",
    ],
    "armed": ["armed"],
}


def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["t"] = (df["timestamp"] - df["timestamp"].iloc[0]).dt.total_seconds()
    return df


def plot_group(df: pd.DataFrame, name: str, cols: list[str], out_dir: Path):
    present = [c for c in cols if c in df.columns]
    if not present:
        print(f"  skip {name}: no cols found")
        return

    fig, axes = plt.subplots(
        len(present), 1, figsize=(12, 3 * len(present)), sharex=True
    )
    if len(present) == 1:
        axes = [axes]

    for ax, col in zip(axes, present):
        ax.plot(df["t"], df[col], linewidth=0.8)
        ax.set_ylabel(col, fontsize=8)
        ax.grid(True, linestyle="--", alpha=0.5)
        # armed overlay
        if "armed" in df.columns and col != "armed":
            for _, seg in df.groupby((df["armed"].diff().fillna(0) != 0).cumsum()):
                if seg["armed"].iloc[0]:
                    ax.axvspan(
                        seg["t"].iloc[0],
                        seg["t"].iloc[-1],
                        color="red",
                        alpha=0.08,
                        label="_armed",
                    )

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(name.replace("_", " ").title(), fontsize=11, y=1.01)
    fig.tight_layout()

    out = out_dir / f"{name}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {out}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_telemetry.py <log.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    out_dir = (
        Path(sys.argv[2]) if len(sys.argv) > 2 else Path(csv_path).parent / "plots"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path} ...")
    df = load(csv_path)
    print(f"  {len(df)} rows | {df['t'].iloc[-1]:.1f}s duration")

    for name, cols in GROUPS.items():
        plot_group(df, name, cols, out_dir)

    print(f"\nDone. Plots in: {out_dir}")


if __name__ == "__main__":
    main()
