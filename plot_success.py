from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def main():
    csv_path = Path("logs/episode_metrics.csv")
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    if df.empty:
        print("CSV is empty, nothing to plot yet.")
        return

    # Episode-based
    plt.figure()
    plt.plot(df["episode"], df["rolling_sr"])
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success Rate")
    plt.title("Rolling Success Rate (window=100)")
    plt.grid(alpha=0.3)

    out_path = csv_path.parent / "success_rate.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Timesteps-based (optional but useful)
    plt.figure()
    plt.plot(df["timesteps"], df["rolling_sr"])
    plt.xlabel("Timesteps")
    plt.ylabel("Rolling Success Rate")
    plt.title("Rolling Success Rate vs Timesteps")
    plt.grid(alpha=0.3)

    out_path2 = csv_path.parent / "success_rate_vs_steps.png"
    plt.savefig(out_path2, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    main()
