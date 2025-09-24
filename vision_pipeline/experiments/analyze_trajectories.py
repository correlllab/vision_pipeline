import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

experiment_dir = os.path.dirname(os.path.realpath(__file__))
csv_path = os.path.join(experiment_dir, "trajectories.csv")


def plot_errorbars(ax, trajectory_list, x, title, y_label, y_range_01=False):
    data = np.array(trajectory_list, dtype=float)
    n_trajectories, n_steps = data.shape
    mean_trajectory = np.mean(data, axis=0)
    min_trajectories = np.min(data, axis=0)
    max_trajectories = np.max(data, axis=0)

    # plot each trajectory
    for trajectory in data:
        ax.plot(x, trajectory, color="grey", alpha=0.5, linewidth=1)

    # error band
    ax.fill_between(x, min_trajectories, max_trajectories, color="lightblue", alpha=0.5)
    ax.plot(x, mean_trajectory, color="orange")

    ax.set_title(title)
    ax.set_xlabel("Timestep (K)")
    ax.set_ylabel(y_label)
    if y_range_01:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(bottom=0)

    return mean_trajectory


def main():
    if not os.path.exists(csv_path):
        # print(f"No trajectory data found at {csv_path}")
        return

    df = pd.read_csv(csv_path)

    # Count how many trajectories per (obj type, pc match method)
    counts = (
        df.groupby(["obj type", "pc match method"])
          .size()
          .reset_index(name="count")
    )
    print("\nTrajectory counts:")
    print(counts.to_string(index=False))

    # Loop through each group and plot
    for (obj_type, pc_match), group in df.groupby(
        ["obj type", "pc match method"]
    ):
        print(f"\nPlotting for {obj_type}, match={pc_match}")

        # Convert JSON strings to arrays
        belief_histories = [json.loads(b) for b in group["belief history"].dropna().values]
        n_points_histories = [json.loads(n) for n in group["n_points history"].dropna().values]

        belief_histories = np.array(belief_histories, dtype=float)
        n_points_histories = np.array(n_points_histories, dtype=float)

        # x-axis: timesteps
        k_range = np.arange(belief_histories.shape[1])

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Plot belief histories
        mean_belief = plot_errorbars(
            axes[0], belief_histories, k_range,
            f"{obj_type} | Belief", "Belief", y_range_01=True
        )

        # Plot point histories
        mean_points = plot_errorbars(
            axes[1], n_points_histories, k_range,
            f"{obj_type} | N Points", "N Points", y_range_01=False
        )

        # Compute final belief mean
        final_mean = mean_belief[-1]

        plt.suptitle(f"{obj_type} | match={pc_match} | final_mean={final_mean:.3f}")
        plt.tight_layout()

        # Save figure with descriptive name
        safe_obj = str(obj_type).replace(" ", "_")
        safe_match = str(pc_match).replace(" ", "_")
        fig_name = f"{safe_obj}_{safe_match}.png"
        fig_path = os.path.join(experiment_dir, fig_name)
        plt.savefig(fig_path, dpi=150)
        print(f"Saved plot -> {fig_path}")

        # Also display interactively
        plt.show(block=False)

    # Keep all plots open until user closes
    plt.show(block=True)


if __name__ == "__main__":
    main()
