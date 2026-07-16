import os
import json
import matplotlib.pyplot as plt
import numpy as np

def get_latest_run_id(sacred_dir):
    run_ids = []
    for d in os.listdir(sacred_dir):
        if d.isdigit():
            run_ids.append(int(d))
    if not run_ids:
        return None
        
    run_ids.sort(reverse=True)
    for run_id in run_ids:
        if os.path.exists(os.path.join(sacred_dir, str(run_id), "info.json")):
            return str(run_id)
            
    return None

def plot_metrics(sacred_dir="results/sacred"):
    latest_run = get_latest_run_id(sacred_dir)
    if not latest_run:
        print(f"No sacred runs found in {sacred_dir}")
        return

    info_path = os.path.join(sacred_dir, latest_run, "info.json")
    if not os.path.exists(info_path):
        print(f"info.json not found in {info_path}")
        return

    print(f"Loading metrics from run ID: {latest_run}")
    with open(info_path, 'r') as f:
        info = json.load(f)

    def extract(lst):
        return [item["value"] if isinstance(item, dict) else item for item in lst]

    fig, axs = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Mean Episode Reward (Return)
    has_reward = False
    if "return_mean" in info and "return_mean_T" in info:
        y_vals = extract(info["return_mean"])
        x_vals = extract(info["return_mean_T"])
        axs[0].plot(x_vals, y_vals, label="Train Return", color='blue', alpha=0.7)
        has_reward = True
    if "test_return_mean" in info and "test_return_mean_T" in info:
        y_vals = extract(info["test_return_mean"])
        x_vals = extract(info["test_return_mean_T"])
        axs[0].plot(x_vals, y_vals, label="Eval Return", color='orange', linewidth=2)
        has_reward = True
        
    if has_reward:
        axs[0].set_title("Mean Episode Reward")
        axs[0].set_xlabel("Timesteps")
        axs[0].set_ylabel("Reward")
        axs[0].legend()
        axs[0].grid(True, alpha=0.3)
    else:
        axs[0].set_title("Mean Episode Reward (No data)")

    # Plot 2: Win Rate (is_success)
    has_win_rate = False
    if "test_is_success_mean" in info and "test_is_success_mean_T" in info:
        # Multiply by 100 to get percentage like MAPPO
        y_vals = extract(info["test_is_success_mean"])
        x_vals = extract(info["test_is_success_mean_T"])
        win_rates = [v * 100 for v in y_vals]
        axs[1].plot(x_vals, win_rates, label="Eval Win Rate", color='green', linewidth=2)
        has_win_rate = True
        
    if has_win_rate:
        axs[1].set_title("Evaluation Win Rate (%)")
        axs[1].set_xlabel("Timesteps")
        axs[1].set_ylabel("Win Rate (%)")
        axs[1].legend()
        axs[1].grid(True, alpha=0.3)
        axs[1].set_ylim([-5, 105])
    else:
        axs[1].set_title("Win Rate (No data)")

    plt.tight_layout()
    output_path = f"results/plot_run_{latest_run}.png"
    plt.savefig(output_path)
    print(f"Plot saved successfully to {output_path}!")
    
    # Print the final stats to terminal
    print("\n--- Final Metrics ---")
    if "test_return_mean" in info and len(info["test_return_mean"]) > 0:
        val = extract(info["test_return_mean"])[-1]
        print(f"Final Eval Return: {val:.2f}")
    if "test_is_success_mean" in info and len(info["test_is_success_mean"]) > 0:
        val = extract(info["test_is_success_mean"])[-1]
        print(f"Final Eval Win Rate: {val*100:.2f}%")
        
if __name__ == "__main__":
    plot_metrics()
