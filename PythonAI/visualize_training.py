"""
Visualize RL Training Progress
Creates publication-ready plots from training logs
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import sys


def plot_training_progress(log_dir='logs', output_file='training_progress.png'):
    """
    Plot comprehensive training progress from CSV logs.
    
    Args:
        log_dir: Directory containing progress.csv
        output_file: Where to save the plot
    """
    
    # Find progress file
    progress_file = os.path.join(log_dir, 'progress.csv')
    
    if not os.path.exists(progress_file):
        print(f"❌ No progress file found at {progress_file}")
        print(f"\nLooked in: {os.path.abspath(log_dir)}")
        print("\nMake sure you've run training first:")
        print("  python train_agent.py --timesteps 50000")
        return False
    
    print(f"📊 Loading data from {progress_file}...")
    
    try:
        data = pd.read_csv(progress_file)
        print(f"✅ Loaded {len(data)} training steps")
    except Exception as e:
        print(f"❌ Failed to load CSV: {e}")
        return False
    
    # Check required columns
    required_cols = ['total_timesteps', 'rollout/ep_rew_mean', 
                     'rollout/ep_len_mean', 'train/policy_loss', 'train/value_loss']
    
    missing_cols = [col for col in required_cols if col not in data.columns]
    if missing_cols:
        print(f"⚠️ Missing columns: {missing_cols}")
        print(f"Available columns: {list(data.columns)}")
        return False
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Detective Game - RL Training Progress', 
                 fontsize=18, fontweight='bold', y=0.995)
    
    # Color scheme
    colors = {
        'reward': '#2E86AB',
        'length': '#A23B72', 
        'policy': '#F18F01',
        'value': '#C73E1D'
    }
    
    timesteps = data['total_timesteps']
    
    # Plot 1: Episode Reward (Top Left)
    ax1 = axes[0, 0]
    reward_mean = data['rollout/ep_rew_mean']
    
    ax1.plot(timesteps, reward_mean, 
             label='Average Reward', color=colors['reward'], linewidth=2.5, alpha=0.9)
    
    # Add smoothed trend line
    if len(reward_mean) > 10:
        # Use pandas rolling mean to smooth without scipy dependency
        window = min(20, max(3, len(reward_mean) // 10))
        smoothed = reward_mean.rolling(window=window, center=True, min_periods=1).mean()
        ax1.plot(timesteps, smoothed,
                 color='red', linewidth=2, linestyle='--', alpha=0.7, label='Smoothed Trend')
    
    # Shaded region for variance
    if len(reward_mean) > 1:
        window = min(20, max(3, len(reward_mean) // 10))
        rolling_std = reward_mean.rolling(window=window, center=True, min_periods=1).std()
        ax1.fill_between(timesteps, 
                         reward_mean - rolling_std,
                         reward_mean + rolling_std,
                         alpha=0.2, color=colors['reward'])
    
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax1.axhline(y=5, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Target: +5')
    ax1.set_xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Average Episode Reward', fontsize=13, fontweight='bold')
    ax1.set_title('Episode Reward Over Time', fontsize=14, fontweight='bold', pad=10)
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    
    # Add final value annotation
    final_reward = reward_mean.iloc[-1]
    ax1.annotate(f'Final: {final_reward:.2f}', 
                xy=(timesteps.iloc[-1], final_reward),
                xytext=(-60, 20), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Episode Length (Top Right)
    ax2 = axes[0, 1]
    ep_length = data['rollout/ep_len_mean']
    
    ax2.plot(timesteps, ep_length,
             color=colors['length'], linewidth=2.5, alpha=0.9)
    ax2.axhline(y=10, color='green', linestyle='--', linewidth=1.5, 
                alpha=0.5, label='Max Questions (10)')
    ax2.set_xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Average Episode Length', fontsize=13, fontweight='bold')
    ax2.set_title('Episode Length Over Time', fontsize=14, fontweight='bold', pad=10)
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax2.set_ylim([0, 11])
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    
    # Plot 3: Policy Loss (Bottom Left)
    ax3 = axes[1, 0]
    policy_loss = data['train/policy_loss']
    
    ax3.plot(timesteps, policy_loss,
             color=colors['policy'], linewidth=2.5, alpha=0.9)
    ax3.set_xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Policy Loss', fontsize=13, fontweight='bold')
    ax3.set_title('Policy Loss Over Time', fontsize=14, fontweight='bold', pad=10)
    ax3.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)
    
    # Plot 4: Value Loss (Bottom Right)
    ax4 = axes[1, 1]
    value_loss = data['train/value_loss']
    
    ax4.plot(timesteps, value_loss,
             color=colors['value'], linewidth=2.5, alpha=0.9)
    ax4.set_xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    ax4.set_ylabel('Value Loss', fontsize=13, fontweight='bold')
    ax4.set_title('Value Loss Over Time', fontsize=14, fontweight='bold', pad=10)
    ax4.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)
    
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved training plot to {output_file}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    print(f"Total Timesteps: {timesteps.iloc[-1]:,.0f}")
    print(f"Initial Reward: {reward_mean.iloc[0]:.2f}")
    print(f"Final Reward: {reward_mean.iloc[-1]:.2f}")
    print(f"Improvement: {reward_mean.iloc[-1] - reward_mean.iloc[0]:+.2f}")
    print(f"Best Reward: {reward_mean.max():.2f}")
    print(f"Avg Episode Length: {ep_length.mean():.1f}")
    print("="*60 + "\n")
    
    # Show plot
    plt.show()
    
    return True


def plot_multiple_runs(log_dirs, labels, output_file='training_comparison.png'):
    """
    Compare multiple training runs.
    
    Args:
        log_dirs: List of log directories
        labels: List of labels for each run
        output_file: Where to save the plot
    """
    
    plt.figure(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#06D6A0']
    
    for i, (log_dir, label) in enumerate(zip(log_dirs, labels)):
        progress_file = os.path.join(log_dir, 'progress.csv')
        
        if os.path.exists(progress_file):
            data = pd.read_csv(progress_file)
            plt.plot(data['total_timesteps'], data['rollout/ep_rew_mean'],
                    label=label, linewidth=2.5, alpha=0.8, color=colors[i % len(colors)])
            print(f"✅ Loaded {label}: {len(data)} steps")
        else:
            print(f"⚠️ Skipping {label}: file not found")
    
    plt.xlabel('Training Timesteps', fontsize=13, fontweight='bold')
    plt.ylabel('Average Episode Reward', fontsize=13, fontweight='bold')
    plt.title('Training Comparison: Multiple Runs', fontsize=16, fontweight='bold')
    plt.legend(fontsize=11, loc='lower right')
    plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.8)
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    
    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"✅ Saved comparison plot to {output_file}")
    
    plt.show()


def main():
    """Main function with command-line interface"""
    
    print("="*60)
    print("📊 TRAINING VISUALIZATION TOOL")
    print("="*60)
    print()
    
    # Check if log directory exists
    if not os.path.exists('logs'):
        print("❌ No 'logs' directory found!")
        print("\nMake sure you're in the python_server directory:")
        print("  cd python_server")
        print("\nAnd have run training:")
        print("  python train_agent.py --timesteps 50000")
        return
    
    # Plot single run
    success = plot_training_progress()
    
    if not success:
        return
    
    # Offer to compare multiple runs
    print("\n📋 To compare multiple training runs:")
    print("Example:")
    print("  plot_multiple_runs(['logs_50k', 'logs_100k'], ")
    print("                     ['50k steps', '100k steps'])")


if __name__ == "__main__":
    main()