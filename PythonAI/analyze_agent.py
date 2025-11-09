"""
Analyze trained agent performance with detailed plots
"""

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from environment import DetectiveEnv
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.facecolor'] = 'white'


def analyze_agent_behavior(model_path='models/detective_agent_final.zip', n_episodes=100):
    """Analyze agent behavior across multiple episodes"""
    
    print(f"Loading agent from {model_path}...")
    model = PPO.load(model_path)
    env = DetectiveEnv()
    
    # Collect statistics
    results = {
        'rewards': [],
        'suspicions': [],
        'contradictions': [],
        'escaped': 0,
        'caught': 0,
        'strategies_used': {i: 0 for i in range(5)},
        'suspicion_trajectory': []
    }
    
    print(f"Running {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        suspicion_history = [info['suspicion_level']]
        
        for step in range(10):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            suspicion_history.append(info['suspicion_level'])
            results['strategies_used'][int(action)] += 1
            
            if terminated or truncated:
                break
        
        results['rewards'].append(episode_reward)
        results['suspicions'].append(info['suspicion_level'])
        results['contradictions'].append(info['contradiction_count'])
        results['suspicion_trajectory'].append(suspicion_history)
        
        if info['caught']:
            results['caught'] += 1
        else:
            results['escaped'] += 1
    
    env.close()
    
    # Create comprehensive plot
    create_analysis_plots(results, n_episodes)
    
    return results


def create_analysis_plots(results, n_episodes):
    """Create comprehensive analysis plots"""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Plot 1: Reward Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(results['rewards'], bins=30, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axvline(np.mean(results['rewards']), color='red', linestyle='--', 
                linewidth=2, label=f'Mean: {np.mean(results["rewards"]):.2f}')
    ax1.set_xlabel('Episode Reward', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Reward Distribution', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Suspicion Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(results['suspicions'], bins=20, color='#A23B72', alpha=0.7, edgecolor='black')
    ax2.axvline(0.65, color='red', linestyle='--', linewidth=2, label='Caught Threshold')
    ax2.axvline(np.mean(results['suspicions']), color='orange', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(results["suspicions"]):.2f}')
    ax2.set_xlabel('Final Suspicion Level', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Final Suspicion Distribution', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Outcome Pie Chart
    ax3 = fig.add_subplot(gs[0, 2])
    escape_rate = results['escaped'] / n_episodes * 100
    colors = ['#06D6A0', '#EF476F']
    ax3.pie([results['escaped'], results['caught']], 
            labels=[f'Escaped\n({escape_rate:.1f}%)', 
                   f'Caught\n({100-escape_rate:.1f}%)'],
            colors=colors, autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax3.set_title('Outcome Distribution', fontsize=12, fontweight='bold')
    
    # Plot 4: Strategy Usage
    ax4 = fig.add_subplot(gs[1, 0])
    strategy_names = ['Deny', 'Partial\nTruth', 'Deflect', 'Admit', 'Cooperate']
    strategy_counts = [results['strategies_used'][i] for i in range(5)]
    bars = ax4.bar(strategy_names, strategy_counts, 
                   color=['#E63946', '#F77F00', '#FCBF49', '#06D6A0', '#118AB2'])
    ax4.set_ylabel('Times Used', fontsize=11)
    ax4.set_title('Strategy Usage Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom', fontsize=10)
    
    # Plot 5: Contradiction Distribution
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.hist(results['contradictions'], bins=range(6), color='#EF476F', 
             alpha=0.7, edgecolor='black', align='left')
    ax5.set_xlabel('Number of Contradictions', fontsize=11)
    ax5.set_ylabel('Frequency', fontsize=11)
    ax5.set_title('Contradictions per Episode', fontsize=12, fontweight='bold')
    ax5.set_xticks(range(6))
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Suspicion Trajectories
    ax6 = fig.add_subplot(gs[1, 2])
    # Plot sample trajectories
    for traj in results['suspicion_trajectory'][:20]:  # First 20 episodes
        ax6.plot(range(len(traj)), traj, alpha=0.3, color='#2E86AB')
    # Plot mean trajectory
    max_len = max(len(t) for t in results['suspicion_trajectory'])
    mean_traj = []
    for i in range(max_len):
        vals = [t[i] for t in results['suspicion_trajectory'] if i < len(t)]
        mean_traj.append(np.mean(vals))
    ax6.plot(range(len(mean_traj)), mean_traj, color='red', linewidth=3, 
             label='Mean', alpha=0.8)
    ax6.axhline(0.65, color='orange', linestyle='--', linewidth=2, 
                label='Caught Threshold')
    ax6.set_xlabel('Question Number', fontsize=11)
    ax6.set_ylabel('Suspicion Level', fontsize=11)
    ax6.set_title('Suspicion Trajectories', fontsize=12, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Reward vs Suspicion Scatter
    ax7 = fig.add_subplot(gs[2, 0])
    colors_scatter = ['#06D6A0' if r > 0 else '#EF476F' for r in results['rewards']]
    ax7.scatter(results['suspicions'], results['rewards'], 
                c=colors_scatter, alpha=0.6, s=50)
    ax7.set_xlabel('Final Suspicion', fontsize=11)
    ax7.set_ylabel('Episode Reward', fontsize=11)
    ax7.set_title('Reward vs Suspicion', fontsize=12, fontweight='bold')
    ax7.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    ax7.axvline(0.65, color='orange', linestyle='--', linewidth=2, alpha=0.5)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Performance Metrics Table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    metrics_data = [
        ['Metric', 'Value'],
        ['Total Episodes', f'{n_episodes}'],
        ['Escape Rate', f'{results["escaped"]/n_episodes*100:.1f}%'],
        ['Caught Rate', f'{results["caught"]/n_episodes*100:.1f}%'],
        ['Avg Reward', f'{np.mean(results["rewards"]):.2f}'],
        ['Avg Final Suspicion', f'{np.mean(results["suspicions"]):.3f}'],
        ['Avg Contradictions', f'{np.mean(results["contradictions"]):.2f}'],
        ['Reward Std Dev', f'{np.std(results["rewards"]):.2f}'],
    ]
    
    table = ax8.table(cellText=metrics_data, cellLoc='left',
                     bbox=[0.2, 0.2, 0.6, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    
    # Style header row
    for i in range(2):
        table[(0, i)].set_facecolor('#2E86AB')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(metrics_data)):
        color = '#F0F0F0' if i % 2 == 0 else 'white'
        for j in range(2):
            table[(i, j)].set_facecolor(color)
    
    ax8.set_title('Performance Summary', fontsize=14, fontweight='bold', pad=20)
    
    # Main title
    fig.suptitle('Detective Game - Agent Performance Analysis', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Save
    plt.savefig('agent_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved analysis plot to agent_analysis.png")
    
    plt.show()


if __name__ == "__main__":
    print("="*60)
    print("ANALYZING AGENT PERFORMANCE")
    print("="*60)
    
    results = analyze_agent_behavior(n_episodes=100)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Escape Rate: {results['escaped']/100*100:.1f}%")
    print(f"Avg Reward: {np.mean(results['rewards']):.2f}")
    print(f"Avg Suspicion: {np.mean(results['suspicions']):.3f}")