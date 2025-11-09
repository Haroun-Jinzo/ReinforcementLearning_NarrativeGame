import os
import time
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure

# Import our custom environment
from environment import DetectiveEnv


class TrainingProgressCallback(BaseCallback):
    """
    Custom callback for tracking training progress and printing updates.
    """
    
    def __init__(self, check_freq: int = 1000, verbose: int = 1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.suspicion_levels = []
        self.caught_count = 0
        self.escaped_count = 0
    
    def _on_step(self) -> bool:
        # This is called at every step
        
        if self.n_calls % self.check_freq == 0:
            # Print progress
            print(f"\n{'='*60}")
            print(f"Training Step: {self.n_calls}")
            
            if len(self.episode_rewards) > 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_length = np.mean(self.episode_lengths[-100:])
                total_episodes = len(self.episode_rewards)
                escape_rate = (self.escaped_count / total_episodes * 100) if total_episodes > 0 else 0
                
                print(f"Episodes Completed: {total_episodes}")
                print(f"Avg Reward (last 100): {avg_reward:.2f}")
                print(f"Avg Episode Length: {avg_length:.1f}")
                print(f"Escape Rate: {escape_rate:.1f}%")
                print(f"Caught: {self.caught_count} | Escaped: {self.escaped_count}")
            
            print(f"{'='*60}\n")
        
        return True
    
    def _on_rollout_end(self) -> None:
        """Called at the end of each rollout"""
        # Extract episode statistics from info buffer
        if len(self.model.ep_info_buffer) > 0:
            for ep_info in self.model.ep_info_buffer:
                if 'r' in ep_info:
                    self.episode_rewards.append(ep_info['r'])
                if 'l' in ep_info:
                    self.episode_lengths.append(ep_info['l'])


class DetectiveTrainer:
    """
    Main training class for the Detective RL agent.
    """
    
    def __init__(self, 
                 total_timesteps: int = 50000,
                 n_envs: int = 4,
                 save_dir: str = "models",
                 log_dir: str = "logs"):
        
        self.total_timesteps = total_timesteps
        self.n_envs = n_envs
        self.save_dir = save_dir
        self.log_dir = log_dir
        
        # Create directories
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(log_dir, exist_ok=True)
        
        # Training statistics
        self.training_start_time = None
        self.stats = {
            'rewards': [],
            'escape_rates': [],
            'suspicion_levels': [],
            'episode_lengths': []
        }
    
    def create_environment(self):
        """Create vectorized environment for parallel training"""
        print("Creating training environments...")
        
        # Create multiple parallel environments
        env = make_vec_env(
            DetectiveEnv,
            n_envs=self.n_envs,
            seed=42
        )
        
        # Wrap with monitor to track episode statistics
        env = VecMonitor(env)
        
        print(f"✅ Created {self.n_envs} parallel environments")
        return env
    
    def create_model(self, env):
        """Create PPO model with custom hyperparameters"""
        print("\nInitializing PPO model...")
        
        # Configure custom logger
        logger = configure(self.log_dir, ["stdout", "csv", "tensorboard"])
        
        # Create PPO model with tuned hyperparameters for our task
        model = PPO(
            policy="MultiInputPolicy",  # For Dict observation space
            env=env,
            
            # Learning rate schedule (start high, decay)
            learning_rate=3e-4,
            
            # Number of steps to collect before update
            n_steps=2048,
            
            # Mini-batch size for training
            batch_size=64,
            
            # Number of epochs per update
            n_epochs=10,
            
            # Discount factor (how much to value future rewards)
            gamma=0.99,
            
            # GAE parameter
            gae_lambda=0.95,
            
            # Clipping parameter for PPO
            clip_range=0.2,
            
            # Entropy coefficient (encourages exploration)
            ent_coef=0.01,
            
            # Value function coefficient
            vf_coef=0.5,
            
            # Max gradient norm
            max_grad_norm=0.5,
            
            verbose=1,
            tensorboard_log=self.log_dir
        )
        
        model.set_logger(logger)
        
        print("✅ PPO model created")
        print(f"   Learning Rate: 3e-4")
        print(f"   Steps per Update: 2048")
        print(f"   Batch Size: 64")
        print(f"   Parallel Envs: {self.n_envs}")
        
        return model
    
    def setup_callbacks(self, env):
        """Setup training callbacks"""
        callbacks = []
        
        # Progress callback
        progress_callback = TrainingProgressCallback(
            check_freq=2000,
            verbose=1
        )
        callbacks.append(progress_callback)
        
        # Checkpoint callback (save model every N steps)
        checkpoint_callback = CheckpointCallback(
            save_freq=10000,
            save_path=os.path.join(self.save_dir, "checkpoints"),
            name_prefix="detective_agent"
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback (test performance periodically)
        eval_env = make_vec_env(DetectiveEnv, n_envs=1, seed=123)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(self.save_dir, "best_model"),
            log_path=os.path.join(self.log_dir, "eval"),
            eval_freq=5000,
            deterministic=True,
            render=False,
            n_eval_episodes=10
        )
        callbacks.append(eval_callback)
        
        return callbacks
    
    def train(self):
        """Main training loop"""
        print("\n" + "="*60)
        print("🚀 STARTING TRAINING")
        print("="*60)
        print(f"Total Timesteps: {self.total_timesteps:,}")
        print(f"Parallel Environments: {self.n_envs}")
        print(f"Save Directory: {self.save_dir}")
        print(f"Log Directory: {self.log_dir}")
        print("="*60 + "\n")
        
        self.training_start_time = time.time()
        
        # Create environment and model
        env = self.create_environment()
        model = self.create_model(env)
        callbacks = self.setup_callbacks(env)
        
        # Train the model
        print("\n🎓 Training started...\n")
        
        try:
            model.learn(
                total_timesteps=self.total_timesteps,
                callback=callbacks,
                progress_bar=True
            )
            
            print("\n✅ Training completed successfully!")
            
        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        
        except Exception as e:
            print(f"\n❌ Training failed: {str(e)}")
            raise
        
        finally:
            # Save final model
            final_path = os.path.join(self.save_dir, "detective_agent_final")
            model.save(final_path)
            print(f"\n💾 Final model saved to: {final_path}")
            
            # Print training summary
            self.print_summary()
            
            # Close environments
            env.close()
    
    def print_summary(self):
        """Print training summary"""
        elapsed_time = time.time() - self.training_start_time
        hours = int(elapsed_time // 3600)
        minutes = int((elapsed_time % 3600) // 60)
        seconds = int(elapsed_time % 60)
        
        print("\n" + "="*60)
        print("📊 TRAINING SUMMARY")
        print("="*60)
        print(f"Total Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
        print(f"Total Timesteps: {self.total_timesteps:,}")
        print(f"Timesteps/Second: {self.total_timesteps/elapsed_time:.1f}")
        print(f"Models Saved: {self.save_dir}/")
        print(f"Logs Saved: {self.log_dir}/")
        print("="*60 + "\n")


def test_trained_agent(model_path: str, n_episodes: int = 5):
    """
    Test a trained agent and visualize its performance.
    
    Args:
        model_path: Path to saved model
        n_episodes: Number of test episodes to run
    """
    print("\n" + "="*60)
    print("🧪 TESTING TRAINED AGENT")
    print("="*60 + "\n")
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create test environment
    env = DetectiveEnv(render_mode='human')
    
    results = {
        'rewards': [],
        'escaped': 0,
        'caught': 0,
        'suspicion_levels': [],
        'contradiction_counts': []
    }
    
    for episode in range(n_episodes):
        print(f"\n{'='*60}")
        print(f"Test Episode {episode + 1}/{n_episodes}")
        print('='*60)
        
        obs, info = env.reset()
        terminated = False
        truncated = False
        episode_reward = 0
        
        while not (terminated or truncated):
            # Get action from trained model
            action, _states = model.predict(obs, deterministic=True)
            
            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            
            # Render
            env.render()
        
        # Record results
        results['rewards'].append(episode_reward)
        results['suspicion_levels'].append(info['suspicion_level'])
        results['contradiction_counts'].append(info['contradiction_count'])
        
        if info['caught']:
            results['caught'] += 1
            outcome = "❌ CAUGHT"
        else:
            results['escaped'] += 1
            outcome = "✅ ESCAPED"
        
        print(f"\n{outcome}")
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Final Suspicion: {info['suspicion_level']:.2f}")
        print(f"Contradictions: {info['contradiction_count']}")
    
    # Print summary
    print("\n" + "="*60)
    print("📊 TEST RESULTS SUMMARY")
    print("="*60)
    print(f"Episodes: {n_episodes}")
    print(f"Escaped: {results['escaped']} ({results['escaped']/n_episodes*100:.1f}%)")
    print(f"Caught: {results['caught']} ({results['caught']/n_episodes*100:.1f}%)")
    print(f"Avg Reward: {np.mean(results['rewards']):.2f}")
    print(f"Avg Suspicion: {np.mean(results['suspicion_levels']):.2f}")
    print(f"Avg Contradictions: {np.mean(results['contradiction_counts']):.2f}")
    print("="*60 + "\n")
    
    env.close()


# ==================== MAIN EXECUTION ====================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Detective Game RL Agent")
    parser.add_argument("--timesteps", type=int, default=50000,
                       help="Total training timesteps (default: 50000)")
    parser.add_argument("--envs", type=int, default=4,
                       help="Number of parallel environments (default: 4)")
    parser.add_argument("--test", type=str, default=None,
                       help="Path to model to test (skip training)")
    parser.add_argument("--test-episodes", type=int, default=5,
                       help="Number of test episodes (default: 5)")
    
    args = parser.parse_args()
    
    if args.test:
        # Test mode
        test_trained_agent(args.test, args.test_episodes)
    else:
        # Training mode
        trainer = DetectiveTrainer(
            total_timesteps=args.timesteps,
            n_envs=args.envs
        )
        trainer.train()
        
        # Automatically test after training
        print("\n🧪 Running post-training test...\n")
        test_trained_agent("models/detective_agent_final", n_episodes=3)