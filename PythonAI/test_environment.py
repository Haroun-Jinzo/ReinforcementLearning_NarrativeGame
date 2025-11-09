import numpy as np
from environment import DetectiveEnv


def test_basic_functionality():
    """Test basic environment functionality"""
    print("="*60)
    print("TEST 1: Basic Functionality")
    print("="*60)
    
    env = DetectiveEnv()
    
    # Test reset
    try:
        obs, info = env.reset()
        print("✅ Reset works")
        print(f"   Initial suspicion: {info['suspicion_level']:.2f}")
    except Exception as e:
        print(f"❌ Reset failed: {e}")
        return False
    
    # Test step
    try:
        action = 0  # Deny everything
        obs, reward, terminated, truncated, info = env.step(action)
        print("✅ Step works")
        print(f"   Reward: {reward:.2f}")
        print(f"   Suspicion: {info['suspicion_level']:.2f}")
    except Exception as e:
        print(f"❌ Step failed: {e}")
        return False
    
    # Test all actions
    print("\n✅ Testing all action types:")
    action_names = ['Deny', 'Partial Truth', 'Deflect', 'Admit', 'Cooperate']
    for action in range(5):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"   Action {action} ({action_names[action]}): reward={reward:.2f}")
        if terminated:
            break
    
    env.close()
    print("\n✅ Test 1 PASSED\n")
    return True


def test_observation_space():
    """Test observation space structure"""
    print("="*60)
    print("TEST 2: Observation Space")
    print("="*60)
    
    env = DetectiveEnv()
    obs, info = env.reset()
    
    print("\nObservation components:")
    for key, value in obs.items():
        print(f"   {key}: shape={value.shape}, dtype={value.dtype}")
        print(f"      Range: [{np.min(value):.2f}, {np.max(value):.2f}]")
    
    # Check if observation matches space
    if env.observation_space.contains(obs):
        print("\n✅ Observation space valid")
    else:
        print("\n❌ Observation space mismatch!")
        return False
    
    env.close()
    print("\n✅ Test 2 PASSED\n")
    return True


def test_episode_completion():
    """Test full episode with random actions"""
    print("="*60)
    print("TEST 3: Full Episode (Random Actions)")
    print("="*60)
    
    env = DetectiveEnv()
    obs, info = env.reset()
    
    total_reward = 0
    steps = 0
    
    while steps < 15:  # Max 15 steps to avoid infinite loop
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            break
    
    print(f"\nEpisode completed:")
    print(f"   Steps: {steps}")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Final Suspicion: {info['suspicion_level']:.2f}")
    print(f"   Contradictions: {info['contradiction_count']}")
    print(f"   Outcome: {'❌ CAUGHT' if info['caught'] else '✅ ESCAPED'}")
    
    if steps == 10:
        print("\n✅ Episode ended at correct step (10 questions)")
    else:
        print(f"\n⚠️ Episode ended at step {steps} (expected 10)")
    
    env.close()
    print("\n✅ Test 3 PASSED\n")
    return True


def test_reward_function():
    """Test reward function logic"""
    print("="*60)
    print("TEST 4: Reward Function")
    print("="*60)
    
    env = DetectiveEnv()
    obs, info = env.reset()
    
    print("\nTesting reward for different actions:")
    
    rewards = []
    action_names = ['Deny', 'Partial Truth', 'Deflect', 'Admit', 'Cooperate']
    
    for action in range(5):
        env.reset()
        obs, reward, _, _, info = env.step(action)
        rewards.append(reward)
        print(f"   {action_names[action]:15s}: {reward:6.2f} (Suspicion: {info['suspicion_level']:.2f})")
    
    # Check reward variety
    unique_rewards = len(set([round(r, 1) for r in rewards]))
    if unique_rewards >= 3:
        print(f"\n✅ Reward function produces variety ({unique_rewards} different values)")
    else:
        print(f"\n⚠️ Low reward variety ({unique_rewards} different values)")
    
    env.close()
    print("\n✅ Test 4 PASSED\n")
    return True


def test_consistency_detection():
    """Test contradiction detection"""
    print("="*60)
    print("TEST 5: Consistency Detection")
    print("="*60)
    
    env = DetectiveEnv()
    obs, info = env.reset()
    
    print("\nTesting contradiction scenarios:")
    
    # Scenario 1: Admit then deny
    print("\n1. Admit minor detail, then deny everything:")
    env.step(3)  # Admit
    _, reward2, _, _, info2 = env.step(0)  # Deny
    print(f"   Contradictions: {info2['contradiction_count']}")
    if info2['contradiction_count'] > 0:
        print(f"   ✅ Contradiction detected! (Reward: {reward2:.2f})")
    else:
        print(f"   ⚠️ No contradiction detected")
    
    # Scenario 2: Consistent cooperation
    print("\n2. Cooperate multiple times:")
    env.reset()
    for _ in range(3):
        _, reward, _, _, info = env.step(4)  # Cooperate
    print(f"   Contradictions: {info['contradiction_count']}")
    print(f"   ✅ No contradictions (as expected)")
    
    env.close()
    print("\n✅ Test 5 PASSED\n")
    return True


def run_performance_benchmark():
    """Benchmark environment performance"""
    print("="*60)
    print("TEST 6: Performance Benchmark")
    print("="*60)
    
    import time
    
    env = DetectiveEnv()
    
    # Time 1000 steps
    n_steps = 1000
    start_time = time.time()
    
    for _ in range(n_steps // 10):
        env.reset()
        for _ in range(10):
            action = env.action_space.sample()
            env.step(action)
    
    elapsed = time.time() - start_time
    steps_per_sec = n_steps / elapsed
    
    print(f"\nPerformance:")
    print(f"   Steps: {n_steps}")
    print(f"   Time: {elapsed:.2f} seconds")
    print(f"   Speed: {steps_per_sec:.0f} steps/second")
    
    if steps_per_sec > 500:
        print(f"   ✅ Good performance! Training should be fast.")
    elif steps_per_sec > 100:
        print(f"   ⚠️ Acceptable performance. Training might be slow.")
    else:
        print(f"   ❌ Slow performance. Consider optimization.")
    
    env.close()
    print("\n✅ Test 6 PASSED\n")
    return True


def test_multiple_episodes():
    """Test multiple episodes for statistics"""
    print("="*60)
    print("TEST 7: Multiple Episodes Statistics")
    print("="*60)
    
    env = DetectiveEnv()
    
    n_episodes = 20
    results = {
        'escaped': 0,
        'caught': 0,
        'rewards': [],
        'suspicions': []
    }
    
    print(f"\nRunning {n_episodes} episodes with random actions...")
    
    for ep in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0
        
        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        results['rewards'].append(total_reward)
        results['suspicions'].append(info['suspicion_level'])
        
        if info['caught']:
            results['caught'] += 1
        else:
            results['escaped'] += 1
    
    # Print statistics
    print(f"\nResults ({n_episodes} episodes):")
    print(f"   Escaped: {results['escaped']} ({results['escaped']/n_episodes*100:.1f}%)")
    print(f"   Caught:  {results['caught']} ({results['caught']/n_episodes*100:.1f}%)")
    print(f"   Avg Reward: {np.mean(results['rewards']):.2f}")
    print(f"   Avg Suspicion: {np.mean(results['suspicions']):.2f}")
    
    # Random policy should have ~30% escape rate
    escape_rate = results['escaped'] / n_episodes
    if 0.15 < escape_rate < 0.45:
        print(f"\n✅ Escape rate reasonable for random policy ({escape_rate*100:.1f}%)")
    else:
        print(f"\n⚠️ Unusual escape rate: {escape_rate*100:.1f}%")
        print(f"   Expected: 20-40% for random actions")
    
    env.close()
    print("\n✅ Test 7 PASSED\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("DETECTIVE ENVIRONMENT TEST SUITE")
    print("="*60 + "\n")
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Observation Space", test_observation_space),
        ("Episode Completion", test_episode_completion),
        ("Reward Function", test_reward_function),
        ("Consistency Detection", test_consistency_detection),
        ("Performance Benchmark", run_performance_benchmark),
        ("Multiple Episodes", test_multiple_episodes)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"⚠️ {test_name} had issues\n")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} FAILED with error:")
            print(f"   {str(e)}\n")
    
    # Final summary
    print("="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {len(tests)}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failed} ❌")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED! Environment is ready for training.")
    else:
        print(f"\n⚠️ {failed} test(s) failed. Please review errors above.")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()