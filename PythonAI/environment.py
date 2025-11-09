import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import json


class DetectiveEnv(gym.Env):
    """
    Custom Environment for training an AI suspect in a detective game.
    
    The agent must:
    - Answer detective's questions strategically
    - Maintain consistency in responses
    - Balance between seeming innocent and not revealing truth
    - Avoid contradictions
    
    Observation Space:
        - conversation_history: Last 5 Q&A pairs (encoded)
        - evidence_revealed: Binary array of which evidence has been shown
        - suspicion_level: Current suspicion (0.0 to 1.0)
        - questions_remaining: How many questions left (0 to 10)
        - contradiction_count: Number of detected contradictions (0 to 5)
        
    Action Space:
        - 0: Deny Everything (act offended, firmly deny)
        - 1: Partial Truth (admit minor details, hide important facts)
        - 2: Deflect (change subject, vague answers, redirect)
        - 3: Admit Minor Detail (seem cooperative, build trust)
        - 4: Full Cooperation (act helpful, maintain innocence)
        
    Rewards:
        - Not caught at end: +10 - (suspicion_level * 5)
        - Caught: -20
        - Contradiction detected: -5 per contradiction
        - Suspicion increased: -2 per increase
        - Suspicion decreased: +1 (successful deflection)
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, case_data: Optional[Dict] = None, render_mode: Optional[str] = None):
        super().__init__()
        
        self.render_mode = render_mode
        
        # Load case data (or use default)
        self.case_data = case_data or self._load_default_case()
        
        # Environment configuration
        self.max_questions = 10
        self.max_evidence = 5
        self.history_size = 5
        
        # Define action space: 5 strategies
        self.action_space = spaces.Discrete(5)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            # History encoding: simplified to 20 dimensions (4 per Q&A pair * 5 pairs)
            'history': spaces.Box(low=0, high=1, shape=(20,), dtype=np.float32),
            
            # Evidence revealed: binary array
            'evidence': spaces.MultiBinary(self.max_evidence),
            
            # Suspicion level: 0 (not suspicious) to 1 (very suspicious)
            'suspicion': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            # Questions remaining: normalized 0 to 1
            'questions_remaining': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
            
            # Contradiction count: normalized 0 to 1
            'contradictions': spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)
        })
        
        # Episode state
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Initialize episode state
        self.current_step = 0
        self.questions_asked = 0
        self.suspicion_level = 0.3  # Start slightly suspicious
        self.contradiction_count = 0
        self.evidence_revealed = np.zeros(self.max_evidence, dtype=np.int8)
        self.conversation_history = []
        self.previous_responses = []  # Track for consistency checking
        self.episode_reward = 0
        self.caught = False
        
        # Generate initial state
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            action: The strategy the AI suspect chooses (0-4)
            
        Returns:
            observation: Current state
            reward: Reward for this action
            terminated: Whether episode ended (caught or escaped)
            truncated: Whether episode was cut off (max steps)
            info: Additional information
        """
        # Validate action
        assert self.action_space.contains(action), f"Invalid action: {action}"
        
        # Simulate detective asking a question
        question_type = self._generate_question()
        
        # Simulate revealing evidence randomly
        if np.random.random() < 0.3 and self.questions_asked > 2:
            unrevealed = np.where(self.evidence_revealed == 0)[0]
            if len(unrevealed) > 0:
                reveal_idx = np.random.choice(unrevealed)
                self.evidence_revealed[reveal_idx] = 1
        
        # Calculate response consistency based on action and history
        consistency_penalty = self._check_consistency(action, question_type)
        
        # Update suspicion based on action
        suspicion_change = self._calculate_suspicion_change(action, question_type)
        old_suspicion = self.suspicion_level
        self.suspicion_level = np.clip(self.suspicion_level + suspicion_change, 0.0, 1.0)
        
        # Store response for future consistency checks
        self.previous_responses.append({
            'action': action,
            'question_type': question_type,
            'suspicion_at_time': self.suspicion_level,
            'evidence_revealed': self.evidence_revealed.copy()
        })
        
        # Update conversation history
        self._update_history(action, question_type)
        
        # Increment counters
        self.questions_asked += 1
        self.current_step += 1
        
        # Calculate reward
        reward = self._calculate_reward(action, suspicion_change, consistency_penalty, old_suspicion)
        self.episode_reward += reward
        
        # Check if episode is done
        terminated = False
        truncated = False
        
        if self.questions_asked >= self.max_questions:
            # End of interrogation - decide if caught
            terminated = True
            final_reward = self._calculate_final_reward()
            reward += final_reward
        
        # Get new observation and info
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _get_observation(self) -> Dict:
        """Get current observation state"""
        # Encode conversation history
        history_encoding = self._encode_history()
        
        # Normalize questions remaining
        questions_norm = np.array([self.questions_asked / self.max_questions], dtype=np.float32)
        
        # Normalize contradictions
        contradictions_norm = np.array([min(self.contradiction_count / 5.0, 1.0)], dtype=np.float32)
        
        return {
            'history': history_encoding,
            'evidence': self.evidence_revealed.copy(),
            'suspicion': np.array([self.suspicion_level], dtype=np.float32),
            'questions_remaining': questions_norm,
            'contradictions': contradictions_norm
        }
    
    def _get_info(self) -> Dict:
        """Get additional information about current state"""
        return {
            'questions_asked': self.questions_asked,
            'suspicion_level': self.suspicion_level,
            'contradiction_count': self.contradiction_count,
            'evidence_revealed_count': int(np.sum(self.evidence_revealed)),
            'episode_reward': self.episode_reward,
            'caught': self.caught
        }
    
    def _encode_history(self) -> np.ndarray:
        """Encode conversation history into fixed-size vector"""
        # Simplified encoding: each Q&A pair gets 4 features
        # [question_type, action_taken, suspicion_change, evidence_count]
        
        encoding = np.zeros(20, dtype=np.float32)  # 5 pairs * 4 features
        
        recent_history = self.previous_responses[-self.history_size:]
        
        for i, response in enumerate(recent_history):
            base_idx = i * 4
            encoding[base_idx] = response['question_type'] / 5.0  # Normalize
            encoding[base_idx + 1] = response['action'] / 4.0  # Normalize
            encoding[base_idx + 2] = response['suspicion_at_time']
            encoding[base_idx + 3] = np.sum(response['evidence_revealed']) / self.max_evidence
        
        return encoding
    
    def _generate_question(self) -> int:
        """
        Simulate detective asking a question.
        Returns question type (0-4):
        0: Direct accusation
        1: About evidence
        2: Timeline/alibi
        3: Relationship question
        4: General/trap question
        """
        # Early game: more general questions
        # Late game: more direct accusations with evidence
        
        if self.questions_asked < 3:
            # Early: avoid direct accusations
            return np.random.choice([2, 3, 4], p=[0.4, 0.3, 0.3])
        elif self.questions_asked < 7:
            # Mid: mix of everything
            return np.random.choice([0, 1, 2, 3, 4], p=[0.1, 0.3, 0.2, 0.2, 0.2])
        else:
            # Late: more accusations and evidence-based
            return np.random.choice([0, 1, 2, 3, 4], p=[0.4, 0.3, 0.15, 0.1, 0.05])
    
    def _check_consistency(self, action: int, question_type: int) -> float:
        """
        Check if current action is consistent with previous responses.
        Returns penalty value (0 = consistent, higher = more inconsistent)
        """
        if len(self.previous_responses) < 2:
            return 0.0  # Not enough history
        
        penalty = 0.0
        
        # Check for flip-flopping strategies
        recent_actions = [r['action'] for r in self.previous_responses[-3:]]
        int_recent_actions = int(recent_actions)
        if len(set(int_recent_actions)) == len(int_recent_actions) and len(int_recent_actions) >= 3:
            # Changed strategy every single time - suspicious
            penalty += 0.3
        
        # Check for contradictory denials
        if action == 0:  # Deny everything
            for prev in self.previous_responses[-3:]:
                if prev['action'] == 3:  # Previously admitted details
                    penalty += 0.5
                    self.contradiction_count += 1
        
        # Check for inconsistent cooperation
        if action == 4:  # Full cooperation
            deny_count = sum(1 for r in self.previous_responses[-5:] if r['action'] == 0)
            if deny_count >= 3:
                # Suddenly cooperative after denying everything?
                penalty += 0.4
                self.contradiction_count += 1
        
        return penalty
    
    def _calculate_suspicion_change(self, action: int, question_type: int) -> float:
        """Calculate how much suspicion changes based on action and question"""
        
        # Base suspicion changes for each strategy
        base_changes = {
            0: 0.08,   # Deny everything - suspicious
            1: 0.02,   # Partial truth - slightly suspicious
            2: 0.05,   # Deflect - moderately suspicious
            3: -0.03,  # Admit minor - seems cooperative
            4: -0.05   # Full cooperation - seems innocent
        }
        action_int = int(action)
        change = base_changes[action_int]
        
        # Modify based on question type
        if question_type == 0:  # Direct accusation
            if action_int == 0:  # Denying direct accusation
                change += 0.05  # Extra suspicious
            elif action_int == 4:  # Being cooperative when accused
                change -= 0.03  # Seems more innocent
        
        elif question_type == 1:  # About evidence
            evidence_count = np.sum(self.evidence_revealed)
            if evidence_count > 2 and action_int == 0:
                # Denying when lots of evidence exists
                change += 0.07
        
        # Late-game effects
        if self.questions_asked > 7:
            if action_int == 2:  # Deflecting late is very suspicious
                change += 0.06
        
        return change
    
    def _update_history(self, action: int, question_type: int):
        """Update conversation history"""
        self.conversation_history.append({
            'step': self.current_step,
            'question_type': question_type,
            'action': action,
            'suspicion': self.suspicion_level
        })
    
    def _calculate_reward(self, action: int, suspicion_change: float, 
                         consistency_penalty: float, old_suspicion: float) -> float:
        """Calculate immediate reward for this step"""
        reward = 0.0
        
        # Penalty for increasing suspicion
        if suspicion_change > 0:
            reward -= suspicion_change * 10  # -0.2 to -0.8
        else:
            # Bonus for decreasing suspicion
            reward += abs(suspicion_change) * 5  # +0.15 to +0.25
        
        # Penalty for inconsistency
        if consistency_penalty > 0:
            reward -= consistency_penalty * 10  # -3 to -5
        
        # Penalty for high suspicion
        if self.suspicion_level > 0.7:
            reward -= 1.0
        elif self.suspicion_level < 0.4:
            reward += 0.5
        
        # Small step penalty to encourage ending episodes
        reward -= 0.1
        
        return reward
    
    def _calculate_final_reward(self) -> float:
        """Calculate final reward at end of interrogation"""
        
        # Determine if caught based on suspicion and contradictions
        self.caught = (self.suspicion_level > 0.65 or 
                      self.contradiction_count >= 3)
        
        if self.caught:
            # Caught - large penalty
            return -20.0
        else:
            # Escaped - reward inversely proportional to suspicion
            escape_bonus = 10.0 - (self.suspicion_level * 5.0)
            
            # Bonus for maintaining low suspicion throughout
            if self.suspicion_level < 0.4:
                escape_bonus += 5.0
            
            # Bonus for no contradictions
            if self.contradiction_count == 0:
                escape_bonus += 3.0
            
            return escape_bonus
    
    def _load_default_case(self) -> Dict:
        """Load default case scenario"""
        return {
            'case_id': 'case_001',
            'title': 'The Missing Artifact',
            'background': 'A valuable artifact has been stolen from the museum.',
            'truth': 'The suspect was at the museum that night and stole the artifact.',
            'evidence': [
                'Security footage shows someone matching description',
                'Fingerprints found at scene',
                'Witness saw suspicious person',
                'Artifact pieces found near suspect\'s home',
                'Bank records show unusual deposits'
            ]
        }
    
    def render(self):
        """Render the environment state"""
        if self.render_mode == 'human' or self.render_mode == 'ansi':
            print("\n" + "="*50)
            print(f"Detective Game - Step {self.current_step}")
            print("="*50)
            print(f"Questions Asked: {self.questions_asked}/{self.max_questions}")
            print(f"Suspicion Level: {self.suspicion_level:.2f}")
            print(f"Contradictions: {self.contradiction_count}")
            print(f"Evidence Revealed: {np.sum(self.evidence_revealed)}/{self.max_evidence}")
            print(f"Episode Reward: {self.episode_reward:.2f}")
            
            if len(self.conversation_history) > 0:
                print("\nRecent Actions:")
                for entry in self.conversation_history[-3:]:
                    action_name = ['Deny', 'Partial Truth', 'Deflect', 'Admit', 'Cooperate'][entry['action']]
                    print(f"  Step {entry['step']}: {action_name} (Suspicion: {entry['suspicion']:.2f})")
            
            print("="*50 + "\n")
    
    def close(self):
        """Clean up environment"""
        pass


# ==================== TESTING FUNCTIONS ====================

def test_environment():
    """Test the environment with random actions"""
    print("Testing Detective Environment...")
    print("="*60)
    
    env = DetectiveEnv(render_mode='human')
    
    # Test reset
    observation, info = env.reset()
    print("✅ Environment reset successful")
    print(f"Initial state: Suspicion={info['suspicion_level']:.2f}")
    
    # Run one episode with random actions
    terminated = False
    truncated = False
    total_reward = 0
    
    while not (terminated or truncated):
        # Random action
        action = env.action_space.sample()
        
        # Step
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        # Render
        env.render()
        
        if terminated:
            print("\n" + "="*60)
            if info['caught']:
                print("❌ CAUGHT! Suspect failed to escape.")
            else:
                print("✅ ESCAPED! Suspect avoided detection.")
            print(f"Final Suspicion: {info['suspicion_level']:.2f}")
            print(f"Total Contradictions: {info['contradiction_count']}")
            print(f"Total Reward: {total_reward:.2f}")
            print("="*60)
    
    env.close()
    print("\n✅ Environment test completed!")


if __name__ == "__main__":
    test_environment()