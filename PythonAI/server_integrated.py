from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import json
import os
from typing import Dict, List
from datetime import datetime

# Import our components
from stable_baselines3 import PPO
from gemini_client import GeminiClient

app = Flask(__name__)
CORS(app)

# Global variables
rl_agent = None
gemini_client = None
sessions = {}

print("="*60)
print("🕵️ DETECTIVE GAME - INTEGRATED SERVER (Week 3)")
print("="*60)


def initialize_ai_components():
    """Initialize RL agent and Gemini client"""
    global rl_agent, gemini_client
    
    print("\n🤖 Initializing AI components...")
    
    # Load RL agent
    try:
        model_path = "models/detective_agent_final.zip"
        if not os.path.exists(model_path):
            print(f"⚠️ Model not found at {model_path}")
            print("   Using random strategy selection as fallback")
            rl_agent = None
        else:
            rl_agent = PPO.load(model_path)
            print(f"✅ RL agent loaded from {model_path}")
    except Exception as e:
        print(f"⚠️ Failed to load RL agent: {e}")
        print("   Using random strategy selection as fallback")
        rl_agent = None
    
    # Initialize Gemini client
    try:
        gemini_client = GeminiClient()
        if gemini_client.test_connection():
            print("✅ Gemini API connected")
        else:
            print("⚠️ Gemini API connection issue")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        print("   Make sure GEMINI_API_KEY is set!")
        gemini_client = None


# Initialize on startup
initialize_ai_components()


# ==================== HELPER FUNCTIONS ====================

def load_case(case_id: str) -> Dict:
    """Load case data from JSON file"""
    case_file = f"cases/{case_id}.json"
    
    if os.path.exists(case_file):
        with open(case_file, 'r') as f:
            return json.load(f)
    else:
        # Return default case if file not found
        return {
            'case_id': case_id,
            'title': 'The Missing Artifact',
            'background': 'A valuable artifact has been stolen from the museum.',
            'truth': 'You stole the artifact and hid it in your apartment.',
            'intro': 'Detective: "Thank you for coming in. I have some questions about last night."',
            'evidence': [
                'Security footage shows someone matching your description',
                'Your fingerprints were found at the scene',
                'A witness saw you leaving the area',
                'Stolen item was found near your residence',
                'Your alibi doesn\'t check out'
            ]
        }


def build_rl_state(session: Dict) -> np.ndarray:
    """
    Convert session data to RL state observation.
    This should match the observation space from environment.py
    """
    # History encoding (20 dimensions)
    history_encoding = np.zeros(20, dtype=np.float32)
    recent_history = session['history'][-5:]  # Last 5 Q&A
    
    for i, qa in enumerate(recent_history):
        base_idx = i * 4
        if base_idx + 3 < len(history_encoding):
            history_encoding[base_idx] = qa.get('question_type', 2) / 5.0
            history_encoding[base_idx + 1] = qa.get('strategy', 2) / 4.0
            history_encoding[base_idx + 2] = session['suspicion']
            history_encoding[base_idx + 3] = sum(session['evidence_revealed']) / 5.0
    
    # Build complete observation dict
    observation = {
        'history': history_encoding,
        'evidence': np.array(session['evidence_revealed'], dtype=np.int8),
        'suspicion': np.array([session['suspicion']], dtype=np.float32),
        'questions_remaining': np.array([session['questions_asked'] / 10.0], dtype=np.float32),
        'contradictions': np.array([session['contradictions'] / 5.0], dtype=np.float32)
    }
    
    return observation


def choose_strategy(session: Dict) -> int:
    """
    Use RL agent to choose strategy, or fallback to random.
    
    Returns:
        Strategy number (0-4)
    """
    if rl_agent is not None:
        try:
            observation = build_rl_state(session)
            action, _states = rl_agent.predict(observation, deterministic=True)
            return int(action)
        except Exception as e:
            print(f"⚠️ RL prediction failed: {e}")
    
    # Fallback: smart random strategy
    # Early game: be cooperative
    # Late game: be more defensive if suspicion is high
    if session['questions_asked'] < 3:
        return np.random.choice([3, 4], p=[0.4, 0.6])  # Admit or cooperate
    elif session['suspicion'] > 0.6:
        return np.random.choice([0, 2], p=[0.5, 0.5])  # Deny or deflect
    else:
        return np.random.randint(0, 5)  # Random


def calculate_suspicion_increase(strategy: int, question_type: int, 
                                 evidence_count: int) -> float:
    """Calculate suspicion increase based on strategy and context"""
    
    base_changes = {
        0: 0.08,   # Deny - suspicious
        1: 0.02,   # Partial truth - slightly suspicious
        2: 0.05,   # Deflect - moderately suspicious
        3: -0.03,  # Admit - seems cooperative
        4: -0.05   # Cooperate - seems innocent
    }
    
    change = base_changes.get(strategy, 0.02)
    
    # Denying when lots of evidence is very suspicious
    if strategy == 0 and evidence_count > 2:
        change += 0.06
    
    return change


# ==================== API ENDPOINTS ====================

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'running',
        'rl_agent': 'loaded' if rl_agent else 'fallback',
        'gemini': 'connected' if gemini_client else 'disconnected',
        'timestamp': datetime.now().isoformat()
    })


@app.route('/start_game', methods=['POST'])
def start_game():
    """Initialize a new game session"""
    try:
        data = request.json
        session_id = data.get('session_id')
        case_id = data.get('case_id', 'case_001')
        
        # Load case data
        case_data = load_case(case_id)
        
        # Initialize session
        sessions[session_id] = {
            'case': case_data,
            'history': [],
            'evidence_revealed': [0] * len(case_data['evidence']),
            'suspicion': 0.3,  # Start slightly suspicious
            'questions_asked': 0,
            'contradictions': 0,
            'caught': False,
            'started_at': datetime.now().isoformat()
        }
        
        print(f"\n🎮 New game started: {session_id}")
        print(f"   Case: {case_data.get('title', 'Unknown')}")
        
        return jsonify({
            'status': 'success',
            'case_title': case_data.get('title', ''),
            'intro': case_data.get('intro', ''),
            'evidence_count': len(case_data['evidence'])
        })
        
    except Exception as e:
        print(f"❌ Error in /start_game: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Process a detective question and generate AI suspect response.
    
    Flow:
    1. Receive question from Unity
    2. Use RL agent to choose strategy
    3. Use Gemini to generate natural language response
    4. Update session state
    5. Return response to Unity
    """
    try:
        data = request.json
        session_id = data['session_id']
        question = data['question']
        evidence_shown = data.get('evidence_shown', [])
        
        if session_id not in sessions:
            return jsonify({
                'status': 'error',
                'message': 'Session not found. Start a new game first.'
            }), 404
        
        session = sessions[session_id]
        
        # Update evidence revealed
        for evi_id in evidence_shown:
            if evi_id < len(session['evidence_revealed']):
                session['evidence_revealed'][evi_id] = 1
        
        evidence_count = sum(session['evidence_revealed'])
        
        # Determine question type (for RL state)
        question_lower = question.lower()
        if any(word in question_lower for word in ['did you', 'were you', 'are you']):
            question_type = 0  # Direct accusation
        elif 'evidence' in question_lower or 'proof' in question_lower:
            question_type = 1  # About evidence
        elif 'when' in question_lower or 'where' in question_lower:
            question_type = 2  # Timeline/alibi
        else:
            question_type = 4  # General question
        
        print(f"\n📨 Question from Unity: \"{question}\"")
        
        # Choose strategy using RL agent
        strategy = choose_strategy(session)
        strategy_name = GeminiClient.STRATEGIES[strategy]['name']
        
        print(f"🤖 RL Agent chose strategy: {strategy} ({strategy_name})")
        
        # Generate response using Gemini
        if gemini_client:
            response_text = gemini_client.generate_response(
                strategy=strategy,
                question=question,
                case_context=session['case'],
                conversation_history=session['history'],
                evidence_revealed=session['evidence_revealed'],
                suspicion_level=session['suspicion']
            )
            print(f"💬 Gemini response: \"{response_text[:100]}...\"")
        else:
            # Fallback if Gemini is not available
            response_text = f"[Fallback] I don't have much to say about that."
            print(f"⚠️ Using fallback response (Gemini unavailable)")
        
        # Update suspicion
        suspicion_change = calculate_suspicion_increase(
            strategy, question_type, evidence_count
        )
        old_suspicion = session['suspicion']
        session['suspicion'] = np.clip(session['suspicion'] + suspicion_change, 0.0, 1.0)
        
        # Detect contradictions (simplified)
        contradiction_detected = False
        if len(session['history']) > 0:
            # Check if denying after previously admitting
            last_strategy = session['history'][-1].get('strategy', -1)
            if last_strategy == 3 and strategy == 0:
                contradiction_detected = True
                session['contradictions'] += 1
        
        # Update history
        session['history'].append({
            'question': question,
            'question_type': question_type,
            'response': response_text,
            'strategy': strategy,
            'strategy_name': strategy_name,
            'suspicion': session['suspicion'],
            'contradiction': contradiction_detected
        })
        
        session['questions_asked'] += 1
        
        print(f"📊 Suspicion: {old_suspicion:.2f} → {session['suspicion']:.2f}")
        
        # Prepare response
        return jsonify({
            'status': 'success',
            'response': response_text,
            'suspicion_level': float(session['suspicion']),
            'questions_remaining': 10 - session['questions_asked'],
            'evidence_revealed': evidence_count,
            'contradiction_detected': contradiction_detected
        })
        
    except Exception as e:
        print(f"❌ Error in /ask_question: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/accuse', methods=['POST'])
def accuse():
    """Handle player accusation and determine outcome"""
    try:
        data = request.json
        session_id = data['session_id']
        
        if session_id not in sessions:
            return jsonify({
                'status': 'error',
                'message': 'Session not found'
            }), 404
        
        session = sessions[session_id]
        
        # Determine if suspect is caught
        caught = (
            session['suspicion'] > 0.45 or
            session['contradictions'] >= 2
        )
        
        session['caught'] = caught
        
        # Find contradictions in history
        contradictions = []
        for i, qa in enumerate(session['history']):
            if qa.get('contradiction', False):
                contradictions.append(f"Step {i+1}: {qa['question']}")
        
        print(f"\n⚖️ Accusation made!")
        print(f"   Result: {'CAUGHT' if caught else 'ESCAPED'}")
        print(f"   Final Suspicion: {session['suspicion']:.2f}")
        print(f"   Contradictions: {session['contradictions']}")
        
        return jsonify({
            'status': 'success',
            'outcome': 'caught' if caught else 'escaped',
            'suspicion_level': float(session['suspicion']),
            'contradictions': contradictions,
            'total_contradictions': session['contradictions'],
            'questions_asked': session['questions_asked']
        })
        
    except Exception as e:
        print(f"❌ Error in /accuse: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/get_evidence', methods=['POST'])
def get_evidence():
    """Get available evidence for the case"""
    try:
        data = request.json
        session_id = data['session_id']
        
        if session_id not in sessions:
            return jsonify({
                'status': 'error',
                'message': 'Session not found'
            }), 404
        
        session = sessions[session_id]
        evidence_list = session['case'].get('evidence', [])
        
        return jsonify({
            'status': 'success',
            'evidence': [
                {
                    'id': i,
                    'description': evidence_list[i],
                    'revealed': bool(session['evidence_revealed'][i])
                }
                for i in range(len(evidence_list))
            ]
        })
        
    except Exception as e:
        print(f"❌ Error in /get_evidence: {e}")
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ==================== SERVER STARTUP ====================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 Starting Integrated Server...")
    print("="*60)
    print("\n📋 Available endpoints:")
    print("   GET  /health        - Check server status")
    print("   POST /start_game    - Start new interrogation")
    print("   POST /ask_question  - Ask suspect a question")
    print("   POST /accuse        - Make accusation")
    print("   POST /get_evidence  - Get case evidence")
    print("\n⚡ Server ready! Waiting for Unity...")
    print("="*60 + "\n")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )