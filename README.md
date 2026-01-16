# 🕵️ Detective Game - AI Interrogation System

An advanced AI-powered detective interrogation game that combines Reinforcement Learning and Large Language Models to simulate realistic interrogation scenarios. This project demonstrates the application of AI in interactive game environments, specifically focused on deception detection and suspect interrogation.

## 📋 Project Overview

This project combines a **Unity-based game interface** with a **Python AI backend** to create an interactive detective game. The system uses:

- **Reinforcement Learning (PPO)**: An RL agent trained to select strategic deception tactics
- **Generative AI (Google Gemini)**: LLM-based natural language generation for suspect responses
- **Custom Game Environment**: A Gymnasium-based environment simulating interrogation scenarios
- **Flask Backend**: A server integrating the RL agent and LLM for real-time gameplay

### Game Concept

In this game, you play as a detective interrogating suspects who are **guilty but attempting to deceive you**. The suspects use various deception strategies:

1. **Deny Aggressively** - Firmly deny all accusations
2. **Mix Truth with Lies** - Admit to minor facts while hiding the main crime
3. **Deflect and Evade** - Change subjects, give vague answers
4. **Blame Others** - Redirect accusations to other people
5. **Show Remorse (False)** - Appear guilty of minor crimes to distract from major ones

## 🏗️ Project Structure

```
ProjectV2/
├── DetectiveGame/           # Unity game project
│   ├── Assets/             # Game assets and scripts
│   ├── ProjectSettings/     # Unity project configuration
│   └── *.csproj           # C# project files
│
└── PythonAI/               # Python AI backend
    ├── train_agent.py      # RL agent training script
    ├── environment.py      # Custom Gymnasium environment
    ├── server_integrated.py # Flask server for game integration
    ├── gemini_client.py     # Google Gemini API client
    ├── models/             # Trained RL models
    ├── logs/               # Training logs and metrics
    ├── cases/              # Detective case data (JSON)
    ├── environments/       # Game environment definitions
    └── requirement.txt     # Python dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Unity (for the game client)
- Google Generative AI API key
- CUDA-capable GPU (optional, for faster training)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd ProjectV2/PythonAI
   ```

2. **Create and activate virtual environment**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirement.txt
   ```

4. **Set up environment variables**
   Create a `.env` file in `PythonAI/` directory:
   ```
   GEMINI_API_KEY=your_google_gemini_api_key_here
   ```

### Running the System

**Option 1: Train a new RL agent**
```bash
python train_agent.py
```

**Option 2: Run the integrated Flask server**
```bash
python server_integrated.py
```

The server will start on `http://localhost:5000`

## 🤖 Key Components

### 1. **RL Agent Training** (`train_agent.py`)

Trains a Proximal Policy Optimization (PPO) agent to select deception strategies during interrogation.

- Uses Stable Baselines3 framework
- Custom environment with detective-specific reward signals
- Tracks escape rate, caught rate, and suspicion levels
- Saves checkpoints and training logs

### 2. **Game Environment** (`environment.py`)

A custom Gymnasium environment simulating the interrogation process.

**Observation Space:**
- Conversation history (20 dimensions)
- Revealed evidence (binary array, max 5)
- Suspicion level (0-1)
- Remaining questions (0-1)
- Contradiction count (0-1)

**Action Space:**
- 5 discrete actions representing different deception strategies

**Reward Signals:**
- Positive reward for maintaining low suspicion
- Bonus for successful escape (answering all questions without admission)
- Penalty for being caught (caught_by_detective flag set)

### 3. **Gemini Client** (`gemini_client.py`)

Generates natural language suspect responses based on:
- Selected deception strategy (from RL agent)
- Detective's question
- Case context and evidence
- Conversation history

Uses carefully crafted prompts to ensure realistic, contextual responses that maintain the suspect's cover story.

### 4. **Flask Server** (`server_integrated.py`)

RESTful API endpoints:
- `POST /start_interrogation` - Initialize a new interrogation session
- `POST /ask_question` - Send a detective question and get suspect response
- `GET /session_status` - Get current session state
- `POST /end_interrogation` - Conclude interrogation and get outcome

## 📊 Training & Metrics

The training pipeline tracks:
- **Escape Rate**: Percentage of interrogations where suspect escaped without admission
- **Caught Rate**: Percentage of interrogations where detective succeeded
- **Average Reward**: Mean reward over training episodes
- **Episode Length**: Number of turns in each interrogation

Training visualizations are saved as:
- `training_progress.png` - Training metrics over time
- `agent_analysis.png` - Strategy distribution analysis

## 📁 Case Files

Detective cases are stored in JSON format in the `cases/` directory:

```json
{
  "case_id": "case_001",
  "title": "The Missing Artifact",
  "background": "...",
  "truth": "You are guilty because...",
  "evidence": ["evidence1", "evidence2", ...],
  "suggested_questions": ["question1", "question2", ...]
}
```

## 🔧 Configuration

### Environment Variables
- `GEMINI_API_KEY`: Google Generative AI API key (required)
- `FLASK_PORT`: Server port (default: 5000)
- `MODEL_PATH`: Path to trained RL model

### Training Hyperparameters (in `train_agent.py`)
- `total_timesteps`: Total training steps
- `learning_rate`: RL learning rate
- `batch_size`: Batch size for PPO
- `n_steps`: Steps per environment step

## 🎮 Game Flow

1. **Detective starts interrogation** → Server initializes case with RL agent
2. **Agent selects strategy** → Based on current game state
3. **LLM generates response** → Gemini creates suspect dialogue
4. **Detective reacts** → Asks follow-up questions, presents evidence
5. **Environment updates** → Suspicion level, contradictions increase
6. **Game ends** → When suspect escapes or is caught

## 🧪 Testing

Run tests with pytest:
```bash
pytest
```

## 📚 Dependencies

- **stable-baselines3**: Reinforcement learning framework
- **gymnasium**: Game environment framework
- **torch**: Deep learning backend
- **google-generativeai**: LLM API
- **flask**: Web server
- **numpy, matplotlib**: Data processing and visualization
- **tensorboard**: Training visualization

See `requirement.txt` for complete list and versions.

## 🎯 Future Improvements

- [ ] Multi-suspect interrogations
- [ ] Evidence presentation mechanics
- [ ] Lawyer/legal system integration
- [ ] Multiple case difficulty levels
- [ ] Player skill rating system
- [ ] Advanced NLP for contradiction detection
- [ ] Neural network-based detective strategy optimization
- [ ] Multiplayer online interrogation mode

## 📖 References

- [Stable Baselines3 Documentation](https://stable-baselines3.readthedocs.io/)
- [Gymnasium Documentation](https://gymnasium.farama.org/)
- [Google Generative AI](https://ai.google.dev/)
- [PPO Algorithm Paper](https://arxiv.org/abs/1707.06347)

## 📝 License

This project is part of the AI for Games course at Polytechnique.

## 👥 Contributors

Developed as a course project for "AI for Games" - Polytechnique (2025)

## 📞 Support

For questions or issues, please open an issue on the GitHub repository.

---

**Note**: The suspect in this game is always guilty of the crime and will attempt to deceive the detective using various strategies. The goal is for the detective (player) to see through the lies and achieve a confession or sufficient evidence of guilt.
