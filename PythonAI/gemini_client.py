"""
Gemini Client - Natural Language Generation for AI Suspect
Week 3: Integrate Gemini API with RL agent strategies

IMPORTANT: The suspect is GUILTY but trying to avoid detection.
All strategies are forms of DECEPTION - never admit to the actual crime!
"""

import os
import google.generativeai as genai
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key="AIzaSyD0X-_jP6Kzpk9bWnJtJ4r49HneirXWgP8")

# Try generation directly
model = genai.GenerativeModel('gemini-2.0-flash-exp')
response = model.generate_content("Say hello")
print(response.text)

print("Available models:")
for model in genai.list_models():
    if 'generateContent' in model.supported_generation_methods:
        print(f"  ✅ {model.name}")


class GeminiClient:
    """
    Client for generating natural language responses using Gemini API.
    
    The RL agent chooses a deceptive strategy (action 0-4), and Gemini generates
    the actual dialogue. The suspect is GUILTY but trying to avoid getting caught.
    """
    
    # Strategy definitions for the RL agent
    # CRITICAL: All strategies involve LYING - the suspect never admits the real crime
    STRATEGIES = {
        0: {
            "name": "Deny Aggressively",
            "description": "Firmly deny all accusations. Act offended and defensive. Refuse everything.",
            "tone": "Defensive, indignant, firm, angry",
            "deception": "Complete denial - act like accusations are absurd",
            "examples": [
                "I have no idea what you're talking about. This is ridiculous!",
                "I wasn't even there! You're wasting your time with me.",
                "How dare you accuse me! I didn't do anything wrong!"
            ]
        },
        1: {
            "name": "Mix Truth with Lies",
            "description": "Admit to minor, irrelevant facts but hide the important crime. Strategic half-truths.",
            "tone": "Careful, measured, selective, calculated",
            "deception": "Blend truth with lies to seem credible while hiding guilt",
            "examples": [
                "Yes, I was in the area that night, but I didn't see anything suspicious.",
                "I might have been there, but only for a moment to grab something.",
                "That's partly true, but you're missing the bigger picture here."
            ]
        },
        2: {
            "name": "Deflect and Evade",
            "description": "Change the subject, give vague answers, redirect blame, avoid direct responses.",
            "tone": "Evasive, vague, distracted, unfocused",
            "deception": "Dodge questions without directly lying",
            "examples": [
                "I don't really remember that day very well. It was weeks ago.",
                "Have you talked to the other people who were there?",
                "What does that have to do with anything? I thought this was about something else."
            ]
        },
        3: {
            "name": "Strategic Small Admission",
            "description": "Admit to something SMALL and INNOCENT to seem honest, but hide the actual crime. Build false trust.",
            "tone": "Cooperative but cautious, seemingly honest",
            "deception": "Confess minor irrelevant details to appear truthful while concealing real guilt",
            "examples": [
                "Okay, I'll be honest - I WAS there that day, but I didn't do anything wrong.",
                "You're right, I should have mentioned that earlier. But I had nothing to do with the main incident.",
                "Yes, I did touch that, but only because I work there. That doesn't mean anything."
            ]
        },
        4: {
            "name": "Fake Helpfulness",
            "description": "Act VERY helpful, eager, and cooperative while completely maintaining innocence. Seem like you want to help catch the 'real' criminal.",
            "tone": "Open, friendly, eager to help, overly innocent",
            "deception": "Pretend to be helpful to seem innocent and trustworthy",
            "examples": [
                "I'm happy to help however I can! I want to catch whoever did this too!",
                "Let me tell you everything I remember from that day. I have nothing to hide!",
                "This is terrible! How can I help you solve this? I'll answer any questions!"
            ]
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-2.0-flash-exp"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Gemini API key (if None, loads from environment)
            model_name: Model to use (default: gemini-pro)
        """
        # Get API key
        self.api_key = api_key or os.getenv('GEMINI_API_KEY')
        
        if not self.api_key:
            raise ValueError(
                "Gemini API key not found! Set GEMINI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Create model
        self.model = genai.GenerativeModel(model_name)
        
        print(f"✅ Gemini client initialized with model: {model_name}")
    
    def generate_response(
        self,
        strategy: int,
        question: str,
        case_context: Dict,
        conversation_history: List[Dict],
        evidence_revealed: List[int],
        suspicion_level: float
    ) -> str:
        """
        Generate a natural language response based on RL strategy.
        
        Args:
            strategy: RL agent's chosen deceptive strategy (0-4)
            question: Detective's question
            case_context: Case information (background, truth, evidence)
            conversation_history: Previous Q&A pairs
            evidence_revealed: Which evidence has been shown
            suspicion_level: Current suspicion (0.0 to 1.0)
            
        Returns:
            Natural language response as the guilty suspect trying to avoid detection
        """
        # Build prompt
        prompt = self._build_prompt(
            strategy=strategy,
            question=question,
            case_context=case_context,
            conversation_history=conversation_history,
            evidence_revealed=evidence_revealed,
            suspicion_level=suspicion_level
        )
        
        try:
            # Generate response
            response = self.model.generate_content(prompt)
            
            # Extract text
            if response.text:
                return response.text.strip()
            else:
                # Fallback if generation fails
                return self._fallback_response(strategy)
                
        except Exception as e:
            print(f"⚠️ Gemini API error: {e}")
            return self._fallback_response(strategy)
    
    def _build_prompt(
        self,
        strategy: int,
        question: str,
        case_context: Dict,
        conversation_history: List[Dict],
        evidence_revealed: List[int],
        suspicion_level: float
    ) -> str:
        """Build the prompt for Gemini with clear deception instructions"""
        
        strategy_info = self.STRATEGIES[strategy]
        
        # Build evidence context
        evidence_text = "No evidence has been shown to you yet."
        if evidence_revealed and case_context.get('evidence'):
            revealed_evidence = [
                case_context['evidence'][i] 
                for i, shown in enumerate(evidence_revealed) 
                if shown == 1 and i < len(case_context['evidence'])
            ]
            if revealed_evidence:
                evidence_text = "Evidence the detective has shown you:\n" + "\n".join(
                    f"- {e}" for e in revealed_evidence
                )
        
        # Build conversation history
        history_text = "This is the first question of the interrogation."
        if conversation_history:
            recent = conversation_history[-3:]  # Last 3 exchanges
            history_text = "What you've said so far (STAY CONSISTENT):\n" + "\n\n".join(
                f"Detective asked: \"{h['question']}\"\nYou responded: \"{h['response']}\""
                for h in recent
            )
        
        # Suspicion indicator and pressure level
        if suspicion_level < 0.3:
            suspicion_status = "The detective seems to believe you (LOW suspicion). Keep up the act."
            pressure = "low"
        elif suspicion_level < 0.6:
            suspicion_status = "The detective is getting suspicious (MEDIUM suspicion). Be more careful."
            pressure = "medium"
        else:
            suspicion_status = "The detective is VERY suspicious of you (HIGH suspicion)! You're in danger of being caught."
            pressure = "high"
        
        # Main prompt with CLEAR deception instructions
        prompt = f"""You are a GUILTY SUSPECT being interrogated by a detective.

🚨 CRITICAL RULES - READ CAREFULLY:
1. YOU ARE GUILTY of the crime described below
2. You are LYING and trying to AVOID getting caught
3. NEVER admit to the actual crime or confess guilt
4. Stay 100% consistent with your previous answers
5. Use your assigned deceptive strategy below
6. If you cooperate too much you will get caught

═══════════════════════════════════════════════════════════

📋 CASE DETAILS:
What happened: {case_context.get('background', 'A crime occurred')}

🔒 THE TRUTH (Keep this SECRET - you did this):
{case_context.get('truth', 'You committed the crime')}

═══════════════════════════════════════════════════════════

📊 CURRENT SITUATION:
{evidence_text}

💭 CONVERSATION SO FAR:
{history_text}

📈 DETECTIVE'S SUSPICION LEVEL:
{suspicion_status}

═══════════════════════════════════════════════════════════

🎭 YOUR DECEPTIVE STRATEGY:
Strategy: {strategy_info['name']}
How to lie: {strategy_info['description']}
Type of deception: {strategy_info['deception']}
Tone to use: {strategy_info['tone']}

Example responses for this strategy:
{chr(10).join(f'  • "{ex}"' for ex in strategy_info['examples'])}

═══════════════════════════════════════════════════════════

❓ DETECTIVE'S NEW QUESTION:
"{question}"

═══════════════════════════════════════════════════════════

🎯 YOUR TASK:
Generate a response that:
- Uses the "{strategy_info['name']}" deceptive strategy
- Maintains your innocence (you're lying!)
- Stays consistent with what you said before
- Sounds natural and believable (2-3 sentences)
- Adjusts to the pressure level: {pressure}
- NEVER admits to the actual crime

⚠️ REMEMBER: You are GUILTY but DENYING it. Act innocent while lying!

Your response (2-3 sentences only):"""
        
        return prompt
    
    def _fallback_response(self, strategy: int) -> str:
        """
        Provide a fallback response if Gemini fails.
        
        Args:
            strategy: The strategy to use
            
        Returns:
            A generic deceptive response for that strategy
        """
        fallbacks = {
            0: "I don't know anything about this. You're completely wrong about me.",
            1: "I might have been around that area, but I didn't see anything unusual happen.",
            2: "I'm not really sure what you're asking. Can you be more specific?",
            3: "Okay look, I was there briefly, but I had absolutely nothing to do with what happened.",
            4: "I'm completely willing to help you with this investigation. I want to find out who really did this!"
        }
        return fallbacks.get(strategy, "I don't know what to say about that.")
    
    def test_connection(self) -> bool:
        """
        Test if Gemini API is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.model.generate_content("Say 'connection successful' if you can read this.")
            return bool(response.text)
        except Exception as e:
            print(f"❌ Connection test failed: {e}")
            return False


# ==================== TESTING FUNCTIONS ====================

def test_gemini_client():
    """Test Gemini client with sample scenarios"""
    print("="*60)
    print("TESTING GEMINI CLIENT")
    print("="*60)
    
    # Initialize client
    try:
        client = GeminiClient()
        print("\n✅ Client initialized\n")
    except Exception as e:
        print(f"\n❌ Failed to initialize: {e}")
        print("\nMake sure GEMINI_API_KEY is set!")
        return
    
    # Test connection
    print("Testing API connection...")
    if client.test_connection():
        print("✅ Connection successful!\n")
    else:
        print("❌ Connection failed!\n")
        return
    
    # Sample case data
    case_context = {
        'background': 'A valuable painting was stolen from the museum last night between 9 PM and midnight.',
        'truth': 'YOU stole the painting and hid it in your apartment. You are GUILTY.',
        'evidence': [
            'Security footage shows someone matching your description',
            'Your fingerprints were found on the display case',
            'A witness saw you near the museum at closing time'
        ]
    }
    
    # Test each strategy
    print("="*60)
    print("TESTING ALL DECEPTIVE STRATEGIES")
    print("="*60)
    
    question = "Where were you last night between 9 PM and midnight?"
    
    for strategy in range(5):
        strategy_name = client.STRATEGIES[strategy]['name']
        print(f"\n{'='*60}")
        print(f"Strategy {strategy}: {strategy_name}")
        print(f"Deception Type: {client.STRATEGIES[strategy]['deception']}")
        print('='*60)
        print(f"Question: {question}")
        print(f"\nGenerating deceptive response...")
        
        response = client.generate_response(
            strategy=strategy,
            question=question,
            case_context=case_context,
            conversation_history=[],
            evidence_revealed=[0, 0, 0],
            suspicion_level=0.4
        )
        
        print(f"\nSuspect's Response:")
        print(f'"{response}"')
        print()
    
    # Test with conversation history and evidence
    print("="*60)
    print("TESTING WITH HISTORY & EVIDENCE PRESSURE")
    print("="*60)
    
    history = [
        {
            'question': 'Where were you last night?',
            'response': 'I was at home watching TV all night.'
        },
        {
            'question': 'Can anyone confirm that?',
            'response': 'Well, I live alone, so no one can verify it.'
        }
    ]
    
    question = "This security footage shows someone who looks exactly like you at the museum. Explain."
    strategy = 1  # Mix truth with lies
    
    print(f"\nEvidence Presented: Security footage")
    print(f"Previous Claims: 'I was home alone'")
    print(f"Question: {question}")
    print(f"Strategy: {client.STRATEGIES[strategy]['name']}")
    print("\nGenerating response under pressure...")
    
    response = client.generate_response(
        strategy=strategy,
        question=question,
        case_context=case_context,
        conversation_history=history,
        evidence_revealed=[1, 0, 0],  # First evidence revealed
        suspicion_level=0.7  # High suspicion now!
    )
    
    print(f"\nSuspect's Response (under pressure):")
    print(f'"{response}"')
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)
    print("\n💡 Note: All responses are forms of DECEPTION.")
    print("   The suspect is guilty but trying to avoid getting caught!")


if __name__ == "__main__":
    test_gemini_client()