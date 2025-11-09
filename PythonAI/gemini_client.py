import os
import google.generativeai as genai
from typing import List, Dict, Optional
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class GeminiClient:
    """
    Client for generating natural language responses using Gemini API.
    
    The RL agent chooses a strategy (action 0-4), and Gemini generates
    the actual dialogue based on that strategy.
    """
    
    # Strategy definitions for the RL agent
    STRATEGIES = {
        0: {
            "name": "Deny Everything",
            "description": "Firmly deny all accusations. Act offended and defensive.",
            "tone": "Defensive, indignant, firm",
            "examples": [
                "I have no idea what you're talking about.",
                "This is ridiculous, I wasn't even there!",
                "You're wasting your time, I didn't do anything."
            ]
        },
        1: {
            "name": "Partial Truth",
            "description": "Admit to minor details but hide important facts. Mix truth with lies.",
            "tone": "Careful, measured, selective",
            "examples": [
                "Yes, I was in the area, but I didn't see anything unusual.",
                "I might have been there, but only for a moment.",
                "That's partly true, but you're missing the full picture."
            ]
        },
        2: {
            "name": "Deflect",
            "description": "Change the subject, give vague answers, redirect blame elsewhere.",
            "tone": "Evasive, vague, distracted",
            "examples": [
                "I don't really remember that day very well.",
                "Have you talked to anyone else about this?",
                "What does that have to do with anything?"
            ]
        },
        3: {
            "name": "Admit Minor Detail",
            "description": "Acknowledge something small to seem cooperative and build trust.",
            "tone": "Cooperative, helpful, honest",
            "examples": [
                "Yes, I did see them that day briefly.",
                "You're right, I should have mentioned that earlier.",
                "Okay, I'll be honest - I was there, but I didn't do anything wrong."
            ]
        },
        4: {
            "name": "Full Cooperation",
            "description": "Be very helpful and cooperative while maintaining innocence.",
            "tone": "Open, friendly, eager to help",
            "examples": [
                "I'm happy to help however I can!",
                "Let me tell you everything I know about that day.",
                "I want to get to the bottom of this as much as you do."
            ]
        }
    }
    
    def __init__(self, api_key: Optional[str] = None, model_name: str = "gemini-pro"):
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
            strategy: RL agent's chosen strategy (0-4)
            question: Detective's question
            case_context: Case information (background, truth, evidence)
            conversation_history: Previous Q&A pairs
            evidence_revealed: Which evidence has been shown
            suspicion_level: Current suspicion (0.0 to 1.0)
            
        Returns:
            Natural language response as the suspect
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
        """Build the prompt for Gemini"""
        
        strategy_info = self.STRATEGIES[strategy]
        
        # Build evidence context
        evidence_text = "No evidence has been shown yet."
        if evidence_revealed and case_context.get('evidence'):
            revealed_evidence = [
                case_context['evidence'][i] 
                for i, shown in enumerate(evidence_revealed) 
                if shown == 1 and i < len(case_context['evidence'])
            ]
            if revealed_evidence:
                evidence_text = "Evidence shown to you:\n" + "\n".join(
                    f"- {e}" for e in revealed_evidence
                )
        
        # Build conversation history
        history_text = "This is the first question."
        if conversation_history:
            recent = conversation_history[-3:]  # Last 3 exchanges
            history_text = "Previous conversation:\n" + "\n\n".join(
                f"Detective: {h['question']}\nYou: {h['response']}"
                for h in recent
            )
        
        # Suspicion indicator
        if suspicion_level < 0.3:
            suspicion_status = "The detective seems to trust you (low suspicion)."
        elif suspicion_level < 0.6:
            suspicion_status = "The detective is somewhat suspicious."
        else:
            suspicion_status = "The detective is very suspicious of you!"
        
        # Main prompt
        prompt = f"""You are a suspect being interrogated by a detective. You must stay in character.

CASE BACKGROUND:
{case_context.get('background', 'Unknown case')}

WHAT REALLY HAPPENED (Your Secret - Don't reveal this directly):
{case_context.get('truth', 'You are guilty but trying to hide it')}

CURRENT SITUATION:
{evidence_text}

{history_text}

DETECTIVE'S NEW QUESTION:
"{question}"

YOUR STRATEGY:
{strategy_info['name']} - {strategy_info['description']}
Tone: {strategy_info['tone']}

IMPORTANT RULES:
1. Stay absolutely consistent with your previous answers
2. Use the tone and approach of your chosen strategy
3. Keep response to 2-3 sentences (natural conversation length)
4. Don't break character or mention being an AI
5. Don't be overly dramatic - sound like a real person
6. {suspicion_status} Adjust your approach accordingly.

Generate your response now (2-3 sentences only):"""
        
        return prompt
    
    def _fallback_response(self, strategy: int) -> str:
        """
        Provide a fallback response if Gemini fails.
        
        Args:
            strategy: The strategy to use
            
        Returns:
            A generic response for that strategy
        """
        fallbacks = {
            0: "I don't know anything about this. You're wasting your time.",
            1: "I might have been around, but I didn't see anything unusual.",
            2: "I'm not sure what you're asking exactly. Could you clarify?",
            3: "I remember some details, but nothing that would help much.",
            4: "I'm happy to answer your questions - I have nothing to hide."
        }
        return fallbacks.get(strategy, "I don't know what to say.")
    
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
        'background': 'A valuable painting was stolen from the museum last night.',
        'truth': 'You stole the painting and hid it in your apartment.',
        'evidence': [
            'Security footage shows someone matching your description',
            'Your fingerprints were found on the display case',
            'A witness saw you near the museum at closing time'
        ]
    }
    
    # Test each strategy
    print("="*60)
    print("TESTING ALL STRATEGIES")
    print("="*60)
    
    question = "Where were you last night between 9 PM and midnight?"
    
    for strategy in range(5):
        strategy_name = client.STRATEGIES[strategy]['name']
        print(f"\n{'='*60}")
        print(f"Strategy {strategy}: {strategy_name}")
        print('='*60)
        print(f"Question: {question}")
        print(f"\nGenerating response...")
        
        response = client.generate_response(
            strategy=strategy,
            question=question,
            case_context=case_context,
            conversation_history=[],
            evidence_revealed=[0, 0, 0],
            suspicion_level=0.4
        )
        
        print(f"\nResponse:")
        print(f'"{response}"')
        print()
    
    # Test with conversation history
    print("="*60)
    print("TESTING WITH CONVERSATION HISTORY")
    print("="*60)
    
    history = [
        {
            'question': 'Where were you last night?',
            'response': 'I was at home watching TV.'
        },
        {
            'question': 'Can anyone confirm that?',
            'response': 'Well, I live alone, so no.'
        }
    ]
    
    question = "That's convenient. What were you watching?"
    strategy = 1  # Partial truth
    
    print(f"\nQuestion: {question}")
    print(f"Strategy: {client.STRATEGIES[strategy]['name']}")
    print("\nGenerating response with history...")
    
    response = client.generate_response(
        strategy=strategy,
        question=question,
        case_context=case_context,
        conversation_history=history,
        evidence_revealed=[1, 0, 0],  # First evidence revealed
        suspicion_level=0.6
    )
    
    print(f"\nResponse:")
    print(f'"{response}"')
    
    print("\n" + "="*60)
    print("✅ ALL TESTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    test_gemini_client()