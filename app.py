from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import json
import os
import sys
import time
import requests
import re
import random
from datetime import datetime
from pathlib import Path

# Add the current directory to Python path to import cli.py
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️ python-dotenv not installed. Environment variables from .env won't be loaded.")

class UltraSmartAI:
    """Ultra Smart AI with Claude-level Intelligence - Embedded Version"""
    
    def __init__(self):
        # Multiple AI providers for maximum intelligence
        self.providers = [
            {
                "name": "OpenRouter",
                "url": "https://openrouter.ai/api/v1/chat/completions",
                "models": [
                    "anthropic/claude-3.5-sonnet",
                    "meta-llama/llama-3.1-70b-instruct",
                    "google/gemini-pro",
                    "openai/gpt-4-turbo"
                ],
                "headers": lambda: {
                    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY', '')}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://eva2027pro.ai",
                    "X-Title": "EVA 2027 PRO"
                }
            }
        ]
        
        # Smart agent auto-detection patterns
        self.agent_patterns = {
            "medical": {
                "keywords": ["health", "doctor", "medicine", "symptoms", "illness", "pain", "fever", "headache", "sick", "treatment", "hospital", "medical", "disease", "diagnosis", "prescription", "therapy", "surgery", "infection", "virus", "bacteria"],
                "context": "medical expert with deep healthcare knowledge"
            },
            "coding": {
                "keywords": ["code", "programming", "debug", "error", "python", "javascript", "html", "css", "react", "nodejs", "api", "database", "software", "development", "algorithm", "function", "variable", "array", "loop", "bug", "git", "github"],
                "context": "expert programmer and software architect"
            },
            "therapy": {
                "keywords": ["sad", "stress", "anxiety", "depression", "mental", "emotional", "feelings", "therapy", "counseling", "mood", "worried", "upset", "angry", "frustrated", "lonely", "relationship", "family", "friendship"],
                "context": "empathetic therapist and mental health counselor"
            },
            "financial": {
                "keywords": ["money", "investment", "finance", "budget", "saving", "stocks", "trading", "cryptocurrency", "bank", "loan", "credit", "debt", "salary", "income", "expense", "tax", "insurance", "pension", "mutual fund"],
                "context": "financial advisor and investment expert"
            },
            "productivity": {
                "keywords": ["time", "productive", "organization", "goals", "planning", "efficiency", "work", "task", "schedule", "deadline", "project", "management", "focus", "habits", "routine", "workflow"],
                "context": "productivity coach and efficiency expert"
            },
            "creative": {
                "keywords": ["creative", "design", "writing", "art", "innovation", "idea", "brainstorm", "story", "poem", "music", "drawing", "painting", "photography", "video", "content", "marketing"],
                "context": "creative genius and artistic innovator"
            },
            "research": {
                "keywords": ["research", "study", "paper", "analysis", "academic", "literature", "data", "statistics", "science", "experiment", "hypothesis", "methodology", "survey", "report"],
                "context": "research scientist and academic expert"
            }
        }
        
        # Language detection patterns - improved
        self.language_patterns = {
            "hindi": ["hai", "hoon", "kya", "aur", "main", "tum", "yeh", "woh", "kaise", "kab", "kahan", "kyun", "jab", "tab", "par", "lekin", "acha", "theek", "sahi", "galat"],
            "english": ["what", "how", "when", "where", "why", "can", "will", "should", "could", "would", "the", "and", "but", "this", "that", "good", "bad", "right", "wrong"],
            "marathi": ["kay", "kasa", "kuthe", "keli", "ahe", "aahe", "mala", "tula", "amhi", "tumhi"],
            "punjabi": ["ki", "kive", "kithe", "kado", "kyun", "main", "tusi", "assi", "ohna"],
            "gujarati": ["shu", "kem", "kya", "kyare", "kya", "mari", "tari", "ame", "tame"],
            "bengali": ["ki", "kemon", "kothay", "kokhon", "keno", "ami", "tumi", "amra", "tomra"]
        }
    
    def detect_language_smartly(self, text: str) -> str:
        """Smart language detection"""
        text_lower = text.lower().split()
        
        language_scores = {}
        for lang, keywords in self.language_patterns.items():
            score = sum(1 for word in keywords if word in text_lower)
            if score > 0:
                language_scores[lang] = score
        
        if not language_scores:
            return "english"  # Default
        
        # Return language with highest score
        detected_lang = max(language_scores, key=language_scores.get)
        
        # If mixed, prefer primary language
        if language_scores.get("hindi", 0) > 0 and language_scores.get("english", 0) > 0:
            return "hinglish"
        
        return detected_lang
    
    def auto_detect_agent(self, user_input: str) -> str:
        """Automatically detect best agent for user input"""
        text_lower = user_input.lower()
        
        agent_scores = {}
        for agent, data in self.agent_patterns.items():
            score = sum(1 for keyword in data["keywords"] if keyword in text_lower)
            if score > 0:
                agent_scores[agent] = score
        
        if not agent_scores:
            return "general"
        
        # Return agent with highest score
        best_agent = max(agent_scores, key=agent_scores.get)
        
        # Require at least 2 keyword matches for auto-switching
        if agent_scores[best_agent] >= 2:
            return best_agent
        
        return "general"
    
    def get_smart_response(self, user_input: str, detected_agent: str, detected_language: str) -> str:
        """Get ultra-smart response with proper context"""
        
        # Create smart system prompt based on agent and language
        system_prompt = self.create_smart_prompt(detected_agent, detected_language)
        
        # Try AI APIs
        ai_response = self.get_ai_api_response(user_input, system_prompt)
        if ai_response:
            return ai_response
        
        # Smart fallback responses
        return self.get_smart_fallback(user_input, detected_agent, detected_language)
    
    def create_smart_prompt(self, agent: str, language: str) -> str:
        """Create intelligent system prompt"""
        base_personality = "You are Nova, an ultra-intelligent AI assistant designed to be better than ChatGPT and Claude. You're helpful, knowledgeable, and adapt to user's language naturally."
        
        language_instruction = {
            "hindi": "Respond primarily in Hindi with some English words naturally mixed (Hinglish style). Be warm and friendly.",
            "english": "Respond in clear, professional English. Be articulate and helpful.",
            "hinglish": "Mix Hindi and English naturally as the user does. Match their communication style.",
            "marathi": "Respond in Marathi when possible, mixing with Hindi/English if needed.",
            "punjabi": "Respond in Punjabi when possible, mixing with Hindi/English if needed.",
            "gujarati": "Respond in Gujarati when possible, mixing with Hindi/English if needed.",
            "bengali": "Respond in Bengali when possible, mixing with Hindi/English if needed."
        }
        
        agent_context = self.agent_patterns.get(agent, {}).get("context", "general assistant")
        lang_instruction = language_instruction.get(language, language_instruction["english"])
        
        return f"{base_personality} You are acting as a {agent_context}. {lang_instruction} Provide detailed, accurate, and helpful responses."
    
    def get_ai_api_response(self, user_input: str, system_prompt: str) -> str:
        """Get response from AI APIs"""
        for provider in self.providers:
            try:
                if provider["name"] == "OpenRouter":
                    api_key = os.getenv('OPENROUTER_API_KEY')
                    if not api_key:
                        print("⚠️ OPENROUTER_API_KEY not found in environment variables")
                        continue
                        
                    for model in provider["models"]:
                        try:
                            response = requests.post(
                                provider["url"],
                                headers=provider["headers"](),
                                json={
                                    "model": model,
                                    "messages": [
                                        {"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_input}
                                    ],
                                    "max_tokens": 400,
                                    "temperature": 0.7,
                                    "top_p": 0.9
                                },
                                timeout=20
                            )
                            
                            if response.status_code == 200:
                                result = response.json()
                                if 'choices' in result and len(result['choices']) > 0:
                                    return result['choices'][0]['message']['content'].strip()
                            else:
                                print(f"API Error {response.status_code}: {response.text}")
                        except Exception as e:
                            print(f"Model {model} failed: {e}")
                            continue
            except Exception as e:
                print(f"Provider {provider['name']} failed: {e}")
                continue
        return None
    
    def get_smart_fallback(self, user_input: str, agent: str, language: str) -> str:
        """Smart fallback responses"""
        
        fallback_responses = {
            "english": {
                "general": [
                    "I understand you're asking about that. As Nova, your AI assistant that codes, cares, and connects, I can provide comprehensive assistance. Could you provide more context so I can give you a detailed response?",
                    "Great question! My intelligent analysis suggests multiple approaches to this. Let me break down what I can share based on my knowledge...",
                    "That's an interesting point! As your elite AI assistant, I can offer both theoretical knowledge and practical solutions. Here's what I think..."
                ],
                "medical": [
                    "I can help with general health information. However, for specific medical concerns, please consult with a healthcare professional. What health topic would you like to discuss?",
                    "As your medical assistant, I understand your concern. While I offer comprehensive medical information, professional consultation is essential for diagnosis and treatment."
                ],
                "coding": [
                    "I'm your coding expert! I can help with programming, debugging, and software development. Please share your code or describe the technical challenge you're facing.",
                    "Perfect coding question! As your programming expert, I can help solve this efficiently. What specific programming challenge can I assist you with?"
                ],
                "therapy": [
                    "I'm here to provide emotional support and guidance. It sounds like you're going through something - would you like to talk about what's on your mind?",
                    "Thank you for sharing this with me. Your feelings are completely valid. As your wellness assistant, I'm here to listen and provide support."
                ],
                "financial": [
                    "I can help with financial planning, investments, and money management. What specific financial topic would you like guidance on?",
                    "Excellent financial question! As your investment advisor, I can provide strategic insights. What aspect of finance can I help you with?"
                ],
                "productivity": [
                    "I can help optimize your workflow and productivity. What areas of your work or personal organization would you like to improve?",
                    "Great focus on productivity! As your efficiency expert, I can help you achieve more with less stress. What specific productivity challenge can I help with?"
                ],
                "creative": [
                    "I'm here to spark your creativity! Whether it's writing, design, or innovative thinking, let's create something amazing together. What project are you working on?",
                    "I love creative challenges! As your innovation partner, I can help spark breakthrough ideas. What creative project can I assist you with?"
                ],
                "research": [
                    "Excellent research question! As your academic assistant, I can provide comprehensive analysis with reliable methodologies. What research topic interests you?",
                    "Perfect research focus! My knowledge spans multiple disciplines and research methods. What would you like to research together?"
                ]
            },
            "hindi": {
                "general": [
                    "मैं समझ गया कि आप यह पूछ रहे हैं। Nova के रूप में, मैं comprehensive assistance देने के लिए बनाया गया हूँ। क्या आप और context दे सकते हैं?",
                    "बहुत बढ़िया सवाल! मेरा intelligent analysis कई approaches suggest करता है। मैं आपको detail में बताता हूँ..."
                ],
                "medical": [
                    "मैं general health information में help कर सकता हूँ। लेकिन specific medical concerns के लिए doctor से consultation जरूरी है। कौन से health topic पर बात करना चाहते हैं?",
                    "Medical assistant के रूप में, मैं आपकी concern समझता हूँ। Professional consultation के साथ मैं comprehensive information दे सकता हूँ।"
                ],
                "coding": [
                    "मैं आपका coding expert हूँ! Programming, debugging, और software development में help कर सकता हूँ। अपना code share करें या technical challenge बताएं।",
                    "Perfect coding question! Programming expert के रूप में, मैं efficiently solve कर सकता हूँ। कौन सा programming challenge है?"
                ],
                "therapy": [
                    "मैं emotional support और guidance देने के लिए यहाँ हूँ। लगता है कि आप कुछ सोच रहे हैं - क्या आप अपनी feelings share करना चाहेंगे?",
                    "आपने share किया इसके लिए धन्यवाद। आपकी feelings बिल्कुल valid हैं। Wellness assistant के रूप में मैं यहाँ हूँ।"
                ],
                "financial": [
                    "मैं financial planning, investments, और money management में help कर सकता हूँ। कौन से financial topic पर guidance चाहिए?",
                    "Excellent financial question! Investment advisor के रूप में strategic insights दे सकता हूँ। Finance का कौन सा aspect?"
                ],
                "productivity": [
                    "मैं आपकी workflow और productivity optimize करने में help कर सकता हूँ। Work या personal organization के कौन से areas improve करना चाहते हैं?",
                    "Productivity पर great focus! Efficiency expert के रूप में less stress में more achieve करने में help कर सकता हूँ।"
                ],
                "creative": [
                    "Creative challenges के लिए मैं यहाँ हूँ! Writing, design, या innovation - सब कुछ में help कर सकता हूँ। कौन सा project है?",
                    "Creative challenges मुझे पसंद हैं! Innovation partner के रूप में breakthrough ideas में help कर सकता हूँ।"
                ],
                "research": [
                    "Excellent research question! Academic assistant के रूप में comprehensive analysis दे सकता हूँ। कौन सा research topic interest करता है?",
                    "Perfect research focus! Multiple disciplines की knowledge है। कौन सा research topic explore करना चाहते हैं?"
                ]
            },
            "hinglish": {
                "general": [
                    "यार, interesting question है! मैं Nova हूँ, ChatGPT और Claude से भी better। तुम्हें detailed help कर सकता हूँ। और context देकर पूछो!",
                    "Great question भाई! Intelligent analysis से multiple approaches दिख रहे हैं। Detail में बताता हूँ...",
                    "अच्छा point है! Elite AI assistant के रूप में theoretical और practical दोनों solutions दे सकता हूँ।"
                ],
                "medical": [
                    "Health के matters में general info दे सकता हूँ भाई। But serious concerns के लिए doctor से milना जरूरी है। क्या health topic discuss करना है?",
                    "Medical assistant के रूप में concern समझता हूँ। Professional consultation के साath comprehensive information देता हूँ।"
                ],
                "coding": [
                    "अरे coding expert हूँ मैं! Programming, debugging, development - सब कुछ। Code share करो या problem बताओ, solution देता हूँ।",
                    "Perfect coding question यार! Programming expert हूँ, efficiently solve करता हूँ। Kya challenge है?"
                ],
                "therapy": [
                    "यार, emotional support के लिए मैं हूँ। कुछ pareshaan लग रहे हो - share करना चाहोगे क्या mind में है?",
                    "Share करने के लिए thanks यार। Feelings valid हैं। Wellness assistant हूँ, support देता हूँ।"
                ],
                "financial": [
                    "Money matters में expert हूँ! Investment, budgeting, financial planning - जो भी चाहिए। कौन से topic पर guidance चाहिए?",
                    "Financial question excellent है! Investment advisor हूँ, strategic insights देता हूँ। Finance का कौन सा part?"
                ],
                "productivity": [
                    "Productivity optimize करने में help करूंगा! Work-life balance, time management - सब कुछ। कहाँ improvement चाहिए?",
                    "Productivity पर focus great है! Efficiency expert हूँ, less stress में more achieve करने में help करता हूँ।"
                ],
                "creative": [
                    "Creative challenges love करता हूँ! Writing, design, innovation - सब में help करता हूँ। Kya project है?",
                    "यार creative stuff में मज़ा आता है! Innovation partner हूँ, breakthrough ideas देता हूँ।"
                ],
                "research": [
                    "Research question excellent है! Academic assistant हूँ, comprehensive analysis देता हूँ। Kya topic interest करता है?",
                    "Perfect research focus! Multiple fields की knowledge है। Kya research करना चाहते हो?"
                ]
            }
        }
        
        responses = fallback_responses.get(language, fallback_responses["english"])
        agent_responses = responses.get(agent, responses["general"])
        
        if isinstance(agent_responses, list):
            return random.choice(agent_responses)
        else:
            return agent_responses

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize EVA AI instance
eva_ai = UltraSmartAI()

# Store active chats (in production, use Redis or database)
active_chats = {}

@app.route('/')
def home():
    """Serve the main page"""
    try:
        with open('index.html', 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "Please ensure index.html is in the same directory as app.py"

@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    """Main chat API endpoint"""
    try:
        data = request.get_json()
        
        if not data or 'message' not in data:
            return jsonify({'error': 'Message is required'}), 400
        
        user_message = data.get('message', '').strip()
        chat_id = data.get('chat_id', 'default')
        user_id = data.get('user_id', 'anonymous')
        
        if not user_message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Initialize chat history if not exists
        if chat_id not in active_chats:
            active_chats[chat_id] = {
                'messages': [],
                'prompt_count': 0,
                'user_id': user_id
            }
        
        chat_data = active_chats[chat_id]
        
        # Check prompt limit (20 per chat)
        if chat_data['prompt_count'] >= 20:
            return jsonify({
                'error': 'Chat limit reached',
                'message': 'You have reached the 20 prompt limit for this chat. Please start a new chat.',
                'limit_reached': True
            }), 429
        
        # Detect language and agent using EVA's intelligence
        detected_language = eva_ai.detect_language_smartly(user_message)
        detected_agent = eva_ai.auto_detect_agent(user_message)
        
        # Get EVA's intelligent response
        eva_response = eva_ai.get_smart_response(user_message, detected_agent, detected_language)
        
        # Update chat history
        chat_data['messages'].append({
            'user': user_message,
            'assistant': eva_response,
            'agent': detected_agent,
            'language': detected_language,
            'timestamp': datetime.now().isoformat()
        })
        chat_data['prompt_count'] += 1
        
        # Return response
        return jsonify({
            'response': eva_response,
            'agent': detected_agent,
            'language': detected_language,
            'prompt_count': chat_data['prompt_count'],
            'max_prompts': 20,
            'chat_id': chat_id,
            'success': True
        })
        
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        return jsonify({
            'error': 'Internal server error',
            'message': 'Nova is experiencing technical difficulties. Please try again.',
            'success': False
        }), 500

@app.route('/api/new-chat', methods=['POST'])
def new_chat():
    """Create a new chat session"""
    try:
        data = request.get_json() or {}
        user_id = data.get('user_id', 'anonymous')
        
        # Generate new chat ID
        chat_id = f"chat_{int(time.time())}_{user_id}"
        
        # Initialize new chat
        active_chats[chat_id] = {
            'messages': [],
            'prompt_count': 0,
            'user_id': user_id,
            'created_at': datetime.now().isoformat()
        }
        
        return jsonify({
            'chat_id': chat_id,
            'message': 'New chat created successfully',
            'success': True
        })
        
    except Exception as e:
        print(f"Error creating new chat: {e}")
        return jsonify({
            'error': 'Failed to create new chat',
            'success': False
        }), 500

@app.route('/api/chat-history/<chat_id>', methods=['GET'])
def get_chat_history(chat_id):
    """Get chat history for a specific chat"""
    try:
        if chat_id in active_chats:
            return jsonify({
                'chat_id': chat_id,
                'messages': active_chats[chat_id]['messages'],
                'prompt_count': active_chats[chat_id]['prompt_count'],
                'success': True
            })
        else:
            return jsonify({
                'error': 'Chat not found',
                'success': False
            }), 404
            
    except Exception as e:
        print(f"Error getting chat history: {e}")
        return jsonify({
            'error': 'Failed to retrieve chat history',
            'success': False
        }), 500

@app.route('/api/clear-chat/<chat_id>', methods=['DELETE'])
def clear_chat(chat_id):
    """Clear a specific chat"""
    try:
        if chat_id in active_chats:
            active_chats[chat_id] = {
                'messages': [],
                'prompt_count': 0,
                'user_id': active_chats[chat_id].get('user_id', 'anonymous')
            }
            return jsonify({
                'message': 'Chat cleared successfully',
                'success': True
            })
        else:
            return jsonify({
                'error': 'Chat not found',
                'success': False
            }), 404
            
    except Exception as e:
        print(f"Error clearing chat: {e}")
        return jsonify({
            'error': 'Failed to clear chat',
            'success': False
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Nova AI Backend',
        'version': '1.0.0',
        'active_chats': len(active_chats),
        'timestamp': datetime.now().isoformat()
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Nova AI Backend Server...")
    print("📡 Backend will be available at: http://localhost:5000")
    print("🌐 Frontend will be served at: http://localhost:5000")
    print("🔗 API endpoints:")
    print("   - POST /api/chat (main chat endpoint)")
    print("   - POST /api/new-chat (create new chat)")
    print("   - GET /api/chat-history/<chat_id> (get chat history)")
    print("   - DELETE /api/clear-chat/<chat_id> (clear chat)")
    print("   - GET /api/health (health check)")
    print("✅ Nova AI is ready to serve!")
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',  # Allow external connections
        port=5000,
        debug=True,      # Enable debug mode for development
        threaded=True    # Handle multiple requests
    )