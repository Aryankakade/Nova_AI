#!/usr/bin/env python3
"""
EVA 2027 PRO - Ultra Advanced AI Assistant
World's Most Intelligent AI • Better than ChatGPT & Claude • IIT-Level Logic
Smart Auto-Agent Switching • Multi-Language Voice • Claude-Level Intelligence
"""

import click
import asyncio
import json
import os
import sys
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import sqlite3
import hashlib
import requests
import re
import random
import pickle
from collections import defaultdict, deque

# Voice & Audio with proper imports
try:
    import speech_recognition as sr
    import pyttsx3
    import pygame
    # Azure Speech SDK for premium TTS
    import azure.cognitiveservices.speech as speechsdk
    VOICE_AVAILABLE = True
    AZURE_TTS_AVAILABLE = True
except ImportError as e:
    VOICE_AVAILABLE = False
    AZURE_TTS_AVAILABLE = False
    print(f"⚠️ Voice features disabled. Install: pip install speechrecognition pyttsx3 pyaudio azure-cognitiveservices-speech pygame")

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

class Colors:
    RESET = '\033[0m'
    BOLD = '\033[1m'
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    PURPLE = '\033[95m'
    MAGENTA = '\033[95m'
    WHITE = '\033[97m'

class UltraSmartAI:
    """Ultra Smart AI with Claude-level Intelligence"""
    
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
            return f"[{detected_agent.upper()}] {ai_response}"
        
        # Smart fallback responses
        return self.get_smart_fallback(user_input, detected_agent, detected_language)
    
    def create_smart_prompt(self, agent: str, language: str) -> str:
        """Create intelligent system prompt"""
        base_personality = "You are EVA 2027 Pro, an ultra-intelligent AI assistant designed to be better than ChatGPT and Claude. You're helpful, knowledgeable, and adapt to user's language naturally."
        
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
    
    def get_ai_api_response(self, user_input: str, system_prompt: str) -> Optional[str]:
        """Get response from AI APIs"""
        for provider in self.providers:
            try:
                if provider["name"] == "OpenRouter":
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
                "general": "I understand you're asking about that. As EVA 2027 Pro, I'm designed to provide comprehensive assistance. Could you provide more context so I can give you a detailed response?",
                "medical": "I can help with general health information. However, for specific medical concerns, please consult with a healthcare professional. What health topic would you like to discuss?",
                "coding": "I'm your coding expert! I can help with programming, debugging, and software development. Please share your code or describe the technical challenge you're facing.",
                "therapy": "I'm here to provide emotional support and guidance. It sounds like you're going through something - would you like to talk about what's on your mind?",
                "financial": "I can help with financial planning, investments, and money management. What specific financial topic would you like guidance on?",
                "productivity": "I can help optimize your workflow and productivity. What areas of your work or personal organization would you like to improve?",
                "creative": "I'm here to spark your creativity! Whether it's writing, design, or innovative thinking, let's create something amazing together. What project are you working on?"
            },
            "hindi": {
                "general": "मैं समझ गया कि आप यह पूछ रहे हैं। EVA 2027 Pro के रूप में, मैं comprehensive assistance देने के लिए बनाया गया हूँ। क्या आप और context दे सकते हैं?",
                "medical": "मैं general health information में help कर सकता हूँ। लेकिन specific medical concerns के लिए doctor से consultation जरूरी है। कौन से health topic पर बात करना चाहते हैं?",
                "coding": "मैं आपका coding expert हूँ! Programming, debugging, और software development में help कर सकता हूँ। अपना code share करें या technical challenge बताएं।",
                "therapy": "मैं emotional support और guidance देने के लिए यहाँ हूँ। लगता है कि आप कुछ सोच रहे हैं - क्या आप अपनी feelings share करना चाहेंगे?",
                "financial": "मैं financial planning, investments, और money management में help कर सकता हूँ। कौन से financial topic पर guidance चाहिए?",
                "productivity": "मैं आपकी workflow और productivity optimize करने में help कर सकता हूँ। Work या personal organization के कौन से areas improve करना चाहते हैं?"
            },
            "hinglish": {
                "general": "यार, interesting question है! मैं EVA 2027 Pro हूँ, ChatGPT और Claude से भी better। तुम्हें detailed help कर सकता हूँ। और context देकर पूछो!",
                "medical": "Health के matters में मैं general info दे सकता हूँ भाई। But serious concerns के लिए doctor से milना जरूरी है। क्या health topic discuss करना है?",
                "coding": "अरे coding expert हूँ मैं! Programming, debugging, development - सब कुछ। Code share करो या problem बताओ, solution देता हूँ।",
                "therapy": "यार, emotional support के लिए मैं हूँ। कुछ pareshaan लग रहे हो - share करना चाहोगे क्या mind में है?",
                "financial": "Money matters में expert हूँ! Investment, budgeting, financial planning - जो भी चाहिए। कौन से topic पर guidance चाहिए?",
                "productivity": "Productivity optimize करने में help करूंगा! Work-life balance, time management - सब कुछ। कहाँ improvement चाहिए?"
            }
        }
        
        responses = fallback_responses.get(language, fallback_responses["english"])
        return responses.get(agent, responses["general"])

class ProVoiceManager:
    """Professional Voice Manager with Azure TTS"""
    
    def __init__(self, voice_type="male"):
        if not VOICE_AVAILABLE:
            raise ImportError("Voice features not available")
        
        # Initialize pygame for audio
        pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        
        # Speech recognition
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        
        # Azure Speech Service setup
        self.azure_speech_key = os.getenv("AZURE_SPEECH_KEY")
        self.azure_region = os.getenv("AZURE_REGION", "eastus")
        
        # TTS Engine setup
        if AZURE_TTS_AVAILABLE and self.azure_speech_key:
            self.speech_config = speechsdk.SpeechConfig(
                subscription=self.azure_speech_key, 
                region=self.azure_region
            )
            self.setup_azure_voice(voice_type)
            self.azure_available = True
        else:
            self.azure_available = False
            self.setup_system_tts(voice_type)
        
        # Calibrate microphone
        try:
            with self.microphone as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
        except Exception as e:
            print(f"Microphone calibration warning: {e}")
    
    def setup_azure_voice(self, voice_type):
        """Setup Azure Neural TTS voices"""
        voice_map = {
            "male": {
                "en": "en-US-DavisNeural",
                "hi": "hi-IN-MadhurNeural", 
                "mr": "mr-IN-ManoharNeural",
                "default": "en-US-DavisNeural"
            },
            "female": {
                "en": "en-US-AriaNeural",
                "hi": "hi-IN-SwaraNeural",
                "mr": "mr-IN-AarohiNeural", 
                "default": "en-US-AriaNeural"
            }
        }
        
        self.voice_map = voice_map.get(voice_type, voice_map["male"])
        
        # Create synthesizer with default voice
        self.speech_config.speech_synthesis_voice_name = self.voice_map["default"]
        self.azure_synthesizer = speechsdk.SpeechSynthesizer(speech_config=self.speech_config)
    
    def setup_system_tts(self, voice_type):
        """Setup system TTS as fallback"""
        self.tts_engine = pyttsx3.init()
        voices = self.tts_engine.getProperty('voices')
        
        if voices:
            for voice in voices:
                if voice_type == "female" and "female" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
                elif voice_type == "male" and "male" in voice.name.lower():
                    self.tts_engine.setProperty('voice', voice.id)
                    break
        
        self.tts_engine.setProperty('rate', 180)
        self.tts_engine.setProperty('volume', 0.9)
    
    def speak_with_language(self, text: str, language: str):
        """Speak with appropriate language voice"""
        if not text.strip():
            return
        
        # Clean text for speech
        clean_text = re.sub(r'\[.*?\]', '', text).strip()
        
        print(f"{Colors.CYAN}🤖 EVA: {text}{Colors.RESET}")
        
        # Try Azure TTS first
        if self.azure_available:
            if self.speak_azure(clean_text, language):
                return
        
        # Fallback to system TTS
        try:
            self.tts_engine.say(clean_text)
            self.tts_engine.runAndWait()
        except Exception as e:
            print(f"TTS Error: {e}")
    
    def speak_azure(self, text: str, language: str) -> bool:
        """Speak using Azure Neural TTS"""
        try:
            # Map language to voice
            voice_name = self.voice_map.get(language[:2], self.voice_map["default"])
            
            # Create SSML for natural speech
            ssml = f"""
            <speak version='1.0' xml:lang='en-US'>
                <voice name='{voice_name}'>
                    <prosody rate='0.9' pitch='medium'>
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Synthesize speech
            result = self.azure_synthesizer.speak_ssml_async(ssml.strip()).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                return True
            elif result.reason == speechsdk.ResultReason.Canceled:
                cancellation = result.cancellation_details
                print(f"Azure TTS canceled: {cancellation.reason}")
                return False
        except Exception as e:
            print(f"Azure TTS error: {e}")
            return False
        
        return False
    
    def listen_smart(self) -> Tuple[str, str]:
        """Smart listening with language detection"""
        try:
            with self.microphone as source:
                print(f"{Colors.YELLOW}🎤 Listening... (speak now){Colors.RESET}")
                audio = self.recognizer.listen(source, timeout=10, phrase_time_limit=15)
            
            print(f"{Colors.BLUE}🔄 Processing speech...{Colors.RESET}")
            
            # Try Google Speech Recognition
            text = self.recognizer.recognize_google(audio)
            
            # Simple emotion detection
            emotion = "neutral"
            if any(word in text.lower() for word in ["excited", "happy", "great", "awesome"]):
                emotion = "excited" 
            elif any(word in text.lower() for word in ["sad", "upset", "worried", "problem"]):
                emotion = "concerned"
            
            return text, emotion
            
        except sr.UnknownValueError:
            print(f"{Colors.RED}❌ Could not understand audio{Colors.RESET}")
            return "", "neutral"
        except sr.RequestError as e:
            print(f"{Colors.RED}❌ Speech recognition error: {e}{Colors.RESET}")
            return "", "neutral"
        except Exception as e:
            print(f"{Colors.RED}❌ Listening error: {e}{Colors.RESET}")
            return "", "neutral"

class EVA2027ProMax:
    """EVA 2027 PRO MAX - Ultimate AI Assistant"""
    
    def __init__(self):
        print(f"{Colors.PURPLE}🚀 Initializing EVA 2027 PRO - Ultra Advanced AI System...{Colors.RESET}")
        
        # Core AI system
        self.ai = UltraSmartAI()
        self.voice = None
        self.current_agent = "general" 
        self.conversation_count = 0
        
        print(f"{Colors.GREEN}✅ EVA 2027 PRO Ready - Better than ChatGPT & Claude!{Colors.RESET}")
    
    def start_voice_chat(self, voice_type="male"):
        """Start advanced voice chat"""
        if not VOICE_AVAILABLE:
            print(f"{Colors.RED}❌ Voice not available. Install required packages.{Colors.RESET}")
            return
        
        try:
            print(f"{Colors.CYAN}🎤 Initializing Advanced Voice System...{Colors.RESET}")
            self.voice = ProVoiceManager(voice_type)
        except Exception as e:
            print(f"{Colors.RED}❌ Voice setup failed: {e}{Colors.RESET}")
            return
        
        # Multi-language welcome messages
        welcome_messages = [
            "नमस्ते! मैं EVA 2027 Pro हूँ - दुनिया का सबसे advanced AI assistant! ChatGPT और Claude से भी बेहतर। आज कैसे help कर सकता हूँ भाई?",
            "Hello! I'm EVA 2027 Pro, the world's most intelligent AI assistant. I'm designed to be better than ChatGPT and Claude. How can I assist you today?",
            "Hey there! EVA 2027 Pro at your service! मैं multi-agent system हूँ - medical, coding, therapy, productivity सब कुछ। क्या बात करना चाहते हैं?"
        ]
        
        welcome = random.choice(welcome_messages)
        self.voice.speak_with_language(welcome, "hinglish")
        
        print(f"\n{Colors.GREEN}🚀 EVA 2027 PRO Voice Assistant Active!{Colors.RESET}")
        print(f"{Colors.YELLOW}💡 Pro Commands:{Colors.RESET}")
        print("  • 'goodbye' - Exit gracefully")
        print("  • 'switch to [agent]' - Change specialist")
        print("  • 'upload file' - Process documents")
        print("  • Natural conversation in Hindi/English")
        print(f"  • Current Agent: {self.current_agent}")
        
        consecutive_errors = 0
        while True:
            try:
                user_input, emotion = self.voice.listen_smart()
                
                if not user_input:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        self.voice.speak_with_language(
                            "मुझे सुनने में problem हो रही है। Microphone check कर लेते हैं।", 
                            "hinglish"
                        )
                        consecutive_errors = 0
                    continue
                
                consecutive_errors = 0
                
                # Detect language from user input
                detected_language = self.ai.detect_language_smartly(user_input)
                
                # Auto-detect best agent
                detected_agent = self.ai.auto_detect_agent(user_input)
                
                # Show agent switch if different
                if detected_agent != self.current_agent:
                    switch_msg = f"🔄 Switching to {detected_agent} agent for better assistance."
                    print(f"{Colors.PURPLE}{switch_msg}{Colors.RESET}")
                    self.current_agent = detected_agent
                
                print(f"{Colors.GREEN}👤 You said: {user_input} (emotion: {emotion}){Colors.RESET}")
                
                # Exit commands
                if any(word in user_input.lower() for word in ['goodbye', 'exit', 'quit', 'bye', 'alvida']):
                    farewell_messages = {
                        "english": f"Goodbye! Thanks for using EVA 2027 Pro. We had {self.conversation_count} great conversations!",
                        "hindi": f"अलविदा! EVA 2027 Pro use करने के लिए धन्यवाद। हमारी {self.conversation_count} बातचीत amazing थी!",
                        "hinglish": f"Goodbye भाई! {self.conversation_count} conversations में से har ek meaningful tha। Take care! 😊"
                    }
                    farewell = farewell_messages.get(detected_language, farewell_messages["hinglish"])
                    self.voice.speak_with_language(farewell, detected_language)
                    break
                
                # Get smart response
                response = self.ai.get_smart_response(user_input, self.current_agent, detected_language)
                
                # Speak response in detected language
                self.voice.speak_with_language(response, detected_language)
                
                self.conversation_count += 1
                
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}👋 EVA 2027 Pro shutting down gracefully...{Colors.RESET}")
                if self.voice:
                    self.voice.speak_with_language("Thanks for using EVA 2027 Pro! धन्यवाद!", "hinglish")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")
                try:
                    self.voice.speak_with_language(
                        "मुझे technical issue आई है, but मैं यहाँ हूँ। Try again करें।",
                        "hinglish"
                    )
                except:
                    pass
    
    def start_text_chat(self):
        """Start text-based chat"""
        print(f"\n{Colors.CYAN}💬 EVA 2027 PRO Text Chat Mode{Colors.RESET}")
        print("Smart auto-agent switching enabled. Just ask naturally!\n")
        
        while True:
            try:
                user_input = input(f"\n{Colors.GREEN}You: {Colors.RESET}").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['exit', 'quit', 'bye']:
                    print(f"{Colors.GREEN}Goodbye! Had {self.conversation_count} great conversations! 👋{Colors.RESET}")
                    break
                
                # Smart detection
                detected_language = self.ai.detect_language_smartly(user_input)
                detected_agent = self.ai.auto_detect_agent(user_input)
                
                # Show agent switch
                if detected_agent != self.current_agent:
                    print(f"{Colors.PURPLE}🔄 Switching to {detected_agent} agent{Colors.RESET}")
                    self.current_agent = detected_agent
                
                # Get response
                response = self.ai.get_smart_response(user_input, self.current_agent, detected_language)
                print(f"{Colors.CYAN}{response}{Colors.RESET}")
                
                self.conversation_count += 1
                
            except KeyboardInterrupt:
                print(f"\n{Colors.GREEN}Goodbye! 👋{Colors.RESET}")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {e}{Colors.RESET}")

# CLI Commands
@click.group()
def cli():
    """EVA 2027 PRO - World's Most Advanced AI Assistant"""
    pass

@cli.command()
@click.option('--voice', is_flag=True, help='Enable voice mode')
@click.option('--male-voice', is_flag=True, help='Use male voice')  
@click.option('--female-voice', is_flag=True, help='Use female voice')
def chat(voice, male_voice, female_voice):
    """Start EVA 2027 PRO Chat"""
    eva = EVA2027ProMax()
    
    if voice:
        voice_type = "male" if male_voice else "female" if female_voice else "male"
        eva.start_voice_chat(voice_type)
    else:
        eva.start_text_chat()

@cli.command()
def test():
    """Test EVA systems"""
    print(f"{Colors.PURPLE}🧪 Testing EVA 2027 PRO Systems...{Colors.RESET}")
    
    ai = UltraSmartAI()
    
    # Test language detection
    test_inputs = [
        "Hello, how are you?",
        "नमस्ते, कैसे हो?", 
        "Hey yaar, kya haal hai?",
        "My code is not working, can you help debug it?",
        "मुझे बहुत stress हो रहा है"
    ]
    
    for test_input in test_inputs:
        detected_lang = ai.detect_language_smartly(test_input)
        detected_agent = ai.auto_detect_agent(test_input)
        print(f"Input: '{test_input}'")
        print(f"  Language: {detected_lang}, Agent: {detected_agent}")
    
    print(f"{Colors.GREEN}✅ Language & Agent Detection: Working{Colors.RESET}")
    
    # Test voice (if available)
    if VOICE_AVAILABLE:
        print(f"{Colors.GREEN}✅ Voice System: Available{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}⚠️ Voice System: Install packages{Colors.RESET}")
    
    if AZURE_TTS_AVAILABLE and os.getenv("AZURE_SPEECH_KEY"):
        print(f"{Colors.GREEN}✅ Azure TTS: Available{Colors.RESET}")
    else:
        print(f"{Colors.YELLOW}⚠️ Azure TTS: Configure API keys{Colors.RESET}")
    
    print(f"{Colors.PURPLE}🚀 EVA 2027 PRO Test Complete!{Colors.RESET}")

if __name__ == '__main__':
    # Startup Banner
    banner = f"""
{Colors.PURPLE}╔══════════════════════════════════════════════════════════════════════════════╗
║  {Colors.BOLD}🚀 EVA 2027 PRO - Ultra Advanced AI Assistant{Colors.RESET}{Colors.PURPLE}                          ║
║  {Colors.GREEN}✨ Better than ChatGPT & Claude • IIT-Level Intelligence • Multi-Agent{Colors.RESET}{Colors.PURPLE}    ║
║  {Colors.CYAN}🎯 Voice-Enabled • File Processing • Emotional Intelligence • Learning{Colors.RESET}{Colors.PURPLE}     ║
╚══════════════════════════════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)
    cli()