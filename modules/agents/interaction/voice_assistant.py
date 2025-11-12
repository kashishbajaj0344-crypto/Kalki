#!/usr/bin/env python3
"""
Voice Assistant Agent (Phase 16)
Implements natural voice interaction with speech recognition and synthesis.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any, Dict, Optional
from dataclasses import dataclass

from ..base_agent import AgentCapability, BaseAgent

logger = logging.getLogger("kalki.agent.voice")


@dataclass
class VoiceConfig:
    """Configuration for voice interface"""
    speech_recognition_engine: str = "google"  # google, sphinx, azure
    text_to_speech_engine: str = "pyttsx3"  # pyttsx3, gtts, azure
    language: str = "en-US"
    voice_rate: int = 200  # words per minute
    voice_volume: float = 0.8  # 0.0 to 1.0
    auto_listen: bool = False
    wake_word: str = "kalki"


class VoiceAssistant(BaseAgent):
    """
    Voice Assistant Agent providing natural speech interaction.

    Capabilities:
    - Speech-to-text recognition
    - Text-to-speech synthesis
    - Voice command processing
    - Wake word detection
    - Continuous conversation
    """

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(
            name="VoiceAssistant",
            capabilities=[AgentCapability.VOICE_ASSISTANT],
            description="Natural voice interaction with speech recognition and synthesis",
            config=config or {}
        )

        self.voice_config = VoiceConfig(**self.config.get("voice", {}))
        self.speech_recognizer = None
        self.tts_engine = None
        self.is_listening = False
        self.conversation_active = False

    async def initialize(self) -> bool:
        """Initialize voice recognition and synthesis engines"""
        try:
            # Initialize speech recognition (optional)
            try:
                await self._init_speech_recognition()
                speech_recognition_ok = True
            except Exception as e:
                self.logger.warning(f"Speech recognition initialization failed: {e}")
                speech_recognition_ok = False

            # Initialize text-to-speech
            try:
                await self._init_text_to_speech()
                text_to_speech_ok = True
            except Exception as e:
                self.logger.warning(f"Text-to-speech initialization failed: {e}")
                text_to_speech_ok = False

            # Success if at least one modality works
            if speech_recognition_ok or text_to_speech_ok:
                self.logger.info(f"{self.name} initialized successfully")
                return True
            else:
                self.logger.error(f"{self.name} failed to initialize any voice capabilities")
                return False

        except Exception as e:
            self.logger.exception(f"Failed to initialize {self.name}: {e}")
            return False

    async def _init_speech_recognition(self):
        """Initialize speech recognition engine"""
        try:
            if self.voice_config.speech_recognition_engine == "google":
                import speech_recognition as sr
                self.speech_recognizer = sr.Recognizer()
                # Test microphone access
                try:
                    with sr.Microphone() as source:
                        self.speech_recognizer.adjust_for_ambient_noise(source, duration=1)
                    self.logger.info("Google Speech Recognition initialized with microphone")
                except Exception as mic_error:
                    self.logger.warning(f"Microphone not available: {mic_error}. Speech recognition will be limited.")
                    self.microphone_available = False
                self.microphone_available = True
            else:
                raise ValueError(f"Unsupported speech recognition engine: {self.voice_config.speech_recognition_engine}")
        except ImportError:
            self.logger.error("speech_recognition library not installed. Install with: pip install SpeechRecognition")
            self.speech_recognizer = None
            self.microphone_available = False
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize speech recognition: {e}")
            self.speech_recognizer = None
            self.microphone_available = False
            raise

    async def _init_text_to_speech(self):
        """Initialize text-to-speech engine"""
        try:
            if self.voice_config.text_to_speech_engine == "pyttsx3":
                import pyttsx3
                self.tts_engine = pyttsx3.init()
                self.tts_engine.setProperty('rate', self.voice_config.voice_rate)
                self.tts_engine.setProperty('volume', self.voice_config.voice_volume)
                self.logger.info("pyttsx3 Text-to-Speech initialized")
            elif self.voice_config.text_to_speech_engine == "gtts":
                from gtts import gTTS
                import pygame
                self.tts_engine = gTTS
                self.audio_player = pygame
                self.audio_player.mixer.init()
                self.logger.info("Google Text-to-Speech initialized")
            else:
                raise ValueError(f"Unsupported TTS engine: {self.voice_config.text_to_speech_engine}")
        except ImportError as e:
            self.logger.error(f"TTS library not installed: {e}. Install pyttsx3 or gtts")
            raise
        except Exception as e:
            self.logger.error(f"Failed to initialize text-to-speech: {e}")
            raise

    async def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute voice assistant tasks"""
        action = task.get("action")
        params = task.get("params", {})

        if action == "listen":
            return await self._listen_for_speech(params)
        elif action == "speak":
            return await self._speak_text(params)
        elif action == "start_conversation":
            return await self._start_conversation(params)
        elif action == "stop_conversation":
            return await self._stop_conversation(params)
        elif action == "process_voice_command":
            return await self._process_voice_command(params)
        else:
            return {"status": "error", "error": f"Unknown action: {action}"}

    async def _listen_for_speech(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Listen for speech input and convert to text"""
        if not hasattr(self, 'microphone_available') or not self.microphone_available:
            return {"status": "error", "error": "Microphone not available. Speech recognition requires microphone access."}

        try:
            timeout = params.get("timeout", 5)
            phrase_time_limit = params.get("phrase_time_limit", 10)

            # Run speech recognition in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(
                None,
                self._recognize_speech_blocking,
                timeout,
                phrase_time_limit
            )

            if text:
                return {
                    "status": "success",
                    "text": text,
                    "confidence": 0.8  # Placeholder - actual engines provide confidence
                }
            else:
                return {"status": "no_speech", "message": "No speech detected"}

        except Exception as e:
            self.logger.exception(f"Speech recognition failed: {e}")
            return {"status": "error", "error": str(e)}

    def _recognize_speech_blocking(self, timeout: int, phrase_time_limit: int) -> Optional[str]:
        """Blocking speech recognition function"""
        try:
            import speech_recognition as sr
            with sr.Microphone() as source:
                self.logger.info("Listening for speech...")
                audio = self.speech_recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_time_limit)

            # Recognize speech using Google Speech Recognition
            text = self.speech_recognizer.recognize_google(audio, language=self.voice_config.language)
            self.logger.info(f"Recognized: {text}")
            return text

        except sr.WaitTimeoutError:
            return None
        except sr.UnknownValueError:
            self.logger.warning("Speech recognition could not understand audio")
            return None
        except sr.RequestError as e:
            self.logger.error(f"Speech recognition service error: {e}")
            return None
        except AttributeError as e:
            if "PyAudio" in str(e):
                self.logger.warning("PyAudio not available - speech recognition disabled")
                return None
            raise
        except Exception as e:
            self.logger.exception(f"Speech recognition error: {e}")
            return None

    async def _speak_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Convert text to speech and play it"""
        try:
            text = params.get("text", "")
            if not text:
                return {"status": "error", "error": "No text provided"}

            # Run TTS in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._speak_text_blocking, text)

            return {"status": "success", "message": "Speech synthesized successfully"}

        except Exception as e:
            self.logger.exception(f"Text-to-speech failed: {e}")
            return {"status": "error", "error": str(e)}

    def _speak_text_blocking(self, text: str):
        """Blocking text-to-speech function"""
        try:
            if self.voice_config.text_to_speech_engine == "pyttsx3":
                self.tts_engine.say(text)
                self.tts_engine.runAndWait()
            elif self.voice_config.text_to_speech_engine == "gtts":
                tts = self.tts_engine(text=text, lang=self.voice_config.language[:2])
                tts.save("temp_speech.mp3")
                self.audio_player.mixer.music.load("temp_speech.mp3")
                self.audio_player.mixer.music.play()
                while self.audio_player.mixer.music.get_busy():
                    time.sleep(0.1)
                # Cleanup
                import os
                os.remove("temp_speech.mp3")

        except Exception as e:
            self.logger.exception(f"TTS playback error: {e}")

    async def _start_conversation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Start continuous voice conversation mode"""
        try:
            self.conversation_active = True
            wake_word = params.get("wake_word", self.voice_config.wake_word)

            # Start conversation loop in background
            asyncio.create_task(self._conversation_loop(wake_word))

            return {"status": "success", "message": f"Voice conversation started. Say '{wake_word}' to begin."}

        except Exception as e:
            self.logger.exception(f"Failed to start conversation: {e}")
            return {"status": "error", "error": str(e)}

    async def _stop_conversation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Stop continuous voice conversation"""
        self.conversation_active = False
        return {"status": "success", "message": "Voice conversation stopped."}

    async def _conversation_loop(self, wake_word: str):
        """Main conversation loop for continuous interaction"""
        try:
            while self.conversation_active:
                # Listen for wake word
                result = await self._listen_for_speech({"timeout": 30, "phrase_time_limit": 5})
                if result.get("status") == "success":
                    text = result["text"].lower()
                    if wake_word.lower() in text:
                        # Wake word detected, start interaction
                        await self._speak_text({"text": "Yes? I'm listening."})

                        # Listen for command
                        command_result = await self._listen_for_speech({"timeout": 10, "phrase_time_limit": 10})
                        if command_result.get("status") == "success":
                            command = command_result["text"]
                            # Process command (would integrate with other agents)
                            await self._process_voice_command({"command": command})
                        else:
                            await self._speak_text({"text": "I didn't catch that. Please try again."})
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting

        except Exception as e:
            self.logger.exception(f"Conversation loop error: {e}")
            self.conversation_active = False

    async def _process_voice_command(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Process voice commands and generate responses"""
        try:
            command = params.get("command", "")
            if not command:
                return {"status": "error", "error": "No command provided"}

            # Simple command processing (would be enhanced with NLP)
            command_lower = command.lower()

            if "hello" in command_lower or "hi" in command_lower:
                response = "Hello! How can I help you today?"
            elif "time" in command_lower:
                from datetime import datetime
                current_time = datetime.now().strftime("%I:%M %p")
                response = f"The current time is {current_time}."
            elif "date" in command_lower:
                from datetime import datetime
                current_date = datetime.now().strftime("%A, %B %d, %Y")
                response = f"Today is {current_date}."
            elif "status" in command_lower:
                response = "All systems operational. I'm ready to assist you."
            else:
                response = f"I heard: {command}. I'm still learning to understand complex commands."

            # Speak the response
            await self._speak_text({"text": response})

            return {
                "status": "success",
                "command": command,
                "response": response
            }

        except Exception as e:
            self.logger.exception(f"Voice command processing failed: {e}")
            return {"status": "error", "error": str(e)}

    async def shutdown(self) -> bool:
        """Clean shutdown of voice assistant"""
        try:
            self.conversation_active = False
            self.is_listening = False

            # Stop TTS engine if needed
            if hasattr(self, 'tts_engine') and self.tts_engine:
                if self.voice_config.text_to_speech_engine == "pyttsx3":
                    self.tts_engine.stop()

            self.logger.info(f"{self.name} shut down successfully")
            return True

        except Exception as e:
            self.logger.exception(f"Shutdown failed for {self.name}: {e}")
            return False