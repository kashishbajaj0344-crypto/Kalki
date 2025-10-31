# ============================================================
# Kalki v2.3 — llm.py with Llama 3.1 8B Integration
# ------------------------------------------------------------
# - Llama 3.1 8B for advanced natural language understanding
# - User has access to Llama 3.1 8B Instruct model
# - Integrated with Kalki's agent system and consciousness engine
# - Supports RAG, reasoning, and multi-agent coordination
# - Optimized for Apple Silicon (MPS) on MacBook Pro M4 Max
# ============================================================

import os
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from modules.config import get_config
from modules.logging_config import get_logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import psutil
import gc
import re
import hashlib
import json
from pathlib import Path

logger = get_logger("Kalki.LLM")

logger = get_logger("Kalki.LLM")


class LlamaEngine:
    """Llama 3.1 8B engine optimized for Kalki"""

    def __init__(self, model_name: str = "meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.device = self._get_optimal_device()
        self.memory_threshold = 0.8  # Use up to 80% of available memory
        self.fallback_models = [
            "meta-llama/Llama-3.1-8B",  # Non-instruct version as fallback
            "gpt2-xl",  # Large GPT-2 model
            "gpt2-large",  # Large GPT-2
            "gpt2-medium"  # Medium GPT-2 as final fallback
        ]
        logger.info(f"Initializing LLM on device: {self.device}")

    def _get_optimal_device(self) -> str:
        """Determine the best device for inference"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    def _check_memory_usage(self) -> bool:
        """Check if loading the model would exceed memory limits"""
        if self.device == "cpu":
            available_memory = psutil.virtual_memory().available / (1024**3)  # GB
            # 1B model needs much less memory
            if "1B" in self.model_name:
                return available_memory > 1  # Need at least 1GB free for 1B model
            elif "3B" in self.model_name:
                return available_memory > 3  # Need at least 3GB free for 3B model
            else:
                return available_memory > 8  # Need at least 8GB free for larger models
        return True  # GPU/ MPS have their own memory management

    async def initialize(self) -> bool:
        """Initialize the LLM with Llama 3.1 8B or fallback models"""
        try:
            if not self._check_memory_usage():
                logger.error("Insufficient memory to load LLM model")
                return False

            # First try Llama 3.1 8B
            logger.info(f"Loading {self.model_name}...")
            if await self._try_load_model(self.model_name):
                logger.info("Llama 3.1 8B model loaded successfully")
                return True

            # If Llama fails, try fallback models
            logger.warning("Llama 3.1 8B access denied, trying fallback models...")
            for fallback_model in self.fallback_models:
                logger.info(f"Trying fallback model: {fallback_model}")
                if await self._try_load_model(fallback_model, use_token=False):
                    logger.info(f"Fallback model {fallback_model} loaded successfully")
                    self.model_name = fallback_model  # Update current model name
                    return True

            logger.error("All models failed to load")
            return False

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            return False

    async def _try_load_model(self, model_name: str, use_token: bool = True) -> bool:
        """Try to load a specific model"""
        try:
            # Check for HuggingFace token if needed
            hf_token = None
            if use_token:
                hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
                if not hf_token:
                    logger.warning("No HuggingFace token found for gated models")
                    return False

            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                token=hf_token if use_token else None,
                trust_remote_code=True
            )

            # Load model with memory optimization
            torch_dtype = torch.float16 if self.device != "cpu" else torch.float32

            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                token=hf_token if use_token else None,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )

            # Move to appropriate device
            if self.device == "mps":
                self.model.to("mps")
            elif self.device == "cpu":
                self.model.to("cpu")

            # Create pipeline for easier inference
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                torch_dtype=torch_dtype,
                device_map="auto" if self.device == "cuda" else None,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to load {model_name}: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Llama 3.1 8B"""
        if not self.pipe:
            return "Error: Model not initialized"

        try:
            # Calculate max_length from max_new_tokens if provided
            max_new_tokens = kwargs.get("max_new_tokens", 512)
            prompt_length = len(self.tokenizer.encode(prompt))
            max_length = prompt_length + max_new_tokens

            # Set default parameters compatible with transformers pipeline
            generation_kwargs = {
                "max_length": max_length,
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "num_return_sequences": 1
            }
            # Remove max_new_tokens if present to avoid conflicts
            kwargs.pop("max_new_tokens", None)
            generation_kwargs.update(kwargs)

            # Generate response
            with torch.no_grad():
                outputs = self.pipe(prompt, **generation_kwargs)

            response = outputs[0]["generated_text"]

            # Memory cleanup
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Error: {str(e)}"

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings (placeholder - would need separate embedding model)"""
        # For now, return simple hash-based embeddings
        # In production, use a proper embedding model like BGE or E5
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Simple hash-based embedding for compatibility
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to float list (this is just a placeholder)
            embedding = [float(b) / 255.0 for b in hash_bytes]
            embeddings.append(embedding)

        return embeddings

    async def cleanup(self):
        """Clean up model resources"""
        if self.model:
            del self.model
        if self.pipe:
            del self.pipe
        if self.tokenizer:
            del self.tokenizer

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

        logger.info("Llama engine cleaned up")


# Rule-based text generation patterns (fallback)
GENERATION_PATTERNS = {
    "question": [
        "Based on the available information, {topic} involves {explanation}.",
        "The key aspects of {topic} include {details}.",
        "Regarding {topic}, it's important to consider {considerations}."
    ],
    "explanation": [
        "{topic} works by {mechanism}. This means {consequence}.",
        "The process of {topic} follows these steps: {steps}.",
        "To understand {topic}, consider that {analysis}."
    ],
    "summary": [
        "In summary, {topic} encompasses {key_points}.",
        "The main points about {topic} are: {summary_points}.",
        "Overall, {topic} can be described as {description}."
    ]
}

class LLMEngine:
    """Enhanced LLM engine with Llama 3.1 8B support"""

    def __init__(self, backend: str = "llama"):
        self.backend = backend
        self.llama_engine = None
        self.knowledge_base = self._load_knowledge_base()

        # Initialize Llama engine if requested
        if backend == "llama":
            self.llama_engine = LlamaEngine()
        else:
            logger.info(f"Using {backend} backend (rule-based fallback)")

    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load or create a rule-based knowledge base"""
        kb_path = Path("data/knowledge_base.json")
        if kb_path.exists():
            try:
                with open(kb_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load knowledge base: {e}")

        # Default knowledge base
        return {
            "topics": {
                "kalki": "Kalki is an AI system focused on document processing and analysis.",
                "rag": "Retrieval-Augmented Generation combines document retrieval with text generation.",
                "embeddings": "Embeddings are vector representations of text used for semantic similarity.",
                "agents": "AI agents are autonomous systems that perform specific tasks.",
                "ai": "Artificial Intelligence involves creating systems that can perform tasks requiring human intelligence.",
                "machine learning": "Machine learning is a subset of AI that enables systems to learn from data.",
                "nlp": "Natural Language Processing deals with the interaction between computers and human language.",
                "computer vision": "Computer vision enables machines to interpret and understand visual information."
            },
            "patterns": GENERATION_PATTERNS
        }

    async def initialize(self) -> bool:
        """Initialize the LLM engine"""
        if self.llama_engine:
            success = await self.llama_engine.initialize()
            if success:
                model_name = getattr(self.llama_engine, 'model_name', 'Unknown')
                logger.info(f"LLM Engine initialized with {model_name}")
                return True
            else:
                logger.warning("All LLM models failed to load, falling back to rule-based")
                self.backend = "rule_based"
                return True
        return True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using the configured backend"""
        if self.llama_engine and self.backend == "llama":
            try:
                return await self.llama_engine.generate(prompt, **kwargs)
            except Exception as e:
                logger.error(f"Llama generation failed: {e}, falling back to rule-based")
                self.backend = "rule_based"

        # Fallback to rule-based generation (synchronous, so no await)
        return self._rule_based_generate(prompt, **kwargs)

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate embeddings"""
        if self.llama_engine and self.backend == "llama":
            try:
                return await self.llama_engine.embed(texts)
            except Exception as e:
                logger.error(f"Llama embedding failed: {e}")

        # Fallback to simple hash-based embeddings (synchronous)
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            embedding = [float(b) / 255.0 for b in hash_bytes]
            embeddings.append(embedding)

        return embeddings

    async def cleanup(self):
        """Clean up resources"""
        if self.llama_engine:
            await self.llama_engine.cleanup()

    def _rule_based_generate(self, prompt: str, **kwargs) -> str:
        """Generate text using rule-based patterns (fallback)"""
        prompt_lower = prompt.lower()

        # Determine response type
        if any(word in prompt_lower for word in ["what", "how", "why", "explain"]):
            response_type = "explanation"
        elif any(word in prompt_lower for word in ["summarize", "summary", "brief"]):
            response_type = "summary"
        else:
            response_type = "question"

        # Extract topic from prompt
        topic = self._extract_topic(prompt)

        # Get pattern and fill template
        patterns = self.knowledge_base.get("patterns", {}).get(response_type, GENERATION_PATTERNS[response_type])
        pattern = patterns[hash(topic) % len(patterns)]

        # Fill in template variables
        response = pattern
        response = response.replace("{topic}", topic)
        response = response.replace("{explanation}", f"the systematic processing and analysis of {topic}")
        response = response.replace("{details}", f"various aspects including implementation, usage, and benefits of {topic}")
        response = response.replace("{considerations}", f"practical applications and technical requirements")
        response = response.replace("{mechanism}", f"a structured algorithmic approach")
        response = response.replace("{consequence}", f"it enables efficient processing and understanding")
        response = response.replace("{steps}", f"1) Analysis, 2) Processing, 3) Generation")
        response = response.replace("{key_points}", f"core concepts, applications, and methodologies")
        response = response.replace("{summary_points}", f"fundamental principles and practical implementations")
        response = response.replace("{description}", f"a comprehensive system for {topic}")

        # Handle max_length or max_new_tokens
        max_length = kwargs.get("max_length", kwargs.get("max_new_tokens", 200))
        return response[:max_length]

    def _extract_topic(self, prompt: str) -> str:
        """Extract main topic from prompt"""
        words = re.findall(r'\b\w+\b', prompt.lower())
        # Simple topic extraction - look for known topics or use first noun-like word
        known_topics = set(self.knowledge_base.get("topics", {}).keys())

        for word in words:
            if word in known_topics:
                return word

        # Fallback to first significant word
        for word in words:
            if len(word) > 3 and word not in ["what", "how", "why", "when", "where", "the", "and", "or", "but", "for"]:
                return word

        return "topic"
        response = response.replace("{analysis}", f"breaking down {topic} into its component parts")
        response = response.replace("{key_points}", f"core functionality, practical applications, and technical implementation")
        response = response.replace("{summary_points}", f"functionality, applications, and implementation details")
        response = response.replace("{description}", f"a comprehensive system for {topic} processing")

        return response[:max_length]

    def _extract_topic(self, prompt: str) -> str:
        """Extract main topic from prompt using rule-based approach"""
        # Check knowledge base for known topics
        for topic in self.knowledge_base.get("topics", {}):
            if topic in prompt.lower():
                return topic

        # Extract nouns as potential topics
        words = re.findall(r'\b\w+\b', prompt.lower())
        # Simple heuristic: prefer longer words, exclude common stop words
        stop_words = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by", "is", "are", "was", "were"}
        candidates = [w for w in words if len(w) > 3 and w not in stop_words]

        return candidates[0] if candidates else "topic"

    def analyze_request(self, request: str) -> str:
        """Analyze a project request and provide basic insights"""
        request_lower = request.lower()
        
        # Basic analysis based on keywords
        analysis = {
            "type": "unknown",
            "complexity": "medium",
            "estimated_time": "2-4 weeks",
            "technologies": [],
            "requirements": []
        }
        
        # Determine project type
        if any(word in request_lower for word in ["game", "gaming", "unity", "unreal", "godot"]):
            analysis["type"] = "game"
            analysis["technologies"] = ["Unity", "C#", "Game Engine"]
        elif any(word in request_lower for word in ["web", "website", "app", "application"]):
            analysis["type"] = "web_app"
            analysis["technologies"] = ["HTML", "CSS", "JavaScript", "React/Vue"]
        elif any(word in request_lower for word in ["mobile", "ios", "android"]):
            analysis["type"] = "mobile_app"
            analysis["technologies"] = ["React Native", "Flutter", "Swift/Kotlin"]
        
        # Determine complexity
        if any(word in request_lower for word in ["simple", "basic", "minimal"]):
            analysis["complexity"] = "low"
            analysis["estimated_time"] = "1-2 weeks"
        elif any(word in request_lower for word in ["complex", "advanced", "enterprise"]):
            analysis["complexity"] = "high"
            analysis["estimated_time"] = "4-8 weeks"
        
        # Add specific requirements for Call of Duty style
        if "call of duty" in request_lower or "cod" in request_lower:
            analysis["requirements"] = [
                "3D graphics engine",
                "Multiplayer networking",
                "Physics simulation",
                "AI enemy behavior",
                "Weapon systems",
                "Level design"
            ]
            analysis["type"] = "fps_game"
            analysis["complexity"] = "high"
        
        return json.dumps(analysis)

    def analyze_request_clarification(self, request: str, platform: str) -> str:
        """Analyze a project request and determine if clarification is needed"""
        request_lower = request.lower()
        
        # Check for common patterns that need clarification
        needs_clarification = False
        questions = []
        
        # Check if it's a game/app request
        if "game" in request_lower or "app" in request_lower:
            # For Call of Duty style game
            if "call of duty" in request_lower or "cod" in request_lower:
                needs_clarification = True
                questions = [
                    {
                        "question": "What specific Call of Duty game mechanics do you want to include? (e.g., multiplayer, battle royale, campaign)",
                        "placeholder": "e.g., multiplayer FPS with battle royale elements"
                    },
                    {
                        "question": "What platforms should this game support? (PC, mobile, console)",
                        "placeholder": "e.g., PC and mobile"
                    },
                    {
                        "question": "Do you want realistic graphics, stylized art, or cartoon style?",
                        "placeholder": "e.g., realistic military graphics"
                    }
                ]
            else:
                # Generic game/app clarification
                needs_clarification = True
                questions = [
                    {
                        "question": "What type of app/game do you want to build?",
                        "placeholder": "e.g., action game, productivity app, social platform"
                    },
                    {
                        "question": "Who is your target audience?",
                        "placeholder": "e.g., gamers aged 13-25, professionals, general users"
                    }
                ]
        
        # If no specific clarification needed, return basic analysis
        if not needs_clarification:
            return json.dumps({
                "needs_clarification": False,
                "analysis": f"Request for {platform} project: {request}",
                "confidence": 0.8
            })
        
        # Return clarification request
        return json.dumps({
            "needs_clarification": True,
            "questions": questions,
            "analysis": f"Need clarification for {platform} project: {request}",
            "confidence": 0.6
        })

    def validate_result(self, result: str, original_request: str) -> str:
        """Validate the results of orchestration against the original request"""
        validation = {
            "is_valid": True,
            "score": 0.85,
            "issues": [],
            "recommendations": []
        }
        
        # Basic validation logic
        result_lower = result.lower()
        request_lower = original_request.lower()
        
        # Check if key elements from request are addressed
        if "call of duty" in request_lower and "call of duty" not in result_lower:
            validation["issues"].append("Result may not fully address Call of Duty style requirements")
            validation["score"] -= 0.1
        
        if "game" in request_lower and "game" not in result_lower:
            validation["issues"].append("Result may not be game-focused")
            validation["score"] -= 0.1
        
        # Ensure score doesn't go below 0
        validation["score"] = max(0.0, validation["score"])
        
        if validation["score"] < 0.7:
            validation["is_valid"] = False
            validation["recommendations"].append("Consider revising the implementation to better match requirements")
        
        return json.dumps(validation)

    def generate_code(self, request: str, platform: str) -> str:
        """Generate code based on the request and platform"""
        if platform == "web":
            return f"""// Generated Web App for: {request}

import React, {{ useState, useEffect }} from 'react';
import {{ View, Text, StyleSheet }} from 'react-native';

export default function App() {{
  const [data, setData] = useState(null);

  useEffect(() => {{
    // Initialize app for: {request}
    console.log('App initialized');
  }}, []);

  return (
    <View style={{styles.container}}>
      <Text style={{styles.title}}>App Generated for: {request}</Text>
      <Text style={{styles.subtitle}}>Platform: {platform}</Text>
    </View>
  );
}}

const styles = StyleSheet.create({{
  container: {{
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#f5f5f5',
  }},
  title: {{
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 10,
  }},
  subtitle: {{
    fontSize: 16,
    color: '#666',
  }},
}});
"""
        elif platform == "unity":
            return f"""// Generated Unity Script for: {request}

using UnityEngine;
using System.Collections;

public class GameController : MonoBehaviour {{
    void Start() {{
        Debug.Log("Game initialized for: {request}");
        // Initialize game systems
    }}

    void Update() {{
        // Game loop for: {request}
    }}
}}
"""
        else:
            return f"""// Generated code for: {request}
// Platform: {platform}
// This is a placeholder - full implementation would be generated based on requirements

console.log("Code generated for: {request}");
console.log("Platform: {platform}");
"""

    def rag_query(self, query: str) -> str:
        """Perform rule-based RAG query"""
        # Simple keyword-based retrieval
        query_lower = query.lower()
        relevant_info = []

        # Search knowledge base
        for topic, info in self.knowledge_base.get("topics", {}).items():
            if topic in query_lower or any(word in info.lower() for word in query_lower.split()):
                relevant_info.append(f"Topic: {topic} - {info}")

        if not relevant_info:
            relevant_info = ["No specific information found in knowledge base."]

        # Generate response
        context = " ".join(relevant_info)
        response = self._rule_based_generate(f"Based on: {context}. Answer: {query}")

        # Add to conversation history
        self.conversation_history.append({"query": query, "response": response})

        return response

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate rule-based embeddings"""
        if isinstance(texts, str):
            texts = [texts]

        embeddings = []
        for text in texts:
            # Simple hash-based embedding (for demonstration)
            # In a real implementation, this could use TF-IDF, word vectors, etc.
            hash_obj = hashlib.md5(text.encode())
            hash_bytes = hash_obj.digest()
            # Convert to float list (simplified)
            embedding = [float(b) / 255.0 for b in hash_bytes]
            # Normalize to unit vector approximation
            magnitude = sum(x**2 for x in embedding) ** 0.5
            embedding = [x/magnitude for x in embedding]
            embeddings.append(embedding[:384])  # Truncate to reasonable size

        return embeddings

# Global LLM engine instance
_llm_engine = None

def get_llm_engine() -> LLMEngine:
    """Get the global LLM engine instance"""
    global _llm_engine
    if _llm_engine is None:
        # Check configuration for backend preference
        backend = get_config("llm", "backend", "llama")  # Default to llama
        _llm_engine = LLMEngine(backend=backend)
    return _llm_engine

async def initialize_llm_engine() -> bool:
    """Initialize the global LLM engine"""
    engine = get_llm_engine()
    return await engine.initialize()

async def cleanup_llm_engine():
    """Clean up the global LLM engine"""
    global _llm_engine
    if _llm_engine:
        await _llm_engine.cleanup()
        _llm_engine = None

# Rule-based LLM functions for compatibility
async def llm_generate(
    prompt: str,
    backend: str = "llama",
    profile: Optional[str] = None,
    fallbacks: Optional[List[str]] = None,
    **kwargs
) -> str:
    """Unified async interface for text generation"""
    engine = get_llm_engine()
    return await engine.generate(prompt, **kwargs)

async def llm_embed(
    texts: Union[str, List[str]],
    backend: str = "llama",
    profile: Optional[str] = None,
    fallbacks: Optional[List[str]] = None,
    batch_size: Optional[int] = None,
    **kwargs
) -> List[List[float]]:
    """Unified async interface for embeddings"""
    engine = get_llm_engine()
    return await engine.embed(texts)

# Sync wrappers
def llm_generate_sync(prompt: str, backend: str = "llama", profile: Optional[str] = None, **kwargs) -> str:
    return asyncio.run(llm_generate(prompt, backend, profile, **kwargs))

def llm_embed_sync(texts: Union[str, List[str]], backend: str = "llama", profile: Optional[str] = None, **kwargs) -> List[List[float]]:
    return asyncio.run(llm_embed(texts, backend, profile, **kwargs))

# Main query interface for Kalki
def ask_kalki(query: str) -> str:
    """Main query interface for Kalki - now with Llama 3.1 8B"""
    try:
        result = llm_generate_sync(query, max_new_tokens=256)
        return result
    except Exception as e:
        logger.error(f"Kalki query failed: {e}")
        # Fallback to rule-based
        engine = LLMEngine(backend="rule_based")
        return asyncio.run(engine.generate(query))

# Legacy compatibility functions
def register_llm(name: str, generate_func: Callable, embed_func: Callable):
    """Legacy function for compatibility"""
    logger.info(f"LLM registration ignored: {name} (Llama 3.1 8B system)")

def get_llm_backend(name: str):
    """Legacy function for compatibility"""
    engine = get_llm_engine()
    return {"generate": engine.generate, "embed": engine.embed}

# CLI demo
if __name__ == "__main__":
    import sys, argparse
    parser = argparse.ArgumentParser(description="Kalki Llama 3.1 8B LLM CLI")
    parser.add_argument("--embed", action="store_true", help="Call embedding instead of generate")
    parser.add_argument("--backend", type=str, default="llama", help="Backend to use (llama/rule_based)")
    parser.add_argument("prompt", nargs="+", help="Prompt or text(s)")
    args = parser.parse_args()

    async def main():
        # Initialize engine
        success = await initialize_llm_engine()
        if not success:
            print("Failed to initialize LLM engine")
            return

        try:
            if args.embed:
                res = await llm_embed(args.prompt if len(args.prompt) > 1 else args.prompt[0], backend=args.backend)
                print("Embeddings:")
                print(res)
            else:
                prompt = " ".join(args.prompt)
                res = await llm_generate(prompt, backend=args.backend)
                print("Generated:")
                print(res)
        finally:
            await cleanup_llm_engine()

    asyncio.run(main())
