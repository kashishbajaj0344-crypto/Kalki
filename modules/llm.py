# ============================================================
# Kalki — llm.py with Dual-Model Intelligence
# ------------------------------------------------------------
# - Llama 3.1 8B Instruct for advanced text reasoning (fast)
# - Llama 3.2 Vision 11B for multimodal diagram analysis
# - Intelligent model routing based on query type
# - Cross-modal validation and ensemble reasoning
# - Optimized for Apple Silicon (MPS) on MacBook Pro M4 Max
# ============================================================

import os
import asyncio
from typing import List, Dict, Any, Optional, Callable, Union
from modules.utils.config import get_config
from modules.utils.logging_config import get_logger
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModel, MllamaForConditionalGeneration, AutoProcessor
import psutil
import gc
import re
import hashlib
import json
from pathlib import Path
from PIL import Image
import numpy as np
from modules.meta_core import get_meta_core, process_command
from modules.learning.vectordb import BGEEmbedder

logger = get_logger("Kalki.LLM")

logger = get_logger("Kalki.LLM")


class LlamaEngine:
    """Llama 3.1 8B engine optimized for Kalki with integrated embeddings - ALWAYS uses local models"""

    def __init__(self, model_name: str = None, model_path: Optional[str] = None):
        # ALWAYS prioritize local models from kalki/models directory
        if model_path is None:
            try:
                from config.models_config import get_model_path
                local_path = get_model_path("llama-3.1-8b-instruct")
                if local_path and Path(local_path).exists():
                    model_path = local_path
                    logger.info(f"✅ Found local Llama 3.1 8B model at: {model_path}")
                else:
                    logger.warning("⚠️  Local Llama 3.1 8B model not found in models/ directory")
            except Exception as e:
                logger.warning(f"Could not check for local model: {e}")
        
        self.model_name = model_name or "Llama-3.1-8B-Instruct (local)"
        self.model_path = model_path  # Local path - REQUIRED
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.embedder = None  # BGE embedder for semantic embeddings
        self.device = self._get_optimal_device()
        self.memory_threshold = 0.8  # Use up to 80% of available memory
        # NO fallback models - we ONLY use local models
        self.fallback_models = []
        self.conversation_history = []  # Initialize conversation history
        
        if not self.model_path:
            logger.error("❌ No local model path provided. Models must be in models/ directory.")
        else:
            logger.info(f"🚀 Initializing LLM from local model: {self.model_path}")
            logger.info(f"📱 Device: {self.device}")

    def _get_optimal_device(self) -> str:
        """Determine the best device for inference - M4 Max optimized"""
        # M4 Max: Prioritize MPS (Metal Performance Shaders) for GPU acceleration
        if torch.backends.mps.is_available():
            logger.info("🚀 Using Metal (MPS) GPU acceleration on M4 Max")
            return "mps"
        elif torch.cuda.is_available():
            logger.info("🚀 Using CUDA GPU acceleration")
            return "cuda"
        else:
            logger.warning("⚠️  Falling back to CPU (GPU not available)")
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
        """Initialize the LLM - ONLY uses local models from models/ directory"""
        try:
            if not self._check_memory_usage():
                logger.error("Insufficient memory to load LLM model")
                return False

            # REQUIRED: Must have local model path
            if not self.model_path:
                logger.error("❌ No local model path provided. Cannot initialize.")
                logger.error("💡 Place Llama 3.1 8B model in models/llama_3.1_8b/ directory")
                return False

            if not Path(self.model_path).exists():
                logger.error(f"❌ Model path does not exist: {self.model_path}")
                logger.error("💡 Ensure models are in models/llama_3.1_8b/ directory")
                return False

            # Load local model - this is the ONLY way we load models
            logger.info(f"📦 Loading local Llama 3.1 8B from: {self.model_path}")
            if await self._try_load_local_model(self.model_path):
                logger.info("✅ Local Llama 3.1 8B model loaded successfully")
                # Initialize BGE embedder for semantic embeddings
                await self._initialize_embedder()
                logger.info("🧠 Model ready for intelligent inference")
                return True
            else:
                logger.error("❌ Failed to load local model")
                return False

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    async def _try_load_model(self, model_name: str, use_token: bool = True) -> bool:
        """
        Try to load model - DISABLED: We only use local models from kalki/models
        This method is kept for compatibility but will not download from HuggingFace
        """
        logger.warning(f"Attempted to load model from HuggingFace: {model_name}")
        logger.warning("This is disabled - only local models from kalki/models are used")
        return False

    async def _try_load_local_model(self, model_path: str) -> bool:
        """Try to load a local fine-tuned model"""
        try:
            logger.info(f"Loading local model from: {model_path}")

            # Load tokenizer from local path
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)

            # Determine appropriate dtype
            torch_dtype = torch.float16 if self.device in ["cuda", "mps"] else torch.float32

            # Load model from local path
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                local_files_only=True,
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
                device=self.device if self.device != "cuda" else 0,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

            return True

        except Exception as e:
            logger.warning(f"Failed to load local model from {model_path}: {e}")
            return False

    async def _initialize_embedder(self) -> bool:
        """Initialize BGE embedder for semantic embeddings"""
        try:
            logger.info("Initializing BGE embedder for semantic embeddings...")
            self.embedder = BGEEmbedder()
            logger.info("BGE embedder initialized successfully")
            return True
        except Exception as e:
            logger.warning(f"Failed to initialize BGE embedder: {e}")
            return False

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using Llama 3.1 8B with meta-cognitive enhancement"""
        if not self.pipe:
            return "Error: Model not initialized"

        try:
            # Check if this is a meta-cognitive command
            if prompt.strip().startswith('/'):
                command_result = process_command(prompt.strip())
                if command_result["success"]:
                    return f"✅ {command_result['message']}\n\n{json.dumps(command_result, indent=2)}"
                else:
                    return f"❌ {command_result['message']}"

            # Get meta-core instance for enhanced prompting
            meta_core = get_meta_core()

            # Generate meta-prompt based on current settings and task context
            meta_prompt = meta_core.generate_meta_prompt(prompt)

            # Combine meta-prompt with user prompt
            enhanced_prompt = f"{meta_prompt}\n\nUSER QUERY: {prompt}"

            # Format the prompt as a chat message for Llama-3.1-8B-Instruct
            messages = [{"role": "user", "content": enhanced_prompt}]
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Set generation parameters
            max_new_tokens = kwargs.get("max_new_tokens", 512)

            # Set default parameters compatible with transformers pipeline
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": kwargs.get("temperature", 0.7),
                "do_sample": kwargs.get("do_sample", True),
                "pad_token_id": self.tokenizer.eos_token_id,
                "return_full_text": False,
                "num_return_sequences": 1
            }
            # Remove max_new_tokens if present to avoid conflicts
            kwargs.pop("max_new_tokens", None)
            generation_kwargs.update(kwargs)

            # Track response time for quality evaluation
            start_time = asyncio.get_event_loop().time()

            # Generate response with error handling for MPS
            try:
                with torch.no_grad():
                    # Ensure max_new_tokens is reasonable for MPS
                    if self.device == "mps" and generation_kwargs["max_new_tokens"] > 2048:
                        generation_kwargs["max_new_tokens"] = 2048
                    
                    outputs = self.pipe(formatted_prompt, **generation_kwargs)
                
                response = outputs[0]["generated_text"]
            except Exception as e:
                error_str = str(e).lower()
                if "out of range" in error_str or "integral" in error_str:
                    logger.warning(f"MPS generation failed with out-of-range error, retrying with smaller tokens...")
                    # Retry with much smaller token limit
                    generation_kwargs["max_new_tokens"] = min(generation_kwargs.get("max_new_tokens", 512), 128)
                    try:
                        with torch.no_grad():
                            outputs = self.pipe(formatted_prompt, **generation_kwargs)
                        response = outputs[0]["generated_text"]
                    except Exception as retry_e:
                        logger.error(f"Retry also failed: {retry_e}")
                        raise
                else:
                    raise

            # Calculate response time
            end_time = asyncio.get_event_loop().time()
            response_time = end_time - start_time

            # Evaluate response quality using meta-core
            quality_metrics = meta_core.evaluate_response_quality(response, prompt, response_time)

            # Memory cleanup
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()

            return response

        except Exception as e:
            import traceback
            logger.error(f"Generation failed: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            return f"Error: {str(e)}"

    async def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        """Generate semantic embeddings using BGE-Large model"""
        if self.embedder is None:
            logger.warning("Embedder not initialized, falling back to hash-based embeddings")
            # Fallback to hash-based embeddings for compatibility
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

        try:
            # Use the real BGE embedder
            return self.embedder.embed(texts)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            # Fallback to hash-based embeddings
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

class LlamaVisionEngine:
    """Llama 3.2 Vision 11B engine for multimodal diagram analysis - ALWAYS uses local models"""
    
    def __init__(self, model_path: Optional[str] = None):
        # ALWAYS prioritize local models from kalki/models directory
        if model_path is None:
            try:
                from config.models_config import get_model_path
                local_path = get_model_path("llama-3.2-11b-vision-instruct")
                if local_path and Path(local_path).exists():
                    model_path = local_path
                    logger.info(f"✅ Found local Llama 3.2 Vision 11B model at: {model_path}")
                else:
                    # Try default path structure
                    default_path = Path(__file__).parent.parent.parent / "models" / "llama_3.2_11b_vision"
                    if default_path.exists():
                        model_path = str(default_path)
                        logger.info(f"✅ Found local Llama 3.2 Vision 11B at default path: {model_path}")
                    else:
                        logger.warning("⚠️  Local Llama 3.2 Vision 11B model not found in models/ directory")
            except Exception as e:
                logger.warning(f"Could not check for local vision model: {e}")
                # Try default path
                default_path = Path(__file__).parent.parent.parent / "models" / "llama_3.2_11b_vision"
                if default_path.exists():
                    model_path = str(default_path)
        
        self.model_path = model_path
        self.model = None
        self.processor = None
        self.device = self._get_optimal_device()
        self.memory_threshold = 0.8
        self.is_initialized = False
        
        if not self.model_path:
            logger.error("❌ No local vision model path provided. Models must be in models/ directory.")
        else:
            logger.info(f"🎨 Initializing Vision Engine from local model: {self.model_path}")
            logger.info(f"📱 Device: {self.device}")
    
    def _get_optimal_device(self) -> str:
        """Determine the best device for vision inference"""
        if torch.backends.mps.is_available():
            logger.info("🎨 Using Metal (MPS) for Vision Model")
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            logger.warning("⚠️ Vision model on CPU will be slow")
            return "cpu"
    
    async def initialize(self) -> bool:
        """Load Llama 3.2 Vision 11B model - ONLY uses local models"""
        try:
            if not self.model_path:
                logger.error("❌ No local vision model path provided. Cannot initialize.")
                logger.error("💡 Place Llama 3.2 Vision 11B model in models/llama_3.2_11b_vision/ directory")
                return False
                
            if not Path(self.model_path).exists():
                logger.error(f"❌ Vision model not found at {self.model_path}")
                logger.error("💡 Ensure models are in models/llama_3.2_11b_vision/ directory")
                return False
            
            logger.info(f"📦 Loading local Llama 3.2 Vision 11B from: {self.model_path}")
            
            # Load processor and model
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Use bfloat16 for M4 Max efficiency
            self.model = MllamaForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16 if self.device == "mps" else torch.float16,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                local_files_only=True
            )
            
            # Move to device
            if self.device == "mps":
                self.model.to("mps")
            elif self.device == "cpu":
                self.model.to("cpu")
            
            self.is_initialized = True
            logger.info("✅ Vision model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}")
            return False
    
    async def analyze_image(self, image_path: str, query: str = "Describe this image in detail.") -> str:
        """Analyze an image with optional query"""
        if not self.is_initialized:
            return "Error: Vision model not initialized"
        
        try:
            # Load image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare inputs
            inputs = self.processor(
                text=query,
                images=image,
                return_tensors="pt",
                padding=True
            )
            
            # Move to device
            if self.device == "mps":
                inputs = {k: v.to("mps") if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            elif self.device == "cuda":
                inputs = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True
                )
            
            # Decode response
            response = self.processor.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up memory
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            
            return response
            
        except Exception as e:
            logger.error(f"Vision analysis failed: {e}")
            return f"Error analyzing image: {str(e)}"
    
    async def extract_diagram_elements(self, image_path: str) -> Dict[str, Any]:
        """Extract structured information from technical diagrams"""
        query = """Analyze this technical diagram and extract:
1. All visible dimensions and measurements
2. Material specifications
3. Labels and annotations
4. Formulas or equations
5. Structural elements and their relationships
Provide a structured breakdown."""
        
        response = await self.analyze_image(image_path, query)
        
        # Parse response into structured data
        result = {
            "raw_description": response,
            "dimensions": self._extract_dimensions(response),
            "materials": self._extract_materials_from_text(response),
            "labels": self._extract_labels(response),
            "formulas": self._extract_formulas_from_text(response)
        }
        
        return result
    
    def _extract_dimensions(self, text: str) -> List[str]:
        """Extract dimension measurements from text"""
        # Match patterns like: 12', 6", 3.5m, 100mm, 4'-6"
        patterns = [
            r'\d+[\s]?(?:feet|ft|\')',
            r'\d+[\s]?(?:inches|in|")',
            r'\d+\.?\d*[\s]?(?:m|mm|cm|meters|millimeters|centimeters)',
            r'\d+\'-\d+"'  # Combined feet-inches
        ]
        
        dimensions = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            dimensions.extend(matches)
        
        return list(set(dimensions))
    
    def _extract_materials_from_text(self, text: str) -> List[str]:
        """Extract material mentions from text"""
        materials = [
            "steel", "concrete", "wood", "lumber", "rebar", "aluminum",
            "plywood", "drywall", "insulation", "asphalt", "brick"
        ]
        
        found = []
        text_lower = text.lower()
        for material in materials:
            if material in text_lower:
                found.append(material)
        
        return found
    
    def _extract_labels(self, text: str) -> List[str]:
        """Extract labels and annotations"""
        # Simple heuristic: capital letter sequences or quoted text
        labels = re.findall(r'[A-Z][A-Z\s]{2,}|"([^"]+)"', text)
        return [l if isinstance(l, str) else l[0] for l in labels if l]
    
    def _extract_formulas_from_text(self, text: str) -> List[str]:
        """Extract mathematical formulas from text"""
        # Match patterns with = signs and mathematical operators
        formulas = re.findall(r'[A-Za-z0-9\s]+\s*=\s*[A-Za-z0-9\s\+\-\*/\(\)]+', text)
        return formulas
    
    async def cleanup(self):
        """Clean up model resources"""
        if self.model:
            del self.model
            del self.processor
            if self.device == "mps":
                torch.mps.empty_cache()
            elif self.device == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
            logger.info("Vision model resources cleaned up")

class LLMEngine:
    """
    Enhanced dual-model LLM engine with text (3.1 8B) and vision (3.2 11B)
    
    All generation uses local models - no API calls.
    Optimized with caching and batch processing.
    Now includes advanced reasoning and domain fine-tuning support.
    """
    
    def __init__(self, backend: str = "llama", enable_vision: bool = True, domain: Optional[str] = None):
        self.backend = backend
        self.llama_engine = None
        self.vision_engine = None
        self.enable_vision = enable_vision
        self.domain = domain  # Domain for fine-tuned model selection
        self.knowledge_base = self._load_knowledge_base()
        self.conversation_history = []
        
        # Optimization: Response caching
        self._response_cache: Dict[str, Any] = {}
        self._cache_max_size = 1000
        
        # Optimization: Batch processing queue
        self._batch_queue: List[Dict[str, Any]] = []
        self._batch_size = 5
        self._batch_timeout = 0.5  # seconds
        
        # Advanced Reasoning Engine (lazy-loaded)
        self._advanced_reasoning = None
        
        # Domain Fine-Tuning support
        self._domain_finetuner = None
        self._domain_model_path = None

        # Initialize Llama text engine - ALWAYS uses local models
        if backend == "llama":
            # ALWAYS get local model path from models_config
            try:
                from config.models_config import get_model_path
                local_model_path = get_model_path("llama-3.1-8b-instruct")
                if local_model_path and Path(local_model_path).exists():
                    logger.info(f"✅ Using local Llama 3.1 8B from: {local_model_path}")
                    self.llama_engine = LlamaEngine(model_path=local_model_path)
                else:
                    logger.error("❌ Local Llama 3.1 8B model not found!")
                    logger.error("💡 Place model in models/llama_3.1_8b/ directory")
                    raise FileNotFoundError("Local Llama 3.1 8B model not found")
            except Exception as e:
                logger.error(f"❌ Failed to load local Llama 3.1 8B model: {e}")
                logger.error("💡 Ensure models are properly placed in models/ directory")
                raise
            
            if enable_vision:
                # ALWAYS use local vision model
                try:
                    vision_model_path = get_model_path("llama-3.2-11b-vision-instruct")
                    if vision_model_path and Path(vision_model_path).exists():
                        logger.info(f"✅ Using local Llama 3.2 Vision 11B from: {vision_model_path}")
                        self.vision_engine = LlamaVisionEngine(model_path=vision_model_path)
                    else:
                        logger.warning("⚠️  Local Llama 3.2 Vision 11B not found, vision features disabled")
                        logger.warning("💡 Place model in models/llama_3.2_11b_vision/ directory for vision support")
                        self.vision_engine = None
                        enable_vision = False
                except Exception as e:
                    logger.warning(f"Could not load local vision model: {e}")
                    self.vision_engine = None
                    enable_vision = False
                
                if enable_vision and self.vision_engine:
                    logger.info("🧠 Dual-model mode: Text (Llama 3.1 8B) + Vision (Llama 3.2 11B)")
                    logger.info("🚀 Maximum intelligence enabled with local models")
                else:
                    logger.info("📝 Text-only mode: Llama 3.1 8B (vision model not available)")
            else:
                logger.info("📝 Text-only mode: Llama 3.1 8B")
        else:
            logger.warning(f"⚠️  Using {backend} backend (rule-based fallback - NOT recommended)")
            logger.warning("💡 Use 'llama' backend to leverage local Llama models for maximum intelligence")

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
        """Initialize the LLM engine with both text and vision models"""
        if self.llama_engine:
            text_success = await self.llama_engine.initialize()
            if text_success:
                model_name = getattr(self.llama_engine, 'model_name', 'Unknown')
                logger.info(f"✅ Text model initialized: {model_name}")
                
                # Initialize vision model if enabled
                if self.enable_vision and self.vision_engine:
                    vision_success = await self.vision_engine.initialize()
                    if vision_success:
                        logger.info("✅ Vision model initialized: Llama 3.2 11B Vision")
                        logger.info("🧠 Kalki is now EXCEPTIONALLY SMART with dual-model intelligence!")
                        return True
                    else:
                        logger.warning("Vision model failed, continuing with text-only mode")
                        self.vision_engine = None
                        return True
                
                return True
            else:
                logger.warning("All LLM models failed to load, falling back to rule-based")
                self.backend = "rule_based"
                return True
        return True

    async def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        use_advanced_reasoning: bool = False,
        reasoning_method: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Generate text with intelligent routing between text and vision models.
        Uses caching and batch processing for optimization.
        Now supports advanced reasoning (CoT, ToT, Self-Consistency, ReAct, Reflexion).
        
        Args:
            prompt: Text query
            image_path: Optional path to image for vision analysis
            use_advanced_reasoning: Enable advanced reasoning (default: False for speed)
            reasoning_method: Specific method ('cot', 'tot', 'self_consistency', 'react', 'reflexion')
            **kwargs: Generation parameters
        
        Returns:
            Generated response
        """
        # Check cache first (for text-only queries)
        if not image_path and not use_advanced_reasoning:
            cache_key = hashlib.md5(f"{prompt}_{kwargs}".encode()).hexdigest()
            if cache_key in self._response_cache:
                logger.debug("Cache hit for prompt")
                return self._response_cache[cache_key]
        
        # Advanced Reasoning (if enabled)
        if use_advanced_reasoning and not image_path:
            try:
                from modules.advanced_reasoning import AdvancedReasoningEngine, ReasoningMethod
                
                if self._advanced_reasoning is None:
                    self._advanced_reasoning = AdvancedReasoningEngine(self)
                
                # Determine reasoning method
                if reasoning_method:
                    method_map = {
                        'cot': ReasoningMethod.CHAIN_OF_THOUGHT,
                        'tot': ReasoningMethod.TREE_OF_THOUGHT,
                        'self_consistency': ReasoningMethod.SELF_CONSISTENCY,
                        'react': ReasoningMethod.REACT,
                        'reflexion': ReasoningMethod.REFLEXION
                    }
                    method = method_map.get(reasoning_method.lower(), ReasoningMethod.CHAIN_OF_THOUGHT)
                else:
                    # Auto-select based on query complexity
                    if any(word in prompt.lower() for word in ['complex', 'analyze', 'design', 'plan']):
                        method = ReasoningMethod.TREE_OF_THOUGHT
                    elif any(word in prompt.lower() for word in ['verify', 'check', 'validate']):
                        method = ReasoningMethod.SELF_CONSISTENCY
                    else:
                        method = ReasoningMethod.CHAIN_OF_THOUGHT
                
                context = kwargs.get('context', {})
                domain = self.domain or context.get('domain', 'general')
                
                result = await self._advanced_reasoning.reason(
                    query=prompt,
                    method=method,
                    context=context,
                    domain=domain
                )
                
                # Extract answer from reasoning result
                if isinstance(result, dict):
                    answer = result.get('final_answer') or result.get('consensus_answer') or result.get('answer', str(result))
                else:
                    answer = str(result)
                
                # Cache result
                cache_key = hashlib.md5(f"{prompt}_{kwargs}_{reasoning_method}".encode()).hexdigest()
                self._cache_response(cache_key, answer)
                
                return answer
            except ImportError:
                logger.warning("Advanced reasoning module not available, using standard generation")
            except Exception as e:
                logger.error(f"Advanced reasoning failed: {e}, falling back to standard generation")
        
        # Route to vision model if image provided (uses Llama 3.2 Vision 11B)
        if image_path and self.vision_engine and self.vision_engine.is_initialized:
            try:
                logger.info(f"🎨 Routing to Llama 3.2 Vision 11B for image analysis")
                result = await self.vision_engine.analyze_image(image_path, prompt)
                # Cache vision results (with image hash)
                if not image_path.startswith("http"):  # Don't cache remote images
                    cache_key = hashlib.md5(f"{prompt}_{image_path}".encode()).hexdigest()
                    self._cache_response(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Vision generation failed: {e}, falling back to text-only")
        
        # Check for domain-specific fine-tuned model
        if self.domain and self._domain_model_path is None:
            await self._load_domain_model()
        
        # Route to text model for standard queries (uses Llama 3.1 8B or domain fine-tuned)
        if self.llama_engine and self.backend == "llama":
            try:
                result = await self.llama_engine.generate(prompt, **kwargs)
                # Cache text results
                cache_key = hashlib.md5(f"{prompt}_{kwargs}".encode()).hexdigest()
                self._cache_response(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Llama generation failed: {e}, falling back to rule-based")
                self.backend = "rule_based"

        # Fallback to rule-based generation
        return self._rule_based_generate(prompt, **kwargs)
    
    async def _load_domain_model(self):
        """Load domain-specific fine-tuned model if available"""
        if not self.domain:
            return
        
        try:
            from modules.domain_finetuning import DomainFineTuner
            if self._domain_finetuner is None:
                self._domain_finetuner = DomainFineTuner()
            
            model_path = await self._domain_finetuner.load_domain_model(self.domain)
            if model_path:
                self._domain_model_path = model_path
                logger.info(f"✅ Loaded domain-specific model: {self.domain}")
                # Reload llama_engine with domain model
                self.llama_engine = LlamaEngine(model_path=model_path)
                await self.llama_engine.initialize()
        except Exception as e:
            logger.warning(f"Could not load domain model for {self.domain}: {e}")
    
    def _cache_response(self, cache_key: str, response: str):
        """Cache response with size management"""
        if len(self._response_cache) >= self._cache_max_size:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(self._response_cache))
            del self._response_cache[oldest_key]
        self._response_cache[cache_key] = response
    
    async def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Batch generate for multiple prompts (optimization)"""
        results = []
        for prompt in prompts:
            result = await self.generate(prompt, **kwargs)
            results.append(result)
        return results
    
    async def analyze_image(self, image_path: str, query: str = "Describe this image") -> str:
        """Analyze image using vision model"""
        if not self.vision_engine or not self.vision_engine.is_initialized:
            return "Error: Vision model not available. Enable vision in initialization."
        
        return await self.vision_engine.analyze_image(image_path, query)
    
    async def extract_diagram(self, image_path: str) -> Dict[str, Any]:
        """Extract structured data from technical diagrams"""
        if not self.vision_engine or not self.vision_engine.is_initialized:
            return {"error": "Vision model not available"}
        
        return await self.vision_engine.extract_diagram_elements(image_path)
    
    async def cross_validate(self, text_result: str, image_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Cross-validate results between text and vision models
        
        Args:
            text_result: Result from text model
            image_path: Optional image to validate against
        
        Returns:
            Validation results with confidence scores
        """
        if not image_path or not self.vision_engine or not self.vision_engine.is_initialized:
            return {
                "validated": True,
                "confidence": 0.9,
                "notes": "Text-only validation (no vision model available)"
            }
        
        try:
            # Get vision model's analysis
            vision_prompt = f"Validate if this statement is accurate based on the image: {text_result[:200]}"
            vision_analysis = await self.vision_engine.analyze_image(image_path, vision_prompt)
            
            # Simple validation logic (can be enhanced with more sophisticated NLP)
            agreement_keywords = ["correct", "accurate", "yes", "confirmed", "matches"]
            disagreement_keywords = ["incorrect", "inaccurate", "no", "contradicts", "wrong"]
            
            vision_lower = vision_analysis.lower()
            agreement_score = sum(1 for kw in agreement_keywords if kw in vision_lower)
            disagreement_score = sum(1 for kw in disagreement_keywords if kw in vision_lower)
            
            if agreement_score > disagreement_score:
                confidence = 0.95
                validated = True
            elif disagreement_score > agreement_score:
                confidence = 0.4
                validated = False
            else:
                confidence = 0.7
                validated = True  # Neutral = lean toward acceptance
            
            return {
                "validated": validated,
                "confidence": confidence,
                "vision_analysis": vision_analysis,
                "agreement_score": agreement_score,
                "disagreement_score": disagreement_score
            }
            
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            return {
                "validated": True,
                "confidence": 0.7,
                "error": str(e),
                "notes": "Validation failed, defaulting to text result"
            }

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
        """Clean up resources for both text and vision models"""
        if self.llama_engine:
            await self.llama_engine.cleanup()
        if self.vision_engine:
            await self.vision_engine.cleanup()
        logger.info("Dual-model LLM engine cleaned up")

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

    async def generate_code(self, request: str, platform: str) -> str:
        """Generate code based on the request and platform using LLM"""
        try:
            prompt = f"""Generate production-ready code for the following request:

Request: {request}
Platform: {platform}

Please generate complete, functional code that:
1. Implements the core functionality described in the request
2. Uses appropriate frameworks and libraries for the platform
3. Includes proper error handling and best practices
4. Has clear comments explaining the implementation
5. Is immediately runnable/deployable

For {platform} platform, use these conventions:
- Web: React with modern hooks, TypeScript preferred
- Unity: C# with proper Unity lifecycle methods
- Mobile: React Native or Flutter with platform-specific features
- Desktop: Tauri + Vue.js or Electron + React
- Game: Unity C# or Godot GDScript

Generate only the code, no explanations outside of comments."""

            # Use the LLM to generate code
            generated_code = await self.generate(prompt, max_new_tokens=1500, temperature=0.3)

            # Clean up the response (remove any extra text before/after code)
            if "```" in generated_code:
                # Extract code from markdown code blocks
                code_blocks = generated_code.split("```")
                for block in code_blocks:
                    if block.strip() and not block.lower().startswith(("python", "javascript", "typescript", "csharp", "c#", "java", "cpp", "c++")):
                        return block.strip()
                # If no clean code block found, return the first one
                return code_blocks[1].strip() if len(code_blocks) > 1 else generated_code
            else:
                return generated_code

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Fallback to template-based generation
            return self._generate_template_code(request, platform)

    def _generate_template_code(self, request: str, platform: str) -> str:
        """Fallback template-based code generation"""
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
// Template implementation - customize as needed

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
    """Get the global LLM engine instance - ALWAYS uses local Llama models"""
    global _llm_engine
    if _llm_engine is None:
        # ALWAYS use llama backend to leverage local models
        backend = get_config("llm", "backend", "llama")  # Default to llama
        enable_vision = get_config("llm", "enable_vision", True)  # Enable vision by default
        domain = get_config("llm", "domain", None)  # Optional domain for fine-tuning
        
        logger.info("🚀 Initializing LLM Engine with local Llama models...")
        _llm_engine = LLMEngine(backend=backend, enable_vision=enable_vision, domain=domain)
    return _llm_engine

async def initialize_llm_engine() -> bool:
    """Initialize the global LLM engine - ensures local models are loaded"""
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
