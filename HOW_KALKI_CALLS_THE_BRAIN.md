# 🧠 How Kalki Calls The Brain: Complete Call Flow

## Overview

Kalki (the body) calls the brain (Llama models) through a **3-layer architecture**:

1. **Component Layer** (Body) → Calls `LLMEngine.generate()`
2. **Interface Layer** (`LLMEngine`) → Routes to appropriate model
3. **Brain Layer** (`LlamaEngine` / `LlamaVisionEngine`) → Executes model inference

---

## 📊 Complete Call Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    KALKI COMPONENTS (Body)                  │
│  GameDevCopilot, ConstructionCopilot, Orchestrator, etc.    │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ Calls: await self.llm.generate(prompt)
                          │
                          ↓
┌─────────────────────────────────────────────────────────────┐
│              LLMEngine (Interface Layer)                     │
│  modules/llm.py - LLMEngine.generate()                       │
│                                                              │
│  • Checks cache                                              │
│  • Routes to vision or text model                           │
│  • Handles advanced reasoning                                │
│  • Manages batching & optimization                           │
└─────────────────────────────────────────────────────────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                     │
        ↓                                     ↓
┌──────────────────────┐        ┌──────────────────────────┐
│  LlamaEngine         │        │  LlamaVisionEngine       │
│  (Text Brain)        │        │  (Vision Brain)          │
│                      │        │                          │
│  Llama 3.1 8B        │        │  Llama 3.2 11B Vision   │
│  models/llama_3.1_8b │        │  models/llama_3.2_11b_   │
│                      │        │    vision/               │
│  • Text generation   │        │  • Image analysis        │
│  • Code generation   │        │  • Diagram parsing       │
│  • Reasoning         │        │  • Visual QA             │
│  • Q&A               │        │  • OCR                   │
└──────────────────────┘        └──────────────────────────┘
        │                                     │
        └─────────────────┬─────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   Model Inference     │
              │   (PyTorch/Transformers)│
              │                        │
              │  • Tokenization        │
              │  • Forward pass        │
              │  • Generation          │
              │  • Decoding            │
              └───────────────────────┘
                          │
                          ↓
              ┌───────────────────────┐
              │   Response Returned   │
              │   (Text/Code/Answer)  │
              └───────────────────────┘
```

---

## 🔍 Step-by-Step Call Flow

### Example: GameDevCopilot Asking a Question

#### Step 1: Component Calls Brain
```python
# modules/game_dev_copilot.py
class GameDevCopilot:
    def __init__(self):
        self.llm = LLMEngine()  # Initialize brain interface
    
    async def answer_question(self, session_id: str, answer: str):
        # ... process answer ...
        
        # CALL THE BRAIN
        response = await self.llm.generate(
            prompt=prompt,
            max_new_tokens=300,
            temperature=0.7
        )
```

#### Step 2: LLMEngine Routes Request
```python
# modules/llm.py
class LLMEngine:
    async def generate(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        use_advanced_reasoning: bool = False,
        **kwargs
    ) -> str:
        # Check cache first
        cache_key = hashlib.md5(f"{prompt}_{kwargs}".encode()).hexdigest()
        if cache_key in self._response_cache:
            return self._response_cache[cache_key]  # Cache hit!
        
        # Route to vision model if image provided
        if image_path and self.vision_engine:
            return await self.vision_engine.analyze_image(image_path, prompt)
        
        # Route to text model (Llama 3.1 8B)
        if self.llama_engine:
            result = await self.llama_engine.generate(prompt, **kwargs)
            self._cache_response(cache_key, result)  # Cache for next time
            return result
```

#### Step 3: LlamaEngine Executes Model Inference
```python
# modules/llm.py
class LlamaEngine:
    async def generate(self, prompt: str, **kwargs) -> str:
        # Format prompt for Llama 3.1 8B Instruct
        messages = [{"role": "user", "content": prompt}]
        formatted_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Set generation parameters
        generation_kwargs = {
            "max_new_tokens": kwargs.get("max_new_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "do_sample": True
        }
        
        # CALL THE ACTUAL MODEL (THE BRAIN)
        result = self.pipe(
            formatted_prompt,
            **generation_kwargs
        )
        
        # Extract generated text
        generated_text = result[0]['generated_text']
        return generated_text
```

#### Step 4: Model Inference (PyTorch/Transformers)
```python
# Inside self.pipe (Transformers pipeline)
# This is where the actual neural network runs:

# 1. Tokenize input
tokens = tokenizer(formatted_prompt, return_tensors="pt").to(device)

# 2. Forward pass through Llama 3.1 8B model
outputs = model.generate(
    **tokens,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)

# 3. Decode tokens to text
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

---

## 💻 Real Code Examples

### Example 1: GameDevCopilot Generating Code

```python
# modules/game_dev_copilot.py (line ~1562)
async def generate_code(self, requirements: ProjectRequirements):
    prompt = f"""Generate Unity C# code for {requirements.game_name}..."""
    
    # CALL BRAIN
    response = await self.llm.generate(
        prompt=prompt,
        task='game_code_generation',
        max_tokens=600
    )
    
    return response
```

**Flow:**
```
GameDevCopilot.generate_code()
    ↓
LLMEngine.generate(prompt="Generate Unity C# code...")
    ↓
LlamaEngine.generate(prompt)
    ↓
self.pipe(formatted_prompt)  # Transformers pipeline
    ↓
Llama 3.1 8B Model Inference (PyTorch)
    ↓
Generated C# code returned
```

### Example 2: Orchestrator Analyzing Task

```python
# modules/orchestrator.py (line ~253)
async def _analyze_task(self, task: Dict[str, Any]):
    analysis_prompt = f"Analyze this task: {task['query']}..."
    
    # CALL BRAIN
    analysis_result = await self.llm_engine.generate(
        analysis_prompt,
        max_tokens=1000,
        use_advanced_reasoning=True  # Uses advanced reasoning!
    )
    
    return analysis_result
```

**Flow:**
```
KalkiOrchestrator._analyze_task()
    ↓
LLMEngine.generate(prompt, use_advanced_reasoning=True)
    ↓
AdvancedReasoningEngine.reason()  # If advanced reasoning enabled
    ↓
LlamaEngine.generate()  # Multiple calls for reasoning
    ↓
Llama 3.1 8B Model Inference
    ↓
Reasoned analysis returned
```

### Example 3: Vision Analysis (YouTube Ingestion)

```python
# modules/youtube_ingestion.py
async def analyze_frames_with_vision(self, frame_path: str):
    # CALL VISION BRAIN
    vision_analysis = await self._llm.vision_engine.analyze_image(
        frame_path,
        "Extract code from this video frame"
    )
    
    return vision_analysis
```

**Flow:**
```
YouTubeIngestionSystem.analyze_frames_with_vision()
    ↓
LLMEngine.vision_engine.analyze_image()
    ↓
LlamaVisionEngine.analyze_image()
    ↓
Llama 3.2 11B Vision Model Inference
    ↓
Visual analysis returned
```

---

## 🔧 Key Methods: How Components Access Brain

### Method 1: Direct LLMEngine Instance
```python
# Most common pattern
class GameDevCopilot:
    def __init__(self):
        self.llm = LLMEngine()  # Create brain interface
    
    async def some_method(self):
        response = await self.llm.generate("What should I ask?")
```

### Method 2: Global LLMEngine Access
```python
# modules/orchestrator.py
from modules.llm import get_llm_engine

class KalkiOrchestrator:
    def __init__(self):
        self.llm_engine = get_llm_engine()  # Get shared instance
    
    async def process_task(self, task):
        result = await self.llm_engine.generate(task['query'])
```

### Method 3: Vision-Specific Access
```python
# For vision tasks
class SomeComponent:
    def __init__(self):
        self.llm = LLMEngine(enable_vision=True)
    
    async def analyze_image(self, image_path: str):
        # Direct vision brain access
        result = await self.llm.vision_engine.analyze_image(
            image_path,
            "Describe this image"
        )
```

---

## 📋 Complete Call Signature

### LLMEngine.generate() - Main Interface

```python
async def generate(
    self,
    prompt: str,                          # The question/request
    image_path: Optional[str] = None,     # For vision tasks
    use_advanced_reasoning: bool = False, # Enable CoT/ToT reasoning
    reasoning_method: Optional[str] = None, # 'cot', 'tot', 'react', etc.
    **kwargs                              # Generation params
) -> str:
    """
    Main method to call the brain.
    
    Routes to:
    - Llama 3.1 8B (text) if no image
    - Llama 3.2 11B Vision (vision) if image provided
    - Advanced reasoning if enabled
    """
```

**Common kwargs:**
- `max_new_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Creativity (0.0-1.0, default: 0.7)
- `do_sample`: Enable sampling (default: True)
- `top_p`: Nucleus sampling (default: 0.9)

### LlamaEngine.generate() - Text Brain

```python
async def generate(
    self,
    prompt: str,
    **kwargs
) -> str:
    """
    Direct call to Llama 3.1 8B text model.
    
    Process:
    1. Format prompt with chat template
    2. Tokenize
    3. Run model inference
    4. Decode tokens to text
    5. Return generated text
    """
```

### LlamaVisionEngine.analyze_image() - Vision Brain

```python
async def analyze_image(
    self,
    image_path: str,
    query: str = "Describe this image"
) -> str:
    """
    Direct call to Llama 3.2 11B Vision model.
    
    Process:
    1. Load and preprocess image
    2. Format multimodal prompt (text + image)
    3. Run vision model inference
    4. Decode response
    5. Return analysis
    """
```

---

## 🎯 Routing Logic

### How LLMEngine Decides Which Brain to Use

```python
# Inside LLMEngine.generate()

# 1. Check cache first
if cache_key in self._response_cache:
    return cached_response  # Fast path!

# 2. Advanced reasoning (if enabled)
if use_advanced_reasoning:
    return await AdvancedReasoningEngine.reason(...)
    # This internally calls LlamaEngine multiple times

# 3. Vision task (if image provided)
if image_path and self.vision_engine:
    return await self.vision_engine.analyze_image(image_path, prompt)
    # Uses Llama 3.2 11B Vision

# 4. Text task (default)
if self.llama_engine:
    return await self.llama_engine.generate(prompt, **kwargs)
    # Uses Llama 3.1 8B

# 5. Fallback (if models unavailable)
return self._rule_based_generate(prompt, **kwargs)
```

---

## 🔄 Complete Example: End-to-End Call

### Scenario: User asks "Build me a solitaire game"

```python
# 1. User input received
user_query = "Build me a solitaire game"

# 2. Unified Chat routes to GameDevCopilot
copilot = GameDevCopilot()  # Has self.llm = LLMEngine()

# 3. GameDevCopilot needs to understand the request
prompt = f"Analyze this game request: {user_query}. What questions should I ask?"

# 4. CALL THE BRAIN
response = await copilot.llm.generate(
    prompt=prompt,
    max_new_tokens=300
)

# 5. Inside LLMEngine.generate():
#    - Checks cache (miss)
#    - Routes to LlamaEngine (text model)
#    - Calls await self.llama_engine.generate(prompt)

# 6. Inside LlamaEngine.generate():
#    - Formats prompt with chat template
#    - Calls self.pipe(formatted_prompt)

# 7. Inside self.pipe (Transformers):
#    - Tokenizes: "Analyze this game request..."
#    - Runs Llama 3.1 8B model forward pass
#    - Generates tokens: "You should ask about platform, engine..."
#    - Decodes tokens to text

# 8. Response flows back:
#    LlamaEngine → LLMEngine → GameDevCopilot → User
#    "You should ask: What platform? What engine? How to monetize?"
```

---

## 📊 Statistics

### How Many Components Call The Brain?

**Found: 237+ components** use `LLMEngine` to call the brain:

- **Copilots**: GameDevCopilot, ConstructionCopilot
- **Agents**: PlannerAgent, ReasoningAgent, MemoryAgent, etc.
- **Engines**: SupremeSynthesisEngine, ConsciousnessEngine
- **Orchestrators**: KalkiOrchestrator, SupremeControlHub
- **Systems**: MetaLearningSystem, AutonomousResearchSystem
- **And many more...**

### Call Patterns

1. **Simple Text Generation**: `await llm.generate(prompt)`
2. **With Parameters**: `await llm.generate(prompt, max_new_tokens=1000, temperature=0.7)`
3. **Vision Tasks**: `await llm.vision_engine.analyze_image(image_path, query)`
4. **Advanced Reasoning**: `await llm.generate(prompt, use_advanced_reasoning=True)`
5. **Batch Processing**: `await llm.generate_batch([prompt1, prompt2, ...])`

---

## ✅ Summary

**How Kalki Calls The Brain:**

1. **Component** calls `await self.llm.generate(prompt)`
2. **LLMEngine** routes to appropriate model (text or vision)
3. **LlamaEngine/LlamaVisionEngine** formats and executes inference
4. **Model** (Llama 3.1 8B or 3.2 11B Vision) generates response
5. **Response** flows back through layers to component

**Key Points:**
- ✅ Single interface: `LLMEngine.generate()`
- ✅ Automatic routing: Text vs Vision
- ✅ Caching: Fast responses for repeated queries
- ✅ Advanced reasoning: Optional enhanced intelligence
- ✅ 237+ components use this pattern

**The brain is always accessible through `LLMEngine`!** 🧠

