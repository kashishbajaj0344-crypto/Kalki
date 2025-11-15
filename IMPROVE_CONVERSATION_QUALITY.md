# 🎯 Improving Kalki's Conversation Quality to Match Ollama

## Problem Statement

**Issue**: Kalki's conversation quality doesn't match Ollama's quality, despite using the same models (Llama 3.1 8B).

**Root Causes to Investigate:**
1. Prompt formatting differences
2. Generation parameter differences
3. Model loading/quantization differences
4. Context/conversation history handling
5. System prompts vs user prompts

---

## 🔍 Current Implementation Analysis

### Current Prompt Formatting

```python
# modules/llm.py - LlamaEngine.generate()
async def generate(self, prompt: str, **kwargs):
    # Get meta-core instance for enhanced prompting
    meta_core = get_meta_core()
    
    # Generate meta-prompt based on current settings
    meta_prompt = meta_core.generate_meta_prompt(prompt)
    
    # Combine meta-prompt with user prompt
    enhanced_prompt = f"{meta_prompt}\n\nUSER QUERY: {prompt}"
    
    # Format as chat message
    messages = [{"role": "user", "content": enhanced_prompt}]
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
```

**Potential Issues:**
- ⚠️ Meta-prompt might be interfering
- ⚠️ "USER QUERY:" prefix might confuse the model
- ⚠️ Single message format (no conversation history)

### Current Generation Parameters

```python
# modules/llm.py
generation_kwargs = {
    "max_new_tokens": kwargs.get("max_new_tokens", 512),  # Default: 512
    "temperature": kwargs.get("temperature", 0.7),  # Default: 0.7
    "do_sample": kwargs.get("do_sample", True),
    "pad_token_id": self.tokenizer.eos_token_id,
    "return_full_text": False,
    "num_return_sequences": 1
}
```

**Potential Issues:**
- ⚠️ `max_new_tokens=512` might be too short for complex conversations
- ⚠️ `temperature=0.7` might need adjustment
- ⚠️ Missing `top_p` and `top_k` for better sampling

---

## 🆚 Ollama vs Current Implementation

### How Ollama Formats Prompts

Ollama uses Llama 3's native chat template directly:
```python
# Ollama's approach (simplified)
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is AI?"},
    {"role": "assistant", "content": "AI is..."},
    {"role": "user", "content": "Tell me more"}
]
formatted = tokenizer.apply_chat_template(messages, tokenize=False)
```

**Key Differences:**
1. ✅ Clean system prompt (not meta-prompt)
2. ✅ Proper conversation history
3. ✅ Native chat template (no extra prefixes)

### How Ollama Sets Generation Parameters

Ollama's defaults (for Llama 3):
```python
{
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
    "num_predict": 2048,  # Much longer than 512!
}
```

**Key Differences:**
1. ✅ `num_predict=2048` (vs our 512)
2. ✅ `top_p=0.9` (nucleus sampling)
3. ✅ `top_k=40` (top-k sampling)
4. ✅ `repeat_penalty=1.1` (prevents repetition)

---

## 🔧 Recommended Fixes

### Fix 1: Improve Prompt Formatting

**Current (Problematic):**
```python
meta_prompt = meta_core.generate_meta_prompt(prompt)
enhanced_prompt = f"{meta_prompt}\n\nUSER QUERY: {prompt}"
messages = [{"role": "user", "content": enhanced_prompt}]
```

**Recommended (Ollama-like):**
```python
# Clean system prompt
system_prompt = "You are a helpful, intelligent assistant. Provide clear, detailed, and thoughtful responses."

# User prompt (no extra prefixes)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": prompt}
]

# Add conversation history if available
if self.conversation_history:
    messages = [{"role": "system", "content": system_prompt}] + self.conversation_history + [{"role": "user", "content": prompt}]

formatted_prompt = self.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

### Fix 2: Improve Generation Parameters

**Current:**
```python
generation_kwargs = {
    "max_new_tokens": 512,  # Too short!
    "temperature": 0.7,
    "do_sample": True,
    # Missing top_p, top_k, repeat_penalty
}
```

**Recommended (Ollama-like):**
```python
generation_kwargs = {
    "max_new_tokens": kwargs.get("max_new_tokens", 2048),  # Much longer
    "temperature": kwargs.get("temperature", 0.7),
    "top_p": kwargs.get("top_p", 0.9),  # Nucleus sampling
    "top_k": kwargs.get("top_k", 40),  # Top-k sampling
    "repetition_penalty": kwargs.get("repetition_penalty", 1.1),  # Prevent repetition
    "do_sample": True,
    "pad_token_id": self.tokenizer.eos_token_id,
    "return_full_text": False,
}
```

### Fix 3: Better Conversation History

**Current:**
```python
self.conversation_history = []  # Exists but not used properly
```

**Recommended:**
```python
# Maintain proper conversation history
if not hasattr(self, 'conversation_history'):
    self.conversation_history = []

# Add to history after generation
self.conversation_history.append({"role": "user", "content": prompt})
self.conversation_history.append({"role": "assistant", "content": response})

# Limit history length (keep last N exchanges)
max_history = 10  # Keep last 10 exchanges
if len(self.conversation_history) > max_history * 2:
    self.conversation_history = self.conversation_history[-max_history * 2:]
```

### Fix 4: Remove Meta-Prompt Interference

**Current:**
```python
meta_prompt = meta_core.generate_meta_prompt(prompt)
enhanced_prompt = f"{meta_prompt}\n\nUSER QUERY: {prompt}"
```

**Recommended:**
```python
# Option 1: Remove meta-prompt entirely (simplest)
# Just use clean user prompt

# Option 2: Make meta-prompt optional
use_meta_prompt = kwargs.get("use_meta_prompt", False)
if use_meta_prompt:
    meta_prompt = meta_core.generate_meta_prompt(prompt)
    enhanced_prompt = f"{meta_prompt}\n\n{prompt}"
else:
    enhanced_prompt = prompt
```

---

## 🎯 Implementation Plan

### Step 1: Update LlamaEngine.generate()

```python
async def generate(self, prompt: str, **kwargs) -> str:
    """Generate text using Llama 3.1 8B with improved conversation quality"""
    
    # Clean system prompt (Ollama-like)
    system_prompt = kwargs.get(
        "system_prompt",
        "You are a helpful, intelligent assistant. Provide clear, detailed, and thoughtful responses."
    )
    
    # Build messages with conversation history
    messages = [{"role": "system", "content": system_prompt}]
    
    # Add conversation history
    if self.conversation_history:
        messages.extend(self.conversation_history)
    
    # Add current user prompt
    messages.append({"role": "user", "content": prompt})
    
    # Format using chat template
    formatted_prompt = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    
    # Improved generation parameters (Ollama-like)
    generation_kwargs = {
        "max_new_tokens": kwargs.get("max_new_tokens", 2048),  # Longer responses
        "temperature": kwargs.get("temperature", 0.7),
        "top_p": kwargs.get("top_p", 0.9),  # Nucleus sampling
        "top_k": kwargs.get("top_k", 40),  # Top-k sampling
        "repetition_penalty": kwargs.get("repetition_penalty", 1.1),  # Prevent repetition
        "do_sample": True,
        "pad_token_id": self.tokenizer.eos_token_id,
        "return_full_text": False,
    }
    
    # Generate
    outputs = self.pipe(formatted_prompt, **generation_kwargs)
    response = outputs[0]["generated_text"]
    
    # Update conversation history
    self.conversation_history.append({"role": "user", "content": prompt})
    self.conversation_history.append({"role": "assistant", "content": response})
    
    # Limit history length
    max_history = kwargs.get("max_history", 10)
    if len(self.conversation_history) > max_history * 2:
        self.conversation_history = self.conversation_history[-max_history * 2:]
    
    return response
```

### Step 2: Update LLMEngine.generate()

```python
async def generate(
    self,
    prompt: str,
    image_path: Optional[str] = None,
    use_advanced_reasoning: bool = False,
    **kwargs
) -> str:
    """Generate with improved defaults"""
    
    # Set better defaults for conversation quality
    kwargs.setdefault("max_new_tokens", 2048)  # Longer responses
    kwargs.setdefault("temperature", 0.7)
    kwargs.setdefault("top_p", 0.9)
    kwargs.setdefault("top_k", 40)
    kwargs.setdefault("repetition_penalty", 1.1)
    
    # Rest of the method...
```

### Step 3: Test Conversation Quality

```python
# Test script
async def test_conversation_quality():
    llm = LLMEngine()
    await llm.initialize()
    
    # Test 1: Simple question
    response1 = await llm.generate("What is artificial intelligence?")
    print(f"Response 1: {response1}")
    
    # Test 2: Follow-up (should use conversation history)
    response2 = await llm.generate("Can you explain that in more detail?")
    print(f"Response 2: {response2}")
    
    # Test 3: Complex reasoning
    response3 = await llm.generate(
        "How does machine learning differ from traditional programming?",
        max_new_tokens=1024
    )
    print(f"Response 3: {response3}")
```

---

## 🔍 Debugging Checklist

### Check 1: Model Loading
```python
# Verify model is loaded correctly
print(f"Model: {llm.llama_engine.model}")
print(f"Tokenizer: {llm.llama_engine.tokenizer}")
print(f"Device: {llm.llama_engine.device}")
```

### Check 2: Prompt Formatting
```python
# Print formatted prompt to see what model receives
formatted = llm.llama_engine.tokenizer.apply_chat_template(...)
print(f"Formatted prompt:\n{formatted}")
```

### Check 3: Generation Parameters
```python
# Print actual generation parameters
print(f"Generation kwargs: {generation_kwargs}")
```

### Check 4: Response Length
```python
# Check if responses are being truncated
response = await llm.generate("Explain quantum computing")
print(f"Response length: {len(response)} tokens")
print(f"Response: {response}")
```

---

## 🎯 Quick Wins

### Immediate Fixes (No Code Changes)

1. **Increase max_new_tokens in calls:**
```python
response = await llm.generate(prompt, max_new_tokens=2048)
```

2. **Add top_p and top_k:**
```python
response = await llm.generate(
    prompt,
    max_new_tokens=2048,
    top_p=0.9,
    top_k=40,
    repetition_penalty=1.1
)
```

3. **Use cleaner prompts:**
```python
# Instead of complex meta-prompts, use simple system prompt
response = await llm.generate(
    prompt,
    system_prompt="You are a helpful assistant."
)
```

---

## 📊 Expected Improvements

### Before (Current)
- Short responses (512 tokens max)
- No conversation history
- Meta-prompt interference
- Missing sampling parameters

### After (Ollama-like)
- Longer, more detailed responses (2048 tokens)
- Proper conversation history
- Clean system prompts
- Better sampling (top_p, top_k, repetition_penalty)

---

## ✅ Action Items

1. **Update `LlamaEngine.generate()`** with:
   - Clean system prompts
   - Conversation history
   - Better generation parameters

2. **Update `LLMEngine.generate()`** with:
   - Better default parameters
   - Conversation history support

3. **Test conversation quality** with:
   - Simple questions
   - Follow-up questions
   - Complex reasoning

4. **Compare with Ollama** to verify quality matches

---

## 🚀 Next Steps

1. Implement the fixes above
2. Test conversation quality
3. Compare with Ollama side-by-side
4. Iterate based on results

**The goal: Match or exceed Ollama's conversation quality while keeping Hugging Face's flexibility!**

