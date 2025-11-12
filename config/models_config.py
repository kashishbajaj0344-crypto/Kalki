"""
Model Configuration for KALKI
Defines available local models and their capabilities
"""

from pathlib import Path
from typing import Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)

# Base paths
HOME = Path.home()
HF_CACHE = HOME / ".cache/huggingface/hub"
LLAMA_STACK = HOME / ".llama/checkpoints"
# Get absolute path to kalki/models directory
# models_config.py is in config/, so models/ is in the parent directory
_models_config_file = Path(__file__).resolve()
KALKI_MODELS = _models_config_file.parent.parent / "models"  # Go up from config/ to root, then to models/

# Model Registry
MODELS = {
    # Text-only models (fast chat, reasoning, validation)
    "llama-3.1-8b-instruct": {
        "name": "Llama 3.1 8B Instruct",
        "type": "text",
        "modalities": ["text"],
        "capabilities": ["chat", "reasoning", "validation", "extraction"],
        "paths": [
            KALKI_MODELS / "llama_3.1_8b",  # ONLY use kalki/models
            KALKI_MODELS / "llama-3.1-8b-instruct",  # Alternative naming
        ],
        "recommended_quantization": "int8",  # or "int4" for more speed
        "memory_footprint": {
            "fp16": "16 GB",
            "int8": "8 GB",
            "int4": "4 GB",
        },
        "speed": "50 tokens/sec (fp16/int8)",
        "use_for": ["daily_chat", "text_extraction", "validation", "copilot_guidance"],
        "priority": 1,  # Primary model
    },
    
    # Multimodal models (vision + text for PDF diagrams)
    "llama-3.2-11b-vision-instruct": {
        "name": "Llama 3.2 11B Vision Instruct",
        "type": "multimodal",
        "modalities": ["text", "image"],
        "capabilities": ["vision", "diagram_analysis", "ocr", "image_understanding"],
        "paths": [
            KALKI_MODELS / "llama_3.2_11b_vision",  # ONLY use kalki/models
            KALKI_MODELS / "llama-3.2-11b-vision-instruct",  # Alternative naming
        ],
        "recommended_quantization": "int4",  # Essential for 36GB RAM
        "memory_footprint": {
            "fp16": "22 GB",
            "int8": "11 GB",
            "int4": "6 GB",
        },
        "speed": "40 tokens/sec (text), 30 sec/image",
        "use_for": ["pdf_diagram_extraction", "image_analysis", "ocr", "visual_qa"],
        "priority": 2,  # Secondary (ingestion only)
    },
    
    # Future: Llama 4 models (remote/placeholder)
    "llama-4-scout": {
        "name": "Llama 4 Scout (MoE)",
        "type": "multimodal",
        "modalities": ["text", "image"],
        "capabilities": ["vision", "long_context", "moe"],
        "paths": [],  # Not available locally
        "memory_footprint": {
            "note": "109B total params - NOT suitable for 36GB RAM local use"
        },
        "use_for": ["remote_only", "heavy_reasoning_optional"],
        "priority": 99,  # Remote fallback only
        "local_viable": False,
    },
}


# Task to Model Mapping
TASK_MODEL_MAP = {
    # Primary tasks (fast text model)
    "chat": "llama-3.1-8b-instruct",
    "reasoning": "llama-3.1-8b-instruct",
    "validation": "llama-3.1-8b-instruct",
    "text_extraction": "llama-3.1-8b-instruct",
    "copilot_guidance": "llama-3.1-8b-instruct",
    "query": "llama-3.1-8b-instruct",
    
    # Vision tasks (multimodal model)
    "pdf_ingestion_with_images": "llama-3.2-11b-vision-instruct",
    "diagram_analysis": "llama-3.2-11b-vision-instruct",
    "image_ocr": "llama-3.2-11b-vision-instruct",
    "visual_qa": "llama-3.2-11b-vision-instruct",
    
    # Construction Copilot tasks (NEW - uses existing models)
    "construction_chat": "llama-3.1-8b-instruct",  # User guidance
    "construction_reasoning": "llama-3.1-8b-instruct",  # Decision support
    "roadmap_generation": "llama-3.1-8b-instruct",  # Timeline/cost
    "property_analysis": "llama-3.1-8b-instruct",  # Zoning/permits
    "consciousness_reasoning": "llama-3.1-8b-instruct",  # WHY explanations
    "meta_learning": "llama-3.1-8b-instruct",  # Learn from outcomes
    "predictive_analytics": "llama-3.1-8b-instruct",  # Risk prediction
    
    # Construction Copilot vision tasks (NEW - uses existing vision model)
    "site_photo_qc": "llama-3.2-11b-vision-instruct",  # Progress tracking
    "blueprint_analysis": "llama-3.2-11b-vision-instruct",  # Design review
    "material_identification": "llama-3.2-11b-vision-instruct",  # Photo ID
    "auto_progress_detection": "llama-3.2-11b-vision-instruct",  # Milestone detection
    "construction_diagram_search": "llama-3.2-11b-vision-instruct",  # Find similar
}


def get_model_path(model_key: str) -> Optional[str]:
    """
    Get the first available path for a model
    ONLY checks kalki/models directory - no fallbacks
    
    Args:
        model_key: Model key from MODELS dict
    
    Returns:
        Path to model or None if not found
    """
    model_info = MODELS.get(model_key)
    if not model_info:
        return None
    
    # ONLY check kalki/models - no HuggingFace fallbacks
    for path in model_info.get("paths", []):
        if isinstance(path, str) and not path.startswith("/"):
            # Skip HuggingFace model IDs - we only use local models
            continue
        
        # Handle Path objects and strings
        if isinstance(path, Path):
            path_obj = path
        else:
            path_obj = Path(path).expanduser()
        
        # Resolve absolute path
        try:
            path_obj = path_obj.resolve()
        except:
            pass
        
        if path_obj.exists():
            # Verify it's actually a model directory (has config.json or model files)
            config_exists = (path_obj / "config.json").exists()
            index_exists = (path_obj / "model.safetensors.index.json").exists()
            model_files = list(path_obj.glob("model-*.safetensors"))
            
            if config_exists or index_exists or len(model_files) > 0:
                logger.info(f"✅ Found model at: {path_obj}")
                return str(path_obj)
            else:
                logger.debug(f"Path exists but no model files found: {path_obj}")
    
    return None


def get_model_for_task(task: str) -> Optional[Dict]:
    """
    Get recommended model for a specific task
    
    Args:
        task: Task name from TASK_MODEL_MAP
    
    Returns:
        Model info dict or None
    """
    model_key = TASK_MODEL_MAP.get(task)
    if not model_key:
        return None
    
    return MODELS.get(model_key)


def check_model_availability() -> Dict[str, bool]:
    """
    Check which models are currently available locally
    
    Returns:
        Dict of {model_key: is_available}
    """
    availability = {}
    
    for model_key, model_info in MODELS.items():
        if not model_info.get("local_viable", True):
            availability[model_key] = False
            continue
        
        path = get_model_path(model_key)
        availability[model_key] = path is not None
    
    return availability


def print_model_status():
    """Print current model availability status"""
    print("\n" + "="*70)
    print("🤖 KALKI Model Status")
    print("="*70)
    
    availability = check_model_availability()
    
    for model_key, model_info in MODELS.items():
        name = model_info["name"]
        is_available = availability.get(model_key, False)
        status = "✅ Available" if is_available else "❌ Not Downloaded"
        
        if not model_info.get("local_viable", True):
            status = "☁️  Remote Only"
        
        print(f"\n{status}: {name}")
        print(f"  Type: {model_info['type'].capitalize()}")
        print(f"  Modalities: {', '.join(model_info['modalities'])}")
        print(f"  Use for: {', '.join(model_info['use_for'][:3])}")
        
        if is_available:
            path = get_model_path(model_key)
            print(f"  Path: {path}")
    
    print("\n" + "="*70)
    print("\n💡 Recommendations:")
    print("  • Llama 3.1 8B: Daily chat, fast responses (REQUIRED)")
    print("  • Llama 3.2 Vision 11B: PDF diagrams (OPTIONAL but recommended)")
    print("="*70)


if __name__ == "__main__":
    print_model_status()


# ═══════════════════════════════════════════════════════════════════════
# CONSTRUCTION COPILOT CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════

# Construction Copilot reuses KALKI's models - NO duplication!
CONSTRUCTION_COPILOT_CONFIG = {
    # Model usage (reuses existing models)
    "primary_model": "llama-3.1-8b-instruct",  # Always loaded
    "primary_quantization": "int8",  # 8GB RAM
    
    "vision_model": "llama-3.2-11b-vision-instruct",  # Load on-demand
    "vision_quantization": "int4",  # 6GB RAM when needed
    
    # Memory footprint
    "total_ram_peak": "14 GB",  # When both loaded simultaneously
    "total_ram_typical": "8 GB",  # Normal operation (text only)
    
    # Usage patterns (expected traffic)
    "daily_guidance": {
        "queries_per_user_per_day": 5,  # 5 questions daily during project
        "avg_tokens_per_query": 500,
        "response_time": "2-3 seconds"
    },
    "photo_qc": {
        "photos_per_project": 30,  # 30 progress photos over project lifecycle
        "avg_time_per_photo": "8-10 seconds",
        "frequency": "2-3x per week"
    },
    "blueprint_ingestion": {
        "frequency": "Once per project (upfront)",
        "time": "5-10 minutes for full blueprint set",
        "pages": "10-50 pages typical"
    },
    
    # Optimization strategy
    "optimization": {
        "strategy": "lazy_load_vision",  # Load vision only when photos uploaded
        "cache_text_responses": True,  # Cache common guidance
        "cache_vision_analyses": True,  # Cache analyzed photos (50-70% speedup)
        "unload_vision_after": "5 minutes idle",  # Free RAM when not in use
    },
    
    # System integration (uses all existing KALKI systems)
    "integrated_systems": [
        "consciousness_engine",  # WHY reasoning
        "meta_learning_system",  # Learn from outcomes
        "multi_agent_consensus",  # Validate decisions
        "autonomous_research",  # Research unknowns
        "visual_knowledge_base",  # Diagram search
        "cross_modal_knowledge_graph",  # Text↔Image links
        "reinforcement_learning_loop",  # Learn from feedback
        "self_evolution_manager",  # Self-improve
    ]
}


def get_construction_copilot_memory_estimate(simultaneous_users: int = 1) -> Dict:
    """
    Estimate memory requirements for Construction Copilot
    
    Args:
        simultaneous_users: Number of concurrent users
        
    Returns:
        Memory breakdown
    """
    base_text = 8  # GB for text model (int8)
    per_vision_session = 6  # GB for vision model (int4)
    kalki_overhead = 2  # GB for knowledge graphs, cache, etc.
    
    # Assume 20% of users using vision simultaneously
    vision_users = max(1, int(simultaneous_users * 0.2))
    
    total_ram = base_text + (per_vision_session * vision_users) + kalki_overhead
    
    return {
        "base_text_model": f"{base_text} GB",
        "vision_sessions": f"{vision_users} sessions × {per_vision_session} GB = {vision_users * per_vision_session} GB",
        "kalki_overhead": f"{kalki_overhead} GB",
        "total_required": f"{total_ram} GB",
        "recommended_ram": f"{total_ram + 4} GB",  # +4GB buffer
        "fits_in_36gb": total_ram <= 32  # Leave 4GB for OS
    }
