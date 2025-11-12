"""
Domain Registry

Central registry for all KALKI domain expertise.
Auto-discovers domain modules and provides unified interface.
"""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
import importlib.util
import logging

from .base_domain import BaseDomain, DomainModule


logger = logging.getLogger(__name__)


class DomainRegistry:
    """
    Central registry for all KALKI domain modules.
    
    Automatically discovers domain modules in modules/domains/ directory
    and provides methods to:
    - List available domains
    - Load specific domains
    - Infer which domain(s) a query needs
    - Query across multiple domains
    """
    
    def __init__(self):
        self.domains: Dict[str, DomainModule] = {}
        self.copilots: Dict[str, Any] = {}  # Store copilots for enhanced processing (lazy-loaded)
        self._copilots_loaded = False  # Flag to track if copilots have been loaded
        self.domain_dir = Path(__file__).parent
        self._discover_domains()
        # Don't load copilots in __init__ to avoid circular imports - load lazily
    
    def _discover_domains(self):
        """Auto-discover all domain modules"""
        logger.info("Discovering KALKI domain modules...")
        
        for domain_path in self.domain_dir.iterdir():
            if not domain_path.is_dir():
                continue
            
            # Skip special directories
            if domain_path.name.startswith('_') or domain_path.name == '__pycache__':
                continue
            
            # Check if it's a valid domain (has __init__.py)
            init_file = domain_path / "__init__.py"
            if not init_file.exists():
                continue
            
            # Extract domain name
            domain_name = domain_path.name.replace("_domain", "")
            
            try:
                domain_module = self._load_domain_module(domain_path, domain_name)
                if domain_module:
                    self.domains[domain_name] = domain_module
                    logger.info(f"✅ Loaded domain: {domain_name}")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load domain {domain_name}: {e}")
    
    def _load_domain_module(self, domain_path: Path, domain_name: str) -> Optional[DomainModule]:
        """
        Load a domain module dynamically.
        
        Looks for a class that inherits from BaseDomain.
        """
        try:
            # Use full folder name for module spec to preserve relative imports
            folder_name = domain_path.name
            
            # Import the domain's __init__.py with correct module name
            spec = importlib.util.spec_from_file_location(
                f"modules.domains.{folder_name}",
                domain_path / "__init__.py"
            )
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                # Look for domain class (should end with "Domain")
                domain_class = None
                for attr_name in dir(module):
                    attr = getattr(module, attr_name)
                    if (isinstance(attr, type) and 
                        issubclass(attr, BaseDomain) and 
                        attr != BaseDomain):
                        domain_class = attr
                        break
                
                if domain_class:
                    # Instantiate the domain
                    domain_instance = domain_class()
                    
                    # Get knowledge stats
                    stats = domain_instance.get_knowledge_stats()
                    
                    return DomainModule(
                        domain=domain_instance,
                        knowledge_stats=stats,
                        is_loaded=True
                    )
        except Exception as e:
            logger.error(f"Error loading domain module {domain_name}: {e}")
            return None
    
    def _discover_copilots(self):
        """Auto-discover copilots for domains (lazy loading to avoid circular imports)"""
        if self._copilots_loaded:
            return  # Already loaded
        
        logger.info("Discovering KALKI copilots...")
        
        # Game Dev Copilot
        try:
            from modules.game_dev_copilot import GameDevCopilot
            self.copilots["game_development"] = GameDevCopilot()
            logger.info("✅ Game Dev Copilot loaded")
        except Exception as e:
            logger.warning(f"⚠️  Game Dev Copilot unavailable: {e}")
        
        # Construction Copilot
        try:
            from modules.construction_copilot_enhanced import EnhancedConstructionCopilot
            self.copilots["construction"] = EnhancedConstructionCopilot()
            logger.info("✅ Construction Copilot loaded")
        except Exception as e:
            logger.warning(f"⚠️  Construction Copilot unavailable: {e}")
        
        self._copilots_loaded = True
    
    def get_domain(self, domain_name: str) -> Optional[BaseDomain]:
        """
        Get a specific domain by name.
        
        Args:
            domain_name: Name of domain (e.g., "construction", "game_development")
        
        Returns:
            Domain instance or None if not found
        """
        module = self.domains.get(domain_name)
        return module.domain if module else None
    
    def get_copilot(self, domain_name: str) -> Optional[Any]:
        """
        Get copilot for domain if available (lazy loads if not already loaded).
        
        Args:
            domain_name: Name of domain (e.g., "construction", "game_development")
        
        Returns:
            Copilot instance or None if not available
        """
        # Lazy load copilots on first access to avoid circular imports
        if not self._copilots_loaded:
            self._discover_copilots()
        
        return self.copilots.get(domain_name)
    
    def has_copilot(self, domain_name: str) -> bool:
        """
        Check if domain has an available copilot (lazy loads if not already loaded).
        
        Args:
            domain_name: Name of domain
        
        Returns:
            True if copilot is available, False otherwise
        """
        # Lazy load copilots on first access to avoid circular imports
        if not self._copilots_loaded:
            self._discover_copilots()
        
        return domain_name in self.copilots
    
    def list_domains(self) -> List[str]:
        """List all available domain names"""
        return list(self.domains.keys())
    
    def get_domain_info(self, domain_name: str) -> Optional[Dict]:
        """
        Get detailed info about a domain.
        
        Returns:
            Dict with name, description, knowledge_stats, deliverables
        """
        module = self.domains.get(domain_name)
        if not module:
            return None
        
        domain = module.domain
        return {
            "name": domain.name,
            "description": domain.description,
            "knowledge_stats": module.knowledge_stats,
            "knowledge_total": sum(module.knowledge_stats.values()),
            "deliverables": [d.name for d in domain.get_deliverable_types()],
            "is_loaded": module.is_loaded
        }
    
    def get_all_domains_info(self) -> Dict[str, Dict]:
        """Get info for all domains"""
        return {
            name: self.get_domain_info(name)
            for name in self.list_domains()
        }
    
    async def infer_domain(
        self,
        query: str,
        llm_client=None
    ) -> List[str]:
        """
        Infer which domain(s) a user query needs.
        
        Uses LLM to classify the query into one or more domains.
        
        Args:
            query: User's natural language query
            llm_client: LLM client for inference (optional, uses heuristics if None)
        
        Returns:
            List of domain names, ordered by relevance
        """
        # Simple keyword-based heuristics for now
        # TODO: Use LLM for better inference
        
        query_lower = query.lower()
        domain_scores = {}
        
        # Construction keywords - comprehensive coverage
        # Construction domain - weighted keyword matching
        # High-weight: Domain-specific technical terms (score x3)
        construction_high_weight = [
            # Building types
            "house", "home", "building", "residential", "commercial", 
            "garage", "shed", "deck", "patio", "addition", "extension",
            "renovation", "remodel", "kitchen", "bathroom", "bedroom", 
            "living room", "basement", "attic", "dormer",
            
            # Structural elements
            "foundation", "footing", "slab", "crawl space",
            "wall", "framing", "stud", "joist", "rafter", "truss", 
            "beam", "column", "post", "girder", "header",
            "floor", "ceiling", "roof", "sheathing", "shear wall",
            
            # Materials
            "concrete", "wood", "lumber", "steel", "brick", "stone",
            "insulation", "drywall", "gypsum", "plywood", "osb",
            
            # Systems
            "plumbing", "electrical", "hvac", "mechanical",
            "ventilation", "drainage", "septic",
            
            # Measurements & specs
            "span", "load", "bearing", "structural", "size", "sizing",
            "dimension", "spacing", "grade", "strength",
            
            # Code & standards
            "code", "permit", "inspection", "compliance", "regulation",
            "building code", "bc building code", "part 9"
        ]
        
        # Low-weight: Generic action words (score x1)
        construction_low_weight = [
            "design", "build", "construct", "install", "create",
            "architect", "engineer", "contractor", "builder",
            "excavation", "framing phase", "rough-in", "finish",
            "how to build", "what size", "construction cost",
            "construction schedule", "site plan", "blueprint"
        ]
        
        construction_score = (
            sum(3 for kw in construction_high_weight if kw in query_lower) +
            sum(1 for kw in construction_low_weight if kw in query_lower)
        )
        if construction_score > 0:
            domain_scores["construction"] = construction_score
        
        # Game development keywords - enhanced with weighting
        # High-weight: Game-specific technical terms (score x3)
        game_dev_high_weight = [
            "unity", "unreal", "godot", "gamemaker", "cocos2d",
            "platformer", "rpg", "fps", "strategy", "puzzle", "arcade", "shooter",
            "sprite", "texture", "shader", "particle", "vfx",
            "level design", "scene", "map editor", "tilemap",
            "character controller", "npc ai", "boss fight",
            "health bar", "hp", "mana", "stamina", "xp",
            "inventory system", "crafting", "skill tree",
            "multiplayer", "netcode", "matchmaking", "server",
            "monetization", "iap", "ads", "freemium",
            "physics engine", "collision", "rigidbody", "realistic physics",
            "game loop", "update loop", "fixed update",
            "state machine", "animation controller",
            "audio mixer", "sound effect", "music loop",
            "procedural generation", "dungeon", "roguelike", "procedural"
        ]
        
        # Medium-weight: Game-specific but common terms (score x2)
        game_dev_medium_weight = [
            "game", "gaming", "physics"
        ]
        
        # Low-weight: Generic game terms (score x1)
        game_dev_low_weight = [
            "player", "score", "menu", "ui",
            "level", "enemy", "weapon", "item", "quest", "algorithm"
        ]
        
        game_dev_score = (
            sum(3 for kw in game_dev_high_weight if kw in query_lower) +
            sum(2 for kw in game_dev_medium_weight if kw in query_lower) +
            sum(1 for kw in game_dev_low_weight if kw in query_lower)
        )
        if game_dev_score > 0:
            domain_scores["game_development"] = game_dev_score
        
        # Robotics keywords - enhanced with weighting
        # High-weight: Robotics-specific technical terms (score x3)
        robotics_high_weight = [
            "arduino", "raspberry pi", "esp32", "microcontroller", "embedded",
            "ros", "robot operating system", "gazebo", "rviz",
            "kinematics", "inverse kinematics", "forward kinematics",
            "pid controller", "control system", "feedback loop",
            "slam", "navigation", "mapping", "localization", "path planning",
            "gripper", "end effector", "manipulator", "robotic arm"
        ]
        
        # Low-weight: Generic robotics terms (score x1)
        robotics_low_weight = [
            "robot", "robotic", "automation", "autonomous",
            "sensor", "motor", "mobile robot"
        ]
        
        robotics_score = (
            sum(3 for kw in robotics_high_weight if kw in query_lower) +
            sum(1 for kw in robotics_low_weight if kw in query_lower)
        )
        if robotics_score > 0:
            domain_scores["robotics"] = robotics_score
        
        # Aerospace keywords - enhanced with weighting
        # High-weight: Aerospace-specific technical terms (score x3)
        aerospace_high_weight = [
            "aircraft", "airplane", "uav", "quadcopter", "multirotor", "hexacopter",
            "aerodynamic", "aerodynamics", "airfoil", "wing", "fuselage",
            "thrust", "lift", "drag coefficient", "angle of attack",
            "propulsion", "turbine", "jet engine", "propeller", "rotor",
            "vtol", "vertical takeoff", "hover", "altitude control",
            "airspeed", "mach number", "g-force", "maneuver",
            "aileron", "rudder", "elevator", "flaps", "control surface",
            "autopilot", "flight controller", "avionics", "nav system",
            "brushless motor", "esc", "lipo battery", "carbon fiber composite"
        ]
        
        # Low-weight: Generic aerospace terms (score x1)
        aerospace_low_weight = [
            "fly", "flying", "flight", "drone", "plane",
            "weight", "force", "velocity", "acceleration",
            "stability", "gps", "battery", "lightweight"
        ]
        
        aerospace_score = (
            sum(3 for kw in aerospace_high_weight if kw in query_lower) +
            sum(1 for kw in aerospace_low_weight if kw in query_lower)
        )
        if aerospace_score > 0:
            domain_scores["aerospace"] = aerospace_score
        
        # Power systems keywords - enhanced with weighting
        # High-weight: Power-specific technical terms (score x3)
        power_high_weight = [
            "lithium ion", "lifepo4", "lead acid", "nimh", "battery chemistry",
            "solar panel", "photovoltaic", "pv array", "solar cell",
            "inverter", "charge controller", "mppt", "pwm",
            "battery management system", "bms", "cell balancing",
            "energy density", "power density", "c-rate", "depth of discharge",
            "grid-tied", "off-grid", "hybrid system", "microgrid",
            "electric vehicle", "ev", "ev battery", "charging station",
            "power electronics", "dc-dc converter", "buck converter", "boost converter",
            "energy storage system", "ess", "renewable energy"
        ]
        
        # Low-weight: Generic power terms (score x1)
        power_low_weight = [
            "battery", "solar", "power", "energy", "voltage", "current",
            "efficiency", "consumption", "runtime", "capacity",
            "grid", "sizing", "power system"
        ]
        
        power_score = (
            sum(3 for kw in power_high_weight if kw in query_lower) +
            sum(1 for kw in power_low_weight if kw in query_lower)
        )
        if power_score > 0:
            domain_scores["power_systems"] = power_score
        
        # Sort by relevance
        sorted_domains = sorted(
            domain_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return domains with score > 0
        return [domain for domain, score in sorted_domains if score > 0]
    
    async def query_across_domains(
        self,
        query: str,
        domains: Optional[List[str]] = None
    ) -> Dict[str, List]:
        """
        Query knowledge across multiple domains.
        
        Args:
            query: Natural language query
            domains: Specific domains to query (if None, queries all)
        
        Returns:
            Dict mapping domain_name -> results
        """
        if domains is None:
            domains = self.list_domains()
        
        results = {}
        for domain_name in domains:
            domain = self.get_domain(domain_name)
            if domain:
                # Each domain would implement its own query method
                # For now, just return placeholder
                results[domain_name] = []
        
        return results
    
    def get_statistics(self) -> Dict[str, any]:
        """Get overall registry statistics"""
        total_knowledge = sum(
            sum(module.knowledge_stats.values())
            for module in self.domains.values()
        )
        
        return {
            "total_domains": len(self.domains),
            "loaded_domains": len([m for m in self.domains.values() if m.is_loaded]),
            "total_knowledge_items": total_knowledge,
            "domains": {
                name: sum(module.knowledge_stats.values())
                for name, module in self.domains.items()
            }
        }
    
    def __repr__(self):
        return f"DomainRegistry({len(self.domains)} domains loaded)"
