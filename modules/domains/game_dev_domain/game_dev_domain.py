"""
Game Development Domain

Specialized domain for game development projects including:
- Game design and prototyping
- Unity/Unreal/Godot development
- Multiplayer/networking
- Monetization and publishing
- Performance optimization
- Asset management
"""

from enum import Enum
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

from modules.domains.base_domain import (
    BaseDomain,
    ProjectStateMachine,
    ProjectPhase,
    ValidationResult,
    KnowledgeExtractor,
    ComplexityScore,
    DeliverableSpec
)


class GameDevPhase(str, Enum):
    """Game development project phases"""
    CONCEPT = "concept"
    PRE_PRODUCTION = "pre_production"
    PROTOTYPE = "prototype"
    PRODUCTION = "production"
    ALPHA = "alpha"
    BETA = "beta"
    POLISH = "polish"
    LAUNCH = "launch"
    POST_LAUNCH = "post_launch"


class GameGenre(str, Enum):
    """Game genres"""
    PLATFORMER = "platformer"
    RPG = "rpg"
    FPS = "fps"
    STRATEGY = "strategy"
    PUZZLE = "puzzle"
    ARCADE = "arcade"
    ADVENTURE = "adventure"
    SIMULATION = "simulation"
    SPORTS = "sports"
    RACING = "racing"


class GameDevProjectStateMachine(ProjectStateMachine):
    """State machine for game development projects with milestone tracking"""
    
    def __init__(self, project_id: str, description: str):
        super().__init__(project_id, description, "game_development")
        self.current_phase = GameDevPhase.CONCEPT
        
        # Budget tracking
        self.budget = {
            "estimated_total": 0,
            "actual_spent": 0,
            "by_phase": {},
            "by_category": {
                "development": 0,
                "art": 0,
                "audio": 0,
                "marketing": 0,
                "infrastructure": 0
            }
        }
        
        # Timeline management
        self.timeline = {
            "start_date": None,
            "target_launch": None,
            "phase_durations": {},
            "actual_phase_durations": {}
        }
        
        # Game-specific details
        self.game_engine = None  # unity, unreal, godot, custom
        self.target_platforms = []  # pc, mobile, console, web
        self.genre = None
        self.team_size = 1
        self.monetization_model = None  # premium, freemium, ads, subscription
        
        # Milestone tracking per phase
        self.milestones = {
            GameDevPhase.CONCEPT: [
                {"name": "Game concept document complete", "complete": False},
                {"name": "Target audience identified", "complete": False},
                {"name": "Core mechanics defined", "complete": False}
            ],
            GameDevPhase.PRE_PRODUCTION: [
                {"name": "Technical design document complete", "complete": False},
                {"name": "Art style established", "complete": False},
                {"name": "Prototype plan finalized", "complete": False},
                {"name": "Development tools set up", "complete": False}
            ],
            GameDevPhase.PROTOTYPE: [
                {"name": "Core gameplay loop implemented", "complete": False},
                {"name": "Basic controls working", "complete": False},
                {"name": "Core mechanics tested", "complete": False},
                {"name": "Proof of fun validated", "complete": False}
            ],
            GameDevPhase.PRODUCTION: [
                {"name": "Game systems implemented", "complete": False},
                {"name": "Level design complete", "complete": False},
                {"name": "Art assets created", "complete": False},
                {"name": "Audio implemented", "complete": False},
                {"name": "UI/UX complete", "complete": False}
            ],
            GameDevPhase.ALPHA: [
                {"name": "All features implemented", "complete": False},
                {"name": "Content complete", "complete": False},
                {"name": "Internal playtesting started", "complete": False},
                {"name": "Major bugs identified", "complete": False}
            ],
            GameDevPhase.BETA: [
                {"name": "Closed beta launched", "complete": False},
                {"name": "Community feedback gathered", "complete": False},
                {"name": "Balancing adjustments made", "complete": False},
                {"name": "Performance optimized", "complete": False}
            ],
            GameDevPhase.POLISH: [
                {"name": "All critical bugs fixed", "complete": False},
                {"name": "Final art pass complete", "complete": False},
                {"name": "Audio polish complete", "complete": False},
                {"name": "Marketing materials prepared", "complete": False}
            ],
            GameDevPhase.LAUNCH: [
                {"name": "Store pages live", "complete": False},
                {"name": "Launch trailer released", "complete": False},
                {"name": "Press kit distributed", "complete": False},
                {"name": "Game released", "complete": False}
            ],
            GameDevPhase.POST_LAUNCH: [
                {"name": "Player feedback monitored", "complete": False},
                {"name": "Patches released", "complete": False},
                {"name": "DLC/updates planned", "complete": False},
                {"name": "Community engagement active", "complete": False}
            ]
        }
    
    async def validate_phase_complete(self, phase: GameDevPhase) -> ValidationResult:
        """Validate if phase requirements are met"""
        errors = []
        warnings = []
        suggestions = []
        
        # Get phase milestones
        phase_milestones = self.milestones.get(phase, [])
        completed = sum(1 for m in phase_milestones if m["complete"])
        total = len(phase_milestones)
        
        # Phase-specific validation
        if phase == GameDevPhase.CONCEPT:
            if not self.genre:
                warnings.append("Game genre not specified")
            if not self.target_platforms:
                warnings.append("Target platforms not specified")
            if completed < total:
                errors.append(f"Only {completed}/{total} concept milestones complete")
        
        elif phase == GameDevPhase.PRE_PRODUCTION:
            if not self.game_engine:
                errors.append("Game engine not selected")
            if completed < total:
                warnings.append(f"Only {completed}/{total} pre-production milestones complete")
        
        elif phase == GameDevPhase.PROTOTYPE:
            if completed < 3:
                errors.append("Core prototype milestones must be complete")
            
        elif phase == GameDevPhase.ALPHA:
            if completed < total:
                warnings.append(f"Alpha not feature complete: {completed}/{total} milestones")
        
        elif phase == GameDevPhase.BETA:
            if completed < 2:
                errors.append("Need community feedback before proceeding")
        
        elif phase == GameDevPhase.LAUNCH:
            if self.budget["actual_spent"] > self.budget["estimated_total"] * 1.2:
                warnings.append(f"Budget exceeded by {(self.budget['actual_spent'] / self.budget['estimated_total'] - 1) * 100:.1f}%")
        
        # Budget warnings
        if self.budget["estimated_total"] > 0:
            percent_spent = (self.budget["actual_spent"] / self.budget["estimated_total"]) * 100
            if percent_spent > 90:
                warnings.append(f"Budget {percent_spent:.1f}% spent")
        
        # Success suggestions
        if not errors:
            if phase == GameDevPhase.PROTOTYPE:
                suggestions.append("Consider user testing your prototype")
            elif phase == GameDevPhase.ALPHA:
                suggestions.append("Start building community before launch")
            elif phase == GameDevPhase.BETA:
                suggestions.append("Focus on player retention metrics")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def advance_phase(self, next_phase: GameDevPhase) -> bool:
        """Advance to next project phase"""
        from datetime import datetime
        
        # Record phase transition
        self.phase_history.append({
            "from": self.current_phase,
            "to": next_phase,
            "timestamp": datetime.now().isoformat()
        })
        
        # Update current phase
        self.current_phase = next_phase
        return True
    
    def get_available_phases(self) -> list:
        """Get all game dev phases"""
        return list(GameDevPhase)
    
    def mark_milestone_complete(self, milestone_name: str) -> bool:
        """Mark a milestone as complete"""
        phase_milestones = self.milestones.get(self.current_phase, [])
        for milestone in phase_milestones:
            if milestone["name"] == milestone_name:
                milestone["complete"] = True
                return True
        return False
    
    def get_phase_progress(self) -> Dict[str, Any]:
        """Get progress statistics for current phase"""
        phase_milestones = self.milestones.get(self.current_phase, [])
        total = len(phase_milestones)
        complete = sum(1 for m in phase_milestones if m["complete"])
        
        return {
            "phase": self.current_phase.value,
            "total_milestones": total,
            "completed_milestones": complete,
            "percent_complete": (complete / total * 100) if total > 0 else 0,
            "milestones": phase_milestones
        }
    
    def update_budget(self, category: str, amount: float):
        """Update budget tracking"""
        self.budget["actual_spent"] += amount
        
        # Update category
        if category in self.budget["by_category"]:
            self.budget["by_category"][category] += amount
        
        # Update phase
        phase_key = self.current_phase.value
        if phase_key not in self.budget["by_phase"]:
            self.budget["by_phase"][phase_key] = {"estimated": 0, "actual": 0}
        self.budget["by_phase"][phase_key]["actual"] += amount
    
    def get_budget_status(self) -> Dict[str, Any]:
        """Get current budget status"""
        total = self.budget["estimated_total"]
        spent = self.budget["actual_spent"]
        percent = (spent / total * 100) if total > 0 else 0
        
        status = "on_budget"
        if percent > 100:
            status = "over_budget"
        elif percent > 90:
            status = "warning"
        
        return {
            "total_budget": total,
            "spent": spent,
            "percent_spent": percent,
            "status": status,
            "by_category": self.budget["by_category"]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for persistence"""
        return {
            "project_id": self.project_id,
            "description": self.description,
            "domain": self.domain,
            "current_phase": self.current_phase.value,
            "phase_history": self.phase_history,
            "budget": self.budget,
            "timeline": self.timeline,
            "game_engine": self.game_engine,
            "target_platforms": self.target_platforms,
            "genre": self.genre.value if self.genre else None,
            "team_size": self.team_size,
            "monetization_model": self.monetization_model,
            "milestones": {
                k.value: v for k, v in self.milestones.items()
            },
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'GameDevProjectStateMachine':
        """Reconstruct from saved dict"""
        state_machine = cls(data["project_id"], data["description"])
        
        # Restore state - handle both formats
        current_phase_str = data["current_phase"]
        if "." in current_phase_str:
            current_phase_str = current_phase_str.split(".")[-1]
        state_machine.current_phase = GameDevPhase(current_phase_str)
        
        state_machine.phase_history = data.get("phase_history", [])
        state_machine.budget = data.get("budget", {})
        state_machine.timeline = data.get("timeline", {})
        state_machine.game_engine = data.get("game_engine")
        state_machine.target_platforms = data.get("target_platforms", [])
        state_machine.team_size = data.get("team_size", 1)
        state_machine.monetization_model = data.get("monetization_model")
        
        # Restore genre
        genre_str = data.get("genre")
        if genre_str:
            state_machine.genre = GameGenre(genre_str)
        
        # Restore milestones
        if "milestones" in data:
            state_machine.milestones = {}
            for k, v in data["milestones"].items():
                phase_key = k.split(".")[-1] if "." in k else k
                state_machine.milestones[GameDevPhase(phase_key)] = v
        
        return state_machine
    
    async def get_contextual_help(self, user_query: str) -> str:
        """Provide help relevant to current game dev phase"""
        phase_help = {
            GameDevPhase.CONCEPT: "Concept phase: Define core game idea, target audience, and unique selling points",
            GameDevPhase.PRE_PRODUCTION: "Pre-production: Plan technical architecture, art style, and development pipeline",
            GameDevPhase.PROTOTYPE: "Prototype phase: Prove core gameplay is fun. Iterate quickly on mechanics",
            GameDevPhase.PRODUCTION: "Production phase: Implement all game systems, create assets, build levels",
            GameDevPhase.ALPHA: "Alpha phase: Feature complete. Focus on bug fixing and content completion",
            GameDevPhase.BETA: "Beta phase: Gather player feedback, balance gameplay, optimize performance",
            GameDevPhase.POLISH: "Polish phase: Final quality pass on all assets and systems",
            GameDevPhase.LAUNCH: "Launch phase: Marketing push, press outreach, store optimization",
            GameDevPhase.POST_LAUNCH: "Post-launch: Monitor metrics, fix issues, engage community, plan updates"
        }
        
        progress = self.get_phase_progress()
        budget_status = self.get_budget_status()
        
        help_text = f"""
Current Phase: {self.current_phase.value}
Progress: {progress['completed_milestones']}/{progress['total_milestones']} milestones ({progress['percent_complete']:.1f}%)
Budget: ${budget_status['spent']:,.2f} spent of ${budget_status['total_budget']:,.2f} ({budget_status['percent_spent']:.1f}%)

{phase_help.get(self.current_phase, 'Game development in progress')}

Pending Milestones:
"""
        
        for milestone in progress['milestones']:
            if not milestone['complete']:
                help_text += f"  • {milestone['name']}\n"
        
        return help_text.strip()


class GameDevelopmentDomain(BaseDomain):
    """Game Development Domain"""
    
    def __init__(self):
        super().__init__(
            name="game_development",
            description="Game development projects - design, prototyping, production, launch"
        )
        
        # Professional systems integration (lazy initialization)
        self._professional_integration = None
        
        # Initialize knowledge extractors
        self.knowledge_extractors = [
            KnowledgeExtractor(
                name="game_mechanics",
                description="Extract gameplay mechanics, systems, and patterns",
                patterns=["mechanic", "gameplay", "system", "pattern", "loop"],
                extractor_func=None,
                storage_db="game_mechanics"
            ),
            KnowledgeExtractor(
                name="engine_docs",
                description="Extract Unity, Unreal, Godot documentation",
                patterns=["unity", "unreal", "godot", "engine", "api"],
                extractor_func=None,
                storage_db="engine_docs"
            ),
            KnowledgeExtractor(
                name="optimization",
                description="Extract performance optimization techniques",
                patterns=["optimization", "performance", "fps", "memory", "profiling"],
                extractor_func=None,
                storage_db="optimization"
            ),
            KnowledgeExtractor(
                name="monetization",
                description="Extract monetization strategies",
                patterns=["monetization", "iap", "ads", "revenue", "pricing"],
                extractor_func=None,
                storage_db="monetization"
            ),
            KnowledgeExtractor(
                name="multiplayer",
                description="Extract networking and multiplayer patterns",
                patterns=["multiplayer", "networking", "netcode", "server", "sync"],
                extractor_func=None,
                storage_db="multiplayer"
            ),
            KnowledgeExtractor(
                name="publishing",
                description="Extract publishing and marketing strategies",
                patterns=["publish", "marketing", "steam", "app store", "launch"],
                extractor_func=None,
                storage_db="publishing"
            )
        ]
    
    def get_knowledge_extractors(self) -> list:
        """Return game dev knowledge extractors"""
        return self.knowledge_extractors
    
    async def create_project(
        self,
        description: str,
        requirements: Optional[Dict[str, Any]] = None
    ) -> GameDevProjectStateMachine:
        """Create a new game development project"""
        import uuid
        project_id = f"game-{uuid.uuid4().hex[:8]}"
        
        project = GameDevProjectStateMachine(project_id, description)
        
        # Apply requirements if provided
        if requirements:
            if "game_engine" in requirements:
                project.game_engine = requirements["game_engine"]
            if "target_platforms" in requirements:
                project.target_platforms = requirements["target_platforms"]
            if "genre" in requirements:
                project.genre = GameGenre(requirements["genre"])
            if "team_size" in requirements:
                project.team_size = requirements["team_size"]
            if "monetization_model" in requirements:
                project.monetization_model = requirements["monetization_model"]
            if "budget" in requirements:
                project.budget["estimated_total"] = requirements["budget"]
        
        return project
    
    def get_deliverable_types(self) -> list:
        """List all deliverables game dev can generate"""
        from modules.domains.base_domain import DeliverableSpec
        
        return [
            DeliverableSpec(
                name="game_design_document",
                description="Comprehensive game design document (GDD)",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["game_mechanics", "design_patterns"]
            ),
            DeliverableSpec(
                name="technical_spec",
                description="Technical architecture specification",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["engine_docs", "optimization"]
            ),
            DeliverableSpec(
                name="asset_list",
                description="Comprehensive asset production list",
                file_types=["json", "xlsx"],
                generator_func=None,
                required_knowledge=[]
            ),
            DeliverableSpec(
                name="monetization_plan",
                description="Monetization and business strategy",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["monetization"]
            ),
            DeliverableSpec(
                name="marketing_plan",
                description="Marketing and launch strategy",
                file_types=["json", "pdf"],
                generator_func=None,
                required_knowledge=["publishing"]
            )
        ]
    
    async def validate_requirements(
        self,
        requirements: Dict[str, Any]
    ) -> ValidationResult:
        """Validate game dev project requirements"""
        errors = []
        warnings = []
        suggestions = []
        
        # Check required fields
        if "game_engine" not in requirements:
            warnings.append("Game engine not specified (Unity recommended for beginners)")
        
        if "target_platforms" not in requirements or not requirements["target_platforms"]:
            errors.append("Target platforms required (e.g., PC, mobile, console)")
        
        if "genre" not in requirements:
            warnings.append("Game genre not specified")
        
        # Validate platform + engine compatibility
        engine = requirements.get("game_engine", "").lower()
        platforms = requirements.get("target_platforms", [])
        
        if "console" in platforms and engine == "gamemaker":
            warnings.append("GameMaker console support is limited")
        
        # Budget validation
        budget = requirements.get("budget", 0)
        team_size = requirements.get("team_size", 1)
        
        if budget > 0 and team_size > 0:
            per_person = budget / team_size
            if per_person < 10000:
                warnings.append(f"Budget may be low: ${per_person:,.0f} per team member")
        
        # Monetization validation
        monetization = requirements.get("monetization_model", "").lower()
        if monetization == "freemium" and "mobile" not in platforms:
            suggestions.append("Freemium model works best on mobile platforms")
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            suggestions=suggestions
        )
    
    async def estimate_complexity(
        self,
        project: GameDevProjectStateMachine
    ) -> ComplexityScore:
        """Estimate game dev project complexity"""
        from modules.domains.base_domain import ComplexityScore
        
        # Start with base complexity
        score = 30.0
        factors = {}
        
        # Genre complexity
        genre_scores = {
            GameGenre.PUZZLE: 20,
            GameGenre.PLATFORMER: 30,
            GameGenre.ARCADE: 25,
            GameGenre.ADVENTURE: 40,
            GameGenre.RPG: 60,
            GameGenre.STRATEGY: 55,
            GameGenre.FPS: 50,
            GameGenre.SIMULATION: 45
        }
        
        if project.genre:
            genre_score = genre_scores.get(project.genre, 30)
            score += genre_score
            factors["genre_complexity"] = genre_score
        
        # Platform complexity
        platform_count = len(project.target_platforms)
        platform_score = platform_count * 10
        score += platform_score
        factors["multi_platform"] = platform_score
        
        # Team size factor (smaller teams = longer dev time)
        if project.team_size < 3:
            team_factor = 15
            score += team_factor
            factors["small_team"] = team_factor
        
        # Multiplayer adds significant complexity
        if project.monetization_model in ["multiplayer", "mmo"]:
            multiplayer_score = 30
            score += multiplayer_score
            factors["multiplayer"] = multiplayer_score
        
        # Cap at 100
        score = min(score, 100)
        
        # Estimate time (in days)
        # Simple formula: score * team_size_factor * genre_factor
        base_days = score * 3  # 3 days per complexity point
        team_factor = 3.0 / max(project.team_size, 1)  # Assume 3-person baseline
        time_estimate = int(base_days * team_factor)
        
        # Estimate cost
        daily_rate = 500  # $500/day per developer
        cost_estimate = time_estimate * daily_rate * project.team_size
        
        # Risk level
        if score < 40:
            risk = "low"
        elif score < 70:
            risk = "medium"
        else:
            risk = "high"
        
        return ComplexityScore(
            overall_score=score,
            time_estimate_days=time_estimate,
            cost_estimate_usd=cost_estimate,
            risk_level=risk,
            factors=factors
        )
    
    def get_knowledge_stats(self) -> Dict[str, int]:
        """Get knowledge base statistics"""
        return {
            "game_mechanics": 0,
            "design_patterns": 0,
            "engine_apis": 0,
            "optimization_techniques": 0,
            "monetization_strategies": 0,
            "multiplayer_patterns": 0
        }
    
    async def generate_deliverables(
        self,
        project: GameDevProjectStateMachine,
        deliverable_types: List[str],
        output_dir: Path
    ) -> Dict[str, Path]:
        """Generate game development deliverables"""
        from modules.domains.game_dev_domain.deliverables_generator import GameDevDeliverablesGenerator
        
        generator = GameDevDeliverablesGenerator()
        generated_files = {}
        
        for deliverable_type in deliverable_types:
            if deliverable_type == "game_design_document":
                result = generator.generate_game_design_document(project)
                if result:
                    file_path = output_dir / "game_design_document.json"
                    with open(file_path, 'w') as f:
                        import json
                        json.dump(result, f, indent=2)
                    generated_files["game_design_document"] = file_path
            
            elif deliverable_type == "technical_spec":
                result = generator.generate_technical_spec(project)
                if result:
                    file_path = output_dir / "technical_spec.json"
                    with open(file_path, 'w') as f:
                        import json
                        json.dump(result, f, indent=2)
                    generated_files["technical_spec"] = file_path
            
            elif deliverable_type == "asset_list":
                result = generator.generate_asset_list(project)
                if result:
                    file_path = output_dir / "asset_list.json"
                    with open(file_path, 'w') as f:
                        import json
                        json.dump(result, f, indent=2)
                    generated_files["asset_list"] = file_path
            
            elif deliverable_type == "monetization_plan":
                result = generator.generate_monetization_plan(project)
                if result:
                    file_path = output_dir / "monetization_plan.json"
                    with open(file_path, 'w') as f:
                        import json
                        json.dump(result, f, indent=2)
                    generated_files["monetization_plan"] = file_path
            
            elif deliverable_type == "marketing_plan":
                result = generator.generate_marketing_plan(project)
                if result:
                    file_path = output_dir / "marketing_plan.json"
                    with open(file_path, 'w') as f:
                        import json
                        json.dump(result, f, indent=2)
                    generated_files["marketing_plan"] = file_path
        
        return generated_files

    async def _get_professional_integration(self):
        """Get or initialize professional integration"""
        if self._professional_integration is None:
            from modules.domains.domain_professional_integration import DomainProfessionalIntegration
            self._professional_integration = DomainProfessionalIntegration(
                domain_name="game_dev"
            )
            await self._professional_integration.initialize()
            
            # Initialize game dev professional roles
            await self._professional_integration.initialize_roles([
                ("GAME_DESIGNER", "CREATIVE_SYNTHESIS"),
                ("PROGRAMMER", "OPTIMIZATION"),
                ("ARTIST", "CREATIVE_SYNTHESIS"),
                ("QA_TESTER", "QUALITY_ASSESSMENT")
            ])
        
        return self._professional_integration
    
    async def get_team_orchestrator(self):
        """Get professional team orchestrator"""
        integration = await self._get_professional_integration()
        return integration.team_orchestrator
    
    async def get_deliverable_generator(self):
        """Get professional deliverable generator"""
        integration = await self._get_professional_integration()
        return integration.deliverable_generator
    
    async def get_cross_learning(self):
        """Get cross-domain learning system"""
        integration = await self._get_professional_integration()
        return integration.cross_learning
    
    async def get_workflow_executor(self):
        """Get professional workflow executor"""
        integration = await self._get_professional_integration()
        return integration.workflow_executor
    
    async def get_quality_framework(self):
        """Get quality assurance framework"""
        integration = await self._get_professional_integration()
        return integration.quality_framework