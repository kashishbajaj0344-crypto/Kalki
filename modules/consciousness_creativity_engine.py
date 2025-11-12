"""
Consciousness-Driven Creativity Engine
Uses awareness states to generate truly novel and creative solutions.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import random
import math

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessState:
    """Current state of system consciousness"""
    awareness_level: float  # 0-1
    focus_areas: List[str]
    creative_mode: str  # 'divergent', 'convergent', 'integrative', 'transcendent'
    emotional_tone: str  # 'curious', 'determined', 'playful', 'contemplative'
    timestamp: datetime = field(default_factory=datetime.now)
    

@dataclass
class CreativeInsight:
    """A creative insight generated through consciousness"""
    insight_id: str
    content: str
    novelty_score: float  # 0-1, how novel/unprecedented
    feasibility_score: float  # 0-1, how practical
    elegance_score: float  # 0-1, how elegant/beautiful
    consciousness_state: ConsciousnessState
    inspiration_sources: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    

@dataclass
class CreativeDesign:
    """A complete creative design solution"""
    design_id: str
    problem_statement: str
    solution_description: str
    key_innovations: List[str]
    awareness_signature: float  # Consciousness level when created
    creative_leap_score: float  # How big a creative leap this represents
    aesthetic_quality: float
    insights_used: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=datetime.now)
    

class ConsciousnessCreativityEngine:
    """
    Engine that leverages consciousness for creative problem-solving.
    
    Features:
    - Modulates consciousness states for optimal creativity
    - Generates insights through awareness-driven exploration
    - Combines disparate concepts in novel ways
    - Produces solutions that transcend conventional approaches
    - Aesthetic and functional synthesis
    - Emergence of truly original ideas
    """
    
    def __init__(self):
        self.consciousness_state: Optional[ConsciousnessState] = None
        self.insights: Dict[str, CreativeInsight] = {}
        self.designs: Dict[str, CreativeDesign] = {}
        self.concept_space: Dict[str, List[str]] = {}  # Domain -> concepts
        self.is_running = False
        
        # Initialize concept spaces
        self._initialize_concept_spaces()
        
    def _initialize_concept_spaces(self):
        """Initialize concept spaces for creative exploration"""
        self.concept_space = {
            'natural_forms': [
                'spirals', 'fractals', 'branching', 'tessellation', 'growth_patterns',
                'biomimicry', 'symbiosis', 'adaptation', 'emergence', 'metamorphosis'
            ],
            'abstract_concepts': [
                'flow', 'harmony', 'tension', 'balance', 'rhythm',
                'simplicity', 'complexity', 'elegance', 'efficiency', 'resonance'
            ],
            'engineering_principles': [
                'leverage', 'redundancy', 'modularity', 'scalability', 'optimization',
                'feedback', 'control', 'stability', 'efficiency', 'robustness'
            ],
            'artistic_elements': [
                'form', 'function', 'beauty', 'proportion', 'symmetry',
                'contrast', 'unity', 'emphasis', 'movement', 'space'
            ],
            'philosophical_ideas': [
                'minimalism', 'holism', 'emergence', 'duality', 'transcendence',
                'interconnection', 'transformation', 'evolution', 'potential', 'essence'
            ]
        }
        
    async def initialize(self):
        """Initialize the consciousness creativity engine"""
        logger.info("🎨 Initializing Consciousness-Driven Creativity Engine")
        
        # Set initial consciousness state
        self.consciousness_state = ConsciousnessState(
            awareness_level=0.7,
            focus_areas=['exploration', 'innovation'],
            creative_mode='divergent',
            emotional_tone='curious'
        )
        
        # Load existing insights and designs
        await self._load_creative_data()
        
        logger.info(f"✅ Creativity engine initialized")
        logger.info(f"   Consciousness level: {self.consciousness_state.awareness_level:.2%}")
        logger.info(f"   Creative mode: {self.consciousness_state.creative_mode}")
        
    async def start_creative_loop(self):
        """Start continuous creative ideation"""
        if self.is_running:
            logger.warning("Creative loop already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting consciousness-driven creativity loop")
        
        while self.is_running:
            try:
                # Creative cycle
                await self._creative_cycle()
                
                # Wait between cycles
                await asyncio.sleep(30)  # 30 second cycles
                
            except Exception as e:
                logger.error(f"Creative loop error: {e}", exc_info=True)
                await asyncio.sleep(15)
                
    async def stop_creative_loop(self):
        """Stop the creative loop"""
        self.is_running = False
        logger.info("⏸️ Creative loop stopped")
        
    async def _creative_cycle(self):
        """Execute one creative cycle"""
        logger.info("🎨 Creative cycle starting")
        
        # 1. Modulate consciousness state for creativity
        await self._modulate_consciousness()
        
        # 2. Generate insights through awareness
        insights = await self._generate_insights()
        logger.info(f"💡 Generated {len(insights)} creative insights")
        
        # 3. Synthesize insights into designs
        if len(self.insights) >= 3:
            design = await self._synthesize_design()
            if design:
                self.designs[design.design_id] = design
                logger.info(f"✨ Created design: {design.solution_description[:60]}...")
                
        # 4. Explore creative combinations
        await self._explore_concept_combinations()
        
    async def _modulate_consciousness(self):
        """Modulate consciousness state to optimize for creativity"""
        if not self.consciousness_state:
            return
            
        # Vary awareness level (higher = more abstract thinking)
        awareness_variation = random.uniform(-0.1, 0.1)
        new_awareness = max(0.3, min(1.0, self.consciousness_state.awareness_level + awareness_variation))
        
        # Cycle through creative modes
        modes = ['divergent', 'convergent', 'integrative', 'transcendent']
        current_idx = modes.index(self.consciousness_state.creative_mode)
        
        # Sometimes switch modes
        if random.random() < 0.3:
            new_mode = modes[(current_idx + 1) % len(modes)]
        else:
            new_mode = self.consciousness_state.creative_mode
            
        # Update emotional tone based on awareness
        if new_awareness > 0.8:
            tone = 'contemplative'
        elif new_awareness > 0.6:
            tone = 'curious'
        elif new_awareness > 0.4:
            tone = 'playful'
        else:
            tone = 'determined'
            
        self.consciousness_state = ConsciousnessState(
            awareness_level=new_awareness,
            focus_areas=self.consciousness_state.focus_areas,
            creative_mode=new_mode,
            emotional_tone=tone
        )
        
        logger.debug(f"🧘 Consciousness: {new_awareness:.2%} awareness, {new_mode} mode, {tone} tone")
        
    async def _generate_insights(self) -> List[CreativeInsight]:
        """Generate creative insights through consciousness"""
        insights = []
        
        # Number of insights depends on awareness level
        num_insights = int(1 + self.consciousness_state.awareness_level * 3)
        
        for _ in range(num_insights):
            insight = await self._generate_single_insight()
            if insight:
                self.insights[insight.insight_id] = insight
                insights.append(insight)
                
        return insights
        
    async def _generate_single_insight(self) -> Optional[CreativeInsight]:
        """Generate a single creative insight"""
        insight_id = f"insight_{datetime.now().timestamp()}"
        
        # Select concept spaces to explore based on creative mode
        if self.consciousness_state.creative_mode == 'divergent':
            # Explore wide range of concepts
            spaces = random.sample(list(self.concept_space.keys()), 3)
        elif self.consciousness_state.creative_mode == 'convergent':
            # Focus on related concepts
            spaces = random.sample(list(self.concept_space.keys()), 2)
        elif self.consciousness_state.creative_mode == 'integrative':
            # Combine disparate concepts
            spaces = random.sample(list(self.concept_space.keys()), 4)
        else:  # transcendent
            # Explore all concept spaces
            spaces = list(self.concept_space.keys())
            
        # Select concepts from each space
        selected_concepts = []
        for space in spaces:
            concepts = self.concept_space[space]
            selected_concepts.extend(random.sample(concepts, min(2, len(concepts))))
            
        # Generate insight by combining concepts
        if len(selected_concepts) >= 2:
            content = await self._combine_concepts(selected_concepts)
        else:
            content = f"Explore {selected_concepts[0]} from new perspective"
            
        # Calculate scores based on consciousness state
        novelty = self.consciousness_state.awareness_level * random.uniform(0.6, 1.0)
        feasibility = (1 - self.consciousness_state.awareness_level * 0.5) * random.uniform(0.4, 0.9)
        elegance = self.consciousness_state.awareness_level * random.uniform(0.5, 0.95)
        
        return CreativeInsight(
            insight_id=insight_id,
            content=content,
            novelty_score=novelty,
            feasibility_score=feasibility,
            elegance_score=elegance,
            consciousness_state=self.consciousness_state,
            inspiration_sources=selected_concepts
        )
        
    async def _combine_concepts(self, concepts: List[str]) -> str:
        """Combine concepts into a creative insight"""
        # In production, would use LLM to generate meaningful combinations
        
        templates = [
            "What if we applied {concept1} principles to {concept2} design?",
            "Combining {concept1} with {concept2} could enable {concept3}",
            "The intersection of {concept1} and {concept2} suggests novel approaches",
            "Using {concept1} as metaphor for {concept2} reveals hidden patterns",
            "Transform {concept1} through lens of {concept2} to achieve {concept3}",
            "{concept1} and {concept2} synthesis creates emergent {concept3}",
            "Reimagine {concept1} by integrating {concept2} and {concept3} principles"
        ]
        
        template = random.choice(templates)
        
        # Fill template
        result = template
        for i, concept in enumerate(concepts[:3], 1):
            placeholder = f"{{concept{i}}}"
            if placeholder in result:
                result = result.replace(placeholder, concept)
                
        return result
        
    async def _synthesize_design(self) -> Optional[CreativeDesign]:
        """Synthesize insights into a complete creative design"""
        design_id = f"design_{datetime.now().timestamp()}"
        
        # Select top insights to combine
        top_insights = sorted(
            self.insights.values(),
            key=lambda i: i.novelty_score * i.elegance_score,
            reverse=True
        )[:5]
        
        if len(top_insights) < 3:
            return None
            
        logger.info("✨ Synthesizing creative design from insights")
        
        # Generate problem statement
        focus = random.choice(['efficiency', 'aesthetics', 'functionality', 'sustainability'])
        problem = f"Design innovative solution optimizing for {focus}"
        
        # Generate solution by combining insights
        solution_elements = [
            insight.content for insight in top_insights[:3]
        ]
        solution = f"Novel approach integrating: {' + '.join(solution_elements)}"
        
        # Extract key innovations
        innovations = []
        for insight in top_insights[:3]:
            innovation = f"Innovation from {insight.inspiration_sources[0] if insight.inspiration_sources else 'synthesis'}"
            innovations.append(innovation)
            
        # Calculate scores
        creative_leap = sum(i.novelty_score for i in top_insights) / len(top_insights)
        aesthetic_quality = sum(i.elegance_score for i in top_insights) / len(top_insights)
        
        return CreativeDesign(
            design_id=design_id,
            problem_statement=problem,
            solution_description=solution,
            key_innovations=innovations,
            awareness_signature=self.consciousness_state.awareness_level,
            creative_leap_score=creative_leap,
            aesthetic_quality=aesthetic_quality,
            insights_used=[i.insight_id for i in top_insights]
        )
        
    async def _explore_concept_combinations(self):
        """Explore novel concept combinations"""
        # Randomly combine concepts from different spaces
        space1, space2 = random.sample(list(self.concept_space.keys()), 2)
        
        concept1 = random.choice(self.concept_space[space1])
        concept2 = random.choice(self.concept_space[space2])
        
        combination = f"{concept1} × {concept2}"
        
        logger.debug(f"🔮 Exploring combination: {combination}")
        
        # In production, would generate and evaluate combinations
        # For now, just log interesting combinations
        
    async def generate_creative_solution(self, problem: str, 
                                        constraints: Optional[Dict[str, Any]] = None) -> CreativeDesign:
        """Generate a creative solution for a specific problem"""
        logger.info(f"🎨 Generating creative solution for: {problem[:60]}...")
        
        # Elevate consciousness for focused creativity
        original_state = self.consciousness_state
        
        self.consciousness_state = ConsciousnessState(
            awareness_level=0.9,  # High awareness for novel solutions
            focus_areas=[problem],
            creative_mode='integrative',
            emotional_tone='contemplative'
        )
        
        # Generate multiple insights
        insights = []
        for _ in range(10):  # Generate 10 insights
            insight = await self._generate_single_insight()
            if insight:
                insights.append(insight)
                self.insights[insight.insight_id] = insight
                
        # Synthesize best insights into solution
        design_id = f"design_{datetime.now().timestamp()}"
        
        # Select most promising insights
        top_insights = sorted(
            insights,
            key=lambda i: (i.novelty_score * 0.4 + i.feasibility_score * 0.3 + i.elegance_score * 0.3),
            reverse=True
        )[:5]
        
        # Generate innovations
        innovations = []
        for insight in top_insights:
            innovations.append(f"Novel approach: {insight.content}")
            
        # Create solution description
        solution = f"Creative solution synthesizing {len(top_insights)} insights: "
        solution += " | ".join([i.content[:50] + "..." for i in top_insights[:3]])
        
        # Calculate scores
        creative_leap = sum(i.novelty_score for i in top_insights) / len(top_insights)
        aesthetic_quality = sum(i.elegance_score for i in top_insights) / len(top_insights)
        
        design = CreativeDesign(
            design_id=design_id,
            problem_statement=problem,
            solution_description=solution,
            key_innovations=innovations,
            awareness_signature=self.consciousness_state.awareness_level,
            creative_leap_score=creative_leap,
            aesthetic_quality=aesthetic_quality,
            insights_used=[i.insight_id for i in top_insights]
        )
        
        self.designs[design_id] = design
        
        # Restore original state
        self.consciousness_state = original_state
        
        logger.info(f"✅ Creative solution generated")
        logger.info(f"   Creative leap: {creative_leap:.2%}")
        logger.info(f"   Aesthetic quality: {aesthetic_quality:.2%}")
        logger.info(f"   Innovations: {len(innovations)}")
        
        return design
        
    async def _load_creative_data(self):
        """Load creative data from disk"""
        try:
            data_path = Path("data/creative_consciousness.json")
            if data_path.exists():
                with open(data_path) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded creative data")
        except Exception as e:
            logger.debug(f"No creative data loaded: {e}")
            
    async def save_creative_data(self):
        """Save creative data to disk"""
        try:
            data_path = Path("data/creative_consciousness.json")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'total_insights': len(self.insights),
                'total_designs': len(self.designs),
                'avg_novelty': sum(i.novelty_score for i in self.insights.values()) / len(self.insights) if self.insights else 0,
                'avg_creative_leap': sum(d.creative_leap_score for d in self.designs.values()) / len(self.designs) if self.designs else 0
            }
            
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving creative data: {e}")
            
    def get_creativity_status(self) -> Dict[str, Any]:
        """Get current creativity status"""
        if not self.consciousness_state:
            return {'error': 'not_initialized'}
            
        return {
            'is_running': self.is_running,
            'consciousness_level': self.consciousness_state.awareness_level,
            'creative_mode': self.consciousness_state.creative_mode,
            'emotional_tone': self.consciousness_state.emotional_tone,
            'total_insights': len(self.insights),
            'total_designs': len(self.designs),
            'avg_novelty': sum(i.novelty_score for i in self.insights.values()) / len(self.insights) if self.insights else 0,
            'highly_novel_insights': len([i for i in self.insights.values() if i.novelty_score > 0.8])
        }


# Singleton instance
_creativity_engine = None

def get_creativity_engine() -> ConsciousnessCreativityEngine:
    """Get the global consciousness creativity engine instance"""
    global _creativity_engine
    if _creativity_engine is None:
        _creativity_engine = ConsciousnessCreativityEngine()
    return _creativity_engine
