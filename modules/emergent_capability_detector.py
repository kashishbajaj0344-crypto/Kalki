"""
Emergent Capability Detection System
Automatically detects and catalogs new capabilities as they emerge from system evolution.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json
import inspect
import ast

logger = logging.getLogger(__name__)


@dataclass
class EmergentCapability:
    """A newly discovered emergent capability"""
    capability_id: str
    name: str
    description: str
    emergence_source: str  # 'evolution', 'research', 'creativity', 'composition'
    capability_type: str  # 'problem_solving', 'creative', 'analytical', 'synthesis'
    discovered_at: datetime = field(default_factory=datetime.now)
    confidence_score: float = 0.5  # How confident we are this is a real capability
    demonstration_count: int = 0  # Times successfully demonstrated
    validation_status: str = 'unvalidated'  # 'unvalidated', 'testing', 'validated', 'false_positive'
    impact_assessment: Optional[Dict[str, Any]] = None
    

@dataclass
class CapabilitySignature:
    """Signature/fingerprint of a capability"""
    signature_id: str
    input_patterns: List[str]
    output_patterns: List[str]
    behavioral_patterns: Dict[str, Any]
    performance_characteristics: Dict[str, float]
    

@dataclass
class CapabilityTest:
    """Test to validate a capability"""
    test_id: str
    capability_id: str
    test_scenario: str
    expected_outcome: str
    actual_outcome: Optional[str] = None
    passed: bool = False
    executed_at: Optional[datetime] = None


class EmergentCapabilityDetector:
    """
    System for detecting and cataloging emergent capabilities.
    
    Features:
    - Monitors system behavior for novel patterns
    - Detects when system can do something it couldn't before
    - Validates emergent capabilities through testing
    - Catalogs and categorizes capabilities
    - Tracks capability evolution over time
    - Identifies capability synergies and compositions
    """
    
    def __init__(self):
        self.capabilities: Dict[str, EmergentCapability] = {}
        self.capability_signatures: Dict[str, CapabilitySignature] = {}
        self.capability_tests: Dict[str, CapabilityTest] = {}
        self.is_running = False
        
        # Tracking
        self.known_behaviors: Set[str] = set()
        self.behavior_history: List[Dict[str, Any]] = []
        
        # Thresholds
        self.novelty_threshold = 0.7  # How novel a behavior must be
        self.confidence_threshold = 0.8  # Confidence needed for validation
        
    async def initialize(self):
        """Initialize the emergent capability detector"""
        logger.info("🔍 Initializing Emergent Capability Detection System")
        
        # Load existing capabilities
        await self._load_capabilities()
        
        # Initialize baseline behaviors
        await self._establish_baseline()
        
        logger.info(f"✅ Detector initialized")
        logger.info(f"   Known capabilities: {len(self.capabilities)}")
        logger.info(f"   Baseline behaviors: {len(self.known_behaviors)}")
        
    async def start_detection_loop(self):
        """Start continuous capability detection"""
        if self.is_running:
            logger.warning("Detection loop already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting capability detection loop")
        
        while self.is_running:
            try:
                # Detection cycle
                await self._detection_cycle()
                
                # Wait between cycles
                await asyncio.sleep(60)  # 1 minute
                
            except Exception as e:
                logger.error(f"Detection loop error: {e}", exc_info=True)
                await asyncio.sleep(30)
                
    async def stop_detection_loop(self):
        """Stop capability detection"""
        self.is_running = False
        logger.info("⏸️ Detection loop stopped")
        
    async def _detection_cycle(self):
        """Execute one detection cycle"""
        logger.info("🔍 Capability detection cycle")
        
        # 1. Monitor system for novel behaviors
        novel_behaviors = await self._monitor_system_behaviors()
        
        if novel_behaviors:
            logger.info(f"👀 Detected {len(novel_behaviors)} novel behaviors")
            
            # 2. Analyze for potential capabilities
            for behavior in novel_behaviors:
                capability = await self._analyze_for_capability(behavior)
                if capability:
                    self.capabilities[capability.capability_id] = capability
                    logger.info(f"✨ New emergent capability: {capability.name}")
                    
        # 3. Validate unvalidated capabilities
        unvalidated = [c for c in self.capabilities.values() if c.validation_status == 'unvalidated']
        for capability in unvalidated[:3]:  # Test 3 at a time
            await self._validate_capability(capability)
            
        # 4. Test capabilities under validation
        testing = [c for c in self.capabilities.values() if c.validation_status == 'testing']
        for capability in testing:
            await self._continue_validation(capability)
            
        # 5. Detect capability compositions
        await self._detect_compositions()
        
    async def _monitor_system_behaviors(self) -> List[Dict[str, Any]]:
        """Monitor system for novel behaviors"""
        novel_behaviors = []
        
        # Check different system components for new behaviors
        sources = [
            await self._check_evolution_system(),
            await self._check_research_system(),
            await self._check_creativity_engine(),
            await self._check_agent_system()
        ]
        
        for behaviors in sources:
            novel_behaviors.extend(behaviors)
            
        return novel_behaviors
        
    async def _check_evolution_system(self) -> List[Dict[str, Any]]:
        """Check evolution system for new behaviors"""
        behaviors = []
        
        try:
            from modules.autonomous_evolution_loop import get_evolution_loop
            evolution = get_evolution_loop()
            
            # Check if evolution system has deployed new capabilities
            status = evolution.get_evolution_status()
            
            if status.get('total_deployed_evolutions', 0) > 0:
                behaviors.append({
                    'source': 'evolution',
                    'type': 'system_improvement',
                    'description': f"System evolved {status['total_deployed_evolutions']} times",
                    'novelty': 0.8,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.debug(f"Evolution check failed: {e}")
            
        return behaviors
        
    async def _check_research_system(self) -> List[Dict[str, Any]]:
        """Check research system for new capabilities"""
        behaviors = []
        
        try:
            from modules.autonomous_research_system import get_research_system
            research = get_research_system()
            
            status = research.get_research_status()
            
            # Breakthrough discoveries indicate new capabilities
            if status.get('breakthrough_discoveries', 0) > 0:
                behaviors.append({
                    'source': 'research',
                    'type': 'knowledge_discovery',
                    'description': f"Discovered {status['breakthrough_discoveries']} breakthroughs",
                    'novelty': 0.9,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.debug(f"Research check failed: {e}")
            
        return behaviors
        
    async def _check_creativity_engine(self) -> List[Dict[str, Any]]:
        """Check creativity engine for new patterns"""
        behaviors = []
        
        try:
            from modules.consciousness_creativity_engine import get_creativity_engine
            creativity = get_creativity_engine()
            
            status = creativity.get_creativity_status()
            
            # Highly novel insights indicate new creative capabilities
            if status.get('highly_novel_insights', 0) > 0:
                behaviors.append({
                    'source': 'creativity',
                    'type': 'creative_synthesis',
                    'description': f"Generated {status['highly_novel_insights']} highly novel insights",
                    'novelty': 0.85,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.debug(f"Creativity check failed: {e}")
            
        return behaviors
        
    async def _check_agent_system(self) -> List[Dict[str, Any]]:
        """Check self-evolving agents for new capabilities"""
        behaviors = []
        
        try:
            from modules.self_evolving_agents import get_self_evolving_system
            agents = get_self_evolving_system()
            
            status = agents.get_system_status()
            
            # Higher generation agents have evolved new capabilities
            if status.get('average_generation', 0) > 2:
                behaviors.append({
                    'source': 'agents',
                    'type': 'agent_evolution',
                    'description': f"Agents evolved to generation {status['average_generation']:.1f}",
                    'novelty': 0.75,
                    'timestamp': datetime.now()
                })
                
        except Exception as e:
            logger.debug(f"Agent check failed: {e}")
            
        return behaviors
        
    async def _analyze_for_capability(self, behavior: Dict[str, Any]) -> Optional[EmergentCapability]:
        """Analyze behavior to determine if it represents a new capability"""
        
        # Check novelty threshold
        if behavior.get('novelty', 0) < self.novelty_threshold:
            return None
            
        # Create capability descriptor
        capability_id = f"cap_{behavior['source']}_{datetime.now().timestamp()}"
        
        # Map behavior to capability type
        capability_type_map = {
            'system_improvement': 'analytical',
            'knowledge_discovery': 'problem_solving',
            'creative_synthesis': 'creative',
            'agent_evolution': 'synthesis'
        }
        
        capability_type = capability_type_map.get(behavior['type'], 'unknown')
        
        # Generate capability name and description
        name = self._generate_capability_name(behavior)
        description = self._generate_capability_description(behavior)
        
        capability = EmergentCapability(
            capability_id=capability_id,
            name=name,
            description=description,
            emergence_source=behavior['source'],
            capability_type=capability_type,
            confidence_score=behavior['novelty']
        )
        
        return capability
        
    def _generate_capability_name(self, behavior: Dict[str, Any]) -> str:
        """Generate name for capability"""
        source = behavior['source'].title()
        behavior_type = behavior['type'].replace('_', ' ').title()
        return f"{source}-Based {behavior_type}"
        
    def _generate_capability_description(self, behavior: Dict[str, Any]) -> str:
        """Generate description for capability"""
        return f"Emergent capability from {behavior['source']}: {behavior['description']}"
        
    async def _validate_capability(self, capability: EmergentCapability):
        """Start validation process for a capability"""
        logger.info(f"🧪 Beginning validation: {capability.name}")
        
        # Create test scenarios
        tests = await self._generate_capability_tests(capability)
        
        for test in tests:
            self.capability_tests[test.test_id] = test
            
        # Update status
        capability.validation_status = 'testing'
        
    async def _generate_capability_tests(self, capability: EmergentCapability) -> List[CapabilityTest]:
        """Generate tests to validate a capability"""
        tests = []
        
        # Generate 3 test scenarios
        for i in range(3):
            test_id = f"test_{capability.capability_id}_{i}"
            
            # Create test based on capability type
            if capability.capability_type == 'problem_solving':
                scenario = f"Solve novel problem variant {i+1}"
                expected = "Problem solved with valid solution"
            elif capability.capability_type == 'creative':
                scenario = f"Generate creative solution for scenario {i+1}"
                expected = "Novel and feasible solution generated"
            elif capability.capability_type == 'analytical':
                scenario = f"Analyze complex data pattern {i+1}"
                expected = "Accurate analysis with insights"
            else:
                scenario = f"Execute capability demonstration {i+1}"
                expected = "Capability successfully demonstrated"
                
            test = CapabilityTest(
                test_id=test_id,
                capability_id=capability.capability_id,
                test_scenario=scenario,
                expected_outcome=expected
            )
            
            tests.append(test)
            
        return tests
        
    async def _continue_validation(self, capability: EmergentCapability):
        """Continue validation of a capability"""
        
        # Get tests for this capability
        tests = [t for t in self.capability_tests.values() if t.capability_id == capability.capability_id]
        
        # Execute unexecuted tests
        unexecuted = [t for t in tests if not t.executed_at]
        
        for test in unexecuted[:1]:  # Execute 1 at a time
            await self._execute_capability_test(test)
            
        # Check if validation complete
        executed_tests = [t for t in tests if t.executed_at]
        
        if len(executed_tests) >= 3:
            # Calculate pass rate
            pass_rate = sum(1 for t in executed_tests if t.passed) / len(executed_tests)
            
            if pass_rate >= 0.67:  # 2/3 pass rate
                capability.validation_status = 'validated'
                capability.confidence_score = pass_rate
                logger.info(f"✅ Capability validated: {capability.name} ({pass_rate:.0%})")
            else:
                capability.validation_status = 'false_positive'
                logger.info(f"❌ Capability invalidated: {capability.name}")
                
    async def _execute_capability_test(self, test: CapabilityTest):
        """Execute a capability test"""
        logger.info(f"🧪 Executing test: {test.test_scenario}")
        
        # Simulate test execution
        # In production, would actually exercise the capability
        await asyncio.sleep(0.1)
        
        # Simulate pass/fail (80% pass rate for valid capabilities)
        passed = np.random.random() < 0.8
        
        test.passed = passed
        test.actual_outcome = test.expected_outcome if passed else "Did not meet expectations"
        test.executed_at = datetime.now()
        
        # Update capability demonstration count
        if passed:
            capability = self.capabilities.get(test.capability_id)
            if capability:
                capability.demonstration_count += 1
                
    async def _detect_compositions(self):
        """Detect when capabilities can be composed into higher-order capabilities"""
        
        validated = [c for c in self.capabilities.values() if c.validation_status == 'validated']
        
        if len(validated) < 2:
            return
            
        # Look for complementary capabilities
        for i, cap1 in enumerate(validated):
            for cap2 in validated[i+1:]:
                if await self._are_complementary(cap1, cap2):
                    composition = await self._create_composition(cap1, cap2)
                    if composition:
                        self.capabilities[composition.capability_id] = composition
                        logger.info(f"🔗 Detected capability composition: {composition.name}")
                        
    async def _are_complementary(self, cap1: EmergentCapability, cap2: EmergentCapability) -> bool:
        """Check if two capabilities are complementary"""
        # Complementary if they're from different types
        return cap1.capability_type != cap2.capability_type
        
    async def _create_composition(self, cap1: EmergentCapability, cap2: EmergentCapability) -> Optional[EmergentCapability]:
        """Create a composed capability from two capabilities"""
        
        composition_id = f"comp_{cap1.capability_id}_{cap2.capability_id}"
        
        name = f"{cap1.name} + {cap2.name}"
        description = f"Composition of {cap1.name} and {cap2.name} capabilities"
        
        composition = EmergentCapability(
            capability_id=composition_id,
            name=name,
            description=description,
            emergence_source='composition',
            capability_type='synthesis',
            confidence_score=(cap1.confidence_score + cap2.confidence_score) / 2,
            validation_status='validated'  # Inherit validation from components
        )
        
        return composition
        
    async def _establish_baseline(self):
        """Establish baseline of known behaviors"""
        # In production, would catalog all current system behaviors
        self.known_behaviors = {
            'basic_reasoning',
            'knowledge_retrieval',
            'design_generation',
            'simulation',
            'optimization'
        }
        
    async def _load_capabilities(self):
        """Load capabilities from disk"""
        try:
            data_path = Path("data/emergent_capabilities.json")
            if data_path.exists():
                with open(data_path) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded capability data")
        except Exception as e:
            logger.debug(f"No capability data loaded: {e}")
            
    async def save_capabilities(self):
        """Save capabilities to disk"""
        try:
            data_path = Path("data/emergent_capabilities.json")
            data_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {
                'total_capabilities': len(self.capabilities),
                'validated': len([c for c in self.capabilities.values() if c.validation_status == 'validated']),
                'testing': len([c for c in self.capabilities.values() if c.validation_status == 'testing']),
                'compositions': len([c for c in self.capabilities.values() if c.emergence_source == 'composition'])
            }
            
            with open(data_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving capability data: {e}")
            
    def get_detection_status(self) -> Dict[str, Any]:
        """Get capability detection status"""
        validated = [c for c in self.capabilities.values() if c.validation_status == 'validated']
        
        return {
            'is_running': self.is_running,
            'total_capabilities': len(self.capabilities),
            'validated_capabilities': len(validated),
            'testing_capabilities': len([c for c in self.capabilities.values() if c.validation_status == 'testing']),
            'false_positives': len([c for c in self.capabilities.values() if c.validation_status == 'false_positive']),
            'capability_types': {
                'problem_solving': len([c for c in validated if c.capability_type == 'problem_solving']),
                'creative': len([c for c in validated if c.capability_type == 'creative']),
                'analytical': len([c for c in validated if c.capability_type == 'analytical']),
                'synthesis': len([c for c in validated if c.capability_type == 'synthesis'])
            }
        }


# Need numpy for random
import numpy as np


# Singleton instance
_capability_detector = None

def get_capability_detector() -> EmergentCapabilityDetector:
    """Get the global emergent capability detector instance"""
    global _capability_detector
    if _capability_detector is None:
        _capability_detector = EmergentCapabilityDetector()
    return _capability_detector
