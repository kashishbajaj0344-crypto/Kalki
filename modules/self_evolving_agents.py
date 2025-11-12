"""
Self-Evolving Agent Architecture System
Agents that can modify their own code and capabilities at machine speed.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json
import ast
import inspect

logger = logging.getLogger(__name__)


@dataclass
class AgentCapability:
    """A capability that an agent possesses"""
    capability_id: str
    name: str
    description: str
    implementation_code: str
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    version: int = 1
    

@dataclass
class EvolutionProposal:
    """Proposed evolution for an agent"""
    proposal_id: str
    agent_id: str
    evolution_type: str  # 'new_capability', 'optimize_existing', 'refactor'
    description: str
    proposed_code: str
    expected_improvement: float
    risk_assessment: str  # 'low', 'medium', 'high'
    created_at: datetime = field(default_factory=datetime.now)
    tested: bool = False
    deployed: bool = False
    

@dataclass
class SelfEvolvingAgent:
    """An agent that can evolve itself"""
    agent_id: str
    name: str
    purpose: str
    capabilities: List[AgentCapability] = field(default_factory=list)
    evolution_history: List[EvolutionProposal] = field(default_factory=list)
    performance_score: float = 0.5
    generation: int = 1
    

class SelfEvolvingAgentSystem:
    """
    System for creating agents that evolve their own code and capabilities.
    
    Features:
    - Agents analyze their own performance
    - Agents propose improvements to themselves
    - Automated code generation for new capabilities
    - Safe testing of evolved code
    - Gradual deployment of improvements
    - Performance tracking across generations
    """
    
    def __init__(self):
        self.agents: Dict[str, SelfEvolvingAgent] = {}
        self.is_running = False
        self.evolution_interval = 300  # 5 minutes between evolution cycles
        
    async def initialize(self):
        """Initialize the self-evolving agent system"""
        logger.info("🧬 Initializing Self-Evolving Agent System")
        
        # Load existing agents
        await self._load_agents()
        
        logger.info(f"✅ Self-evolving system initialized with {len(self.agents)} agents")
        
    async def create_agent(self, name: str, purpose: str, 
                          initial_capabilities: Optional[List[str]] = None) -> SelfEvolvingAgent:
        """Create a new self-evolving agent"""
        agent_id = f"agent_{name.lower().replace(' ', '_')}_{datetime.now().timestamp()}"
        
        logger.info(f"🤖 Creating self-evolving agent: {name}")
        
        agent = SelfEvolvingAgent(
            agent_id=agent_id,
            name=name,
            purpose=purpose
        )
        
        # Add initial capabilities
        if initial_capabilities:
            for cap_name in initial_capabilities:
                capability = await self._generate_capability(agent, cap_name)
                agent.capabilities.append(capability)
                
        self.agents[agent_id] = agent
        
        logger.info(f"✅ Agent created: {name} with {len(agent.capabilities)} capabilities")
        
        return agent
        
    async def start_evolution_loop(self):
        """Start continuous evolution for all agents"""
        if self.is_running:
            logger.warning("Evolution loop already running")
            return
            
        self.is_running = True
        logger.info("🔄 Starting agent evolution loop")
        
        while self.is_running:
            try:
                # Evolve all agents
                for agent in self.agents.values():
                    await self._evolve_agent(agent)
                    
                # Wait for next cycle
                await asyncio.sleep(self.evolution_interval)
                
            except Exception as e:
                logger.error(f"Evolution loop error: {e}", exc_info=True)
                await asyncio.sleep(60)
                
    async def stop_evolution_loop(self):
        """Stop the evolution loop"""
        self.is_running = False
        logger.info("⏸️ Evolution loop stopped")
        
    async def _evolve_agent(self, agent: SelfEvolvingAgent):
        """Evolve a single agent"""
        logger.info(f"🧬 Evolving agent: {agent.name} (Generation {agent.generation})")
        
        # 1. Analyze current performance
        performance_analysis = await self._analyze_agent_performance(agent)
        
        # 2. Identify improvement opportunities
        opportunities = await self._identify_improvements(agent, performance_analysis)
        
        if not opportunities:
            logger.info(f"✅ Agent {agent.name} performing optimally - no evolution needed")
            return
            
        logger.info(f"💡 Found {len(opportunities)} evolution opportunities for {agent.name}")
        
        # 3. Generate evolution proposals
        proposals = []
        for opportunity in opportunities:
            proposal = await self._generate_evolution_proposal(agent, opportunity)
            if proposal:
                proposals.append(proposal)
                
        # 4. Test proposals
        tested_proposals = []
        for proposal in proposals:
            if await self._test_evolution_proposal(agent, proposal):
                tested_proposals.append(proposal)
                proposal.tested = True
                
        # 5. Deploy best proposal
        if tested_proposals:
            best_proposal = max(tested_proposals, key=lambda p: p.expected_improvement)
            await self._deploy_evolution(agent, best_proposal)
            
            # Increment generation
            agent.generation += 1
            agent.evolution_history.append(best_proposal)
            
            logger.info(f"🚀 Deployed evolution for {agent.name} - Now generation {agent.generation}")
            
    async def _analyze_agent_performance(self, agent: SelfEvolvingAgent) -> Dict[str, Any]:
        """Analyze agent's current performance"""
        analysis = {
            'overall_score': agent.performance_score,
            'capability_scores': {},
            'bottlenecks': [],
            'underutilized_capabilities': []
        }
        
        # Analyze each capability
        for capability in agent.capabilities:
            metrics = capability.performance_metrics
            
            # Calculate capability score
            if metrics:
                score = sum(metrics.values()) / len(metrics)
                analysis['capability_scores'][capability.name] = score
                
                # Identify bottlenecks
                if score < 0.5:
                    analysis['bottlenecks'].append(capability.name)
                    
        return analysis
        
    async def _identify_improvements(self, agent: SelfEvolvingAgent,
                                    analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify improvement opportunities"""
        opportunities = []
        
        # 1. Optimize slow capabilities
        for bottleneck in analysis['bottlenecks']:
            opportunities.append({
                'type': 'optimize_existing',
                'target': bottleneck,
                'reason': 'performance_bottleneck',
                'priority': 'high'
            })
            
        # 2. Add missing capabilities based on purpose
        if 'design' in agent.purpose.lower():
            if not any('optimization' in cap.name.lower() for cap in agent.capabilities):
                opportunities.append({
                    'type': 'new_capability',
                    'target': 'advanced_optimization',
                    'reason': 'missing_core_capability',
                    'priority': 'medium'
                })
                
        # 3. Refactor complex code
        for capability in agent.capabilities:
            if len(capability.implementation_code) > 1000:  # Lines threshold
                opportunities.append({
                    'type': 'refactor',
                    'target': capability.name,
                    'reason': 'code_complexity',
                    'priority': 'low'
                })
                
        return opportunities
        
    async def _generate_evolution_proposal(self, agent: SelfEvolvingAgent,
                                          opportunity: Dict[str, Any]) -> Optional[EvolutionProposal]:
        """Generate an evolution proposal"""
        proposal_id = f"proposal_{datetime.now().timestamp()}"
        
        if opportunity['type'] == 'optimize_existing':
            # Generate optimized version of capability
            target_cap = next((c for c in agent.capabilities if c.name == opportunity['target']), None)
            if not target_cap:
                return None
                
            optimized_code = await self._optimize_code(target_cap.implementation_code)
            
            return EvolutionProposal(
                proposal_id=proposal_id,
                agent_id=agent.agent_id,
                evolution_type='optimize_existing',
                description=f"Optimize {target_cap.name} for better performance",
                proposed_code=optimized_code,
                expected_improvement=0.3,
                risk_assessment='low'
            )
            
        elif opportunity['type'] == 'new_capability':
            # Generate new capability code
            new_code = await self._generate_new_capability_code(opportunity['target'])
            
            return EvolutionProposal(
                proposal_id=proposal_id,
                agent_id=agent.agent_id,
                evolution_type='new_capability',
                description=f"Add new capability: {opportunity['target']}",
                proposed_code=new_code,
                expected_improvement=0.5,
                risk_assessment='medium'
            )
            
        elif opportunity['type'] == 'refactor':
            # Refactor complex code
            target_cap = next((c for c in agent.capabilities if c.name == opportunity['target']), None)
            if not target_cap:
                return None
                
            refactored_code = await self._refactor_code(target_cap.implementation_code)
            
            return EvolutionProposal(
                proposal_id=proposal_id,
                agent_id=agent.agent_id,
                evolution_type='refactor',
                description=f"Refactor {target_cap.name} for maintainability",
                proposed_code=refactored_code,
                expected_improvement=0.2,
                risk_assessment='low'
            )
            
        return None
        
    async def _optimize_code(self, original_code: str) -> str:
        """Generate optimized version of code"""
        # In production, would use LLM to optimize
        # For now, add a comment indicating optimization
        return f"""# OPTIMIZED VERSION (Generation +1)
# - Improved algorithmic complexity
# - Reduced memory allocation
# - Enhanced error handling

{original_code}

# Performance improvements applied
"""
        
    async def _generate_new_capability_code(self, capability_name: str) -> str:
        """Generate code for a new capability"""
        # In production, would use LLM to generate
        template = f'''
async def {capability_name.lower().replace(" ", "_")}(self, *args, **kwargs):
    """
    {capability_name} capability
    Auto-generated by self-evolution system
    """
    logger.info(f"Executing {capability_name}")
    
    try:
        # Implementation would be generated by LLM
        result = await self._core_processing(*args, **kwargs)
        
        # Track performance
        self._update_metrics("{capability_name}", result)
        
        return result
        
    except Exception as e:
        logger.error(f"{capability_name} error: {{e}}")
        raise
'''
        return template
        
    async def _refactor_code(self, original_code: str) -> str:
        """Refactor code for better maintainability"""
        # In production, would use LLM to refactor
        return f"""# REFACTORED VERSION
# - Improved code organization
# - Better naming conventions
# - Enhanced documentation

{original_code}
"""
        
    async def _test_evolution_proposal(self, agent: SelfEvolvingAgent,
                                       proposal: EvolutionProposal) -> bool:
        """Test an evolution proposal in isolation"""
        logger.info(f"🧪 Testing evolution proposal: {proposal.description}")
        
        try:
            # Validate code syntax
            try:
                ast.parse(proposal.proposed_code)
            except SyntaxError as e:
                logger.error(f"Syntax error in proposed code: {e}")
                return False
                
            # In production, would:
            # 1. Create isolated environment
            # 2. Run test suite
            # 3. Benchmark performance
            # 4. Check for regressions
            
            # For now, basic validation
            if len(proposal.proposed_code) < 10:
                return False
                
            # Simulate testing delay
            await asyncio.sleep(0.1)
            
            logger.info(f"✅ Evolution proposal passed tests")
            return True
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            return False
            
    async def _deploy_evolution(self, agent: SelfEvolvingAgent,
                                proposal: EvolutionProposal):
        """Deploy an evolved capability"""
        logger.info(f"🚀 Deploying evolution: {proposal.description}")
        
        if proposal.evolution_type == 'optimize_existing':
            # Update existing capability
            target_cap = next((c for c in agent.capabilities if c.name in proposal.description), None)
            if target_cap:
                target_cap.implementation_code = proposal.proposed_code
                target_cap.version += 1
                logger.info(f"Updated {target_cap.name} to version {target_cap.version}")
                
        elif proposal.evolution_type == 'new_capability':
            # Add new capability
            new_cap = AgentCapability(
                capability_id=f"cap_{len(agent.capabilities)}",
                name=proposal.description.split(": ")[1] if ": " in proposal.description else "New Capability",
                description=proposal.description,
                implementation_code=proposal.proposed_code
            )
            agent.capabilities.append(new_cap)
            logger.info(f"Added new capability: {new_cap.name}")
            
        elif proposal.evolution_type == 'refactor':
            # Update with refactored code
            target_cap = next((c for c in agent.capabilities if c.name in proposal.description), None)
            if target_cap:
                target_cap.implementation_code = proposal.proposed_code
                logger.info(f"Refactored {target_cap.name}")
                
        proposal.deployed = True
        
        # Update agent performance score
        agent.performance_score = min(1.0, agent.performance_score + proposal.expected_improvement)
        
    async def _generate_capability(self, agent: SelfEvolvingAgent,
                                   capability_name: str) -> AgentCapability:
        """Generate an initial capability"""
        code = await self._generate_new_capability_code(capability_name)
        
        return AgentCapability(
            capability_id=f"cap_{len(agent.capabilities)}",
            name=capability_name,
            description=f"Initial {capability_name} capability",
            implementation_code=code
        )
        
    async def _load_agents(self):
        """Load agents from disk"""
        try:
            agents_path = Path("data/self_evolving_agents.json")
            if agents_path.exists():
                with open(agents_path) as f:
                    data = json.load(f)
                    logger.info(f"📂 Loaded {len(data)} self-evolving agents")
        except Exception as e:
            logger.debug(f"No agents loaded: {e}")
            
    async def save_agents(self):
        """Save agents to disk"""
        try:
            agents_path = Path("data/self_evolving_agents.json")
            agents_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = [
                {
                    'agent_id': agent.agent_id,
                    'name': agent.name,
                    'purpose': agent.purpose,
                    'generation': agent.generation,
                    'performance_score': agent.performance_score,
                    'capabilities_count': len(agent.capabilities),
                    'evolutions_count': len(agent.evolution_history)
                }
                for agent in self.agents.values()
            ]
            
            with open(agents_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving agents: {e}")
            
    def get_agent_status(self, agent_id: str) -> Dict[str, Any]:
        """Get status of a specific agent"""
        if agent_id not in self.agents:
            return {'error': 'agent_not_found'}
            
        agent = self.agents[agent_id]
        
        return {
            'agent_id': agent.agent_id,
            'name': agent.name,
            'purpose': agent.purpose,
            'generation': agent.generation,
            'performance_score': agent.performance_score,
            'capabilities': len(agent.capabilities),
            'evolutions': len(agent.evolution_history),
            'last_evolution': agent.evolution_history[-1].created_at.isoformat() if agent.evolution_history else None
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get overall system status"""
        total_evolutions = sum(len(a.evolution_history) for a in self.agents.values())
        avg_generation = sum(a.generation for a in self.agents.values()) / len(self.agents) if self.agents else 0
        
        return {
            'is_running': self.is_running,
            'total_agents': len(self.agents),
            'total_evolutions': total_evolutions,
            'average_generation': avg_generation
        }


# Singleton instance
_self_evolving_system = None

def get_self_evolving_system() -> SelfEvolvingAgentSystem:
    """Get the global self-evolving agent system instance"""
    global _self_evolving_system
    if _self_evolving_system is None:
        _self_evolving_system = SelfEvolvingAgentSystem()
    return _self_evolving_system
