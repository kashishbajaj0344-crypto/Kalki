"""
Multi-Agent Consensus System
Validates decisions through 3-agent voting with majority consensus
"""

import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.MultiAgentConsensus")


@dataclass
class ConsensusResult:
    """Result of multi-agent consensus"""
    decision: str  # "approved", "rejected", "requires_modification"
    confidence: float  # 0.0-1.0
    votes: Dict[str, Dict[str, Any]]  # agent_name -> vote details
    reasoning: List[str]  # Combined reasoning from all agents
    timestamp: datetime
    unanimous: bool


class MultiAgentConsensusSystem:
    """
    Validates critical decisions through 3-agent voting.
    Ensures quality through diverse perspectives and majority consensus.
    """
    
    def __init__(self, llm_engine):
        """
        Initialize multi-agent consensus system
        
        Args:
            llm_engine: LLM engine for agent reasoning
        """
        self.llm = llm_engine
        self.consensus_history = []
        
        # Define 3 specialized agents with different perspectives
        self.agents = {
            "feasibility_agent": {
                "role": "Technical Feasibility Validator",
                "focus": "practicality, cost, timeline, complexity",
                "bias": "conservative - prefers proven approaches"
            },
            "quality_agent": {
                "role": "Quality and Safety Validator", 
                "focus": "code quality, building codes, safety, compliance",
                "bias": "strict - prioritizes safety and standards"
            },
            "innovation_agent": {
                "role": "Innovation and Efficiency Validator",
                "focus": "efficiency, modern techniques, optimization, user experience",
                "bias": "progressive - explores better alternatives"
            }
        }
        
        logger.info("Multi-Agent Consensus System initialized with 3 specialized agents")
    
    async def validate_decision(
        self, 
        decision: str,
        context: Dict[str, Any],
        require_unanimous: bool = False
    ) -> ConsensusResult:
        """
        Validate decision through multi-agent consensus
        
        Args:
            decision: Decision to validate
            context: Context information (project details, constraints, etc.)
            require_unanimous: Whether all 3 agents must agree (default: False, 2/3 majority)
            
        Returns:
            ConsensusResult with decision, confidence, votes, reasoning
        """
        logger.info(f"🗳️  Starting multi-agent validation: {decision[:100]}...")
        
        # Get votes from all 3 agents in parallel
        vote_tasks = [
            self._get_agent_vote(agent_name, agent_config, decision, context)
            for agent_name, agent_config in self.agents.items()
        ]
        
        votes = await asyncio.gather(*vote_tasks)
        
        # Combine votes into structured format
        vote_dict = {
            agent_name: vote 
            for agent_name, vote in zip(self.agents.keys(), votes)
        }
        
        # Calculate consensus
        approvals = sum(1 for v in vote_dict.values() if v['decision'] == 'approve')
        rejections = sum(1 for v in vote_dict.values() if v['decision'] == 'reject')
        modifications = sum(1 for v in vote_dict.values() if v['decision'] == 'modify')
        
        # Determine final decision
        unanimous = (approvals == 3 or rejections == 3)
        
        if require_unanimous and not unanimous:
            final_decision = "requires_modification"
            confidence = 0.5
        elif approvals >= 2:
            final_decision = "approved"
            confidence = approvals / 3.0
        elif rejections >= 2:
            final_decision = "rejected"
            confidence = rejections / 3.0
        else:
            final_decision = "requires_modification"
            confidence = 0.5
        
        # Combine reasoning from all agents
        all_reasoning = []
        for agent_name, vote in vote_dict.items():
            all_reasoning.append(f"{agent_name}: {vote['reasoning']}")
        
        result = ConsensusResult(
            decision=final_decision,
            confidence=confidence,
            votes=vote_dict,
            reasoning=all_reasoning,
            timestamp=datetime.now(),
            unanimous=unanimous
        )
        
        # Store in history
        self.consensus_history.append({
            'decision_input': decision,
            'context': context,
            'result': result,
            'timestamp': result.timestamp
        })
        
        logger.info(f"✅ Consensus reached: {final_decision} (confidence: {confidence:.1%})")
        logger.info(f"   Votes: {approvals} approve, {rejections} reject, {modifications} modify")
        
        return result
    
    async def _get_agent_vote(
        self,
        agent_name: str,
        agent_config: Dict[str, Any],
        decision: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Get vote from a single agent
        
        Returns:
            Dict with 'decision' (approve/reject/modify), 'confidence', 'reasoning', 'concerns'
        """
        prompt = f"""You are the {agent_config['role']} in a 3-agent consensus system.

Your focus areas: {agent_config['focus']}
Your bias: {agent_config['bias']}

DECISION TO VALIDATE:
{decision}

PROJECT CONTEXT:
{self._format_context(context)}

Your task: Evaluate this decision from your specialized perspective.

Respond in this EXACT format:
VOTE: [approve/reject/modify]
CONFIDENCE: [0.0-1.0]
REASONING: [Your detailed reasoning in 2-3 sentences]
CONCERNS: [Any concerns or suggestions, or "None"]
ALTERNATIVES: [Better alternatives if vote is reject/modify, or "None"]
"""
        
        try:
            response = await self.llm.generate(
                prompt=prompt,
                task='construction_reasoning',
                max_tokens=300
            )
            
            # Parse response
            vote = self._parse_vote(response)
            
            logger.debug(f"   {agent_name}: {vote['decision']} (confidence: {vote['confidence']:.1%})")
            
            return vote
            
        except Exception as e:
            logger.error(f"Agent {agent_name} vote failed: {e}")
            # Default to cautious response
            return {
                'decision': 'modify',
                'confidence': 0.5,
                'reasoning': f"Error during evaluation: {str(e)}",
                'concerns': "Unable to complete full evaluation",
                'alternatives': "None"
            }
    
    def _parse_vote(self, response: str) -> Dict[str, Any]:
        """Parse agent vote response"""
        lines = response.strip().split('\n')
        
        vote = {
            'decision': 'modify',  # Default
            'confidence': 0.5,
            'reasoning': '',
            'concerns': '',
            'alternatives': ''
        }
        
        for line in lines:
            line = line.strip()
            if line.startswith('VOTE:'):
                decision = line.replace('VOTE:', '').strip().lower()
                if decision in ['approve', 'reject', 'modify']:
                    vote['decision'] = decision
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf = float(line.replace('CONFIDENCE:', '').strip())
                    vote['confidence'] = max(0.0, min(1.0, conf))
                except:
                    pass
            elif line.startswith('REASONING:'):
                vote['reasoning'] = line.replace('REASONING:', '').strip()
            elif line.startswith('CONCERNS:'):
                vote['concerns'] = line.replace('CONCERNS:', '').strip()
            elif line.startswith('ALTERNATIVES:'):
                vote['alternatives'] = line.replace('ALTERNATIVES:', '').strip()
        
        return vote
    
    def _format_context(self, context: Dict[str, Any]) -> str:
        """Format context dictionary as readable text"""
        lines = []
        for key, value in context.items():
            if isinstance(value, dict):
                lines.append(f"{key}:")
                for k, v in value.items():
                    lines.append(f"  - {k}: {v}")
            elif isinstance(value, list):
                lines.append(f"{key}: {', '.join(map(str, value))}")
            else:
                lines.append(f"{key}: {value}")
        return '\n'.join(lines)
    
    async def get_consensus_explanation(
        self,
        result: ConsensusResult,
        include_dissent: bool = True
    ) -> str:
        """
        Generate human-readable explanation of consensus
        
        Args:
            result: ConsensusResult to explain
            include_dissent: Whether to include dissenting opinions
            
        Returns:
            Formatted explanation string
        """
        explanation = [
            f"**Multi-Agent Consensus: {result.decision.upper()}**",
            f"Confidence: {result.confidence:.1%}",
            f"Unanimous: {'Yes' if result.unanimous else 'No'}",
            "",
            "**Agent Votes:**"
        ]
        
        for agent_name, vote in result.votes.items():
            explanation.append(f"- {agent_name}: {vote['decision']} ({vote['confidence']:.1%})")
        
        if include_dissent:
            explanation.append("\n**Key Reasoning:**")
            for reasoning in result.reasoning:
                explanation.append(f"- {reasoning}")
        
        return '\n'.join(explanation)
    
    def get_consensus_statistics(self) -> Dict[str, Any]:
        """Get statistics about consensus history"""
        if not self.consensus_history:
            return {
                'total_decisions': 0,
                'approval_rate': 0.0,
                'average_confidence': 0.0,
                'unanimous_rate': 0.0
            }
        
        total = len(self.consensus_history)
        approvals = sum(1 for h in self.consensus_history 
                       if h['result'].decision == 'approved')
        unanimous = sum(1 for h in self.consensus_history 
                       if h['result'].unanimous)
        avg_confidence = sum(h['result'].confidence 
                            for h in self.consensus_history) / total
        
        return {
            'total_decisions': total,
            'approval_rate': approvals / total,
            'average_confidence': avg_confidence,
            'unanimous_rate': unanimous / total,
            'by_decision': {
                'approved': sum(1 for h in self.consensus_history 
                               if h['result'].decision == 'approved'),
                'rejected': sum(1 for h in self.consensus_history 
                               if h['result'].decision == 'rejected'),
                'requires_modification': sum(1 for h in self.consensus_history 
                                            if h['result'].decision == 'requires_modification')
            }
        }
    
    async def analyze(
        self,
        decision: str,
        context: Dict[str, Any],
        require_unanimous: bool = False,
        agents: List[str] = None,
        domain: str = None
    ) -> Dict[str, Any]:
        """
        Analyze decision with multi-agent consensus (returns dict format)
        
        Args:
            decision: Decision to analyze
            context: Context information
            require_unanimous: Whether all 3 agents must agree
            agents: Optional list of agent types (ignored, uses built-in 3 agents)
            domain: Optional domain specification (added to context)
            
        Returns:
            Dict with 'agreement', 'individual_analyses', 'conflicts', 'recommendation'
        """
        if domain:
            context = {**context, 'domain': domain}
        
        result = await self.validate_decision(decision, context, require_unanimous)
        
        # Convert ConsensusResult to dict format expected by enhanced copilot
        agreement_score = result.confidence  # Confidence = agreement level
        
        # Extract individual agent analyses
        individual_analyses = []
        conflicts = []
        
        for agent_name, vote in result.votes.items():
            individual_analyses.append({
                'agent': agent_name,
                'decision': vote['decision'],
                'confidence': vote['confidence'],
                'reasoning': vote['reasoning'],
                'concerns': vote.get('concerns', '')
            })
            
            # Track conflicts (rejections or modifications)
            if vote['decision'] in ['reject', 'modify']:
                conflicts.append({
                    'agent': agent_name,
                    'issue': vote['reasoning'],
                    'alternatives': vote.get('alternatives', '')
                })
        
        return {
            'agreement': agreement_score,
            'recommendation': result.decision,  # 'approved', 'rejected', or 'requires_modification'
            'individual_analyses': individual_analyses,
            'conflicts': conflicts,
            'unanimous': result.unanimous,
            'reasoning': result.reasoning
        }
