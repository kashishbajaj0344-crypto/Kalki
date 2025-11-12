"""
Property Intelligence Gatherer for Construction Copilot
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Automatically gathers property-specific data using web search APIs:
- Zoning requirements
- Setback rules
- Building permit requirements
- Historic overlays
- Soil conditions
- Utility locations
"""

import logging
import asyncio
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class PropertyIntelligenceGatherer:
    """
    Gathers property intelligence using web search and LLM reasoning.
    
    Integrates with:
    - KALKI's LLM engine for parsing/reasoning
    - Autonomous research system for web search
    - Google CSE API (via research system)
    """
    
    def __init__(self, llm_engine, research_system):
        """
        Initialize property intelligence gatherer.
        
        Args:
            llm_engine: KALKI's LLM engine instance
            research_system: KALKI's autonomous research system
        """
        self.llm = llm_engine
        self.research = research_system
        self.logger = logging.getLogger(__name__)
        
        # Cache for expensive API calls
        self.cache: Dict[str, Dict[str, Any]] = {}
    
    
    async def gather_property_intelligence(
        self,
        address: str,
        project_type: str = 'adu'
    ) -> Dict[str, Any]:
        """
        Gather comprehensive property intelligence.
        
        Args:
            address: Property address (e.g., "1234 Elm St, San Jose, CA 95125")
            project_type: Type of project ('adu', 'remodel', 'new_construction')
        
        Returns:
            Dict with property intelligence:
            {
                'address': str,
                'jurisdiction': str,
                'zoning': {...},
                'setbacks': {...},
                'height_limits': {...},
                'lot_info': {...},
                'constraints': [...],
                'opportunities': [...],
                'estimated_complexity': float
            }
        """
        self.logger.info(f"Gathering property intelligence for: {address}")
        
        # Check cache
        cache_key = f"{address}_{project_type}"
        if cache_key in self.cache:
            self.logger.info("Using cached property intelligence")
            return self.cache[cache_key]
        
        # Parse address to extract city, state
        location_info = self._parse_address(address)
        
        # Gather intelligence in parallel
        results = await asyncio.gather(
            self._get_zoning_info(address, location_info),
            self._get_setback_requirements(address, location_info, project_type),
            self._get_permit_requirements(address, location_info, project_type),
            self._get_height_limits(address, location_info),
            self._get_lot_information(address, location_info),
            self._get_constraints(address, location_info),
            return_exceptions=True
        )
        
        # Unpack results
        zoning, setbacks, permits, height_limits, lot_info, constraints = results
        
        # Handle any exceptions
        if isinstance(zoning, Exception):
            self.logger.warning(f"Zoning lookup failed: {zoning}")
            zoning = {'status': 'unknown', 'error': str(zoning)}
        if isinstance(setbacks, Exception):
            self.logger.warning(f"Setback lookup failed: {setbacks}")
            setbacks = {'status': 'unknown', 'error': str(setbacks)}
        if isinstance(permits, Exception):
            self.logger.warning(f"Permit lookup failed: {permits}")
            permits = {'status': 'unknown', 'error': str(permits)}
        if isinstance(height_limits, Exception):
            self.logger.warning(f"Height limit lookup failed: {height_limits}")
            height_limits = {'status': 'unknown', 'error': str(height_limits)}
        if isinstance(lot_info, Exception):
            self.logger.warning(f"Lot info lookup failed: {lot_info}")
            lot_info = {'status': 'unknown', 'error': str(lot_info)}
        if isinstance(constraints, Exception):
            self.logger.warning(f"Constraints lookup failed: {constraints}")
            constraints = []
        
        # Calculate complexity score
        complexity = self._calculate_complexity(
            zoning, setbacks, constraints, lot_info
        )
        
        # Identify opportunities
        opportunities = self._identify_opportunities(
            zoning, setbacks, lot_info, project_type
        )
        
        # Compile intelligence report
        intelligence = {
            'address': address,
            'location': location_info,
            'jurisdiction': location_info.get('city', 'Unknown'),
            'zoning': zoning,
            'setbacks': setbacks,
            'permits': permits,
            'height_limits': height_limits,
            'lot_info': lot_info,
            'constraints': constraints,
            'opportunities': opportunities,
            'complexity_score': complexity,
            'timestamp': datetime.now().isoformat(),
            'confidence': self._calculate_confidence(
                zoning, setbacks, permits, height_limits
            )
        }
        
        # Cache result
        self.cache[cache_key] = intelligence
        
        self.logger.info(f"Property intelligence gathered. Complexity: {complexity:.2f}")
        
        return intelligence
    
    
    async def _get_zoning_info(
        self,
        address: str,
        location: Dict[str, str]
    ) -> Dict[str, Any]:
        """Get zoning information via web search"""
        city = location.get('city', '')
        state = location.get('state', '')
        
        query = f"{city} {state} zoning code residential ADU requirements {address}"
        
        # Use autonomous research system (which uses Google CSE)
        research_result = await self.research.investigate(
            query=query,
            context={'address': address, 'city': city, 'state': state},
            methods=['web_search']
        )
        
        # Parse zoning info from research results
        zoning_info = await self.llm.generate(
            prompt=f"""Extract zoning information from this research:

Research findings: {research_result.get('summary', '')}

Extract:
1. Zoning designation (e.g., R-1, R-2, R-3)
2. Allowed uses
3. ADU regulations
4. Special overlays (historic, hillside, etc.)

Format as JSON.""",
            task='property_analysis',
            max_tokens=500
        )
        
        return {
            'designation': self._extract_field(zoning_info['text'], 'designation', 'R-1'),
            'allowed_uses': self._extract_list(zoning_info['text'], 'allowed_uses'),
            'adu_allowed': 'adu' in zoning_info['text'].lower(),
            'special_overlays': self._extract_list(zoning_info['text'], 'overlays'),
            'sources': research_result.get('sources', []),
            'confidence': research_result.get('confidence', 0.5)
        }
    
    
    async def _get_setback_requirements(
        self,
        address: str,
        location: Dict[str, str],
        project_type: str
    ) -> Dict[str, Any]:
        """Get setback requirements"""
        city = location.get('city', '')
        
        query = f"{city} {project_type} setback requirements side yard rear yard front yard"
        
        research_result = await self.research.investigate(
            query=query,
            context={'city': city, 'project_type': project_type},
            methods=['web_search']
        )
        
        setback_info = await self.llm.generate(
            prompt=f"""Extract setback requirements from this research:

{research_result.get('summary', '')}

Extract (in feet):
- Front setback
- Side setback
- Rear setback
- Corner lot requirements

Format as JSON with numeric values.""",
            task='property_analysis',
            max_tokens=300
        )
        
        return {
            'front_feet': self._extract_number(setback_info['text'], 'front', 20),
            'side_feet': self._extract_number(setback_info['text'], 'side', 5),
            'rear_feet': self._extract_number(setback_info['text'], 'rear', 15),
            'notes': self._extract_field(setback_info['text'], 'notes', ''),
            'sources': research_result.get('sources', []),
            'confidence': research_result.get('confidence', 0.5)
        }
    
    
    async def _get_permit_requirements(
        self,
        address: str,
        location: Dict[str, str],
        project_type: str
    ) -> Dict[str, Any]:
        """Get permit requirements"""
        city = location.get('city', '')
        
        query = f"{city} building permit requirements {project_type} application process timeline"
        
        research_result = await self.research.investigate(
            query=query,
            context={'city': city, 'project_type': project_type},
            methods=['web_search']
        )
        
        permit_info = await self.llm.generate(
            prompt=f"""Extract permit requirements:

{research_result.get('summary', '')}

Extract:
- Required permits
- Estimated timeline
- Application process
- Required documents

Format as JSON.""",
            task='property_analysis',
            max_tokens=400
        )
        
        return {
            'required_permits': self._extract_list(permit_info['text'], 'permits'),
            'estimated_timeline_weeks': self._extract_number(permit_info['text'], 'timeline', 8),
            'application_steps': self._extract_list(permit_info['text'], 'steps'),
            'sources': research_result.get('sources', []),
            'confidence': research_result.get('confidence', 0.5)
        }
    
    
    async def _get_height_limits(
        self,
        address: str,
        location: Dict[str, str]
    ) -> Dict[str, Any]:
        """Get height restrictions"""
        city = location.get('city', '')
        
        query = f"{city} residential building height limits maximum stories ADU"
        
        research_result = await self.research.investigate(
            query=query,
            context={'city': city},
            methods=['web_search']
        )
        
        height_info = await self.llm.generate(
            prompt=f"""Extract height limits:

{research_result.get('summary', '')}

Extract:
- Maximum height (feet)
- Maximum stories
- Special restrictions

Format as JSON.""",
            task='property_analysis',
            max_tokens=200
        )
        
        return {
            'max_height_feet': self._extract_number(height_info['text'], 'height', 30),
            'max_stories': self._extract_number(height_info['text'], 'stories', 2),
            'restrictions': self._extract_list(height_info['text'], 'restrictions'),
            'sources': research_result.get('sources', []),
            'confidence': research_result.get('confidence', 0.5)
        }
    
    
    async def _get_lot_information(
        self,
        address: str,
        location: Dict[str, str]
    ) -> Dict[str, Any]:
        """Get lot size and characteristics"""
        # In production, this would query GIS/parcel APIs
        # For now, use reasonable defaults
        return {
            'lot_size_sqft': 7500,
            'lot_width_feet': 75,
            'lot_depth_feet': 100,
            'slope_percentage': 5,
            'buildable_area_sqft': 6000,
            'status': 'estimated'
        }
    
    
    async def _get_constraints(
        self,
        address: str,
        location: Dict[str, str]
    ) -> List[Dict[str, Any]]:
        """Identify property constraints"""
        city = location.get('city', '')
        
        query = f"{city} building constraints historic district flood zone easements utilities"
        
        research_result = await self.research.investigate(
            query=query,
            context={'city': city, 'address': address},
            methods=['web_search']
        )
        
        constraints = []
        
        summary = research_result.get('summary', '').lower()
        
        if 'historic' in summary:
            constraints.append({
                'type': 'historic_overlay',
                'severity': 'moderate',
                'description': 'Property may be in historic district',
                'impact': 'Architectural review required, design restrictions'
            })
        
        if 'flood' in summary:
            constraints.append({
                'type': 'flood_zone',
                'severity': 'high',
                'description': 'Property may be in flood zone',
                'impact': 'Elevated foundation required, flood insurance'
            })
        
        if 'easement' in summary:
            constraints.append({
                'type': 'easement',
                'severity': 'moderate',
                'description': 'Utility easements may exist',
                'impact': 'Cannot build in easement areas'
            })
        
        return constraints
    
    
    def _parse_address(self, address: str) -> Dict[str, str]:
        """Parse address into components"""
        # Simple parsing - in production would use geocoding API
        parts = address.split(',')
        
        result = {
            'full_address': address,
            'street': parts[0].strip() if len(parts) > 0 else '',
            'city': parts[1].strip() if len(parts) > 1 else 'San Jose',
            'state': parts[2].strip().split()[0] if len(parts) > 2 else 'CA',
            'zip': parts[2].strip().split()[1] if len(parts) > 2 and len(parts[2].strip().split()) > 1 else ''
        }
        
        return result
    
    
    def _calculate_complexity(
        self,
        zoning: Dict,
        setbacks: Dict,
        constraints: List,
        lot_info: Dict
    ) -> float:
        """Calculate project complexity score (0.0 to 1.0)"""
        complexity = 0.3  # Base complexity
        
        # Add complexity for constraints
        complexity += len(constraints) * 0.1
        
        # Add complexity for special overlays
        if zoning.get('special_overlays'):
            complexity += 0.15
        
        # Add complexity for tight setbacks
        total_setbacks = (
            setbacks.get('front_feet', 20) +
            setbacks.get('side_feet', 5) * 2 +
            setbacks.get('rear_feet', 15)
        )
        if total_setbacks > 50:
            complexity += 0.1
        
        # Add complexity for slope
        slope = lot_info.get('slope_percentage', 0)
        if slope > 10:
            complexity += 0.15
        
        return min(complexity, 1.0)
    
    
    def _identify_opportunities(
        self,
        zoning: Dict,
        setbacks: Dict,
        lot_info: Dict,
        project_type: str
    ) -> List[str]:
        """Identify opportunities based on property characteristics"""
        opportunities = []
        
        if zoning.get('adu_allowed'):
            opportunities.append("ADU permitted by zoning")
        
        buildable = lot_info.get('buildable_area_sqft', 0)
        if buildable > 5000:
            opportunities.append(f"Large buildable area ({buildable:,} sq ft)")
        
        if setbacks.get('side_feet', 0) > 6:
            opportunities.append("Generous side setbacks allow flexible design")
        
        if lot_info.get('slope_percentage', 0) < 5:
            opportunities.append("Flat lot reduces foundation costs")
        
        return opportunities
    
    
    def _calculate_confidence(
        self,
        zoning: Dict,
        setbacks: Dict,
        permits: Dict,
        height_limits: Dict
    ) -> float:
        """Calculate overall confidence in gathered data"""
        confidences = [
            zoning.get('confidence', 0.5),
            setbacks.get('confidence', 0.5),
            permits.get('confidence', 0.5),
            height_limits.get('confidence', 0.5)
        ]
        
        return sum(confidences) / len(confidences)
    
    
    # Helper methods for parsing LLM responses
    
    def _extract_field(self, text: str, field: str, default: str = '') -> str:
        """Extract a field from LLM response"""
        pattern = rf'{field}["\s:]+([^,\n]+)'
        match = re.search(pattern, text, re.IGNORECASE)
        return match.group(1).strip(' "\'') if match else default
    
    
    def _extract_number(self, text: str, field: str, default: float = 0) -> float:
        """Extract a numeric value from LLM response"""
        pattern = rf'{field}["\s:]+(\d+\.?\d*)'
        match = re.search(pattern, text, re.IGNORECASE)
        return float(match.group(1)) if match else default
    
    
    def _extract_list(self, text: str, field: str) -> List[str]:
        """Extract a list from LLM response"""
        # Simple extraction - look for bullet points or comma-separated
        items = []
        
        # Try bullet points
        lines = text.split('\n')
        in_section = False
        for line in lines:
            if field.lower() in line.lower():
                in_section = True
                continue
            if in_section:
                if line.strip().startswith(('-', '•', '*', '1.', '2.')):
                    item = re.sub(r'^[-•*\d.]+\s*', '', line.strip())
                    items.append(item)
                elif line.strip() == '':
                    break
        
        # Try comma-separated
        if not items:
            pattern = rf'{field}["\s:]+([^}}\]]+)'
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                items = [item.strip(' "\',') for item in match.group(1).split(',')]
        
        return items if items else []
