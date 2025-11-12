"""
Construction Domain Module

Expertise in building design, construction management, and delivery.
Handles residential and commercial construction projects in BC (and beyond).
"""

import sys
from pathlib import Path

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from modules.domains.construction_domain.construction_domain import ConstructionDomain

__all__ = ["ConstructionDomain"]
