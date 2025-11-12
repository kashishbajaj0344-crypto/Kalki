"""
KALKI Domain System

Multi-domain expertise architecture. Each domain (construction, game dev, 
robotics, aerospace, etc.) is a pluggable module that provides:
- Domain-specific knowledge extraction
- Project state machines for domain workflows
- Deliverables generation
- Validation and estimation

The core KALKI system (20 phases) remains domain-agnostic.
"""

from .base_domain import BaseDomain, DomainModule
from .domain_registry import DomainRegistry

__all__ = ["BaseDomain", "DomainModule", "DomainRegistry"]
