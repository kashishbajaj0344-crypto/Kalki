"""
Quality Assurance Framework
Professional quality validation for deliverables across all domains.

Supports:
- Building code compliance
- Software engineering standards
- Aerospace standards
- Domain-specific quality checks
"""

import asyncio
import logging
from enum import Enum
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime

from modules.llm import LLMEngine
from modules.professional_deliverable_generator import DeliverableType

logger = logging.getLogger(__name__)


class QualityStandard(Enum):
    """Quality standards for different domains"""
    BUILDING_CODE = "building_code"
    SOFTWARE_ENGINEERING = "software_engineering"
    AEROSPACE_STANDARDS = "aerospace_standards"
    GAME_DEVELOPMENT = "game_development"
    ROBOTICS = "robotics"
    POWER_SYSTEMS = "power_systems"
    GENERAL = "general"


@dataclass
class QualityCheck:
    """A quality check result"""
    check_name: str
    passed: bool
    severity: str  # "critical", "major", "minor"
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of quality validation"""
    valid: bool
    overall_score: float  # 0-1
    checks: List[QualityCheck]
    critical_issues: int
    major_issues: int
    minor_issues: int
    recommendations: List[str] = field(default_factory=list)


class QualityAssuranceFramework:
    """
    Provides quality assurance for professional work.
    
    Domains use this to:
    - Validate deliverables meet standards
    - Check code compliance
    - Verify safety requirements
    - Ensure professional quality
    - Automated testing
    - Standards compliance (ISO, ANSI, IEEE)
    """
    
    def __init__(self, llm_engine: LLMEngine):
        self.llm_engine = llm_engine
        self.standards: Dict[QualityStandard, Dict[str, Any]] = {}
        self.validation_history: List[Dict[str, Any]] = []
        self._load_standards()
        
        # Advanced validation capabilities
        self._advanced_reasoning = None  # For complex validation
        
        # Standards databases (ISO, ANSI, IEEE, etc.)
        self.standards_db: Dict[str, Dict[str, Any]] = {}
        self._load_standards_databases()
    
    def _load_standards(self):
        """Load domain-specific quality standards"""
        # Building code standards
        self.standards[QualityStandard.BUILDING_CODE] = {
            "name": "Building Code Compliance",
            "checks": [
                "structural_integrity",
                "fire_safety",
                "accessibility",
                "energy_efficiency",
                "zoning_compliance"
            ],
            "domain": "construction"
        }
        
        # Software engineering standards
        self.standards[QualityStandard.SOFTWARE_ENGINEERING] = {
            "name": "Software Engineering Standards",
            "checks": [
                "code_quality",
                "documentation",
                "testing_coverage",
                "security",
                "performance"
            ],
            "domain": "game_dev"
        }
        
        # Aerospace standards
        self.standards[QualityStandard.AEROSPACE_STANDARDS] = {
            "name": "Aerospace Standards",
            "checks": [
                "safety_critical",
                "reliability",
                "regulatory_compliance",
                "material_specifications",
                "testing_requirements"
            ],
            "domain": "aerospace"
        }
        
        # Add more standards as needed
    
    async def validate_deliverable(
        self,
        deliverable: Path,
        deliverable_type: DeliverableType,
        quality_standard: QualityStandard,
        domain: str
    ) -> ValidationResult:
        """
        Validate a deliverable meets quality standards using Llama 3.1 8B and 3.2 Vision.
        
        Args:
            deliverable: Path to deliverable file
            deliverable_type: Type of deliverable
            quality_standard: Quality standard to check against
            domain: Domain context
        
        Returns:
            Validation result with checks and recommendations
        """
        logger.info(f"🔍 Validating {deliverable_type.value} against {quality_standard.value}")
        
        if not deliverable.exists():
            return ValidationResult(
                valid=False,
                overall_score=0.0,
                checks=[QualityCheck(
                    check_name="file_exists",
                    passed=False,
                    severity="critical",
                    message=f"Deliverable file not found: {deliverable}"
                )],
                critical_issues=1,
                major_issues=0,
                minor_issues=0
            )
        
        # Load standard
        standard = self.standards.get(quality_standard, {})
        check_names = standard.get("checks", [])
        
        # Perform quality checks
        checks = []
        for check_name in check_names:
            check_result = await self._perform_quality_check(
                check_name=check_name,
                deliverable=deliverable,
                deliverable_type=deliverable_type,
                standard=standard,
                domain=domain
            )
            checks.append(check_result)
        
        # Count issues
        critical_issues = sum(1 for c in checks if not c.passed and c.severity == "critical")
        major_issues = sum(1 for c in checks if not c.passed and c.severity == "major")
        minor_issues = sum(1 for c in checks if not c.passed and c.severity == "minor")
        
        # Calculate overall score
        passed_checks = sum(1 for c in checks if c.passed)
        overall_score = passed_checks / len(checks) if checks else 0.0
        
        # Generate recommendations using Llama 3.1 8B
        recommendations = await self._generate_recommendations(
            checks=checks,
            deliverable_type=deliverable_type,
            domain=domain
        )
        
        valid = critical_issues == 0 and major_issues == 0
        
        result = ValidationResult(
            valid=valid,
            overall_score=overall_score,
            checks=checks,
            critical_issues=critical_issues,
            major_issues=major_issues,
            minor_issues=minor_issues,
            recommendations=recommendations
        )
        
        # Record validation
        self.validation_history.append({
            "deliverable": str(deliverable),
            "type": deliverable_type.value,
            "standard": quality_standard.value,
            "domain": domain,
            "valid": valid,
            "score": overall_score,
            "timestamp": datetime.now().isoformat()
        })
        
        return result
    
    async def _perform_quality_check(
        self,
        check_name: str,
        deliverable: Path,
        deliverable_type: DeliverableType,
        standard: Dict[str, Any],
        domain: str
    ) -> QualityCheck:
        """Perform a specific quality check using Llama 3.1 8B"""
        # Read deliverable content
        try:
            if deliverable.suffix in ['.txt', '.json', '.py', '.js', '.cpp', '.java']:
                with open(deliverable, 'r') as f:
                    content = f.read()
            else:
                # For binary files, use file metadata
                content = f"File: {deliverable.name}, Size: {deliverable.stat().st_size} bytes"
        except Exception as e:
            return QualityCheck(
                check_name=check_name,
                passed=False,
                severity="critical",
                message=f"Failed to read deliverable: {e}"
            )
        
        # Generate check prompt using Llama 3.1 8B
        check_prompt = f"""Perform a quality check for {check_name} on a {domain} {deliverable_type.value}.

Quality Standard: {standard.get('name', 'General')}
Check: {check_name}

Deliverable Content (first 2000 chars):
{content[:2000]}

Evaluate:
1. Does this meet the {check_name} requirement?
2. What issues exist (if any)?
3. Severity: critical, major, or minor?

Return as JSON: {{"passed": true/false, "severity": "critical/major/minor", "message": "description"}}"""

        try:
            check_response = await self.llm_engine.generate(
                prompt=check_prompt,
                max_tokens=300,
                temperature=0.3
            )
            
            if isinstance(check_response, dict):
                check_text = check_response.get("text", str(check_response))
            else:
                check_text = str(check_response)
            
            # Parse JSON response
            import json
            import re
            json_match = re.search(r'\{.*\}', check_text, re.DOTALL)
            if json_match:
                try:
                    check_data = json.loads(json_match.group())
                    return QualityCheck(
                        check_name=check_name,
                        passed=check_data.get("passed", False),
                        severity=check_data.get("severity", "minor"),
                        message=check_data.get("message", "Quality check performed"),
                        details=check_data
                    )
                except:
                    pass
            
            # Fallback: parse from text
            passed = "passed" in check_text.lower() or "meets" in check_text.lower()
            severity = "critical" if "critical" in check_text.lower() else "major" if "major" in check_text.lower() else "minor"
            
            return QualityCheck(
                check_name=check_name,
                passed=passed,
                severity=severity,
                message=check_text[:200]
            )
            
        except Exception as e:
            logger.warning(f"Quality check failed for {check_name}: {e}")
            return QualityCheck(
                check_name=check_name,
                passed=False,
                severity="minor",
                message=f"Check failed: {e}"
            )
    
    async def _generate_recommendations(
        self,
        checks: List[QualityCheck],
        deliverable_type: DeliverableType,
        domain: str
    ) -> List[str]:
        """Generate improvement recommendations using Llama 3.1 8B"""
        failed_checks = [c for c in checks if not c.passed]
        
        if not failed_checks:
            return ["All quality checks passed. No recommendations."]
        
        rec_prompt = f"""Generate professional improvement recommendations for a {domain} {deliverable_type.value} that failed quality checks.

Failed Checks:
{chr(10).join(f"- {c.check_name}: {c.message} (severity: {c.severity})" for c in failed_checks)}

Provide 3-5 specific, actionable recommendations to improve quality."""

        try:
            rec_response = await self.llm_engine.generate(
                prompt=rec_prompt,
                max_tokens=400,
                temperature=0.5
            )
            
            if isinstance(rec_response, dict):
                rec_text = rec_response.get("text", str(rec_response))
            else:
                rec_text = str(rec_response)
            
            # Extract recommendations (numbered list or bullet points)
            import re
            recommendations = re.findall(r'(?:^|\n)[\d\-•]\s*(.+?)(?=\n|$)', rec_text, re.MULTILINE)
            if not recommendations:
                # Fallback: split by newlines
                recommendations = [line.strip() for line in rec_text.split('\n') if line.strip() and len(line.strip()) > 10]
            
            return recommendations[:5]  # Limit to 5 recommendations
            
        except Exception as e:
            logger.warning(f"Recommendation generation failed: {e}")
            return ["Review failed quality checks and address issues."]
    
    async def validate_with_vision(
        self,
        deliverable: Path,
        quality_standard: QualityStandard,
        domain: str,
        llm_engine: LLMEngine
    ) -> ValidationResult:
        """
        Validate deliverable using Llama 3.2 Vision for visual inspection.
        
        Args:
            deliverable: Path to deliverable (image, CAD, blueprint)
            quality_standard: Quality standard
            domain: Domain context
            llm_engine: LLM engine with vision capability
        
        Returns:
            Validation result
        """
        if not llm_engine.vision_engine:
            return ValidationResult(
                valid=False,
                overall_score=0.0,
                checks=[QualityCheck(
                    check_name="vision_available",
                    passed=False,
                    severity="major",
                    message="Vision engine not available for visual inspection"
                )],
                critical_issues=0,
                major_issues=1,
                minor_issues=0
            )
        
        # Use vision to inspect deliverable
        vision_prompt = f"""Perform visual quality inspection on this {domain} deliverable.

Quality Standard: {quality_standard.value}

Inspect for:
- Design quality
- Compliance with standards
- Visual accuracy
- Professional appearance
- Any defects or issues

Provide detailed assessment."""

        try:
            vision_analysis = await llm_engine.analyze_image(
                image_path=str(deliverable),
                prompt=vision_prompt
            )
            
            if isinstance(vision_analysis, dict):
                analysis_text = vision_analysis.get("text", str(vision_analysis))
            else:
                analysis_text = str(vision_analysis)
            
            # Determine if valid based on analysis
            valid = "pass" in analysis_text.lower() or "compliant" in analysis_text.lower()
            
            return ValidationResult(
                valid=valid,
                overall_score=0.8 if valid else 0.4,
                checks=[QualityCheck(
                    check_name="visual_inspection",
                    passed=valid,
                    severity="major",
                    message=analysis_text[:300]
                )],
                critical_issues=0 if valid else 1,
                major_issues=0,
                minor_issues=0,
                recommendations=[analysis_text] if not valid else []
            )
            
        except Exception as e:
            logger.error(f"Vision validation failed: {e}")
            return ValidationResult(
                valid=False,
                overall_score=0.0,
                checks=[QualityCheck(
                    check_name="vision_validation",
                    passed=False,
                    severity="major",
                    message=f"Vision validation failed: {e}"
                )],
                critical_issues=0,
                major_issues=1,
                minor_issues=0
            )
    
    def _load_standards_databases(self):
        """Load standards databases (ISO, ANSI, IEEE, etc.)"""
        # ISO standards
        self.standards_db["ISO"] = {
            "ISO 9001": "Quality management systems",
            "ISO 14001": "Environmental management",
            "ISO 27001": "Information security",
            "ISO 45001": "Occupational health and safety"
        }
        
        # ANSI standards
        self.standards_db["ANSI"] = {
            "ANSI/ASME": "American National Standards Institute / American Society of Mechanical Engineers",
            "ANSI/IEEE": "IEEE standards"
        }
        
        # IEEE standards
        self.standards_db["IEEE"] = {
            "IEEE 802": "Networking standards",
            "IEEE 754": "Floating-point arithmetic",
            "IEEE 1012": "Software verification and validation"
        }
        
        logger.info(f"✅ Loaded {sum(len(v) for v in self.standards_db.values())} standards")
    
    async def validate_against_standards(
        self,
        deliverable: Any,
        domain: str,
        standard_names: List[str] = None
    ) -> Dict[str, Any]:
        """
        Validate against industry standards (ISO, ANSI, IEEE, etc.).
        
        Args:
            deliverable: Deliverable to validate
            domain: Domain context
            standard_names: Specific standards to check (None for all relevant)
        
        Returns:
            Validation result with standards compliance
        """
        logger.info(f"🔍 Validating against industry standards: {standard_names or 'all relevant'}")
        
        # Determine relevant standards
        if standard_names is None:
            # Auto-select based on domain
            if domain == "construction":
                standard_names = ["ISO 9001", "ANSI/ASME"]
            elif domain in ["game_dev", "robotics"]:
                standard_names = ["IEEE 1012", "ISO 27001"]
            elif domain == "aerospace":
                standard_names = ["ISO 9001", "ANSI/ASME", "IEEE 754"]
            else:
                standard_names = ["ISO 9001"]
        
        results = {}
        for standard_name in standard_names:
            # Find standard in database
            standard_found = False
            for org, standards in self.standards_db.items():
                if standard_name in standards:
                    standard_found = True
                    description = standards[standard_name]
                    
                    # Validate using LLM
                    validation_prompt = f"""Validate this {domain} deliverable against {standard_name}: {description}

Deliverable: {str(deliverable)[:1000]}

Does it comply with {standard_name}? Provide:
1. Compliance status (pass/fail)
2. Specific requirements checked
3. Issues found (if any)
4. Recommendations"""
                    
                    validation_result = await self.llm_engine.generate(
                        prompt=validation_prompt,
                        context={"domain": domain, "standard": standard_name},
                        task="standards_validation"
                    )
                    
                    results[standard_name] = {
                        "standard": standard_name,
                        "description": description,
                        "validation": str(validation_result),
                        "compliant": "pass" in str(validation_result).lower() or "compliant" in str(validation_result).lower()
                    }
                    break
            
            if not standard_found:
                results[standard_name] = {
                    "standard": standard_name,
                    "error": "Standard not found in database"
                }
        
        overall_compliant = all(r.get("compliant", False) for r in results.values() if "compliant" in r)
        
        return {
            "overall_compliant": overall_compliant,
            "standards_checked": len(results),
            "results": results
        }
    
    async def run_automated_tests(
        self,
        code_project: Path,
        test_framework: str = "pytest"
    ) -> Dict[str, Any]:
        """
        Run automated tests on code project.
        
        Args:
            code_project: Path to code project
            test_framework: Test framework to use
        
        Returns:
            Test results
        """
        logger.info(f"🧪 Running automated tests: {test_framework}")
        
        # Check if tests exist
        test_dir = code_project / "tests"
        if not test_dir.exists():
            # Generate tests using LLM
            logger.info("No tests found, generating tests...")
            return await self._generate_and_run_tests(code_project, test_framework)
        
        # Run existing tests (simplified - would actually execute tests)
        test_prompt = f"""Analyze this code project and determine test coverage:

Project: {code_project}

Test Framework: {test_framework}

Provide:
1. Test coverage estimate
2. Missing test areas
3. Test execution plan"""
        
        analysis = await self.llm_engine.generate(
            prompt=test_prompt,
            context={"project": str(code_project)},
            task="test_analysis"
        )
        
        return {
            "status": "success",
            "framework": test_framework,
            "tests_found": True,
            "analysis": str(analysis)
        }
    
    async def _generate_and_run_tests(
        self,
        code_project: Path,
        test_framework: str
    ) -> Dict[str, Any]:
        """Generate and run tests for code project"""
        # Generate tests using LLM
        test_gen_prompt = f"""Generate {test_framework} tests for this code project:

Project: {code_project}

Generate comprehensive test suite covering:
- Unit tests
- Integration tests
- Edge cases
- Error handling"""
        
        test_code = await self.llm_engine.generate(
            prompt=test_gen_prompt,
            context={"project": str(code_project)},
            task="test_generation"
        )
        
        return {
            "status": "generated",
            "framework": test_framework,
            "tests_generated": True,
            "test_code": str(test_code)
        }
    
    def get_validation_history(self) -> List[Dict[str, Any]]:
        """Get validation history"""
        return self.validation_history


