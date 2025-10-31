#!/usr/bin/env python3
"""
Ethics Storage: Persistent datastore for ethical evaluations and risk patterns
"""
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from cryptography.fernet import Fernet

from ...config import ROOT


class EthicsStorage:
    """Persistent storage for ethical evaluations, risk assessments, and simulations"""

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path or ROOT / "data" / "ethics_log.json")
        self.encryption_key = self._get_or_create_key()
        self.cipher = Fernet(self.encryption_key)

        # In-memory caches
        self.evaluations = []
        self.assessments = []
        self.simulations = []
        self.risk_patterns = {}

        # Load existing data
        self._load_data()

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key for sensitive ethics data"""
        key_path = self.storage_path.parent / "ethics_key"
        if key_path.exists():
            return key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            key_path.write_bytes(key)
            return key

    def _load_data(self):
        """Load data from persistent storage"""
        try:
            if self.storage_path.exists():
                encrypted_data = self.storage_path.read_bytes()
                decrypted_data = self.cipher.decrypt(encrypted_data)
                data = json.loads(decrypted_data.decode())

                self.evaluations = data.get("evaluations", [])
                self.assessments = data.get("assessments", [])
                self.simulations = data.get("simulations", [])
                self.risk_patterns = data.get("risk_patterns", {})
        except Exception as e:
            print(f"Failed to load ethics data: {e}")
            # Initialize empty if loading fails
            self.evaluations = []
            self.assessments = []
            self.simulations = []
            self.risk_patterns = {}

    def _save_data(self):
        """Save data to persistent storage"""
        try:
            data = {
                "evaluations": self.evaluations,
                "assessments": self.assessments,
                "simulations": self.simulations,
                "risk_patterns": self.risk_patterns,
                "last_updated": datetime.utcnow().isoformat()
            }

            json_data = json.dumps(data, indent=2)
            encrypted_data = self.cipher.encrypt(json_data.encode())
            self.storage_path.write_bytes(encrypted_data)
        except Exception as e:
            print(f"Failed to save ethics data: {e}")

    def store_evaluation(self, evaluation: Dict[str, Any]):
        """Store an ethical evaluation"""
        evaluation["stored_at"] = datetime.utcnow().isoformat()
        evaluation["id"] = f"eval_{len(self.evaluations)}"
        self.evaluations.append(evaluation)
        self._update_risk_patterns(evaluation)
        self._save_data()

    def store_assessment(self, assessment: Dict[str, Any]):
        """Store a risk assessment"""
        assessment["stored_at"] = datetime.utcnow().isoformat()
        assessment["id"] = f"assess_{len(self.assessments)}"
        self.assessments.append(assessment)
        self._update_risk_patterns(assessment)
        self._save_data()

    def store_simulation(self, simulation: Dict[str, Any]):
        """Store a simulation result"""
        simulation["stored_at"] = datetime.utcnow().isoformat()
        simulation["id"] = f"sim_{len(self.simulations)}"
        self.simulations.append(simulation)
        self._save_data()

    def _update_risk_patterns(self, record: Dict[str, Any]):
        """Update risk pattern analysis based on new records"""
        # Extract patterns from evaluations and assessments
        if "violations" in record:
            for violation in record["violations"]:
                if violation not in self.risk_patterns:
                    self.risk_patterns[violation] = {"count": 0, "severity_sum": 0}
                self.risk_patterns[violation]["count"] += 1
                if "severity" in record:
                    self.risk_patterns[violation]["severity_sum"] += record["severity"]

        if "risk_level" in record:
            risk_key = f"risk_{record['risk_level']}"
            if risk_key not in self.risk_patterns:
                self.risk_patterns[risk_key] = {"count": 0}
            self.risk_patterns[risk_key]["count"] += 1

    def get_evaluation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent evaluation history"""
        return self.evaluations[-limit:]

    def get_assessment_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent assessment history"""
        return self.assessments[-limit:]

    def get_simulation_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent simulation history"""
        return self.simulations[-limit:]

    def get_risk_patterns(self) -> Dict[str, Any]:
        """Get aggregated risk patterns"""
        return self.risk_patterns.copy()

    def find_similar_evaluations(self, query: Dict[str, Any], threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Find evaluations similar to the query"""
        similar = []
        for evaluation in self.evaluations:
            similarity = self._calculate_similarity(query, evaluation)
            if similarity >= threshold:
                similar.append({**evaluation, "similarity": similarity})
        return sorted(similar, key=lambda x: x["similarity"], reverse=True)

    def _calculate_similarity(self, query: Dict[str, Any], evaluation: Dict[str, Any]) -> float:
        """Calculate similarity between query and evaluation"""
        # Simple similarity based on matching keys and values
        matches = 0
        total = len(query)

        for key, value in query.items():
            if key in evaluation and evaluation[key] == value:
                matches += 1

        return matches / total if total > 0 else 0

    def get_ethics_summary(self) -> Dict[str, Any]:
        """Get summary statistics of ethical oversight"""
        total_evaluations = len(self.evaluations)
        total_assessments = len(self.assessments)
        total_simulations = len(self.simulations)

        ethical_violations = sum(len(e.get("violations", [])) for e in self.evaluations)
        high_risk_assessments = sum(1 for a in self.assessments if a.get("risk_level") == "high")

        return {
            "total_evaluations": total_evaluations,
            "total_assessments": total_assessments,
            "total_simulations": total_simulations,
            "ethical_violations": ethical_violations,
            "high_risk_assessments": high_risk_assessments,
            "violation_rate": ethical_violations / total_evaluations if total_evaluations > 0 else 0,
            "high_risk_rate": high_risk_assessments / total_assessments if total_assessments > 0 else 0,
            "risk_patterns": self.risk_patterns
        }

    def clear_old_data(self, days_to_keep: int = 90):
        """Clear data older than specified days"""
        cutoff_date = datetime.utcnow().timestamp() - (days_to_keep * 24 * 60 * 60)

        self.evaluations = [
            e for e in self.evaluations
            if datetime.fromisoformat(e.get("stored_at", "2000-01-01")).timestamp() > cutoff_date
        ]

        self.assessments = [
            a for a in self.assessments
            if datetime.fromisoformat(a.get("stored_at", "2000-01-01")).timestamp() > cutoff_date
        ]

        self.simulations = [
            s for s in self.simulations
            if datetime.fromisoformat(s.get("stored_at", "2000-01-01")).timestamp() > cutoff_date
        ]

        self._save_data()