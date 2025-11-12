"""
Project Persistence System

Saves and loads domain-specific projects to/from disk.
Supports JSON-based storage with full project state preservation.
"""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

from modules.domains.base_domain import ProjectStateMachine

logger = logging.getLogger(__name__)


class ProjectPersistence:
    """Manages project persistence across KALKI restarts"""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or Path("data/projects")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # SQLite database for project metadata and quick queries
        self.db_path = self.storage_dir / "projects.db"
        self._init_database()
    
    def _init_database(self):
        """Initialize projects database"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                project_id TEXT PRIMARY KEY,
                domain TEXT NOT NULL,
                description TEXT,
                current_phase TEXT,
                created_at TEXT,
                updated_at TEXT,
                status TEXT DEFAULT 'active',
                metadata TEXT
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_domain ON projects(domain)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_status ON projects(status)
        """)
        conn.commit()
        conn.close()
    
    def save_project(self, project: ProjectStateMachine) -> bool:
        """
        Save project to disk.
        
        Args:
            project: ProjectStateMachine instance
        
        Returns:
            Success boolean
        """
        try:
            # Serialize project state
            project_data = {
                "project_id": project.project_id,
                "description": project.description,
                "domain": project.domain,
                "current_phase": project.current_phase.value if hasattr(project.current_phase, 'value') else str(project.current_phase),
                "phase_history": [
                    {
                        "from": h.get("from").value if hasattr(h.get("from"), 'value') else str(h.get("from", "")),
                        "to": h.get("to").value if hasattr(h.get("to"), 'value') else str(h.get("to", "")),
                        "timestamp": h.get("timestamp")
                    }
                    for h in project.phase_history
                ],
                "metadata": project.metadata,
                "issues": project.issues,
                "milestones": project.milestones,
                "created_at": datetime.now().isoformat(),
                "updated_at": datetime.now().isoformat()
            }
            
            # Add domain-specific attributes
            # Construction domain
            if hasattr(project, 'location'):
                project_data['location'] = project.location
            if hasattr(project, 'building_type'):
                project_data['building_type'] = project.building_type
            if hasattr(project, 'size_sqft'):
                project_data['size_sqft'] = project.size_sqft
            if hasattr(project, 'stories'):
                project_data['stories'] = project.stories
            
            # Game dev domain
            if hasattr(project, 'game_engine'):
                project_data['game_engine'] = project.game_engine
            if hasattr(project, 'target_platforms'):
                project_data['target_platforms'] = project.target_platforms
            if hasattr(project, 'genre'):
                project_data['genre'] = project.genre.value if hasattr(project.genre, 'value') else project.genre
            if hasattr(project, 'team_size'):
                project_data['team_size'] = project.team_size
            if hasattr(project, 'monetization_model'):
                project_data['monetization_model'] = project.monetization_model
            
            # Common attributes
            if hasattr(project, 'budget'):
                project_data['budget'] = project.budget
            if hasattr(project, 'timeline'):
                project_data['timeline'] = project.timeline
            
            # Save as JSON file
            project_file = self.storage_dir / f"{project.project_id}.json"
            with open(project_file, 'w') as f:
                json.dump(project_data, f, indent=2)
            
            # Update database index
            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO projects 
                (project_id, domain, description, current_phase, created_at, updated_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                project.project_id,
                project.domain,
                project.description,
                project.current_phase.value if hasattr(project.current_phase, 'value') else str(project.current_phase),
                project_data['created_at'],
                project_data['updated_at'],
                json.dumps(project.metadata)
            ))
            conn.commit()
            conn.close()
            
            logger.info(f"Saved project {project.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving project: {e}")
            return False
    
    def load_project(self, project_id: str, domain: str = None) -> Optional[Dict]:
        """
        Load project from disk.
        
        Args:
            project_id: Project ID
            domain: Domain hint (optional, speeds up loading)
        
        Returns:
            Project data dict or None if not found
        """
        try:
            project_file = self.storage_dir / f"{project_id}.json"
            
            if not project_file.exists():
                logger.warning(f"Project {project_id} not found")
                return None
            
            with open(project_file, 'r') as f:
                project_data = json.load(f)
            
            return project_data
            
        except Exception as e:
            logger.error(f"Error loading project: {e}")
            return None
    
    def list_projects(
        self,
        domain: Optional[str] = None,
        status: str = 'active',
        limit: int = 100
    ) -> List[Dict]:
        """
        List projects with optional filtering.
        
        Args:
            domain: Filter by domain
            status: Filter by status (active, completed, archived)
            limit: Maximum results
        
        Returns:
            List of project summaries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            
            query = "SELECT * FROM projects WHERE status = ?"
            params = [status]
            
            if domain:
                query += " AND domain = ?"
                params.append(domain)
            
            query += " ORDER BY updated_at DESC LIMIT ?"
            params.append(limit)
            
            cursor = conn.execute(query, params)
            projects = [dict(row) for row in cursor.fetchall()]
            conn.close()
            
            return projects
            
        except Exception as e:
            logger.error(f"Error listing projects: {e}")
            return []
    
    def save_project_state(self, project_state: Any, domain: Optional[str] = None, status: Optional[str] = None) -> bool:
        """
        Persist a project state dataclass/dict for orchestration layers like the construction copilot.
        """
        try:
            if hasattr(project_state, "to_dict"):
                state_dict = project_state.to_dict()
                project_id = state_dict["project_id"]
            elif isinstance(project_state, dict):
                state_dict = project_state
                project_id = state_dict["project_id"]
            else:
                raise ValueError("project_state must be dict-like or provide to_dict()")

            domain = domain or state_dict.get("domain", "general")
            if status is None:
                completion = state_dict.get("completion_percentage", 0)
                status = 'completed' if completion >= 1.0 else 'active'

            return self._save_project_state_dict(project_id, domain, state_dict, status)

        except Exception as e:
            logger.error(f"Error saving project state: {e}")
            return False

    def load_project_state(self, project_id: str) -> Optional[Dict[str, Any]]:
        """
        Load a previously saved project state dictionary.
        """
        try:
            project_file = self.storage_dir / f"{project_id}.json"
            if not project_file.exists():
                logger.warning(f"Project state file not found: {project_file}")
                return None

            with open(project_file) as f:
                return json.load(f)

        except Exception as e:
            logger.error(f"Error loading project state {project_id}: {e}")
            return None

    def _save_project_state_dict(
        self,
        project_id: str,
        domain: str,
        state: Dict[str, Any],
        status: str
    ) -> bool:
        """Persist project state dictionary and maintain metadata index."""
        try:
            project_file = self.storage_dir / f"{project_id}.json"
            with open(project_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            conn = sqlite3.connect(self.db_path)
            conn.execute("""
                INSERT OR REPLACE INTO projects 
                (project_id, domain, description, current_phase, created_at, updated_at, status, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                project_id,
                domain,
                state.get('address') or state.get('project_type', ''),
                state.get('current_stage', 'unknown'),
                state.get('start_date', datetime.now().isoformat()),
                datetime.now().isoformat(),
                status,
                json.dumps({
                    "project_type": state.get('project_type'),
                    "completion_percentage": state.get('completion_percentage'),
                    "address": state.get('address')
                })
            ))
            conn.commit()
            conn.close()

            logger.info(f"Saved project state {project_id} ({domain})")
            return True

        except Exception as e:
            logger.error(f"Error saving project state {project_id}: {e}")
            return False
    
    def delete_project(self, project_id: str) -> bool:
        """
        Delete project (mark as archived, don't actually delete).
        
        Args:
            project_id: Project ID
        
        Returns:
            Success boolean
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.execute(
                "UPDATE projects SET status = 'archived', updated_at = ? WHERE project_id = ?",
                (datetime.now().isoformat(), project_id)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Archived project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error archiving project: {e}")
            return False
    
    def get_project_stats(self) -> Dict[str, Any]:
        """Get project statistics"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Total projects
            cursor = conn.execute("SELECT COUNT(*) FROM projects WHERE status = 'active'")
            total = cursor.fetchone()[0]
            
            # By domain
            cursor = conn.execute("""
                SELECT domain, COUNT(*) as count 
                FROM projects 
                WHERE status = 'active'
                GROUP BY domain
            """)
            by_domain = {row[0]: row[1] for row in cursor.fetchall()}
            
            # By phase
            cursor = conn.execute("""
                SELECT current_phase, COUNT(*) as count 
                FROM projects 
                WHERE status = 'active'
                GROUP BY current_phase
            """)
            by_phase = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_projects": total,
                "by_domain": by_domain,
                "by_phase": by_phase
            }
            
        except Exception as e:
            logger.error(f"Error getting project stats: {e}")
            return {}


# Global instance
_persistence = None


def get_project_persistence() -> ProjectPersistence:
    """Get or create global project persistence instance"""
    global _persistence
    if _persistence is None:
        _persistence = ProjectPersistence()
    return _persistence
