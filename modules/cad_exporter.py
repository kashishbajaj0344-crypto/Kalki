# ============================================================
# Kalki v2.3 — cad_exporter.py
# ------------------------------------------------------------
# CAD Export Module for STL/OBJ Export from OpenSCAD
# - Export OpenSCAD files to STL/OBJ formats using OpenSCAD CLI
# - Batch export functionality for multiple files
# - Quality and format validation
# - Integration with Kalki's design pipeline
# ============================================================

import os
import subprocess
import asyncio
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import shutil

from modules.utils.config import get_config
from modules.utils.logging_config import get_logger

logger = get_logger("Kalki.CADExporter")

class CADExporter:
    """CAD export engine for converting OpenSCAD files to various formats"""

    def __init__(self):
        self.openscad_path = self._find_openscad()
        self.freecad_integration = None
        self.supported_formats = ['stl', 'obj', 'off', 'amf', '3mf', 'csg', 'png', 'svg', 'pdf']
        self.output_dir = Path("output/cad")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _get_freecad_integration(self):
        """Lazy load FreeCAD integration"""
        if self.freecad_integration is None:
            try:
                from modules.freecad_integration import get_freecad_integration
                self.freecad_integration = get_freecad_integration()
            except ImportError:
                self.freecad_integration = None
        return self.freecad_integration

    def _find_openscad(self) -> Optional[str]:
        """Find OpenSCAD executable in system PATH"""
        openscad_cmd = "openscad"

        # Check if openscad is available
        try:
            # Try running with --version first (should work in headless mode)
            result = subprocess.run([openscad_cmd, "--version"],
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info(f"Found OpenSCAD: {openscad_cmd}")
                return openscad_cmd
        except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
            pass

        # Try common installation paths
        common_paths = [
            "/usr/bin/openscad",
            "/usr/local/bin/openscad",
            "/opt/homebrew/bin/openscad",  # macOS Homebrew
            "/Applications/OpenSCAD.app/Contents/MacOS/OpenSCAD",  # macOS app bundle
            "C:\\Program Files\\OpenSCAD\\openscad.exe",  # Windows
        ]

        for path in common_paths:
            if os.path.exists(path):
                logger.info(f"Found OpenSCAD at: {path}")
                return path

        logger.warning("OpenSCAD not found. Install from https://openscad.org/downloads.html")
        logger.info("CAD files are still generated as .scad files for manual export")
        return None

    def is_gui_available(self) -> bool:
        """Check if GUI environment is available for OpenSCAD"""
        import os

        # Check for display environment variables
        display_vars = ['DISPLAY', 'WAYLAND_DISPLAY']
        has_display = any(os.environ.get(var) for var in display_vars)

        # Check if we're on macOS and have GUI access
        if os.name == 'posix' and os.uname().sysname == 'Darwin':
            # On macOS, check if we can access the window server
            try:
                result = subprocess.run(['pgrep', '-x', 'WindowServer'],
                                      capture_output=True, timeout=5)
                has_display = result.returncode == 0
            except:
                has_display = False

        return has_display

    def is_available(self) -> bool:
        """Check if OpenSCAD is available for export"""
        return self.openscad_path is not None and self.is_gui_available()

    async def export_file(self, input_file: str, output_format: str = 'stl',
                         output_file: Optional[str] = None,
                         quality_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export a single OpenSCAD file to specified format

        Args:
            input_file: Path to input .scad file
            output_format: Output format ('stl', 'obj', 'off', etc.)
            output_file: Optional output file path
            quality_settings: Quality settings for export

        Returns:
            Dict with export results
        """
        if not self.is_available():
            if not self.openscad_path:
                return {
                    "status": "error",
                    "error": "OpenSCAD not available",
                    "input_file": input_file
                }
            else:
                return {
                    "status": "error",
                    "error": "OpenSCAD requires GUI environment. Please run in graphical environment or install virtual display (X11/Xvfb)",
                    "input_file": input_file,
                    "openscad_path": self.openscad_path,
                    "gui_available": self.is_gui_available()
                }

        input_path = Path(input_file)
        if not input_path.exists():
            return {
                "status": "error",
                "error": f"Input file not found: {input_file}",
                "input_file": input_file
            }

        if output_format not in self.supported_formats:
            return {
                "status": "error",
                "error": f"Unsupported format: {output_format}. Supported: {self.supported_formats}",
                "input_file": input_file
            }

        # Generate output filename if not provided
        if output_file is None:
            output_name = input_path.stem + f".{output_format}"
            output_path = self.output_dir / output_name
        else:
            output_path = Path(output_file)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            # Build OpenSCAD command
            cmd = [self.openscad_path, "-o", str(output_path), str(input_path)]

            # Add quality settings if provided
            if quality_settings:
                if 'fn' in quality_settings:
                    cmd.extend(['--render', f"fn={quality_settings['fn']}"])
                if 'fa' in quality_settings:
                    cmd.extend(['--render', f"fa={quality_settings['fa']}"])
                if 'fs' in quality_settings:
                    cmd.extend(['--render', f"fs={quality_settings['fs']}"])

            logger.info(f"Exporting {input_file} to {output_path} (format: {output_format})")

            # Run OpenSCAD export
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if process.returncode == 0:
                # Verify output file was created
                if output_path.exists() and output_path.stat().st_size > 0:
                    file_size = output_path.stat().st_size
                    logger.info(f"Successfully exported {output_path} ({file_size} bytes)")

                    return {
                        "status": "success",
                        "input_file": str(input_path),
                        "output_file": str(output_path),
                        "format": output_format,
                        "file_size": file_size,
                        "command": " ".join(cmd)
                    }
                else:
                    return {
                        "status": "error",
                        "error": "Output file was not created or is empty",
                        "input_file": str(input_path),
                        "output_file": str(output_path)
                    }
            else:
                error_msg = stderr.decode().strip() if stderr else "Unknown error"
                logger.error(f"OpenSCAD export failed: {error_msg}")

                return {
                    "status": "error",
                    "error": f"OpenSCAD export failed: {error_msg}",
                    "input_file": str(input_path),
                    "output_file": str(output_path),
                    "stdout": stdout.decode().strip() if stdout else "",
                    "stderr": error_msg
                }

        except Exception as e:
            logger.exception(f"Error during CAD export: {e}")
            return {
                "status": "error",
                "error": str(e),
                "input_file": str(input_path),
                "output_file": str(output_path)
            }

    async def export_batch(self, input_files: List[str], output_format: str = 'stl',
                          quality_settings: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Export multiple OpenSCAD files in batch

        Args:
            input_files: List of input .scad file paths
            output_format: Output format for all files
            quality_settings: Quality settings for all exports

        Returns:
            List of export results
        """
        results = []

        for input_file in input_files:
            result = await self.export_file(input_file, output_format, None, quality_settings)
            results.append(result)

            # Small delay between exports to avoid overwhelming the system
            await asyncio.sleep(0.1)

        return results

    async def export_project(self, project_dir: str, output_format: str = 'stl',
                           quality_settings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Export all OpenSCAD files in a project directory

        Args:
            project_dir: Directory containing .scad files
            output_format: Output format
            quality_settings: Quality settings

        Returns:
            Dict with batch export results
        """
        project_path = Path(project_dir)
        if not project_path.exists():
            return {
                "status": "error",
                "error": f"Project directory not found: {project_dir}"
            }

        # Find all .scad files
        scad_files = list(project_path.glob("**/*.scad"))
        if not scad_files:
            return {
                "status": "error",
                "error": f"No .scad files found in {project_dir}"
            }

        scad_file_paths = [str(f) for f in scad_files]
        logger.info(f"Found {len(scad_file_paths)} .scad files in {project_dir}")

        # Export all files
        results = await self.export_batch(scad_file_paths, output_format, quality_settings)

        # Summarize results
        successful = [r for r in results if r.get("status") == "success"]
        failed = [r for r in results if r.get("status") == "error"]

        return {
            "status": "completed",
            "project_dir": str(project_path),
            "total_files": len(results),
            "successful_exports": len(successful),
            "failed_exports": len(failed),
            "results": results,
            "output_format": output_format
        }

    def get_supported_formats(self) -> List[str]:
        """Get list of supported export formats"""
        return self.supported_formats.copy()

    def validate_export(self, output_file: str, expected_format: str) -> Dict[str, Any]:
        """
        Validate an exported file

        Args:
            output_file: Path to exported file
            expected_format: Expected format

        Returns:
            Validation results
        """
        output_path = Path(output_file)

        if not output_path.exists():
            return {
                "valid": False,
                "error": "File does not exist",
                "file": output_file
            }

        if output_path.stat().st_size == 0:
            return {
                "valid": False,
                "error": "File is empty",
                "file": output_file
            }

        # Basic format validation
        if expected_format == 'stl':
            # Check if file starts with STL header
            try:
                with open(output_path, 'rb') as f:
                    header = f.read(80)
                    if header.startswith(b'solid ') or header.startswith(b'\x00\x00\x00\x00'):
                        return {"valid": True, "file": output_file, "format": "stl"}
            except Exception as e:
                return {"valid": False, "error": str(e), "file": output_file}

        elif expected_format == 'obj':
            # Check if file contains OBJ format markers
            try:
                with open(output_path, 'r') as f:
                    content = f.read(1024)
                    if 'v ' in content or 'f ' in content:
                        return {"valid": True, "file": output_file, "format": "obj"}
            except Exception as e:
                return {"valid": False, "error": str(e), "file": output_file}

        # Generic validation for other formats
        return {
            "valid": True,
            "file": output_file,
            "format": expected_format,
            "file_size": output_path.stat().st_size
        }

# Global CAD exporter instance
_cad_exporter = None

def get_cad_exporter() -> CADExporter:
    """Get the global CAD exporter instance"""
    global _cad_exporter
    if _cad_exporter is None:
        _cad_exporter = CADExporter()
    return _cad_exporter

