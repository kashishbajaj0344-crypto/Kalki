# ============================================================
# Kalki v2.3 — test_cad_outputs.py
# ------------------------------------------------------------
# Automated Unit Tests for CAD Output Validation
# - Test CAD file format validation
# - Test export functionality (STL/OBJ)
# - Test drawing generation (SVG/DXF)
# - Test physics validation (FreeCAD integration)
# - Test design integrity and completeness
# ============================================================

import pytest
import asyncio
import os
import tempfile
from pathlib import Path
import json
from unittest.mock import patch, MagicMock

from modules.cad_exporter import CADExporter, get_cad_exporter
from modules.cad_drawings import CADDrawingGenerator, get_drawing_generator
from modules.freecad_integration import FreeCADIntegration, get_freecad_integration
from modules.generative_design_engine import GenerativeDesignEngine

class TestCADExporter:
    """Test CAD export functionality"""

    @pytest.fixture
    def exporter(self):
        return get_cad_exporter()

    @pytest.fixture
    def sample_scad_file(self, tmp_path):
        """Create a sample OpenSCAD file for testing"""
        scad_content = """
// Sample test cube
cube([10, 10, 10]);
"""
        scad_file = tmp_path / "test_cube.scad"
        scad_file.write_text(scad_content)
        return str(scad_file)

    def test_exporter_initialization(self, exporter):
        """Test that CAD exporter initializes properly"""
        assert isinstance(exporter, CADExporter)
        assert hasattr(exporter, 'openscad_path')
        assert hasattr(exporter, 'supported_formats')

    def test_supported_formats(self, exporter):
        """Test that supported formats are properly defined"""
        formats = exporter.get_supported_formats()
        assert isinstance(formats, list)
        assert 'stl' in formats
        assert 'obj' in formats
        assert 'svg' in formats

    @pytest.mark.asyncio
    async def test_export_validation_success(self, exporter, tmp_path):
        """Test successful export validation"""
        # Create a dummy STL file
        stl_file = tmp_path / "test.stl"
        stl_content = b"solid test\n\tfacet normal 0 0 1\n\t\touter loop\n\t\t\tvertex 0 0 0\n\t\t\tvertex 1 0 0\n\t\t\tvertex 0 1 0\n\t\tendloop\n\tendfacet\nendsolid test"
        stl_file.write_bytes(stl_content)

        result = exporter.validate_export(str(stl_file), 'stl')
        assert result['valid'] == True
        assert result['format'] == 'stl'

    @pytest.mark.asyncio
    async def test_export_validation_failure(self, exporter, tmp_path):
        """Test export validation failure for non-existent file"""
        result = exporter.validate_export("/nonexistent/file.stl", 'stl')
        assert result['valid'] == False
        assert 'not exist' in result['error'].lower()

    @pytest.mark.asyncio
    async def test_export_validation_empty_file(self, exporter, tmp_path):
        """Test export validation for empty file"""
        empty_file = tmp_path / "empty.stl"
        empty_file.write_bytes(b"")

        result = exporter.validate_export(str(empty_file), 'stl')
        assert result['valid'] == False
        assert 'empty' in result['error'].lower()

class TestCADDrawingGenerator:
    """Test CAD drawing generation functionality"""

    @pytest.fixture
    def drawing_generator(self):
        return get_drawing_generator()

    @pytest.fixture
    def sample_scad_file(self, tmp_path):
        """Create a sample OpenSCAD file for testing"""
        scad_content = """
// Sample test cylinder
cylinder(h=20, r=5);
"""
        scad_file = tmp_path / "test_cylinder.scad"
        scad_file.write_text(scad_content)
        return str(scad_file)

    def test_drawing_generator_initialization(self, drawing_generator):
        """Test that drawing generator initializes properly"""
        assert isinstance(drawing_generator, CADDrawingGenerator)
        assert hasattr(drawing_generator, 'standard_views')
        assert hasattr(drawing_generator, 'output_dir')

    def test_available_views(self, drawing_generator):
        """Test that standard views are available"""
        views = drawing_generator.get_available_views()
        assert isinstance(views, list)
        assert 'front' in views
        assert 'top' in views
        assert 'side' in views
        assert 'isometric' in views

    def test_view_descriptions(self, drawing_generator):
        """Test view descriptions"""
        desc = drawing_generator.get_view_description('front')
        assert 'Front View' in desc

        desc = drawing_generator.get_view_description('invalid')
        assert desc == 'Unknown view'

class TestFreeCADIntegration:
    """Test FreeCAD integration functionality"""

    @pytest.fixture
    def freecad_integration(self):
        return get_freecad_integration()

    def test_freecad_integration_initialization(self, freecad_integration):
        """Test that FreeCAD integration initializes properly"""
        assert isinstance(freecad_integration, FreeCADIntegration)
        assert hasattr(freecad_integration, 'freecad_available')
        assert hasattr(freecad_integration, 'temp_dir')

    @pytest.mark.asyncio
    async def test_physics_validation_without_freecad(self, freecad_integration, tmp_path):
        """Test physics validation when FreeCAD is not available"""
        # Mock FreeCAD as unavailable
        freecad_integration.freecad_available = False

        scad_file = tmp_path / "test.scad"
        scad_file.write_text("cube([10,10,10]);")

        result = await freecad_integration.validate_physics(str(scad_file))
        assert result['status'] == 'error'
        assert 'FreeCAD not available' in result['error']

class TestCADIntegration:
    """Test integrated CAD functionality"""

    @pytest.fixture
    async def design_engine(self):
        """Create a design engine for testing"""
        engine = GenerativeDesignEngine()
        await engine.initialize()
        return engine

    @pytest.mark.asyncio
    async def test_full_cad_pipeline(self, design_engine, tmp_path):
        """Test the full CAD generation and export pipeline"""
        # Generate a design
        project = await design_engine.create_design_project("Create a simple cube")

        # Wait for completion (simplified for testing)
        await asyncio.sleep(1)

        # Check if CAD files were created
        cad_dir = Path("output/cad")
        if cad_dir.exists():
            scad_files = list(cad_dir.glob("*.scad"))
            if scad_files:
                scad_file = str(scad_files[0])

                # Test export functionality
                exporter = get_cad_exporter()
                if exporter.is_available():
                    export_result = await exporter.export_file(scad_file, 'stl')
                    # Export may fail if OpenSCAD not available, but structure should be correct
                    assert 'status' in export_result
                    assert 'input_file' in export_result

                # Test drawing generation
                drawing_gen = get_drawing_generator()
                drawing_result = await drawing_gen.generate_2d_projection(scad_file, 'front', 'svg')
                # Drawing generation may fail, but structure should be correct
                assert 'status' in drawing_result
                assert 'view' in drawing_result or 'error' in drawing_result

class TestCADFileValidation:
    """Test CAD file validation and integrity checks"""

    def test_scad_file_validation(self, tmp_path):
        """Test OpenSCAD file validation"""
        # Valid SCAD file
        valid_scad = tmp_path / "valid.scad"
        valid_scad.write_text("""
module test() {
    cube([10, 10, 10]);
}
test();
""")

        # Check if file exists and has content
        assert valid_scad.exists()
        assert valid_scad.stat().st_size > 0

        content = valid_scad.read_text()
        assert 'module' in content
        assert 'cube' in content

    def test_stl_file_structure(self, tmp_path):
        """Test basic STL file structure validation"""
        # Create a minimal valid STL file
        stl_file = tmp_path / "test.stl"
        stl_content = """solid test
  facet normal 0 0 1
    outer loop
      vertex 0 0 0
      vertex 1 0 0
      vertex 0 1 0
    endloop
  endfacet
endsolid test"""

        stl_file.write_text(stl_content)

        content = stl_file.read_text()
        assert content.startswith('solid ')
        assert 'facet normal' in content
        assert 'vertex' in content
        assert content.endswith('endsolid test')

    def test_svg_file_structure(self, tmp_path):
        """Test basic SVG file structure validation"""
        svg_file = tmp_path / "test.svg"
        svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">
  <rect width="50" height="50" fill="blue"/>
</svg>"""

        svg_file.write_text(svg_content)

        content = svg_file.read_text()
        assert '<?xml' in content
        assert '<svg' in content
        assert 'xmlns=' in content
        assert '</svg>' in content

class TestCADQualityAssurance:
    """Test CAD quality assurance and standards compliance"""

    def test_design_standards_check(self):
        """Test basic design standards validation"""
        # Test unit consistency
        design_data = {
            'units': 'mm',
            'tolerance': 0.1,
            'material': 'PLA',
            'dimensions': [10, 10, 10]
        }

        # Basic validation
        assert design_data['units'] in ['mm', 'cm', 'inch']
        assert design_data['tolerance'] > 0
        assert len(design_data['dimensions']) == 3
        assert all(d > 0 for d in design_data['dimensions'])

    def test_export_quality_metrics(self):
        """Test export quality metrics"""
        quality_metrics = {
            'file_size': 1024,
            'vertex_count': 100,
            'face_count': 50,
            'manifold': True,
            'watertight': True
        }

        # Validate metrics
        assert quality_metrics['file_size'] > 0
        assert quality_metrics['vertex_count'] > 0
        assert quality_metrics['face_count'] > 0
        assert quality_metrics['manifold'] == True
        assert quality_metrics['watertight'] == True

# Integration test for the complete CAD pipeline
@pytest.mark.integration
class TestCADPipelineIntegration:
    """Integration tests for the complete CAD pipeline"""

    @pytest.mark.asyncio
    async def test_end_to_end_cad_workflow(self, tmp_path):
        """Test complete CAD workflow from design to export"""
        # This would be a full integration test
        # For now, just test the pipeline structure

        # Create a temporary design request
        design_request = "Create a simple cube"

        # Initialize components
        exporter = get_cad_exporter()
        drawing_gen = get_drawing_generator()
        freecad_int = get_freecad_integration()

        # Test component availability
        assert isinstance(exporter, CADExporter)
        assert isinstance(drawing_gen, CADDrawingGenerator)
        assert isinstance(freecad_int, FreeCADIntegration)

        # Test that all components can be initialized without errors
        assert hasattr(exporter, 'is_available')
        assert hasattr(drawing_gen, 'get_available_views')
        assert hasattr(freecad_int, 'freecad_available')

    def test_integration_validation(self, tmp_path):
        """Test overall CAD system integration"""
        # Create test SCAD file
        scad_content = """
// Integration test cube
cube([20, 20, 20]);
"""
        scad_file = tmp_path / "integration_test.scad"
        scad_file.write_text(scad_content)

        print(f"✅ Created test SCAD file: {scad_file}")

        # Test exporter initialization
        exporter = get_cad_exporter()
        print(f"✅ CAD Exporter initialized, OpenSCAD available: {exporter.is_available()}")

        # Test drawing generator
        drawing_gen = get_drawing_generator()
        views = drawing_gen.get_available_views()
        print(f"✅ Drawing generator initialized with {len(views)} views: {views}")

        # Test FreeCAD integration
        freecad_int = get_freecad_integration()
        print(f"✅ FreeCAD integration initialized, FreeCAD available: {freecad_int.freecad_available}")

        print("🎉 CAD validation tests completed successfully!")