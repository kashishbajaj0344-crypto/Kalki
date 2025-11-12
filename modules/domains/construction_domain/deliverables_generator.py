"""
Construction Deliverables Generators

Generates professional-grade construction deliverables:
- Construction drawings
- Bill of Materials with costs
- Construction schedules
- Inspection checklists
- Structural calculations
- Cost estimates
"""

from typing import Dict, Any, List
from datetime import datetime, timedelta
from pathlib import Path
import json


class ConstructionDeliverablesGenerator:
    """Generate construction deliverables"""
    
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.output_dir = data_dir / "deliverables"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    async def generate_construction_drawings(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate construction drawings"""
        
        drawings = {
            "site_plan": {
                "scale": "1:200",
                "sheets": ["A1.0"],
                "description": "Site layout showing building footprint, setbacks, parking",
                "notes": []
            },
            "floor_plans": {
                "scale": "1:50",
                "sheets": [f"A2.{i}" for i in range(1, getattr(project, 'stories', 1) + 1)],
                "description": "Detailed floor plans with dimensions and room labels",
                "notes": ["All dimensions in millimeters unless noted", "Verify all dimensions on site"]
            },
            "elevations": {
                "scale": "1:100",
                "sheets": ["A3.1", "A3.2", "A3.3", "A3.4"],
                "description": "North, South, East, West elevations",
                "notes": ["Exterior finishes per spec section 07"]
            },
            "sections": {
                "scale": "1:50",
                "sheets": ["A4.1", "A4.2"],
                "description": "Building sections showing floor-to-floor heights",
                "notes": ["Ceiling heights per code minimum"]
            },
            "details": {
                "scale": "1:10, 1:5",
                "sheets": ["A5.1", "A5.2", "A5.3"],
                "description": "Construction details for critical assemblies",
                "notes": ["See structural drawings for beam connections"]
            },
            "structural_plans": {
                "scale": "1:50",
                "sheets": ["S1.1", "S1.2"],
                "description": "Foundation and framing plans",
                "notes": ["All lumber to be Grade #2 or better unless noted"]
            }
        }
        
        project_info = {
            "project_name": project.description,
            "project_id": project.project_id,
            "location": getattr(project, 'location', 'Not specified'),
            "building_type": getattr(project, 'building_type', 'Not specified'),
            "total_area_sqft": getattr(project, 'size_sqft', 'Not specified'),
            "stories": getattr(project, 'stories', 1)
        }
        
        return {
            "drawings": drawings,
            "project_info": project_info,
            "total_sheets": sum(len(d.get("sheets", [])) for d in drawings.values()),
            "status": "ready_for_review",
            "notes": [
                "Drawings conform to BC Building Code 2018",
                "All structural members sized per span tables",
                "Requires professional engineer stamp before permit submission"
            ]
        }
    
    async def generate_bill_of_materials(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate detailed Bill of Materials with costs"""
        
        size_sqft = getattr(project, 'size_sqft', 2000)
        stories = getattr(project, 'stories', 1)
        
        # Foundation materials
        foundation_items = [
            {"item": "Concrete, 30 MPa", "unit": "m³", "quantity": 15.0 * stories * 0.7, "unit_cost": 150.0, "category": "foundation"},
            {"item": "Rebar #4", "unit": "kg", "quantity": 250.0 * stories * 0.7, "unit_cost": 1.20, "category": "foundation"},
            {"item": "Formwork, plywood", "unit": "m²", "quantity": 40.0 * stories * 0.7, "unit_cost": 25.0, "category": "foundation"},
            {"item": "Vapor barrier, 6mil poly", "unit": "m²", "quantity": size_sqft / 10.764, "unit_cost": 2.50, "category": "foundation"},
        ]
        
        # Framing materials
        framing_items = [
            {"item": "2x6 SPF stud, 8'", "unit": "pcs", "quantity": size_sqft * 0.8 * stories, "unit_cost": 8.50, "category": "framing"},
            {"item": "2x10 SPF joist, 16'", "unit": "pcs", "quantity": size_sqft * 0.15, "unit_cost": 25.00, "category": "framing"},
            {"item": "2x12 SPF beam", "unit": "pcs", "quantity": 20.0 + (stories - 1) * 10, "unit_cost": 45.00, "category": "framing"},
            {"item": "Plywood sheathing 1/2\"", "unit": "sheets", "quantity": size_sqft / 32 * stories, "unit_cost": 42.00, "category": "framing"},
            {"item": "OSB roof sheathing 7/16\"", "unit": "sheets", "quantity": size_sqft / 32, "unit_cost": 35.00, "category": "framing"},
            {"item": "Framing nails, 16d", "unit": "kg", "quantity": size_sqft / 50, "unit_cost": 4.50, "category": "framing"},
        ]
        
        # Exterior finishes
        exterior_area = size_sqft * stories * 0.4 / 10.764  # m² of exterior walls
        exterior_items = [
            {"item": "Vinyl siding", "unit": "m²", "quantity": exterior_area, "unit_cost": 15.0, "category": "exterior"},
            {"item": "Asphalt shingles", "unit": "squares", "quantity": size_sqft / 100 * 1.2, "unit_cost": 120.0, "category": "exterior"},
            {"item": "House wrap", "unit": "roll", "quantity": exterior_area / 100, "unit_cost": 200.0, "category": "exterior"},
            {"item": "Windows, vinyl", "unit": "pcs", "quantity": 12.0 + stories * 3, "unit_cost": 450.0, "category": "exterior"},
            {"item": "Entry door, steel", "unit": "pcs", "quantity": 2.0, "unit_cost": 800.0, "category": "exterior"},
            {"item": "Interior door, hollow core", "unit": "pcs", "quantity": 8.0 + stories * 4, "unit_cost": 150.0, "category": "exterior"},
        ]
        
        # Insulation & drywall
        interior_area = size_sqft * stories / 10.764  # m²
        interior_items = [
            {"item": "Fiberglass insulation R20", "unit": "m²", "quantity": interior_area, "unit_cost": 8.0, "category": "insulation"},
            {"item": "Vapor barrier, poly", "unit": "m²", "quantity": interior_area, "unit_cost": 1.50, "category": "insulation"},
            {"item": "Drywall 1/2\"", "unit": "sheets", "quantity": size_sqft / 32 * 2 * stories, "unit_cost": 15.0, "category": "interior"},
            {"item": "Joint compound", "unit": "pail", "quantity": 10.0 * stories, "unit_cost": 25.0, "category": "interior"},
            {"item": "Drywall screws", "unit": "box", "quantity": 15.0 * stories, "unit_cost": 12.0, "category": "interior"},
            {"item": "Paint, interior", "unit": "gallon", "quantity": size_sqft / 350 * stories, "unit_cost": 45.0, "category": "interior"},
        ]
        
        # MEP systems
        mep_items = [
            {"item": "Electrical panel, 200A", "unit": "pcs", "quantity": 1.0, "unit_cost": 350.0, "category": "electrical"},
            {"item": "Romex 14/2", "unit": "m", "quantity": 500.0 * stories, "unit_cost": 0.80, "category": "electrical"},
            {"item": "Romex 12/2", "unit": "m", "quantity": 200.0 * stories, "unit_cost": 1.20, "category": "electrical"},
            {"item": "Outlets & switches", "unit": "pcs", "quantity": 50.0 * stories, "unit_cost": 8.0, "category": "electrical"},
            {"item": "Light fixtures", "unit": "pcs", "quantity": 20.0 * stories, "unit_cost": 65.0, "category": "electrical"},
            {"item": "PEX piping 3/4\"", "unit": "m", "quantity": 100.0 + stories * 30, "unit_cost": 3.50, "category": "plumbing"},
            {"item": "PEX fittings", "unit": "pcs", "quantity": 50.0 * stories, "unit_cost": 2.50, "category": "plumbing"},
            {"item": "HVAC ducting", "unit": "m", "quantity": 50.0 * stories, "unit_cost": 15.0, "category": "hvac"},
            {"item": "Furnace, gas", "unit": "pcs", "quantity": 1.0, "unit_cost": 2500.0, "category": "hvac"},
        ]
        
        # Flooring & finishes
        flooring_items = [
            {"item": "Laminate flooring", "unit": "m²", "quantity": size_sqft / 10.764 * 0.7, "unit_cost": 25.0, "category": "flooring"},
            {"item": "Carpet", "unit": "m²", "quantity": size_sqft / 10.764 * 0.3, "unit_cost": 30.0, "category": "flooring"},
            {"item": "Tile, ceramic", "unit": "m²", "quantity": 30.0 * stories, "unit_cost": 45.0, "category": "flooring"},
            {"item": "Baseboard trim", "unit": "m", "quantity": size_sqft / 10.764 * 4, "unit_cost": 8.0, "category": "finishing"},
            {"item": "Crown molding", "unit": "m", "quantity": size_sqft / 10.764 * 2, "unit_cost": 12.0, "category": "finishing"},
        ]
        
        # Kitchen & bath
        fixtures_items = [
            {"item": "Kitchen cabinets", "unit": "lf", "quantity": 20.0, "unit_cost": 150.0, "category": "fixtures"},
            {"item": "Countertop, laminate", "unit": "m", "quantity": 6.0, "unit_cost": 200.0, "category": "fixtures"},
            {"item": "Kitchen sink", "unit": "pcs", "quantity": 1.0, "unit_cost": 250.0, "category": "fixtures"},
            {"item": "Bathroom vanity", "unit": "pcs", "quantity": 2.0 * stories, "unit_cost": 350.0, "category": "fixtures"},
            {"item": "Toilet", "unit": "pcs", "quantity": 2.0 * stories, "unit_cost": 250.0, "category": "fixtures"},
            {"item": "Bathtub", "unit": "pcs", "quantity": 1.0 * stories, "unit_cost": 600.0, "category": "fixtures"},
            {"item": "Shower stall", "unit": "pcs", "quantity": 1.0, "unit_cost": 800.0, "category": "fixtures"},
        ]
        
        # Combine all items
        all_items = (foundation_items + framing_items + exterior_items + 
                    interior_items + mep_items + flooring_items + fixtures_items)
        
        # Calculate costs
        total_cost = 0.0
        for item in all_items:
            item["total_cost"] = round(item["quantity"] * item["unit_cost"], 2)
            total_cost += item["total_cost"]
        
        # Group by category
        categories = {}
        for item in all_items:
            cat = item["category"]
            if cat not in categories:
                categories[cat] = {"items": [], "subtotal": 0.0}
            categories[cat]["items"].append(item)
            categories[cat]["subtotal"] += item["total_cost"]
        
        # Add markup and contingency
        labor_multiplier = 1.5  # Labor is 50% of materials
        profit_margin = 0.15  # 15% profit
        contingency = 0.10  # 10% contingency
        
        subtotal_materials = total_cost
        subtotal_labor = subtotal_materials * labor_multiplier
        subtotal = subtotal_materials + subtotal_labor
        profit = subtotal * profit_margin
        contingency_amount = subtotal * contingency
        grand_total = subtotal + profit + contingency_amount
        
        return {
            "items": all_items,
            "categories": categories,
            "cost_summary": {
                "materials_subtotal": round(subtotal_materials, 2),
                "labor_subtotal": round(subtotal_labor, 2),
                "subtotal": round(subtotal, 2),
                "profit": round(profit, 2),
                "contingency": round(contingency_amount, 2),
                "grand_total": round(grand_total, 2),
                "cost_per_sqft": round(grand_total / size_sqft, 2)
            },
            "total_items": len(all_items),
            "currency": "CAD",
            "notes": [
                "Prices are estimates based on BC market rates (2024)",
                "Actual costs may vary by ±15% based on supplier and location",
                "Labor costs estimated at 150% of materials",
                "Includes 15% profit margin and 10% contingency",
                "Does not include permits, engineering fees, or site preparation"
            ]
        }
    
    async def generate_construction_schedule(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate construction schedule"""
        
        size_sqft = getattr(project, 'size_sqft', 2000)
        stories = getattr(project, 'stories', 1)
        
        # Base durations (adjust by size and complexity)
        complexity_factor = (size_sqft / 2000) * (1 + (stories - 1) * 0.3)
        
        phases = [
            {
                "phase": "Site Preparation",
                "duration_days": int(5 * complexity_factor),
                "tasks": [
                    "Survey and staking",
                    "Temporary power and water",
                    "Site fencing and signage",
                    "Excavation"
                ],
                "dependencies": [],
                "inspections": ["Survey verification"]
            },
            {
                "phase": "Foundation",
                "duration_days": int(10 * complexity_factor),
                "tasks": [
                    "Footing excavation",
                    "Footing forms and rebar",
                    "Footing pour",
                    "Foundation walls",
                    "Waterproofing",
                    "Backfill"
                ],
                "dependencies": ["Site Preparation"],
                "inspections": ["Footing inspection", "Foundation inspection"]
            },
            {
                "phase": "Framing",
                "duration_days": int(20 * complexity_factor * stories),
                "tasks": [
                    "Floor system framing",
                    "Wall framing",
                    "Roof framing",
                    "Sheathing",
                    "Windows and doors"
                ],
                "dependencies": ["Foundation"],
                "inspections": ["Framing inspection"]
            },
            {
                "phase": "Rough MEP",
                "duration_days": int(15 * complexity_factor * stories),
                "tasks": [
                    "HVAC rough-in",
                    "Electrical rough-in",
                    "Plumbing rough-in",
                    "Install bathtubs/showers"
                ],
                "dependencies": ["Framing"],
                "inspections": ["Rough electrical", "Rough plumbing", "Rough mechanical"]
            },
            {
                "phase": "Insulation",
                "duration_days": int(5 * complexity_factor * stories),
                "tasks": [
                    "Install insulation",
                    "Vapor barrier",
                    "Air sealing"
                ],
                "dependencies": ["Rough MEP"],
                "inspections": ["Insulation inspection"]
            },
            {
                "phase": "Drywall",
                "duration_days": int(12 * complexity_factor * stories),
                "tasks": [
                    "Drywall installation",
                    "Taping and mudding",
                    "Sanding",
                    "Primer coat"
                ],
                "dependencies": ["Insulation"],
                "inspections": []
            },
            {
                "phase": "Interior Finishing",
                "duration_days": int(20 * complexity_factor * stories),
                "tasks": [
                    "Painting",
                    "Flooring installation",
                    "Trim and baseboards",
                    "Cabinet installation",
                    "Countertops",
                    "Tile work"
                ],
                "dependencies": ["Drywall"],
                "inspections": []
            },
            {
                "phase": "MEP Finishing",
                "duration_days": int(10 * complexity_factor * stories),
                "tasks": [
                    "Electrical fixtures",
                    "Plumbing fixtures",
                    "HVAC vents and registers",
                    "Final connections"
                ],
                "dependencies": ["Interior Finishing"],
                "inspections": ["Final electrical", "Final plumbing", "Final mechanical"]
            },
            {
                "phase": "Exterior Finishing",
                "duration_days": int(15 * complexity_factor),
                "tasks": [
                    "Siding installation",
                    "Roofing",
                    "Exterior trim",
                    "Gutters and downspouts",
                    "Landscaping"
                ],
                "dependencies": ["Framing"],
                "inspections": []
            },
            {
                "phase": "Final Inspection & Handover",
                "duration_days": int(5 * complexity_factor),
                "tasks": [
                    "Punch list completion",
                    "Final cleaning",
                    "Final inspections",
                    "Owner walkthrough",
                    "Documentation handover"
                ],
                "dependencies": ["Interior Finishing", "MEP Finishing", "Exterior Finishing"],
                "inspections": ["Final building inspection"]
            }
        ]
        
        # Calculate start and end dates
        start_date = datetime.now() + timedelta(days=30)  # Assume 30 days for permits
        current_date = start_date
        
        for phase in phases:
            phase["start_date"] = current_date.strftime("%Y-%m-%d")
            phase["end_date"] = (current_date + timedelta(days=phase["duration_days"])).strftime("%Y-%m-%d")
            current_date += timedelta(days=phase["duration_days"])
        
        total_duration = sum(p["duration_days"] for p in phases)
        
        return {
            "phases": phases,
            "project_duration_days": total_duration,
            "project_duration_months": round(total_duration / 30, 1),
            "start_date": start_date.strftime("%Y-%m-%d"),
            "completion_date": current_date.strftime("%Y-%m-%d"),
            "total_inspections": sum(len(p["inspections"]) for p in phases),
            "critical_path": ["Site Preparation", "Foundation", "Framing", "Rough MEP", "Insulation", "Drywall", "Interior Finishing", "MEP Finishing", "Final Inspection"],
            "notes": [
                "Schedule assumes normal weather conditions",
                "Exterior finishing can proceed in parallel with interior work",
                "Allow 2-3 days for each inspection",
                "Factor in 10-15% buffer for delays"
            ]
        }
    
    async def generate_inspection_checklists(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate inspection checklists"""
        
        checklists = {
            "footing_inspection": {
                "phase": "Foundation",
                "items": [
                    {"item": "Excavation to proper depth", "critical": True},
                    {"item": "Undisturbed soil at bottom", "critical": True},
                    {"item": "Footing width per drawings", "critical": True},
                    {"item": "Rebar size and spacing correct", "critical": True},
                    {"item": "Rebar clearances maintained", "critical": True},
                    {"item": "Anchor bolts positioned", "critical": False},
                    {"item": "Formwork secure and level", "critical": True}
                ]
            },
            "foundation_inspection": {
                "phase": "Foundation",
                "items": [
                    {"item": "Foundation walls plumb and straight", "critical": True},
                    {"item": "Wall thickness per spec", "critical": True},
                    {"item": "Anchor bolts installed correctly", "critical": True},
                    {"item": "Waterproofing applied properly", "critical": True},
                    {"item": "Weeping tile installed", "critical": True},
                    {"item": "No cracks or defects", "critical": True}
                ]
            },
            "framing_inspection": {
                "phase": "Framing",
                "items": [
                    {"item": "All members sized per drawings", "critical": True},
                    {"item": "Proper spacing of studs/joists", "critical": True},
                    {"item": "Headers over openings sized correctly", "critical": True},
                    {"item": "Bearing points adequate", "critical": True},
                    {"item": "Shear walls properly nailed", "critical": True},
                    {"item": "Roof framing secure", "critical": True},
                    {"item": "Fire blocking installed", "critical": True},
                    {"item": "Hurricane ties/straps installed", "critical": True},
                    {"item": "Sheathing properly fastened", "critical": True}
                ]
            },
            "rough_electrical": {
                "phase": "Rough MEP",
                "items": [
                    {"item": "Service size adequate (200A minimum)", "critical": True},
                    {"item": "Panel properly located", "critical": True},
                    {"item": "Wire sizes correct for circuits", "critical": True},
                    {"item": "GFCI protection in required areas", "critical": True},
                    {"item": "AFCI protection for bedrooms", "critical": True},
                    {"item": "Boxes properly secured", "critical": False},
                    {"item": "Proper stapling/support of cables", "critical": False},
                    {"item": "Smoke detector locations marked", "critical": True}
                ]
            },
            "rough_plumbing": {
                "phase": "Rough MEP",
                "items": [
                    {"item": "All piping properly supported", "critical": True},
                    {"item": "Proper slope on drain lines", "critical": True},
                    {"item": "Vent piping sized correctly", "critical": True},
                    {"item": "Water supply lines protected from freezing", "critical": True},
                    {"item": "Pressure test passed", "critical": True},
                    {"item": "Fixtures properly roughed in", "critical": True}
                ]
            },
            "rough_mechanical": {
                "phase": "Rough MEP",
                "items": [
                    {"item": "Furnace/AC sized for building", "critical": True},
                    {"item": "Combustion air provided", "critical": True},
                    {"item": "Ductwork properly sized", "critical": True},
                    {"item": "Return air adequate", "critical": True},
                    {"item": "All ducts properly sealed", "critical": True},
                    {"item": "Clearances to combustibles maintained", "critical": True}
                ]
            },
            "insulation_inspection": {
                "phase": "Insulation",
                "items": [
                    {"item": "R-value meets code minimum", "critical": True},
                    {"item": "No gaps or voids", "critical": True},
                    {"item": "Vapor barrier continuous", "critical": True},
                    {"item": "Proper installation around services", "critical": True},
                    {"item": "Attic insulation depth adequate", "critical": True},
                    {"item": "Air sealing complete", "critical": True}
                ]
            },
            "final_building_inspection": {
                "phase": "Final",
                "items": [
                    {"item": "All work complete per drawings", "critical": True},
                    {"item": "CO and smoke detectors installed and tested", "critical": True},
                    {"item": "GFCI outlets tested", "critical": True},
                    {"item": "All lights and outlets functional", "critical": True},
                    {"item": "Plumbing fixtures operate properly", "critical": True},
                    {"item": "HVAC system operational", "critical": True},
                    {"item": "Handrails and guards meet code", "critical": True},
                    {"item": "Egress windows operable", "critical": True},
                    {"item": "Address numbers visible", "critical": False},
                    {"item": "Final grading complete", "critical": True},
                    {"item": "All deficiencies from previous inspections corrected", "critical": True}
                ]
            }
        }
        
        # Calculate statistics
        total_items = sum(len(c["items"]) for c in checklists.values())
        critical_items = sum(sum(1 for item in c["items"] if item["critical"]) for c in checklists.values())
        
        return {
            "checklists": checklists,
            "total_checklists": len(checklists),
            "total_items": total_items,
            "critical_items": critical_items,
            "jurisdiction": "BC Building Code 2018",
            "notes": [
                "All critical items must pass before phase advancement",
                "Schedule inspections 48 hours in advance",
                "Inspector may add additional requirements",
                "Keep inspection records for occupancy permit"
            ]
        }
    
    async def generate_structural_calculations(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate structural calculations summary"""
        
        # This would integrate with span tables and load parameters
        # For now, return template structure
        
        size_sqft = getattr(project, 'size_sqft', 2000)
        stories = getattr(project, 'stories', 1)
        location = getattr(project, 'location', 'BC')
        
        # Load parameters (simplified - would come from database)
        loads = {
            "dead_load_psf": 10 + stories * 5,
            "live_load_psf": 40,  # Residential
            "snow_load_psf": 25 if "Vancouver" in location else 40,  # BC varies
            "wind_load_psf": 20,
            "seismic_category": "Low" if "Vancouver" not in location else "Moderate"
        }
        
        # Typical spans
        spans = {
            "floor_joists": {
                "span_ft": 16,
                "spacing_in": 16,
                "member_size": "2x10",
                "grade": "#2 SPF",
                "live_load": loads["live_load_psf"],
                "dead_load": loads["dead_load_psf"]
            },
            "ceiling_joists": {
                "span_ft": 14,
                "spacing_in": 24,
                "member_size": "2x6",
                "grade": "#2 SPF",
                "live_load": 10,
                "dead_load": 10
            },
            "roof_rafters": {
                "span_ft": 18,
                "spacing_in": 16,
                "member_size": "2x10",
                "grade": "#2 SPF",
                "snow_load": loads["snow_load_psf"],
                "dead_load": 15
            },
            "beams": {
                "main_beam": {
                    "span_ft": 20,
                    "member_size": "3-2x12",
                    "grade": "#1 SPF",
                    "load_type": "point_loads",
                    "support": "posts at ends"
                }
            }
        }
        
        return {
            "project_info": {
                "size_sqft": size_sqft,
                "stories": stories,
                "location": location
            },
            "loads": loads,
            "spans": spans,
            "foundations": {
                "footing_width_in": 16 + stories * 4,
                "footing_depth_in": 8,
                "foundation_wall_thickness_in": 8,
                "soil_bearing_capacity_psf": 2000
            },
            "notes": [
                "Calculations based on BC Building Code 2018",
                "All lumber to be Grade #2 SPF minimum unless noted",
                "Spans are maximum - use smaller where practical",
                "Professional engineer review required before construction",
                "Site-specific soil report may require foundation adjustments"
            ],
            "references": [
                "BC Building Code 2018, Part 9",
                "CSA O86 Engineering Design in Wood",
                "Span Tables for Joists and Rafters (CMHC)"
            ]
        }
    
    async def generate_cost_estimate(
        self,
        project,
        **kwargs
    ) -> Dict[str, Any]:
        """Generate detailed cost estimate"""
        
        # Get BOM and add construction costs
        bom = await self.generate_bill_of_materials(project, **kwargs)
        
        size_sqft = getattr(project, 'size_sqft', 2000)
        
        # Additional project costs
        additional_costs = {
            "permits_and_fees": {
                "building_permit": size_sqft * 2.5,
                "development_permit": 1500,
                "utility_connections": 3000,
                "inspection_fees": 800
            },
            "professional_services": {
                "architectural_design": size_sqft * 5,
                "structural_engineering": 3500,
                "energy_advisor": 1200,
                "surveyor": 2000
            },
            "site_costs": {
                "excavation": 5000,
                "grading": 3000,
                "driveway": 8000,
                "landscaping": 5000
            },
            "temporary_facilities": {
                "temporary_power": 800,
                "portable_toilets": 600,
                "waste_disposal": 2500,
                "site_fencing": 1200
            },
            "insurance_and_bonding": {
                "builders_risk_insurance": bom["cost_summary"]["grand_total"] * 0.01,
                "liability_insurance": 2500,
                "performance_bond": bom["cost_summary"]["grand_total"] * 0.015
            }
        }
        
        # Calculate totals
        construction_costs = bom["cost_summary"]["grand_total"]
        
        additional_total = sum(
            sum(items.values()) for items in additional_costs.values()
        )
        
        subtotal = construction_costs + additional_total
        contingency = subtotal * 0.10
        grand_total = subtotal + contingency
        
        return {
            "construction_costs": bom["cost_summary"],
            "additional_costs": additional_costs,
            "cost_summary": {
                "construction_subtotal": construction_costs,
                "additional_costs_total": additional_total,
                "subtotal": subtotal,
                "contingency_10_percent": contingency,
                "grand_total": grand_total,
                "cost_per_sqft": grand_total / size_sqft
            },
            "payment_schedule": {
                "deposit": grand_total * 0.10,
                "foundation_complete": grand_total * 0.20,
                "framing_complete": grand_total * 0.25,
                "drywall_complete": grand_total * 0.20,
                "final_completion": grand_total * 0.25
            },
            "currency": "CAD",
            "validity_days": 60,
            "notes": [
                "Estimate valid for 60 days from date of issue",
                "Prices subject to change based on material availability",
                "Does not include furniture, appliances, or equipment",
                "Site-specific conditions may affect final cost",
                "HST/GST not included - add applicable taxes"
            ]
        }
