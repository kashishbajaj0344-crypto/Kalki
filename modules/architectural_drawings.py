"""
Architectural Drawing Generator
===============================
Generates professional architectural drawings including:
- Floor plans (all levels)
- Elevations (North, South, East, West)
- Building sections
- Site plans
- Detail drawings
"""

import os
import json
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import math

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import Rectangle, Circle, FancyBboxPatch, Polygon, Wedge
    from matplotlib.lines import Line2D
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False


class ArchitecturalDrawingGenerator:
    """Generate professional architectural drawings"""
    
    def __init__(self):
        self.scale = 1/48  # 1/4" = 1'-0" scale (1:48)
        self.wall_thickness = 6  # inches
        self.door_width = 36  # inches
        self.window_height = 48  # inches
        self.ceiling_height = 108  # inches (9 feet)
        
    def generate_complete_set(self, building_data: Dict[str, Any], output_dir: Path) -> List[str]:
        """Generate complete architectural drawing set"""
        
        if not MATPLOTLIB_AVAILABLE or not NUMPY_AVAILABLE:
            return []
        
        drawings = []
        
        # Extract building parameters
        width_ft = building_data.get("width_ft", 30)
        depth_ft = building_data.get("depth_ft", 50)
        levels = building_data.get("levels", 3)
        
        # Generate floor plans for each level
        for level in range(1, levels + 1):
            floor_plan_path = self._generate_floor_plan(
                width_ft, depth_ft, level, levels, output_dir
            )
            if floor_plan_path:
                drawings.append(floor_plan_path)
        
        # Generate elevations
        elevations = self._generate_elevations(width_ft, depth_ft, levels, output_dir)
        drawings.extend(elevations)
        
        # Generate building sections
        sections = self._generate_sections(width_ft, depth_ft, levels, output_dir)
        drawings.extend(sections)
        
        # Generate site plan
        site_plan = self._generate_site_plan(width_ft, depth_ft, output_dir)
        if site_plan:
            drawings.append(site_plan)
        
        return drawings
    
    def _generate_floor_plan(self, width_ft: float, depth_ft: float, 
                            level: int, total_levels: int, output_dir: Path) -> Optional[str]:
        """Generate floor plan for a specific level"""
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_aspect('equal')
        
        # Convert to inches
        width = width_ft * 12
        depth = depth_ft * 12
        
        # Draw exterior walls
        self._draw_wall(ax, 0, 0, width, 0, exterior=True)  # Bottom
        self._draw_wall(ax, width, 0, width, depth, exterior=True)  # Right
        self._draw_wall(ax, width, depth, 0, depth, exterior=True)  # Top
        self._draw_wall(ax, 0, depth, 0, 0, exterior=True)  # Left
        
        # Interior layout based on level
        if level == 1:
            # Ground floor: Entry, living, dining, kitchen, powder room
            self._draw_ground_floor_layout(ax, width, depth)
        elif level == 2:
            # Second floor: Bedrooms, bathrooms
            self._draw_second_floor_layout(ax, width, depth)
        else:
            # Upper floors: Additional bedrooms or open space
            self._draw_upper_floor_layout(ax, width, depth, level)
        
        # Add dimensions
        self._add_dimensions(ax, width, depth)
        
        # Add title block
        self._add_title_block(ax, f"FLOOR PLAN - LEVEL {level}", 
                             f"A-{100 + level}", width, depth)
        
        # Add scale and north arrow
        self._add_scale_bar(ax, 0, -width * 0.15)
        self._add_north_arrow(ax, width * 0.9, -width * 0.1)
        
        # Add room labels
        if level == 1:
            self._add_room_labels_ground(ax, width, depth)
        elif level == 2:
            self._add_room_labels_second(ax, width, depth)
        
        # Set limits and remove axes
        ax.set_xlim(-width * 0.1, width * 1.1)
        ax.set_ylim(-width * 0.2, depth * 1.1)
        ax.axis('off')
        
        # Save
        output_path = output_dir / f"Floor_Plan_Level_{level}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def _generate_elevations(self, width_ft: float, depth_ft: float, 
                            levels: int, output_dir: Path) -> List[str]:
        """Generate all building elevations"""
        
        elevations = []
        
        # Front (South) Elevation
        front = self._generate_elevation(width_ft, levels, "SOUTH ELEVATION", 
                                        "front", output_dir)
        if front:
            elevations.append(front)
        
        # Rear (North) Elevation
        rear = self._generate_elevation(width_ft, levels, "NORTH ELEVATION", 
                                       "rear", output_dir)
        if rear:
            elevations.append(rear)
        
        # Right (East) Elevation
        right = self._generate_elevation(depth_ft, levels, "EAST ELEVATION", 
                                        "right", output_dir)
        if right:
            elevations.append(right)
        
        # Left (West) Elevation
        left = self._generate_elevation(depth_ft, levels, "WEST ELEVATION", 
                                       "left", output_dir)
        if left:
            elevations.append(left)
        
        return elevations
    
    def _generate_elevation(self, width_ft: float, levels: int, 
                          title: str, side: str, output_dir: Path) -> Optional[str]:
        """Generate a single elevation drawing"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect('equal')
        
        width = width_ft * 12
        height = levels * self.ceiling_height
        
        # Draw building outline
        building = Rectangle((0, 0), width, height, 
                            fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(building)
        
        # Add floor lines
        for level in range(1, levels):
            y = level * self.ceiling_height
            ax.plot([0, width], [y, y], 'k-', linewidth=1)
        
        # Add windows and doors based on side
        if side == "front":
            # Entry door on ground floor
            door_x = width / 2 - self.door_width / 2
            self._draw_door_elevation(ax, door_x, 0, self.door_width, 84)
            
            # Windows on each floor
            for level in range(levels):
                y_base = level * self.ceiling_height
                # 3 windows per floor
                for i in range(3):
                    x = width * (0.2 + i * 0.3)
                    self._draw_window_elevation(ax, x, y_base + 24, 48, 48)
        else:
            # Windows on other sides
            for level in range(levels):
                y_base = level * self.ceiling_height
                num_windows = 2 if width < 500 else 3
                for i in range(num_windows):
                    x = width * (0.25 + i * 0.4) if num_windows == 2 else width * (0.2 + i * 0.3)
                    self._draw_window_elevation(ax, x, y_base + 24, 48, 48)
        
        # Add roof
        roof_points = np.array([
            [0, height],
            [width/2, height + width * 0.15],
            [width, height],
            [width, height],
            [0, height]
        ])
        roof = Polygon(roof_points, fill=False, edgecolor='black', linewidth=2)
        ax.add_patch(roof)
        
        # Add foundation line
        ax.plot([0, width], [-12, -12], 'k-', linewidth=2)
        
        # Add grade line
        ax.plot([-width*0.1, width*1.1], [0, 0], 'k--', linewidth=1)
        ax.text(-width*0.05, -6, "GRADE", fontsize=8, ha='right')
        
        # Add height dimensions
        self._add_elevation_dimensions(ax, width, height, levels)
        
        # Add title block
        self._add_title_block(ax, title, f"A-{200 + ['front', 'rear', 'right', 'left'].index(side)}", 
                             width, height)
        
        # Add scale
        self._add_scale_bar(ax, 0, -width * 0.15)
        
        ax.set_xlim(-width * 0.15, width * 1.15)
        ax.set_ylim(-width * 0.2, height * 1.2)
        ax.axis('off')
        
        output_path = output_dir / f"Elevation_{title.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def _generate_sections(self, width_ft: float, depth_ft: float, 
                          levels: int, output_dir: Path) -> List[str]:
        """Generate building sections"""
        
        sections = []
        
        # Longitudinal section
        section = self._generate_section(width_ft, levels, "LONGITUDINAL SECTION", 
                                        "longitudinal", output_dir)
        if section:
            sections.append(section)
        
        # Transverse section
        section = self._generate_section(depth_ft, levels, "TRANSVERSE SECTION", 
                                        "transverse", output_dir)
        if section:
            sections.append(section)
        
        return sections
    
    def _generate_section(self, width_ft: float, levels: int, 
                         title: str, direction: str, output_dir: Path) -> Optional[str]:
        """Generate a building section"""
        
        fig, ax = plt.subplots(figsize=(16, 10))
        ax.set_aspect('equal')
        
        width = width_ft * 12
        height = levels * self.ceiling_height
        
        # Draw exterior walls (cut through)
        wall_pattern = Rectangle((0, 0), self.wall_thickness, height,
                                fill=True, facecolor='black', edgecolor='black')
        ax.add_patch(wall_pattern)
        
        wall_pattern2 = Rectangle((width - self.wall_thickness, 0), self.wall_thickness, height,
                                 fill=True, facecolor='black', edgecolor='black')
        ax.add_patch(wall_pattern2)
        
        # Draw floor slabs
        for level in range(levels + 1):
            y = level * self.ceiling_height
            if level == 0:
                # Foundation/ground floor slab (thicker)
                floor = Rectangle((0, y - 8), width, 8,
                                fill=True, facecolor='gray', edgecolor='black')
            else:
                # Upper floor slabs
                floor = Rectangle((0, y - 6), width, 6,
                                fill=True, facecolor='gray', edgecolor='black')
            ax.add_patch(floor)
        
        # Draw interior spaces
        for level in range(levels):
            y_base = level * self.ceiling_height
            # Interior floor space
            interior = Rectangle((self.wall_thickness, y_base), 
                                width - 2 * self.wall_thickness, self.ceiling_height - 6,
                                fill=False, edgecolor='black', linewidth=0.5, linestyle='--')
            ax.add_patch(interior)
            
            # Ceiling line
            ax.plot([self.wall_thickness, width - self.wall_thickness],
                   [y_base + self.ceiling_height - 6, y_base + self.ceiling_height - 6],
                   'k--', linewidth=0.5)
        
        # Add roof structure
        roof_thickness = 10
        roof = Rectangle((0, height), width, roof_thickness,
                        fill=True, facecolor='lightgray', edgecolor='black', linewidth=2)
        ax.add_patch(roof)
        
        # Add foundation
        foundation = Rectangle((- width * 0.05, -24), width * 1.1, 24,
                              fill=True, facecolor='darkgray', edgecolor='black', linewidth=2)
        ax.add_patch(foundation)
        
        # Add height dimensions and labels
        for level in range(levels + 1):
            y = level * self.ceiling_height
            # Dimension line
            ax.plot([-width * 0.08, -width * 0.04], [y, y], 'k-', linewidth=0.5)
            # Level label
            if level > 0:
                ax.text(-width * 0.1, y - self.ceiling_height/2, 
                       f"LEVEL {level}\n{self.ceiling_height/12:.0f}'-0\"",
                       ha='right', va='center', fontsize=8)
        
        # Dimension lines
        ax.annotate('', xy=(-width * 0.06, height), xytext=(-width * 0.06, 0),
                   arrowprops=dict(arrowstyle='<->', lw=1))
        ax.text(-width * 0.12, height/2, f"TOTAL HEIGHT\n{height/12:.0f}'-0\"",
               ha='center', va='center', rotation=90, fontsize=9)
        
        # Add title block
        self._add_title_block(ax, title, 
                             f"A-{300 + (0 if direction == 'longitudinal' else 1)}", 
                             width, height)
        
        # Add scale
        self._add_scale_bar(ax, 0, -width * 0.15)
        
        ax.set_xlim(-width * 0.2, width * 1.15)
        ax.set_ylim(-width * 0.2, height * 1.2)
        ax.axis('off')
        
        output_path = output_dir / f"Section_{title.replace(' ', '_')}.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    def _generate_site_plan(self, width_ft: float, depth_ft: float, 
                           output_dir: Path) -> Optional[str]:
        """Generate site plan"""
        
        fig, ax = plt.subplots(figsize=(14, 14))
        ax.set_aspect('equal')
        
        # Site dimensions (building + setbacks)
        site_width = width_ft * 12 * 2
        site_depth = depth_ft * 12 * 2
        
        # Property lines
        property_rect = Rectangle((0, 0), site_width, site_depth,
                                 fill=False, edgecolor='black', 
                                 linewidth=2, linestyle='--')
        ax.add_patch(property_rect)
        
        # Building footprint (centered)
        building_x = (site_width - width_ft * 12) / 2
        building_y = (site_depth - depth_ft * 12) / 2
        
        building = Rectangle((building_x, building_y), 
                            width_ft * 12, depth_ft * 12,
                            fill=True, facecolor='lightgray', 
                            edgecolor='black', linewidth=2)
        ax.add_patch(building)
        
        # Driveway
        driveway_width = 120  # 10 feet
        driveway = Rectangle((building_x - driveway_width - 24, building_y + depth_ft * 6),
                            driveway_width, depth_ft * 6,
                            fill=True, facecolor='darkgray', edgecolor='black')
        ax.add_patch(driveway)
        
        # Walkway to front door
        walkway_width = 48  # 4 feet
        walkway_x = building_x + (width_ft * 12 - walkway_width) / 2
        walkway = Rectangle((walkway_x, 0), walkway_width, building_y,
                           fill=True, facecolor='tan', edgecolor='black')
        ax.add_patch(walkway)
        
        # Add landscaping (trees and shrubs)
        self._add_landscaping(ax, site_width, site_depth, building_x, building_y,
                             width_ft * 12, depth_ft * 12)
        
        # Add setback dimensions
        self._add_site_dimensions(ax, site_width, site_depth, building_x, building_y,
                                 width_ft * 12, depth_ft * 12)
        
        # Add title block
        self._add_title_block(ax, "SITE PLAN", "A-001", site_width, site_depth)
        
        # Add scale and north arrow
        self._add_scale_bar(ax, site_width * 0.05, site_depth * 0.05)
        self._add_north_arrow(ax, site_width * 0.9, site_depth * 0.9)
        
        # Legend
        self._add_site_legend(ax, site_width * 0.05, site_depth * 0.8)
        
        ax.set_xlim(-site_width * 0.1, site_width * 1.05)
        ax.set_ylim(-site_depth * 0.15, site_depth * 1.05)
        ax.axis('off')
        
        output_path = output_dir / "Site_Plan.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return str(output_path)
    
    # Helper drawing methods
    def _draw_wall(self, ax, x1, y1, x2, y2, exterior=False):
        """Draw a wall line"""
        thickness = self.wall_thickness if exterior else self.wall_thickness * 0.7
        # Draw wall as thick line
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=thickness/2)
    
    def _draw_door(self, ax, x, y, width, swing_direction='right'):
        """Draw a door in plan view"""
        # Door opening
        ax.plot([x, x + width], [y, y], 'k-', linewidth=1)
        
        # Door swing arc
        if swing_direction == 'right':
            arc = Wedge((x, y), width, 0, 90, 
                       fill=False, edgecolor='black', linewidth=0.5)
        else:
            arc = Wedge((x + width, y), width, 90, 180,
                       fill=False, edgecolor='black', linewidth=0.5)
        ax.add_patch(arc)
    
    def _draw_window(self, ax, x, y, width):
        """Draw a window in plan view"""
        # Window represented as line with marks
        ax.plot([x, x + width], [y, y], 'k-', linewidth=2)
        # Glass marks
        num_marks = max(2, int(width / 24))
        for i in range(num_marks + 1):
            mark_x = x + (width / num_marks) * i
            ax.plot([mark_x, mark_x], [y - 2, y + 2], 'k-', linewidth=0.5)
    
    def _draw_door_elevation(self, ax, x, y, width, height):
        """Draw door in elevation"""
        door = Rectangle((x, y), width, height,
                        fill=True, facecolor='white', edgecolor='black', linewidth=1.5)
        ax.add_patch(door)
        # Door panels
        panel_height = height / 2
        ax.plot([x + 4, x + width - 4], [y + panel_height, y + panel_height], 
               'k-', linewidth=0.5)
    
    def _draw_window_elevation(self, ax, x, y, width, height):
        """Draw window in elevation"""
        window = Rectangle((x, y), width, height,
                          fill=True, facecolor='lightblue', edgecolor='black', linewidth=1)
        ax.add_patch(window)
        # Mullions
        ax.plot([x + width/2, x + width/2], [y, y + height], 'k-', linewidth=0.5)
        ax.plot([x, x + width], [y + height/2, y + height/2], 'k-', linewidth=0.5)
    
    def _draw_ground_floor_layout(self, ax, width, depth):
        """Draw ground floor interior walls and features"""
        # Entry foyer
        self._draw_wall(ax, width * 0.3, 0, width * 0.3, depth * 0.25, exterior=False)
        self._draw_door(ax, width * 0.45, 0, self.door_width, 'right')
        
        # Living/Dining separation
        self._draw_wall(ax, 0, depth * 0.6, width * 0.5, depth * 0.6, exterior=False)
        
        # Kitchen wall
        self._draw_wall(ax, width * 0.65, depth * 0.4, width * 0.65, depth, exterior=False)
        self._draw_door(ax, width * 0.65, depth * 0.65, self.door_width, 'left')
        
        # Powder room
        self._draw_wall(ax, width * 0.75, depth * 0.75, width, depth * 0.75, exterior=False)
        self._draw_wall(ax, width * 0.75, depth * 0.75, width * 0.75, depth, exterior=False)
        self._draw_door(ax, width * 0.75, depth * 0.8, 30, 'right')
        
        # Windows
        self._draw_window(ax, width * 0.1, 0, 60)
        self._draw_window(ax, width * 0.1, depth, 60)
        self._draw_window(ax, 0, depth * 0.3, 60)
        self._draw_window(ax, width, depth * 0.3, 60)
    
    def _draw_second_floor_layout(self, ax, width, depth):
        """Draw second floor interior walls"""
        # Master bedroom
        self._draw_wall(ax, 0, depth * 0.55, width * 0.6, depth * 0.55, exterior=False)
        self._draw_door(ax, width * 0.25, depth * 0.55, self.door_width, 'right')
        
        # Master bathroom
        self._draw_wall(ax, width * 0.45, depth * 0.55, width * 0.45, depth * 0.8, exterior=False)
        self._draw_wall(ax, width * 0.45, depth * 0.8, width * 0.6, depth * 0.8, exterior=False)
        self._draw_door(ax, width * 0.45, depth * 0.65, 30, 'left')
        
        # Bedroom 2
        self._draw_wall(ax, width * 0.6, 0, width * 0.6, depth * 0.45, exterior=False)
        self._draw_door(ax, width * 0.6, depth * 0.2, 30, 'right')
        
        # Shared bathroom
        self._draw_wall(ax, width * 0.75, 0, width * 0.75, depth * 0.3, exterior=False)
        self._draw_wall(ax, width * 0.6, depth * 0.3, width, depth * 0.3, exterior=False)
        self._draw_door(ax, width * 0.8, depth * 0.3, 30, 'left')
    
    def _draw_upper_floor_layout(self, ax, width, depth, level):
        """Draw upper floor layout"""
        # Open floor plan with optional dividers
        self._draw_wall(ax, width * 0.5, 0, width * 0.5, depth * 0.4, exterior=False)
        self._draw_door(ax, width * 0.5, depth * 0.15, self.door_width, 'right')
    
    def _add_dimensions(self, ax, width, depth):
        """Add dimension lines and measurements"""
        offset = -30
        
        # Width dimension
        ax.annotate('', xy=(width, offset), xytext=(0, offset),
                   arrowprops=dict(arrowstyle='<->', lw=1))
        ax.text(width/2, offset - 10, f"{width/12:.0f}'-0\"", 
               ha='center', fontsize=9, weight='bold')
        
        # Depth dimension
        ax.annotate('', xy=(width + 30, depth), xytext=(width + 30, 0),
                   arrowprops=dict(arrowstyle='<->', lw=1))
        ax.text(width + 45, depth/2, f"{depth/12:.0f}'-0\"", 
               ha='left', va='center', rotation=90, fontsize=9, weight='bold')
    
    def _add_elevation_dimensions(self, ax, width, height, levels):
        """Add dimensions to elevation"""
        offset_x = width + width * 0.08
        
        # Total height
        ax.annotate('', xy=(offset_x, height), xytext=(offset_x, 0),
                   arrowprops=dict(arrowstyle='<->', lw=1))
        ax.text(offset_x + 20, height/2, f"{height/12:.0f}'-0\"\nTOTAL", 
               ha='left', va='center', rotation=90, fontsize=8)
        
        # Floor-to-floor heights
        for level in range(levels):
            y = level * self.ceiling_height
            ax.plot([width * 1.02, offset_x - 5], [y, y], 'k-', linewidth=0.5)
    
    def _add_site_dimensions(self, ax, site_width, site_depth, 
                            building_x, building_y, building_width, building_depth):
        """Add site setback dimensions"""
        # Front setback
        front_setback = building_y
        ax.annotate('', xy=(building_x + building_width/2, building_y), 
                   xytext=(building_x + building_width/2, 0),
                   arrowprops=dict(arrowstyle='<->', lw=1, color='red'))
        ax.text(building_x + building_width/2 + 30, building_y/2, 
               f"FRONT SETBACK\n{front_setback/12:.0f}'-0\"",
               ha='left', fontsize=8, color='red')
        
        # Side setback (left)
        side_setback = building_x
        ax.annotate('', xy=(building_x, building_y + building_depth/2),
                   xytext=(0, building_y + building_depth/2),
                   arrowprops=dict(arrowstyle='<->', lw=1, color='red'))
        ax.text(building_x/2, building_y + building_depth/2 + 30,
               f"SIDE SETBACK\n{side_setback/12:.0f}'-0\"",
               ha='center', fontsize=8, color='red')
    
    def _add_landscaping(self, ax, site_width, site_depth, building_x, building_y,
                        building_width, building_depth):
        """Add trees and landscaping symbols"""
        # Trees
        tree_positions = [
            (site_width * 0.1, site_depth * 0.2),
            (site_width * 0.9, site_depth * 0.2),
            (site_width * 0.1, site_depth * 0.8),
            (site_width * 0.9, site_depth * 0.8),
        ]
        
        for x, y in tree_positions:
            # Tree canopy
            circle = Circle((x, y), 36, fill=True, facecolor='lightgreen', 
                          edgecolor='darkgreen', linewidth=1)
            ax.add_patch(circle)
            # Trunk
            trunk = Circle((x, y), 6, fill=True, facecolor='brown', edgecolor='black')
            ax.add_patch(trunk)
    
    def _add_site_legend(self, ax, x, y):
        """Add site plan legend"""
        legend_items = [
            ("Property Line", "k--"),
            ("Building Footprint", "lightgray"),
            ("Driveway", "darkgray"),
            ("Walkway", "tan"),
        ]
        
        ax.text(x, y + 60, "LEGEND", fontsize=10, weight='bold')
        
        for idx, (label, style) in enumerate(legend_items):
            y_pos = y - idx * 25
            if isinstance(style, str) and style.startswith('light') or style in ['darkgray', 'tan']:
                rect = Rectangle((x, y_pos - 8), 30, 16, fill=True, 
                               facecolor=style, edgecolor='black')
                ax.add_patch(rect)
            else:
                ax.plot([x, x + 30], [y_pos, y_pos], style, linewidth=2)
            
            ax.text(x + 40, y_pos, label, fontsize=8, va='center')
    
    def _add_title_block(self, ax, title, drawing_number, width, height):
        """Add professional title block"""
        block_width = width * 0.3
        block_height = 100
        block_x = width - block_width
        block_y = -150
        
        # Title block border
        title_block = FancyBboxPatch((block_x, block_y), block_width, block_height,
                                     boxstyle="round,pad=5", 
                                     fill=True, facecolor='white',
                                     edgecolor='black', linewidth=2)
        ax.add_patch(title_block)
        
        # Title
        ax.text(block_x + block_width/2, block_y + 70, title,
               ha='center', fontsize=14, weight='bold')
        
        # Drawing number
        ax.text(block_x + 10, block_y + 45, f"DRAWING NO: {drawing_number}",
               fontsize=10, weight='bold')
        
        # Date and scale
        ax.text(block_x + 10, block_y + 25, f"DATE: {datetime.now().strftime('%Y-%m-%d')}",
               fontsize=8)
        ax.text(block_x + 10, block_y + 10, "SCALE: 1/4\" = 1'-0\"",
               fontsize=8)
        
        # Project info
        ax.text(block_x + block_width - 10, block_y + 10, 
               "KALKI AI DESIGN SYSTEM",
               ha='right', fontsize=7, style='italic')
    
    def _add_scale_bar(self, ax, x, y):
        """Add graphic scale bar"""
        scale_length = 120  # 10 feet at 1/4" scale
        
        # Scale bar
        ax.plot([x, x + scale_length], [y, y], 'k-', linewidth=3)
        
        # Tick marks
        for i in range(11):
            tick_x = x + (scale_length / 10) * i
            tick_height = 5 if i % 5 == 0 else 3
            ax.plot([tick_x, tick_x], [y - tick_height, y + tick_height], 'k-', linewidth=1)
        
        # Labels
        ax.text(x, y - 15, "0", ha='center', fontsize=7)
        ax.text(x + scale_length/2, y - 15, "5'", ha='center', fontsize=7)
        ax.text(x + scale_length, y - 15, "10'", ha='center', fontsize=7)
        
        ax.text(x + scale_length/2, y - 30, "GRAPHIC SCALE", 
               ha='center', fontsize=8, weight='bold')
    
    def _add_north_arrow(self, ax, x, y):
        """Add north arrow"""
        arrow_length = 50
        
        # Arrow
        ax.annotate('', xy=(x, y + arrow_length), xytext=(x, y),
                   arrowprops=dict(arrowstyle='->', lw=2, color='black'))
        
        # Circle
        circle = Circle((x, y), 15, fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(circle)
        
        # N label
        ax.text(x, y + arrow_length + 15, 'N', ha='center', va='bottom', 
               fontsize=12, weight='bold')
    
    def _add_room_labels_ground(self, ax, width, depth):
        """Add room labels for ground floor"""
        labels = [
            ("ENTRY", width * 0.45, depth * 0.12),
            ("LIVING\nROOM", width * 0.25, depth * 0.35),
            ("DINING\nROOM", width * 0.25, depth * 0.75),
            ("KITCHEN", width * 0.82, depth * 0.65),
            ("POWDER\nROOM", width * 0.87, depth * 0.87),
        ]
        
        for label, x, y in labels:
            ax.text(x, y, label, ha='center', va='center', 
                   fontsize=9, style='italic', weight='bold')
    
    def _add_room_labels_second(self, ax, width, depth):
        """Add room labels for second floor"""
        labels = [
            ("MASTER\nBEDROOM", width * 0.25, depth * 0.75),
            ("MASTER\nBATH", width * 0.52, depth * 0.67),
            ("BEDROOM 2", width * 0.8, depth * 0.2),
            ("BATH", width * 0.82, depth * 0.15),
            ("HALLWAY", width * 0.7, depth * 0.5),
        ]
        
        for label, x, y in labels:
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=9, style='italic', weight='bold')


def generate_architectural_drawings(building_specs: Dict[str, Any], output_dir: Path) -> List[str]:
    """
    Main entry point for generating architectural drawings
    
    Args:
        building_specs: Dictionary with building parameters
        output_dir: Output directory for drawings
        
    Returns:
        List of generated drawing file paths
    """
    generator = ArchitecturalDrawingGenerator()
    return generator.generate_complete_set(building_specs, output_dir)
