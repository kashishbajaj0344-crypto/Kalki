"""
FOUNDATION PHASE IMPLEMENTATION - Complete Step-by-Step Guidance
This module implements all 11 foundation steps with expert-level detail
"""

from modules.construction_copilot import NextStep
from typing import Dict, Any

def foundation_step_1_excavation(project_state) -> NextStep:
    """Step 1: Site Excavation"""
    return NextStep(
        step_number=1,
        title="Foundation Step 1: Excavate Building Site",
        description="""
🏗️ **EXCAVATION - Preparing Your Building Site**

**What You're Doing:**
Removing topsoil and digging down to solid bearing soil for your foundation.

**Step-by-Step Process:**

1. **CALL 811 FIRST** (CRITICAL!)
   - Call at least 3 business days before digging
   - Utility companies will mark underground lines
   - This is FREE and REQUIRED BY LAW
   - Failure to call can result in:
     * Hitting gas line (explosion risk)
     * Hitting electric (electrocution)
     * Hitting water (flooding)
     * $10,000+ fines

2. **Verify Building Setbacks**
   - Check your approved site plan
   - Measure from property lines
   - Ensure you're digging in correct location
   - Typical setbacks: 25ft front, 10ft sides, 20ft rear

3. **Mark Foundation Corners**
   - Use surveyor's stakes (should be set from site analysis)
   - Verify with tape measure (diagonal measurements should match)
   - Spray paint outline on ground
   - Add 2 feet around perimeter for working space

4. **Hire Excavator**
   - **PROFESSIONAL REQUIRED**: This is specialized equipment
   - Equipment: Track excavator (200-series) + dump truck
   - Cost: $150-250/hour (expect 4-8 hours for 2000 SF house)
   - Total estimated cost: $1,500-3,000

5. **Excavation Depth**
   - Footing bottom depth: Below frost line
     * Check local building code for frost depth
     * Typical: 12" (warm climates) to 48" (cold climates)
   - Add footing thickness (typically 12-18")
   - Example: 36" frost line + 12" footing = dig 48" deep

6. **Excavation Process**
   - Remove and stockpile topsoil (6-12" deep) - SAVE THIS!
   - Dig to bearing soil (clay, sand, or rock - NOT loose fill)
   - Grade excavation level (± 1")
   - Slope bottom for drainage (1% toward perimeter drain)

7. **Soil Verification**
   - Geotechnical engineer should verify bearing capacity
   - If soft soil encountered: May need deeper excavation
   - If rock encountered: May need blasting or jack-hammering
   - Undisturbed soil is critical - don't over-excavate

8. **Drainage Preparation**
   - Install perimeter drain pipes (4" perforated PVC)
   - Slope pipes 1/4" per foot minimum
   - Direct to daylight drain or sump pit
   - Wrap in filter fabric

9. **Gravel Base (if needed)**
   - Some soils require 6" compacted gravel base
   - Use 3/4" crushed rock
   - Compact in 3" lifts with plate compactor
   - Creates level, stable working surface

10. **Final Inspection**
    - Verify depth with tape measure (multiple locations)
    - Check level with 8-foot level
    - Ensure all organic matter removed (roots, topsoil)
    - Get building inspector approval (required in most areas)

**SAFETY WARNINGS:**
⚠️ NEVER enter trench deeper than 4 feet without shoring
⚠️ Keep machinery 10+ feet from trench edge (cave-in risk)
⚠️ Watch for groundwater - may need pumping/dewatering
⚠️ Check weather - rain can cause cave-ins
⚠️ Hard hat required when equipment operating

**SUCCESS CRITERIA:**
✅ Excavation matches building plans (dimensions ±3")
✅ Depth correct per frost line requirement
✅ Bottom is level and undisturbed soil
✅ Drainage system installed
✅ Inspector approved (if required)
✅ Working space clear of debris
""",
        why_now="Foundation must sit on solid, undisturbed soil below frost line. This prevents settling, cracking, and frost heave.",
        estimated_cost=2500.0,
        estimated_duration_days=1,
        requires_professional=True,
        professional_type="Excavation Contractor",
        requires_permit=True,
        permit_type="Building Permit (should already have from permitting phase)",
        safety_warnings=[
            "CALL 811 BEFORE DIGGING - Hitting utilities can be fatal",
            "Never enter excavation >4ft deep without shoring/sloping",
            "Keep equipment 10ft from trench edge",
            "Watch for groundwater and cave-in risks",
            "Hard hat required around heavy equipment"
        ],
        material_list=[
            {"item": "Perimeter drain pipe", "quantity": "Linear feet of perimeter + 20%", "unit": "4\" perforated PVC", "cost_per_unit": 2.50},
            {"item": "Filter fabric", "quantity": "Enough to wrap drain", "unit": "roll", "cost_per_unit": 45.0},
            {"item": "Drain gravel", "quantity": "1 yard per 40 LF", "unit": "cubic yard", "cost_per_unit": 35.0}
        ],
        tool_list=[
            "Spray paint (for marking)",
            "100-ft tape measure",
            "8-foot level",
            "Stakes and string line"
        ],
        reference_documents=[
            "Approved site plan",
            "Foundation drawings",
            "Geotechnical report (soil bearing capacity)",
            "Local building code (frost depth requirements)"
        ],
        video_tutorials=[
            "How to mark foundation corners",
            "Understanding excavation depths",
            "Perimeter drain installation"
        ],
        success_criteria=[
            "811 utility locate completed (keep ticket)",
            "Excavation dimensions match plans (±3 inches)",
            "Depth correct per frost line + footing thickness",
            "Bottom is level undisturbed soil",
            "Perimeter drains installed and sloped",
            "Building inspector approval obtained",
            "Topsoil stockpiled for later landscaping"
        ]
    )

def foundation_step_2_footing_layout(project_state) -> NextStep:
    """Step 2: Layout Footing Forms"""
    return NextStep(
        step_number=2,
        title="Foundation Step 2: Layout Footing Forms",
        description="""
📐 **FOOTING LAYOUT - Marking Where Concrete Goes**

**What You're Doing:**
Creating precise layout for footing forms using string lines and batter boards.

**Why Critical:**
If your footings are off by even 2 inches, your entire house will be crooked.
This is the most precision-critical step of the entire project.

**Step-by-Step Process:**

1. **Set Up Batter Boards**
   - Build batter boards 4-6 feet outside foundation corners
   - Use 2x4 stakes (3 feet long) driven 18" into ground
   - Attach 1x4 horizontal boards
   - Top should be level (use laser level or water level)
   - These are your reference points - don't disturb them!

2. **Establish Building Lines**
   - Run mason's string between batter boards
   - First line: Front wall of house
   - Second line: Side wall
   - Check for square using 3-4-5 method:
     * Measure 3ft along one line
     * Measure 4ft along perpendicular line
     * Diagonal should be exactly 5ft
   - For larger: 6-8-10 or 9-12-15

3. **Mark All Corners**
   - Where strings intersect = building corners
   - Use plumb bob to transfer to ground
   - Drive stake at each corner
   - Spray paint "X" on ground
   - Run diagonal measurements:
     * Measure corner A to corner C
     * Measure corner B to corner D
     * These MUST be equal (±1/4")

4. **Calculate Footing Width**
   - Typical residential: 16-20" wide footing
   - Load bearing capacity from geotech report
   - Formula: Footing width = (Load in pounds) ÷ (Soil capacity in PSF)
   - Your engineer should specify this
   - Common: 16" wide for 1-story, 20" for 2-story

5. **Mark Footing Edges**
   - From building line, measure half of footing width each direction
   - Example: 16" footing → 8" inside, 8" outside building line
   - Mark inner edge and outer edge with spray paint
   - Continuous line around entire perimeter

6. **Mark Interior Footings**
   - Load-bearing walls need footings
   - Columns need square footings (24"x24" typical)
   - Beam pockets (where beams sit on foundation wall)
   - Check structural drawings for all locations

7. **Verify Elevation**
   - All footings should be at same elevation
   - Use laser level or transit
   - Mark elevation on stakes
   - Top of stake = top of footing

8. **Final Check**
   - Double-check all dimensions
   - Verify corners are square (diagonals equal)
   - Verify footing widths
   - Take photos for your records
   - Have building inspector verify layout (some jurisdictions require)

**PRO TIPS:**
💡 Use a laser level - worth renting for $50/day
💡 Check square multiple times - errors here are expensive to fix
💡 Take photos of string layout - may need to re-establish if disturbed
💡 Don't trust tape measures alone - verify with diagonals

**SUCCESS CRITERIA:**
✅ All corners marked and squared
✅ Diagonal measurements equal (±1/4")
✅ Footing width marked on ground
✅ Elevations marked
✅ Interior footings located
✅ String lines taut and undisturbed
✅ Photos documented
""",
        why_now="Accurate layout ensures square, level foundation. Errors here compound through entire structure.",
        estimated_cost=150.0,  # Materials only, can DIY
        estimated_duration_days=1,
        requires_professional=False,
        professional_type=None,
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Watch for tripping hazards (strings, stakes)",
            "Be careful near excavation edge",
            "Wear safety glasses when driving stakes"
        ],
        material_list=[
            {"item": "2x4 lumber", "quantity": "12 pieces @ 8ft", "unit": "each", "cost_per_unit": 8.0},
            {"item": "1x4 lumber", "quantity": "8 pieces @ 8ft", "unit": "each", "cost_per_unit": 6.0},
            {"item": "Mason's string", "quantity": "1 roll", "unit": "1000 ft", "cost_per_unit": 25.0},
            {"item": "Stakes", "quantity": "20", "unit": "wooden stakes", "cost_per_unit": 2.0},
            {"item": "Spray paint", "quantity": "4 cans", "unit": "bright orange", "cost_per_unit": 5.0}
        ],
        tool_list=[
            "Laser level or water level (rent $50/day)",
            "100-ft tape measure",
            "Plumb bob",
            "String line",
            "Hammer or small sledge",
            "Calculator",
            "Level (4-foot minimum)"
        ],
        reference_documents=[
            "Foundation plan (architect/engineer drawings)",
            "Building dimensions from plans",
            "Footing width specifications"
        ],
        video_tutorials=[
            "Setting up batter boards",
            "3-4-5 squaring method",
            "Using a plumb bob",
            "Reading foundation plans"
        ],
        success_criteria=[
            "Batter boards set and level",
            "Building lines established",
            "All corners square (3-4-5 method verified)",
            "Diagonals equal (±1/4 inch)",
            "Footing edges marked",
            "Interior footings located",
            "Elevation marks on stakes",
            "Layout matches foundation plan"
        ]
    )


# Additional steps (3-11) would follow similar pattern
# For brevity, showing structure for remaining steps:

def foundation_step_3_footing_forms(project_state) -> NextStep:
    """Step 3: Build & Install Footing Forms"""
    return NextStep(
        step_number=3,
        title="Foundation Step 3: Build & Install Footing Forms",
        description="""
📏 **FOOTING FORMS - Creating the Concrete Mold**

**What You're Doing:**
Building wooden forms that will hold the concrete in place while it cures.

**Materials Needed:**
- 2x8 or 2x10 lumber (depends on footing depth)
- 2x4 stakes (every 2-3 feet)
- 1x4 bracing (diagonal supports)
- 16d duplex nails (easy removal)
- Form release oil (prevents concrete sticking)

**Step-by-Step Process:**

1. **Calculate Lumber Quantity**
   - Measure perimeter of foundation
   - Double it (forms on both sides)
   - Add 20% for waste and stakes
   - Example: 160 LF perimeter → 320 LF lumber + 64 LF = 384 LF total

2. **Cut Lumber to Length**
   - 8-foot or 16-foot sections work best
   - Cut splices at diagonal (45°) for strength
   - Sand rough edges
   - Keep cut ends square

3. **Install Outer Form Boards**
   - Start at one corner
   - Align top edge with footing height marks
   - Drive 2x4 stakes every 2-3 feet
   - Stakes on OUTSIDE of form (easy removal)
   - Nail forms to stakes (3-4 nails per stake)

4. **Install Inner Form Boards**
   - Space exactly footing width from outer form
   - Example: 16" footing = 16" between inside faces
   - Use spacer blocks to maintain width
   - Stake on inside (will be buried)

5. **Brace the Forms**
   - Forms must withstand 2-3 tons of concrete pressure
   - Install 1x4 diagonal braces every 4 feet
   - Angle braces 45° from top of form to ground
   - Stake braces securely

6. **Level Check**
   - Top of forms = top of finished footing
   - Use laser level or water level
   - Adjust stakes up/down as needed
   - Target: ±1/4" level around entire perimeter

7. **Square Check**
   - Measure diagonals (corner to corner)
   - Must be equal (±1/2")
   - Adjust forms if needed
   - Re-stake after adjustments

8. **Seal Joints**
   - Where boards meet, seal gaps
   - Use foam backer rod or caulk
   - Prevents concrete leakage
   - Check bottom edge especially

9. **Apply Form Release**
   - Spray or brush oil on inside of forms
   - Prevents concrete bonding to wood
   - Makes removal much easier
   - Let dry 10 minutes

10. **Final Inspection**
    - Walk entire perimeter
    - Check all stakes are secure
    - Verify width at multiple points
    - Confirm level
    - Look for gaps or weak spots

**PRO TIPS:**
💡 Use duplex nails - double head makes removal easy
💡 Don't over-tighten - forms will bow inward
💡 Wet forms before pour - prevents water absorption
💡 Keep forms at least 1 week after pour for strength

**SUCCESS CRITERIA:**
✅ Forms aligned with layout marks (±1")
✅ Width correct at all points (±1/4")
✅ Level within ±1/4" around perimeter
✅ No gaps or holes in forms
✅ Stakes secure - forms don't move when pushed
✅ Corners square (diagonals equal)
""",
        why_now="Forms create the shape for concrete footings and must be precise for square, level foundation",
        estimated_cost=800.0,
        estimated_duration_days=2,
        requires_professional=False,
        professional_type=None,
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Wear gloves handling lumber - splinters",
            "Safety glasses when hammering",
            "Watch for protruding nails",
            "Bend over nails or cut flush"
        ],
        material_list=[
            {"item": "2x8 or 2x10 lumber", "quantity": "384 LF (for 160 LF foundation)", "unit": "linear foot", "cost_per_unit": 1.50},
            {"item": "2x4 stakes", "quantity": "80 stakes (every 2-3 feet)", "unit": "8-foot stake", "cost_per_unit": 4.0},
            {"item": "1x4 bracing", "quantity": "160 LF", "unit": "linear foot", "cost_per_unit": 0.80},
            {"item": "16d duplex nails", "quantity": "5 pounds", "unit": "pound", "cost_per_unit": 8.0},
            {"item": "Form release oil", "quantity": "1 gallon", "unit": "gallon", "cost_per_unit": 25.0}
        ],
        tool_list=[
            "Circular saw or hand saw",
            "Hammer or nail gun",
            "Tape measure (25-foot)",
            "4-foot level",
            "Laser level (rent $50/day)",
            "String line",
            "Pencil or chalk",
            "Sledgehammer (for stakes)"
        ],
        reference_documents=[
            "Foundation plan (footing dimensions)",
            "Building code (footing width/depth requirements)"
        ],
        video_tutorials=[
            "How to build concrete forms",
            "Checking forms for level and square"
        ],
        success_criteria=[
            "Forms follow layout marks (±1 inch)",
            "Footing width correct everywhere (±1/4 inch)",
            "Top of forms level (±1/4 inch)",
            "No gaps where concrete could leak",
            "All stakes and braces secure",
            "Corners square (diagonal measurements equal)",
            "Form release oil applied"
        ]
    )


def foundation_step_4_rebar(project_state) -> NextStep:
    """Step 4: Install Rebar (Reinforcement)"""
    return NextStep(
        step_number=4,
        title="Foundation Step 4: Install Rebar Reinforcement",
        description="""
🔩 **REBAR INSTALLATION - Strengthening Your Foundation**

**What You're Doing:**
Installing steel reinforcement bars that give concrete tensile strength.

**Why It Matters:**
Concrete is strong in compression but weak in tension. Rebar prevents cracking and structural failure.

**Code Requirements:**
- **Minimum**: #4 rebar (1/2" diameter)
- **Horizontal**: 2 bars continuous around perimeter
- **Vertical**: #4 bars every 4-6 feet (ties to wall rebar later)
- **Spacing**: 3" from bottom, 3" from sides, 3" from top
- **Laps**: 40 bar diameters minimum (20" for #4 rebar)

**Materials Needed:**
- #4 rebar (1/2" diameter) - horizontal
- #5 rebar (5/8" diameter) - vertical (optional, stronger)
- Rebar chairs (hold bars off ground)
- Tie wire (16-gauge)
- Rebar couplers (for long runs)

**Step-by-Step Process:**

1. **Calculate Rebar Quantity**
   - Perimeter × 2 (two horizontal bars) × 1.2 (waste/laps)
   - Example: 160 LF × 2 × 1.2 = 384 LF
   - Rebar comes in 20-foot lengths
   - Need: 384 ÷ 20 = 20 pieces

2. **Cut Rebar to Length**
   - Use hacksaw, angle grinder, or bolt cutters
   - Wear safety glasses (metal sparks!)
   - Cut ends square
   - Deburr sharp edges

3. **Install Rebar Chairs**
   - Place every 3-4 feet
   - Height: 3" (code minimum)
   - Creates proper concrete cover
   - Prevents rust and maintains strength

4. **Lay Bottom Horizontal Bars**
   - Place first bar 3" from inside of form
   - Place second bar 3" from outside of form
   - Overlap splices 20" minimum
   - Support on rebar chairs

5. **Tie Bars Together**
   - Use 16-gauge tie wire
   - Wrap around intersection
   - Twist 3-4 times tight
   - Bend tail down (no sharp points up)

6. **Install Vertical Bars (Dowels)**
   - Every 4-6 feet around perimeter
   - Height: 30-40" (extends into wall)
   - Tie to horizontal bars
   - These connect footing to foundation wall

7. **Check Spacing**
   - 3" minimum from all form faces
   - Use spacer blocks if needed
   - Walk entire perimeter
   - Adjust bars as needed

8. **Add Corner Reinforcement**
   - Bend rebar around corners (preferred)
   - OR lap 40 bar diameters past corner
   - Corners take highest stress
   - Extra care here prevents cracks

9. **Inspect for Compliance**
   - All bars supported on chairs
   - Proper spacing maintained
   - All intersections tied
   - No bars touching forms
   - Vertical bars plumb

10. **Final Check Before Pour**
    - Walk forms, check no bars moved
    - Retie any loose connections
    - Remove any debris on bars
    - Take photos for your records

**SAFETY WARNINGS:**
⚠️ Wear leather gloves - rebar has sharp edges
⚠️ Safety glasses when cutting (sparks!)
⚠️ Watch for tripping hazards (rebar on ground)
⚠️ Bent rebar stores energy - can spring back
⚠️ Lift with legs, not back (rebar is heavy)

**CODE COMPLIANCE NOTES:**
- IRC Section R403.1.3: Minimum #4 rebar required
- 3" concrete cover minimum (rust protection)
- Continuous around perimeter (no breaks)
- Laps must be 40 bar diameters
- Inspector will check before pour approval

**SUCCESS CRITERIA:**
✅ All horizontal bars supported on chairs
✅ 3" spacing from all form sides maintained
✅ All bar intersections tied with wire
✅ Vertical dowels every 4-6 feet
✅ Laps minimum 20" (for #4 rebar)
✅ No bars touching forms
✅ Corners properly reinforced
✅ Inspector approval obtained
""",
        why_now="Rebar must be installed before concrete pour. Adds critical tensile strength to footings.",
        estimated_cost=600.0,
        estimated_duration_days=1,
        requires_professional=False,
        professional_type=None,
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Wear leather gloves - sharp edges",
            "Safety glasses when cutting rebar",
            "Lift with legs - rebar is heavy (20 LB per 20-foot piece)",
            "Watch for tripping on rebar",
            "Bent rebar can spring - be careful"
        ],
        material_list=[
            {"item": "#4 rebar (1/2\" diameter)", "quantity": "20 pieces @ 20 feet", "unit": "20-foot piece", "cost_per_unit": 12.0},
            {"item": "Rebar chairs (3\" height)", "quantity": "100 pieces", "unit": "each", "cost_per_unit": 0.50},
            {"item": "16-gauge tie wire", "quantity": "1 roll (1000 feet)", "unit": "roll", "cost_per_unit": 25.0},
            {"item": "#5 vertical dowels", "quantity": "40 pieces @ 40\"", "unit": "each", "cost_per_unit": 4.0}
        ],
        tool_list=[
            "Angle grinder or hacksaw (for cutting)",
            "Bolt cutters (for tie wire)",
            "Tape measure",
            "Rebar bender (rent if needed)",
            "Leather work gloves",
            "Safety glasses"
        ],
        reference_documents=[
            "IRC Section R403.1.3 (footing reinforcement)",
            "Foundation plan (rebar schedule)",
            "Structural engineer's notes"
        ],
        video_tutorials=[
            "How to tie rebar properly",
            "Rebar spacing requirements",
            "Using rebar chairs"
        ],
        success_criteria=[
            "All bars on rebar chairs (3\" clearance)",
            "3\" spacing from all form sides",
            "All intersections tied with wire",
            "Vertical dowels every 4-6 feet",
            "Laps minimum 20 inches",
            "Corners properly reinforced",
            "No bars touching forms",
            "Building inspector approved (REQUIRED)"
        ]
    )


# Continue with steps 5-11...
def foundation_step_5_inspection(project_state) -> NextStep:
    """Step 5: Footing Inspection"""
    return NextStep(
        step_number=5,
        title="Foundation Step 5: Pre-Pour Footing Inspection",
        description="""
📋 **FOOTING INSPECTION - Critical Checkpoint**

**What's Happening:**
Building inspector verifies your footings meet code BEFORE concrete pour.

**Why Required:**
Once concrete is poured, mistakes are permanent and expensive. Inspector catches problems now.

**What Inspector Checks:**

1. **Excavation Depth**
   - Below frost line (jurisdiction-specific)
   - Bearing on undisturbed soil
   - Proper width and depth per plans

2. **Form Installation**
   - Proper height and level
   - Secure and won't shift during pour
   - Correct width (16-20" typical)
   - Corners square

3. **Rebar Placement**
   - Correct size (#4 minimum)
   - Proper spacing (3" clearance all sides)
   - Tied at intersections
   - Vertical dowels in place
   - Laps sufficient (20" minimum for #4)

4. **Drainage**
   - Perimeter drains installed
   - Proper slope
   - Connected to daylight or sump

5. **Site Conditions**
   - No standing water
   - No debris in forms
   - Ready for concrete

**How to Schedule:**
- Call building department 2-3 days ahead
- Have permit number ready
- Confirm inspection type: "Footing inspection"
- Get inspector name and expected time window

**What to Have Ready:**
- Building permit (posted on site)
- Foundation plans
- Rebar placement drawings
- Engineer's specifications (if applicable)
- Your questions list

**Common Failure Reasons:**
- ❌ Insufficient excavation depth
- ❌ Rebar too close to forms (<3")
- ❌ Missing vertical dowels
- ❌ Forms not level
- ❌ Inadequate bracing
- ❌ Rebar not tied properly

**If You Fail:**
- Inspector will note deficiencies
- Fix problems (usually same day)
- Call for re-inspection
- Do NOT pour until approved

**Pro Tips:**
💡 Be on site when inspector arrives
💡 Ask questions - they're helpful!
💡 Take photos of approved work
💡 Get inspector's card for future questions
💡 Some inspectors prefer morning appointments

**Timeline:**
- Schedule: 2-3 days in advance
- Inspection: 15-30 minutes on site
- Approval: Same day (or correction list)
- Next step: Schedule concrete pour

**SUCCESS CRITERIA:**
✅ Inspector signs off on permit card
✅ "Approved" or "Passed" marked
✅ Any notes or conditions documented
✅ Copy of approval in your records
✅ Ready to schedule concrete pour
""",
        why_now="Code requires inspection before concrete pour. Catches errors while they're still fixable.",
        estimated_cost=0.0,  # Included in permit fee
        estimated_duration_days=1,  # Waiting for inspector
        requires_professional=True,
        professional_type="Building Inspector (from jurisdiction)",
        requires_permit=True,
        permit_type="Building Permit (footing inspection)",
        safety_warnings=[
            "Don't enter excavation without proper shoring",
            "Keep site clean for inspector access",
            "Have fire extinguisher on site (some jurisdictions require)"
        ],
        material_list=[],  # No materials needed
        tool_list=[],  # No tools needed
        reference_documents=[
            "Building permit",
            "Foundation plans",
            "Rebar schedule",
            "Engineer's stamp (if required)"
        ],
        video_tutorials=[
            "What building inspectors look for",
            "How to prepare for footing inspection"
        ],
        success_criteria=[
            "Inspector arrives and completes inspection",
            "Excavation depth verified",
            "Forms checked and approved",
            "Rebar placement approved",
            "Drainage verified",
            "Permit card signed",
            "\"Approved\" or \"Passed\" marked",
            "Ready to order concrete"
        ]
    )


def foundation_step_6_concrete_pour(project_state) -> NextStep:
    """Step 6: Pour Concrete Footings"""
    return NextStep(
        step_number=6,
        title="Foundation Step 6: Pour Concrete Footings",
        description="""
🚚 **CONCRETE POUR - The Big Day!**

**What You're Doing:**
Filling forms with concrete to create permanent foundation footings.

**Concrete Specifications:**
- **Strength**: 3000 PSI minimum (3500-4000 recommended)
- **Mix**: "Foundation mix" or "Footing mix"
- **Slump**: 4-5 inches (for good flow)
- **Air entrainment**: Required in freeze climates
- **Quantity**: Calculate cubic yards needed

**Calculating Concrete Volume:**
Formula: Width (ft) × Depth (ft) × Length (ft) ÷ 27 = Cubic Yards

Example for 160 LF perimeter:
- 16" wide = 1.33 ft
- 12" deep = 1 ft  
- 160 LF length
- 1.33 × 1 × 160 ÷ 27 = 7.9 CY
- **Order 9 CY** (always add 10% for waste)

**Concrete Costs:**
- Ready-mix: $125-150 per cubic yard
- Delivery: $100-200 (one-time)
- Pump truck: $800-1200 (if needed for access)
- **Total for 9 CY**: $1,400-1,800

**Day Before Pour:**
1. Confirm concrete order (time, quantity, specifications)
2. Check weather (no rain, temp above 40°F)
3. Gather crew (need 3-4 people minimum)
4. Rent vibrator (removes air bubbles)
5. Get wheelbarrows, shovels, rakes ready

**Pour Day Process:**

1. **Morning Preparation** (7 AM)
   - Re-check forms (any movement overnight?)
   - Wet down forms (prevents water absorption)
   - Clear path for concrete truck
   - Set up wheelbarrows if needed
   - Brief crew on plan

2. **Truck Arrives** (8 AM, be ready!)
   - Verify concrete specs on ticket
   - Check slump (4-5 inches)
   - Direct driver where to position chute
   - **START POURING IMMEDIATELY**

3. **During Pour** (fast-paced, 60-90 minutes)
   - Pour in 2-3 foot sections
   - Have one person direct concrete placement
   - Have 2 people spread with rakes
   - One person vibrates (removes air pockets)
   - Keep concrete moving - don't let it set up
   - Fill forms to top (level with form boards)

4. **Vibration** (Critical!)
   - Insert vibrator every 18-24 inches
   - Hold 10-15 seconds until air bubbles stop
   - Don't over-vibrate (segregates mix)
   - Don't touch rebar with vibrator
   - Work systematically around perimeter

5. **Screeding** (Leveling)
   - Use 2x4 across top of forms
   - Sawing motion, working forward
   - Fills low spots, removes high spots
   - Two people, one on each end
   - Repeat until smooth and level

6. **Floating** (Smoothing)
   - After bleed water disappears (15-30 min)
   - Use bull float or hand float
   - Smooth trowel marks
   - Fills small voids
   - Don't overwork - weakens surface

7. **Final Checks**
   - All forms filled to top
   - No voids or air pockets
   - Level across all sections
   - Vertical dowels still plumb
   - Edges clean

**After Pour:**
1. **Protect from elements**
   - Cover with plastic if rain expected
   - Shade if hot sun (prevents rapid drying)
   - Don't let freeze (below 40°F)

2. **Curing** (Critical for strength!)
   - Keep moist for 7 days minimum
   - Spray water 2-3 times daily
   - OR cover with wet burlap
   - OR apply curing compound

3. **Wait period**
   - 3 days: Can walk on it
   - 7 days: Can strip forms, start walls
   - 28 days: Full strength achieved

**SAFETY WARNINGS:**
⚠️ Concrete is caustic - wear rubber gloves and boots
⚠️ Long sleeves/pants (burns skin on contact)
⚠️ Safety glasses (splashing)
⚠️ Have water hose ready to wash off any splashes
⚠️ Concrete is HEAVY - lift carefully
⚠️ Slippery when wet - watch footing

**WHAT CAN GO WRONG:**
- Form blowout (too much pressure) → Brace better
- Wrong concrete mix delivered → Refuse it, don't pour
- Rain during pour → Cover immediately, may need tent
- Truck can't reach → Need pump truck ($1200)
- Running short → Have ready-mix plant on standby

**PRO TIPS:**
💡 Start early (cooler, more time)
💡 Have extra wheelbarrows (keep concrete moving)
💡 Order 10% extra concrete (always)
💡 Save concrete tickets (proof for inspector)
💡 Take photos before, during, after

**SUCCESS CRITERIA:**
✅ Correct concrete mix delivered (3000 PSI+)
✅ All forms filled completely
✅ No voids or air pockets (vibrated properly)
✅ Level within ±1/4" across entire pour
✅ Vertical dowels remain plumb
✅ Clean finish on top surface
✅ Curing protection in place
""",
        why_now="Concrete transforms your forms into permanent, structural footings that support entire house",
        estimated_cost=2500.0,  # 9 CY @ $150/CY + delivery + vibrator rental
        estimated_duration_days=1,  # Plus 7 days curing
        requires_professional=True,  # Can DIY but professional concrete crew recommended
        professional_type="Concrete contractor (recommended) or experienced DIY crew",
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Wear rubber gloves and boots - concrete burns skin",
            "Long sleeves and pants required",
            "Safety glasses for splash protection",
            "Have water hose ready for washing skin",
            "Lift with legs - concrete is very heavy",
            "Slippery surfaces - watch your footing"
        ],
        material_list=[
            {"item": "Ready-mix concrete (3000 PSI)", "quantity": "9 cubic yards", "unit": "cubic yard", "cost_per_unit": 145.0},
            {"item": "Delivery fee", "quantity": "1", "unit": "delivery", "cost_per_unit": 150.0},
            {"item": "Concrete vibrator rental", "quantity": "1 day", "unit": "day", "cost_per_unit": 75.0},
            {"item": "Plastic sheeting (curing)", "quantity": "200 SF", "unit": "square foot", "cost_per_unit": 0.25}
        ],
        tool_list=[
            "Concrete vibrator (rent)",
            "Wheelbarrows (2-3)",
            "Shovels (4-5)",
            "Rakes (2-3)",
            "2x4 screed board (10-12 feet)",
            "Bull float or hand floats",
            "Rubber boots",
            "Rubber gloves",
            "Water hose"
        ],
        reference_documents=[
            "Concrete delivery ticket (save for inspector)",
            "Foundation plan (pour locations)",
            "Weather forecast (no rain, above 40°F)"
        ],
        video_tutorials=[
            "How to pour concrete footings",
            "Using a concrete vibrator",
            "Proper concrete curing techniques"
        ],
        success_criteria=[
            "Correct concrete specs (3000+ PSI, proper slump)",
            "All forms filled to top",
            "Vibrated properly (no air voids)",
            "Level surface (±1/4 inch)",
            "Vertical dowels still plumb and in place",
            "Clean finish",
            "Curing protection applied",
            "Concrete ticket saved",
            "7-day curing period maintained"
        ]
    )


def foundation_step_7_strip_forms(project_state) -> NextStep:
    """Step 7: Strip Footing Forms"""
    return NextStep(
        step_number=7,
        title="Foundation Step 7: Strip Footing Forms & Prep for Walls",
        description="""
🔨 **FORM REMOVAL - Revealing Your Footings**

**Wait Time: Minimum 7 days after pour**

**Why Wait:**
- Concrete reaches 70% strength at 7 days
- Too early = damage to concrete
- Too late = forms hard to remove

**What You're Doing:**
Carefully removing wooden forms and preparing footings for foundation walls.

**Tools Needed:**
- Hammer or pry bar
- Cat's paw (nail puller)
- Reciprocating saw (for stubborn pieces)
- Wire brush
- Shovel
- Broom

**Step-by-Step Process:**

1. **Inspection First**
   - Walk perimeter, look for cracks
   - Check corners (high stress areas)
   - Look for voids or honeycomb
   - Minor cracks (<1/16") are normal
   - Major cracks = call structural engineer

2. **Remove Bracing First**
   - Take down diagonal braces
   - Remove top ties and spacers
   - Work systematically (one section at a time)
   - Stack lumber for reuse or disposal

3. **Remove Outer Stakes**
   - Pull stakes away from concrete
   - Use pry bar if stuck
   - Don't damage concrete edge
   - Save stakes (reuse for walls)

4. **Remove Outer Form Boards**
   - Start at one end
   - Tap boards away from concrete
   - If stuck, use pry bar gently
   - Watch for nails - pull or bend over
   - Stack boards (can resell or reuse)

5. **Remove Inner Stakes & Forms**
   - These were buried in pour
   - May need more force
   - Cut flush if necessary
   - Clean concrete surface

6. **Clean the Footings**
   - Wire brush any concrete drips
   - Remove form release residue
   - Sweep top surface clean
   - Remove any debris from top

7. **Inspect Concrete Quality**
   **Look for:**
   - Smooth surfaces (good finish)
   - No large voids (honeycomb)
   - Proper dimensions (width, height)
   - Vertical dowels intact and plumb
   - Corners square and clean

**Common Issues & Fixes:**

**Honeycomb (air pockets):**
- Small areas: Patch with cement mortar
- Large areas: Call engineer
- Cause: Under-vibration during pour

**Form marks:**
- Normal and acceptable
- Will be buried anyway
- No repair needed

**Corner chips:**
- Minor (<1" deep): Patch with mortar
- Major: May need engineering review

**Vertical dowels bent:**
- Straighten if possible
- May need to drill and epoxy new dowels
- Critical for wall connection

8. **Backfill Around Footings**
   - Fill gaps between footing and soil
   - Use excavated soil
   - Compact in 6" lifts
   - Don't bury dowels
   - Slope away from footings for drainage

9. **Mark Wall Layout**
   - Snap chalk lines for wall placement
   - Measure from plans
   - Mark door openings
   - Mark plumbing penetrations
   - Double check dimensions

10. **Document & Photograph**
    - Photo entire perimeter
    - Close-ups of corners
    - Dowel locations
    - Any repairs made
    - Keep for your records

**Lumber Disposal Options:**
1. **Resell**: Post on Craigslist ($0.50/board)
2. **Reuse**: Save for wall forms
3. **Firewood**: If untreated
4. **Dumpster**: Last resort

**Next Steps Prep:**
- Order blocks or concrete for walls
- Schedule wall form lumber delivery
- Confirm wall inspection timing
- Prepare for waterproofing materials

**PRO TIPS:**
💡 Wait 7 full days minimum (longer if cold weather)
💡 Remove forms on cool morning (easier)
💡 Save good lumber - you'll need for walls
💡 Take lots of photos (useful later)
💡 Check for any permit requirements before walls

**SUCCESS CRITERIA:**
✅ All forms and bracing removed
✅ Footings clean and inspected
✅ No major defects found
✅ Vertical dowels intact and plumb
✅ Backfill placed and compacted
✅ Wall layout marked on footings
✅ Lumber stacked or disposed
✅ Photos documented
✅ Ready to start foundation walls
""",
        why_now="Forms must be removed before building foundation walls. Reveals concrete quality for inspection.",
        estimated_cost=0.0,  # DIY labor only
        estimated_duration_days=1,
        requires_professional=False,
        professional_type=None,
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Wear gloves - splinters and sharp concrete edges",
            "Safety glasses - flying debris from prying",
            "Watch for protruding nails",
            "Don't hit concrete hard - can chip edges",
            "Lift properly - lumber is heavy when wet"
        ],
        material_list=[
            {"item": "Concrete patching mortar", "quantity": "1 bag (if needed)", "unit": "60-pound bag", "cost_per_unit": 15.0}
        ],
        tool_list=[
            "Hammer",
            "Pry bar",
            "Cat's paw (nail puller)",
            "Wire brush",
            "Broom",
            "Shovel",
            "Tape measure",
            "Chalk line",
            "Camera/phone for photos"
        ],
        reference_documents=[
            "Foundation plan (wall layout)",
            "Photos from before concrete pour (comparison)"
        ],
        video_tutorials=[
            "How to strip concrete forms",
            "Inspecting concrete quality",
            "Patching minor concrete defects"
        ],
        success_criteria=[
            "Waited minimum 7 days since pour",
            "All forms and bracing removed",
            "Footings visually inspected",
            "No major defects (large cracks, voids)",
            "Minor defects patched",
            "Vertical dowels undamaged",
            "Backfill placed",
            "Wall layout marked",
            "Lumber disposed/stored",
            "Photos taken",
            "Ready for foundation walls"
        ]
    )


def foundation_step_8_walls(project_state) -> NextStep:
    """Step 8: Foundation Walls"""
    return NextStep(
        step_number=8,
        title="Foundation Step 8: Build Foundation Walls",
        description="""
🧱 **FOUNDATION WALLS - Building the Vertical Support**

**What You're Doing:**
Building walls on top of footings to create the foundation perimeter. These walls support the entire house.

**Two Options:**

**OPTION A: Concrete Block (CMU)**
- Faster installation
- Easier for DIY
- Lower cost
- Good for simple foundations
- **Most popular for residential**

**OPTION B: Poured Concrete**
- Stronger walls
- Better waterproofing
- Requires forms + pump
- Professional recommended
- Better for high water table

**CONCRETE BLOCK METHOD (Most Common):**

**1. Calculate Materials:**
```
Wall Height: Typically 8 feet (6 courses of 16" blocks)
Block Count: (Perimeter in feet × 0.75) blocks per course
Example: 160 ft perimeter × 0.75 = 120 blocks per course
6 courses × 120 = 720 blocks total
Add 5% waste = 756 blocks

Mortar: 1 bag per 12 blocks = 63 bags (80-pound bags)
Rebar: Vertical every 4 feet, horizontal every 2 courses
```

**2. Materials List:**
- 8x8x16" concrete blocks (CMU): 720+ blocks
- Type S mortar mix: 60-70 bags (80 lb each)
- #4 vertical rebar: Every 4 feet in cells
- #4 horizontal rebar: Every 2 courses in bond beam
- Grout: For filling rebar cells (2-3 yards)
- Wall ties: If using brick veneer later

**3. Block Laying Process:**

**Day 1: First Course (CRITICAL)**
- Snap chalk lines on footings (inner/outer edges)
- Dry lay first course (no mortar) to check layout
- Mark all block positions
- Mix mortar: 3 parts sand, 1 part cement, water
- Consistency: Like peanut butter (not runny)
- Start at corners first
- Full mortar bed on footing (1" thick)
- Set corner blocks, check level both ways
- Build up corners 3-4 courses (leads)
- String line between corners
- Lay blocks between corners
- Butter ends of each block
- Tap into place, check with level
- 3/8" mortar joints (consistent)
- Tool joints after mortar thumbprint-hard (30-45 min)

**Day 2-3: Remaining Courses**
- Continue building corners first
- Work between corners
- Insert horizontal rebar every 2 courses
- Place vertical rebar in designated cells
- Keep walls plumb (check every 3 blocks)
- Keep courses level (check constantly)
- Clean excess mortar as you go

**Day 4: Bond Beam (Top Course)**
- Use U-block or knock out webs
- Install 2 horizontal #4 rebars
- Fill with grout
- Set anchor bolts every 4-6 feet (embed 7")
- Space bolts 6-12" from corners
- Keep bolts 1-3/4" from edge
- Align bolts for sill plate installation

**4. Vertical Rebar & Grout:**
- Vertical rebar every 4 feet minimum
- At corners, both sides of openings
- Overlap with footing dowels (40 diameters)
- Grout rebar cells after 3-4 courses
- Use flowable grout (high slump)
- Consolidate with vibrator or rebar
- Pour in lifts (don't fill all at once)

**5. Window/Door Openings:**
- Frame with wood (temporary support)
- Use lintel blocks above openings
- Install steel lintel or bond beam
- Minimum bearing: 8" each side
- Keep vertical rebar beside openings

**POURED CONCRETE METHOD:**

**1. Build Forms:**
- Similar to footing forms but vertical
- Use 3/4" plywood + 2x4 studs @ 16"
- Walers (horizontal bracing) every 2 feet
- Snap ties every 18-24" (hold walls apart)
- Form thickness = wall width (8-12")
- Check plumb with 4-foot level
- Brace every 4 feet with 2x4 kickers

**2. Rebar Installation:**
- #4 or #5 vertical bars every 12-16"
- #4 horizontal bars every 16-24"
- 2" minimum cover all sides
- Wire tie all intersections
- Dowel into footings (already done)
- Use rebar chairs and spacers

**3. Concrete Pour:**
- 3000 PSI minimum (same as footings)
- Pour in 2-3 foot lifts
- Vibrate every 18-24" (critical)
- Don't pour too fast (blow outs)
- Set anchor bolts in wet concrete
- Screed top smooth and level

**4. Curing & Stripping:**
- Wait 3 days minimum before stripping
- Keep moist for 7 days
- Strip carefully (forms can stick)
- Patch any voids with mortar

**CRITICAL MEASUREMENTS:**

**Wall Thickness:**
- 8" typical for 1-story
- 10" for 2-story or heavy loads
- 12" for basements with high backfill

**Wall Height:**
- 8 feet typical (6 courses of block)
- Must clear frost line
- Check local code requirements

**Anchor Bolts:**
- 1/2" diameter minimum
- 7" embedment into concrete/grout
- 6 feet maximum spacing
- 12" maximum from corners/ends
- 1-3/4" from edge (center of sill plate)

**INSPECTION POINTS:**
🔍 **REQUIRED INSPECTION** before backfill
- Wall height and thickness
- Anchor bolt placement
- Rebar installation
- Grout consolidation (if block)
- Plumb and level
- Waterproofing prep

**COST BREAKDOWN:**

**Block Walls:**
- Blocks (720 @ $3 each): $2,160
- Mortar (63 bags @ $12): $756
- Rebar & grout: $800
- Anchor bolts: $150
- Tools/misc: $200
- **Total DIY: ~$4,000**
- **With Mason: $8,000-12,000** (labor $4-8K)

**Poured Walls:**
- Concrete (8 yards @ $150): $1,200
- Form lumber/hardware: $1,500
- Rebar: $800
- Pump truck: $800-1,200
- Anchor bolts: $150
- **Total DIY: ~$4,500**
- **With Contractor: $12,000-18,000**

**PRO TIPS:**
💡 Block is easier for DIY (more forgiving)
💡 Rent a mortar mixer ($50/day) - worth it!
💡 Keep mortar workable (re-temper if needed)
💡 Clean tools immediately (mortar sets fast)
💡 Work in cool weather if possible (mortar sets slower)
💡 Cover work at end of each day (rain/freeze protection)
💡 String line is your friend (keeps walls straight)
💡 Check level constantly (fix immediately, not later)
💡 Don't scrimp on rebar (it's insurance)

**SAFETY WARNINGS:**
⚠️ Block walls are HEAVY - use proper lifting (bend knees)
⚠️ Mortar is caustic - wear gloves and long sleeves
⚠️ Scaffold required above 4 feet - NEVER stand on blocks
⚠️ Secure all bracing on poured forms (blow-outs are dangerous)
⚠️ Hard hat when working under forms or scaffold

**COMMON MISTAKES:**
❌ Rushing first course (foundation for everything)
❌ Inconsistent mortar joints (weakens wall)
❌ Not checking level frequently
❌ Forgetting anchor bolts (can't fix later!)
❌ Poor rebar placement (defeats the purpose)
❌ Inadequate bracing on forms (blow-outs)

**SUCCESS CRITERIA:**
✅ Walls built to correct height
✅ Corners plumb (within 1/4" in 8 feet)
✅ Courses level (within 1/4" in 10 feet)
✅ Mortar joints consistent (3/8")
✅ Rebar installed per code
✅ Cells grouted solid
✅ Anchor bolts placed correctly
✅ Top surface level and clean
✅ No major voids or defects
✅ Inspection passed
✅ Ready for waterproofing
""",
        why_now="Walls must be built on footings before waterproofing and backfill. Creates the foundation structure.",
        estimated_cost=8000.0,  # $4K DIY materials + $4K labor if hiring mason
        estimated_duration_days=5,
        requires_professional="Recommended",
        professional_type="Mason (for block) or Concrete Contractor (for poured)",
        requires_permit=False,  # Usually covered by foundation permit
        permit_type=None,
        safety_warnings=[
            "HEAVY LIFTING - Blocks are 30-40 lbs each, use proper technique",
            "Caustic materials - Wear gloves, long sleeves, safety glasses",
            "Scaffold required above 4 feet - Never stand on blocks",
            "Hard hat when working below scaffold or forms",
            "Secure all form bracing - Blow-outs are extremely dangerous"
        ],
        material_list=[
            {"item": "8x8x16 concrete blocks (CMU)", "quantity": "720", "unit": "blocks", "cost_per_unit": 3.0},
            {"item": "Type S mortar mix", "quantity": "63", "unit": "80-lb bags", "cost_per_unit": 12.0},
            {"item": "#4 rebar (vertical)", "quantity": "40", "unit": "10-ft bars", "cost_per_unit": 8.0},
            {"item": "#4 rebar (horizontal)", "quantity": "30", "unit": "10-ft bars", "cost_per_unit": 8.0},
            {"item": "Concrete grout", "quantity": "2.5", "unit": "cubic yards", "cost_per_unit": 200.0},
            {"item": "Anchor bolts (1/2\" x 10\")", "quantity": "30", "unit": "bolts", "cost_per_unit": 5.0},
            {"item": "Rebar tie wire", "quantity": "1", "unit": "roll", "cost_per_unit": 25.0}
        ],
        tool_list=[
            "Mortar mixer (rent $50/day)",
            "Mortar tubs (2-3)",
            "Trowel (brick trowel)",
            "4-foot level",
            "2-foot level",
            "String line",
            "Chalk line",
            "Tape measure",
            "Mason's hammer",
            "Jointing tool",
            "Wire cutters",
            "Rebar bender (if needed)",
            "Scaffold or ladder",
            "Wheelbarrow",
            "Shovel",
            "Buckets"
        ],
        reference_documents=[
            "Foundation plan (wall heights, openings)",
            "IRC Section R404 (Foundation Walls)",
            "Block manufacturer specs",
            "Local code requirements"
        ],
        video_tutorials=[
            "How to lay concrete block",
            "Building block corners",
            "Installing rebar in block walls",
            "Grouting block walls",
            "Setting anchor bolts"
        ],
        success_criteria=[
            "Walls at correct height (8 feet typical)",
            "Corners plumb (±1/4\" in 8 feet)",
            "Courses level (±1/4\" in 10 feet)",
            "Mortar joints consistent (3/8\")",
            "Vertical rebar every 4 feet",
            "Horizontal rebar every 2 courses",
            "All rebar cells grouted",
            "Anchor bolts installed correctly",
            "Top surface level and smooth",
            "No major voids or cracks",
            "Inspection passed",
            "Ready for waterproofing"
        ]
    )


def foundation_step_9_waterproofing(project_state) -> NextStep:
    """Step 9: Waterproofing & Drainage"""
    return NextStep(
        step_number=9,
        title="Foundation Step 9: Waterproofing & Drainage",
        description="""
💧 **WATERPROOFING - Protecting Your Foundation from Water**

**What You're Doing:**
Applying waterproofing membrane and installing drainage to keep foundation dry. **Critical for longevity!**

**Why This Matters:**
- Water damage is #1 foundation problem
- Moisture causes mold, rot, structural issues
- Much harder to fix later (excavate again)
- Small investment now, huge protection forever
- Required by code in most areas

**COMPLETE WATERPROOFING SYSTEM:**

**1. Damp-proofing vs Waterproofing:**

**Damp-proofing (Minimum):**
- Black tar/asphalt coating
- Resists moisture vapor
- $100-200 in materials
- Code minimum for dry sites

**Waterproofing (Recommended):**
- Membrane or liquid rubber
- Stops water intrusion
- $500-800 in materials
- Required for wet sites, basements
- **Much better protection**

**2. Surface Preparation (CRITICAL):**

**Block Walls:**
- Fill all mortar voids with parging
- Parge coat: 1/2" cement plaster
- Mix: 1 part cement, 3 parts sand
- Smooth finish (no rough spots)
- Let cure 3 days before waterproofing

**Poured Walls:**
- Already smooth
- Fill any voids with mortar
- Wire brush any loose material
- Rinse and let dry

**Footing Transition:**
- Cove at wall-footing joint
- Use mortar to create rounded corner
- Prevents membrane tearing

**3. Waterproofing Application:**

**METHOD A: Membrane (Best):**
- Roll-on or sheet membrane
- Products: Tremco, Carlisle, Grace Perm-A-Barrier
- Start at bottom, work up
- Overlap seams 4-6 inches
- Roll out air bubbles
- Coat to 6" above grade
- Protect top edge (termination bar)

**METHOD B: Liquid Rubber:**
- Black liquid applied with brush/roller
- Products: Liquid Rubber Foundation Sealant
- Two coats minimum (let first dry)
- Easier for DIY
- 40-60 mils thickness when dry
- Cover to 6" above grade

**METHOD C: Spray-on (Professional):**
- Polyurethane foam + membrane
- Professional application
- Most expensive but best
- $3-5 per square foot

**4. Drainage System (ESSENTIAL):**

**Perimeter Drain Tile:**
- 4" perforated PVC pipe
- Slope 1/4" per foot minimum
- Place at footing level (outside)
- Holes facing DOWN (yes, down!)
- Wrap in filter fabric sock
- Surround with washed gravel

**Gravel Bed:**
- 6-8" crushed stone under pipe
- Continue 12" above pipe
- Creates drainage layer
- Use 3/4" washed stone (not pea gravel)

**Filter Fabric:**
- Wrap gravel in geotextile fabric
- Prevents soil infiltration
- Keeps system flowing for decades

**Drain Outlet:**
- Pipe to daylight (if sloped lot)
- Sump pump (if flat lot)
- Never let pipe outlet get clogged

**5. Installation Steps:**

**Step 1: Parge Block Walls (if needed)**
- Mix parging mortar
- Apply 1/2" coat with trowel
- Smooth finish
- Keep moist 3 days (spray with water)

**Step 2: Install Drain Pipe**
- Dig trench at footing edge
- 6" deep, 12" wide
- Add 4" crushed stone base
- Lay perforated pipe (holes down)
- Slope to outlet (1/4" per foot)
- Wrap in filter fabric sock

**Step 3: Apply Waterproofing**
- Start at footing (bottom)
- Work upward
- Apply evenly (no thin spots)
- Overlap seams
- Two coats minimum
- Extend 6" above final grade

**Step 4: Add Protection Board (Optional but Recommended)**
- 1/2" foam board over membrane
- Protects during backfill
- Adds insulation (R-value)
- Products: Owens Corning FOAMULAR, DuPont Styrofoam

**Step 5: Complete Drainage**
- Fill around pipe with gravel
- 12" above pipe minimum
- Wrap gravel in filter fabric
- Top with 6" sand or soil

**BASEMENT SPECIFIC:**

**Interior Drainage (Belt & Suspenders):**
- Interior perimeter drain at footing
- French drain to sump pit
- Sump pump with battery backup
- Required in high water table areas

**Floor Waterproofing:**
- 6 mil poly vapor barrier under slab
- Overlap seams 12"
- Tape all seams
- Continue up walls 6"

**Exterior Insulation (Cold Climates):**
- 2" foam board over waterproofing
- Reduces heat loss
- Prevents frost
- R-10 minimum in cold zones

**MATERIALS CHECKLIST:**

**Waterproofing:**
- [ ] Parging mortar (if block): 10-15 bags
- [ ] Waterproofing membrane or liquid: 800 SF coverage
- [ ] Primer (if required by membrane): 5 gallons
- [ ] Termination bar: 160 LF
- [ ] Protection board (optional): 800 SF
- [ ] Mastic/adhesive: As needed

**Drainage:**
- [ ] 4" perforated PVC pipe: 160 LF + 20% (curves/fittings)
- [ ] 3/4" washed stone: 8-10 cubic yards
- [ ] Filter fabric (geotextile): 400 SF (covers gravel)
- [ ] Pipe fittings: Elbows, tees, couplings
- [ ] Outlet pipe (solid 4" PVC): 20-50 LF to daylight

**COST BREAKDOWN:**

**Damp-proofing (Minimum):**
- Tar/asphalt coating: $150
- Drain pipe & gravel: $800
- Filter fabric: $100
- **Total: ~$1,000-1,200**

**Waterproofing (Recommended):**
- Parging materials: $200
- Membrane/liquid rubber: $600
- Protection board: $400
- Drain system: $900
- **Total: ~$2,000-2,500 DIY**
- **Professional: $4,000-6,000**

**TIMELINE:**
- Day 1: Parge walls (if needed)
- Days 2-4: Curing (wait 3 days)
- Day 5: Install drain pipe in gravel
- Day 6: Apply waterproofing (1st coat)
- Day 7: Apply waterproofing (2nd coat)
- Day 8: Install protection board
- Day 9: Complete drainage system
- **Total: 5-9 days** (including curing)

**PRO TIPS:**
💡 Don't skip waterproofing - regrets are expensive
💡 Use waterproofing, not just damp-proofing (worth extra $500)
💡 Drain pipe holes face DOWN (counter-intuitive but correct)
💡 Filter fabric is essential (prevents clogging)
💡 Apply waterproofing below grade + 6" above
💡 Protection board saves membrane during backfill
💡 Test drain before backfilling (run water through)
💡 Take photos before covering (future reference)

**COMMON MISTAKES:**
❌ Using tar paper instead of proper waterproofing
❌ Not parging rough block walls first
❌ Thin spots in waterproofing application
❌ Drain pipe installed upside-down (holes up)
❌ No filter fabric (system clogs in 5-10 years)
❌ Not sloping pipe properly
❌ Backfilling before waterproofing cures

**SAFETY:**
⚠️ Waterproofing chemicals - Wear gloves, ventilate area
⚠️ Working in excavation - Shore trench walls if >5 feet
⚠️ Heavy materials - Proper lifting technique

**SUCCESS CRITERIA:**
✅ Walls cleaned and prepared
✅ Block walls parged smooth (if applicable)
✅ Parging cured 3+ days
✅ Waterproofing applied (2 coats minimum)
✅ Coverage complete (footings to 6\" above grade)
✅ No thin spots or gaps
✅ Protection board installed (optional)
✅ Drain pipe at footing level
✅ Pipe sloped to outlet (1/4\" per foot)
✅ Gravel bed around pipe (6\" below, 12\" above)
✅ Filter fabric wrapped around gravel
✅ Outlet clear and functional
✅ System tested (water flows)
✅ Photos documented
✅ Ready for backfill
""",
        why_now="Must waterproof before backfill. Impossible to access later without excavating again.",
        estimated_cost=2000.0,  # DIY waterproofing system
        estimated_duration_days=7,
        requires_professional=False,
        professional_type="Waterproofing specialist (optional, recommended for basements)",
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Waterproofing chemicals - Wear gloves, respirator in enclosed spaces",
            "Working in excavation - Shore trench walls if over 5 feet deep",
            "Heavy gravel bags - Proper lifting technique",
            "Slippery surfaces when wet - Wear appropriate footwear"
        ],
        material_list=[
            {"item": "Waterproofing membrane or liquid rubber", "quantity": "800", "unit": "SF coverage", "cost_per_unit": 0.75},
            {"item": "Parging mortar (if block walls)", "quantity": "12", "unit": "80-lb bags", "cost_per_unit": 15.0},
            {"item": "4\" perforated drain pipe", "quantity": "180", "unit": "linear feet", "cost_per_unit": 2.50},
            {"item": "3/4\" washed stone", "quantity": "9", "unit": "cubic yards", "cost_per_unit": 50.0},
            {"item": "Filter fabric (geotextile)", "quantity": "400", "unit": "square feet", "cost_per_unit": 0.25},
            {"item": "Protection board (optional)", "quantity": "800", "unit": "square feet", "cost_per_unit": 0.50},
            {"item": "Primer (if required)", "quantity": "5", "unit": "gallons", "cost_per_unit": 30.0}
        ],
        tool_list=[
            "Trowel (for parging)",
            "Paint roller and pan (for liquid waterproofing)",
            "Paintbrush (detail work)",
            "Wheelbarrow",
            "Shovel",
            "Rake",
            "Level (check pipe slope)",
            "Tape measure",
            "Utility knife",
            "Caulk gun (for mastic)"
        ],
        reference_documents=[
            "IRC Section R406 (Foundation Waterproofing)",
            "Waterproofing product installation instructions",
            "Local code requirements (high water table areas)"
        ],
        video_tutorials=[
            "How to parge block foundation walls",
            "Applying foundation waterproofing",
            "Installing perimeter drain tile",
            "Foundation drainage systems"
        ],
        success_criteria=[
            "Walls cleaned and prepared",
            "Block walls parged (if applicable)",
            "Parging cured 3+ days",
            "Waterproofing applied 2+ coats",
            "Complete coverage (footing to 6\" above grade)",
            "No gaps or thin spots",
            "Protection board installed",
            "Drain pipe at footing level",
            "Pipe sloped 1/4\" per foot minimum",
            "Gravel bed installed (6\" below, 12\" above pipe)",
            "Filter fabric in place",
            "Outlet clear and tested",
            "Photos documented",
            "Ready for backfill"
        ]
    )


def foundation_step_10_backfill(project_state) -> NextStep:
    """Step 10: Backfill & Compaction"""
    return NextStep(
        step_number=10,
        title="Foundation Step 10: Backfill & Grade",
        description="""
⛏️ **BACKFILL - Filling Around Foundation & Final Grading**

**What You're Doing:**
Filling the excavation around foundation walls and grading for proper drainage away from house.

**Why Careful Backfilling Matters:**
- Improper backfill can damage waterproofing
- Poor compaction causes settling
- Wrong slope causes water problems
- Foundation movement if done too soon
- Sets the stage for final landscaping

**CRITICAL WAITING PERIOD:**

**Concrete Walls:**
- Wait 7 days minimum after pour
- Concrete at 70% strength
- Can proceed after waterproofing cures

**Block Walls:**
- Wait 7 days after grouting
- Grout must cure fully
- Backfill too soon = wall collapse risk

**BACKFILL MATERIALS:**

**Best to Worst:**

**1. Crushed Stone/Gravel (Best but $$$):**
- Drains instantly
- No expansion/contraction
- Easy to compact
- Never settles
- $40-60 per yard
- Use for first 2-3 feet

**2. Sand (Good):**
- Drains well
- Easy to compact
- Doesn't expand
- $30-40 per yard
- Good for next 2-3 feet

**3. Excavated Soil (OK if clay-free):**
- Free (you have it)
- Must be clean (no topsoil, organics, debris)
- Compact carefully
- Can settle over time
- Screen out rocks, roots, trash

**4. Clay (Avoid if Possible):**
- Expands when wet
- Shrinks when dry
- Hard to compact
- Holds water
- Can damage foundation
- Only use if no alternative (compact well)

**NEVER USE:**
❌ Topsoil (organics decompose)
❌ Organic material (wood, roots, grass)
❌ Large rocks (damage waterproofing)
❌ Construction debris
❌ Frozen soil

**BACKFILL PROCESS:**

**Step 1: Protect Waterproofing**
- Install protection board if not already done
- Place cardboard over drain (temporary)
- Have helper watch for damage
- No heavy equipment against walls

**Step 2: First Lift (0-2 feet)**
- Start with 6" layer
- Place material gently (don't dump)
- Keep 12" away from wall initially
- Use hand shovel near wall
- Compact with hand tamper or plate compactor
- Compact to 95% density
- Check for voids under footings

**Step 3: Subsequent Lifts (Every 6-12 inches)**
- Add 6-12" layer
- Spread evenly around entire perimeter
- Don't pile high on one side (uneven pressure)
- Compact each lift before next
- Water lightly if very dry (helps compaction)
- Continue to final grade

**Step 4: Final Grading**
- Last 6-12" is topsoil (for planting)
- Slope AWAY from house
- Minimum 6" drop in first 10 feet
- Ideal: 1" per foot for first 6 feet, then 1/2" per foot
- Use laser level or long level + stakes

**COMPACTION:**

**Why It's Critical:**
- Prevents settling (cracks in walls, floors)
- Provides lateral support to wall
- Minimizes frost heave
- Stable base for slab/crawlspace

**Equipment:**

**Hand Tamper:**
- $40-60 to buy
- Good for tight spaces
- Slow but effective
- Tiring (rent plate compactor if possible)

**Plate Compactor (Rental):**
- $75-100 per day
- 4,000-5,000 lbs force
- Walk-behind vibrating plate
- Compacts 6-8" lifts
- **Much faster, worth the rental**

**How to Compact:**
- Make 3-4 passes over each area
- Overlap passes 50%
- Watch for bouncing (means compacted)
- Sinking = needs more passes
- Spray light water if very dry soil
- Don't over-water (makes mud)

**SPECIAL SITUATIONS:**

**Basement:**
- Same process
- Extra care with waterproofing
- May need interior french drain
- Floor slab after backfill complete

**Crawlspace:**
- Backfill exterior
- Grade interior to drain
- 6 mil poly vapor barrier on ground
- Vent or condition per code

**Slab on Grade:**
- Less backfill (walls shorter)
- Interior fill extra critical (under slab)
- Compact to 95%+ (slab support)

**Utilities:**
- Backfill after utility rough-in
- Protect pipes during compaction
- Sand around pipes (not gravel)
- No compaction directly over pipes

**FINAL GRADING:**

**Slope Requirements:**
- 6" drop in first 10 feet (minimum code)
- Better: 1" per foot for 6 feet = 6" drop in 6 feet
- Continue 1/2" per foot beyond that
- Direct water to swale, street, or drain

**How to Check:**
- Laser level (rent $50/day) - most accurate
- Water level (DIY with clear tubing)
- String line + line level
- Long level + straightedge

**Mark elevations:**
- Stake every 10 feet
- Mark finish grade on stakes
- Fill or cut to grade
- Compact as you go

**DRAINAGE AWAY FROM HOUSE:**

**Swales:**
- Low areas that channel water
- 6-12" deep
- Grass-lined
- Direct to street, ditch, or drain

**Gutters & Downspouts:**
- Essential for complete system
- 4-6" extensions minimum
- Better: Underground to 10+ feet away

**Splash Blocks:**
- Under downspouts
- Direct water away
- Cheap insurance

**TIMELINE:**
- Day 1: First lift (0-2 feet), compact
- Day 2: Middle lifts (2-6 feet), compact
- Day 3: Final lift to grade
- Day 4: Fine grading, topsoil
- Day 5: Seed/sod if ready
- **Total: 3-5 days** (depends on depth)

**COST BREAKDOWN:**

**Materials:**
- Gravel (3 yards @ $50): $150
- Sand (3 yards @ $35): $105
- Topsoil (5 yards @ $40): $200
- **Total materials: $450** (if buying fill)
- OR **$0** if reusing excavated soil

**Equipment Rental:**
- Plate compactor (3 days @ $90): $270
- Laser level (1 day @ $50): $50
- **Total rental: $320**

**Professional:**
- Excavator w/ operator: $150-250/hour (4-6 hours)
- **Total professional: $800-1,500**

**Grand Total:**
- **DIY: $450-770** (reuse soil + rent compactor)
- **Professional: $1,200-2,000**

**PRO TIPS:**
💡 Rent plate compactor - SO much faster than hand tamper
💡 Backfill in lifts (6-12") - never dump whole pile at once
💡 Compact each lift - don't skip this!
💡 Slope away from house - most important grading rule
💡 Place best material (gravel/sand) at bottom - drains near foundation
💡 Water slightly if soil very dry - helps compaction
💡 Don't backfill too soon - wall needs to cure
💡 Protect waterproofing - watch for tears/punctures
💡 Final grade 6" below siding - prevents rot

**COMMON MISTAKES:**
❌ Backfilling too soon (wall not cured)
❌ No compaction (settles later)
❌ Large rocks damage waterproofing
❌ Dumping all at once (uneven pressure on wall)
❌ Using topsoil as backfill (organic material)
❌ Wrong slope (water toward house)
❌ Soil too wet or frozen
❌ Not protecting waterproofing

**SAFETY:**
⚠️ Equipment operation - Read manual, training important
⚠️ Heavy lifting - Proper technique, ask for help
⚠️ Working in excavation - Trench can collapse if re-entering
⚠️ Underground utilities - Already marked but stay aware

**INSPECTION:**
🔍 Some jurisdictions require inspection before backfill
- Call inspector before starting
- They verify waterproofing, drainage
- Get approval, then backfill

**SUCCESS CRITERIA:**
✅ Waited minimum cure time (7+ days)
✅ Waterproofing protected (board or care)
✅ Backfill placed in 6-12\" lifts
✅ Each lift compacted (3-4 passes)
✅ No voids or air pockets
✅ Final grade slopes away from house
✅ Minimum 6\" drop in first 10 feet
✅ Topsoil in place (top 6-12\")
✅ Drainage paths clear
✅ No settlement after 2-3 weeks
✅ Ready for final landscaping
✅ Foundation ready for framing
""",
        why_now="Backfill completes foundation protection and enables framing to begin. Must be done carefully to avoid damage.",
        estimated_cost=800.0,  # DIY with equipment rental
        estimated_duration_days=4,
        requires_professional=False,
        professional_type="Excavator operator (optional, faster if available)",
        requires_permit=False,
        permit_type=None,
        safety_warnings=[
            "Heavy equipment - Proper training required for plate compactor",
            "Heavy lifting - Shoveling is strenuous, pace yourself",
            "Trench collapse risk - Don't enter deep excavation without shoring",
            "Underground utilities - Stay aware of marked lines"
        ],
        material_list=[
            {"item": "Crushed stone (for first 2-3 feet)", "quantity": "3", "unit": "cubic yards", "cost_per_unit": 50.0},
            {"item": "Sand (for middle section)", "quantity": "3", "unit": "cubic yards", "cost_per_unit": 35.0},
            {"item": "Topsoil (final 6-12 inches)", "quantity": "5", "unit": "cubic yards", "cost_per_unit": 40.0},
            {"item": "Grass seed or sod", "quantity": "1", "unit": "lot", "cost_per_unit": 150.0}
        ],
        tool_list=[
            "Plate compactor (rent $90/day)",
            "Hand tamper (backup/tight spaces)",
            "Shovel",
            "Rake",
            "Wheelbarrow",
            "Laser level (rent $50/day)",
            "Stakes and string",
            "Tape measure"
        ],
        reference_documents=[
            "IRC Section R401.3 (Drainage)",
            "Foundation plan (final grade elevations)",
            "Site plan (drainage swales)"
        ],
        video_tutorials=[
            "How to backfill a foundation",
            "Using a plate compactor",
            "Final grading for drainage",
            "Foundation backfill best practices"
        ],
        success_criteria=[
            "Waited 7+ days after concrete/grout",
            "Waterproofing protected during backfill",
            "Backfill placed in 6-12\" lifts",
            "Each lift compacted (3-4 passes minimum)",
            "No voids or air pockets",
            "Final grade slopes away (6\" drop in 10 feet minimum)",
            "Topsoil layer in place",
            "Drainage swales/paths clear",
            "No visible settlement",
            "Foundation ready for framing"
        ]
    )


def foundation_step_11_final_inspection(project_state) -> NextStep:
    """Step 11: Final Foundation Inspection"""
    return NextStep(
        step_number=11,
        title="Foundation Step 11: Final Foundation Inspection",
        description="""
✅ **FINAL INSPECTION - Foundation Approval & Moving Forward**

**What You're Doing:**
Getting building inspector's approval that foundation meets code. **Required before framing can begin.**

**Why This Matters:**
- Legal requirement (can't frame without approval)
- Verifies structural soundness
- Confirms code compliance
- Protects your investment
- Needed for financing/insurance
- Peace of mind before spending more $$

**BEFORE CALLING INSPECTOR:**

**Complete Your Checklist:**

**Foundation Work:**
- [ ] Footings poured and cured
- [ ] Foundation walls built to height
- [ ] Anchor bolts installed (1/2", every 4-6 feet)
- [ ] Waterproofing applied
- [ ] Drainage system installed
- [ ] Backfill completed
- [ ] Final grading sloped away

**Paperwork:**
- [ ] Permit posted/visible
- [ ] Foundation plans on site
- [ ] Previous inspection approvals
- [ ] Any engineering letters (if required)

**Site Cleanup:**
- [ ] Debris removed
- [ ] Materials organized
- [ ] Safe access for inspector
- [ ] Forms stripped and removed

**SCHEDULE INSPECTION:**

**How to Schedule:**
1. **Call building department** (48-72 hours ahead typical)
2. **Online portal** (many jurisdictions)
3. **Automated phone system**

**What to Say:**
"I need to schedule a final foundation inspection for permit #[your permit number] at [address]. The foundation is complete and ready for inspection."

**They'll Ask:**
- Permit number
- Property address
- Your contact phone
- Preferred date/time window
- Type of foundation (slab, crawlspace, basement)

**Inspector Will Come:**
- Usually 1-3 days after request
- Morning typically (8-11 AM)
- 15-30 minute inspection
- You should be present

**WHAT INSPECTOR CHECKS:**

**Footings:**
- ✅ Width and depth per plan
- ✅ Below frost line
- ✅ Clean (no mud, debris)
- ✅ Level

**Walls:**
- ✅ Height correct
- ✅ Thickness per code (8-12")
- ✅ Plumb (vertical)
- ✅ No major cracks or voids

**Anchor Bolts:**
- ✅ 1/2" diameter minimum
- ✅ 7" embedment into concrete/grout
- ✅ Maximum 6 feet spacing
- ✅ Within 12" of corners/ends
- ✅ 1-3/4" from edge (centered for sill plate)
- ✅ Threaded portion extends 2-1/2"+ above wall

**Waterproofing:**
- ✅ Applied below grade
- ✅ Complete coverage
- ✅ Extends 6" above final grade

**Drainage:**
- ✅ Perimeter drain installed
- ✅ Proper slope away from house
- ✅ Outlet clear

**Backfill:**
- ✅ Properly compacted
- ✅ Graded for drainage
- ✅ No settlement visible

**Openings:**
- ✅ Window/door bucks in place (if applicable)
- ✅ Lintels over openings
- ✅ Proper sizing per plans

**COMMON INSPECTION ISSUES:**

**Anchor Bolts (Most Common):**
- ❌ Wrong spacing (>6 feet)
- ❌ Too close to edge (<1-3/4")
- ❌ Not enough embedment (<7")
- ❌ Missing at corners
- **Fix:** Drill and epoxy new bolts (inspector may allow)

**Grading:**
- ❌ Soil too close to siding (<6")
- ❌ Wrong slope (toward house)
- **Fix:** Re-grade before framing

**Waterproofing:**
- ❌ Visible gaps or thin spots
- ❌ Doesn't extend above grade
- **Fix:** Apply additional coat (if accessible)

**Backfill:**
- ❌ Settlement visible
- ❌ Not compacted
- **Fix:** Add soil and compact

**IF INSPECTION FAILS:**

**Don't Panic:**
- Very common to need minor fixes
- Inspector will note what's wrong
- Fix issues and call for re-inspection
- No additional fee usually

**Common Fixes:**
- Add/relocate anchor bolts
- Patch concrete defects
- Improve drainage
- Add backfill, re-grade

**Re-inspection:**
- Fix noted items
- Call for re-inspection
- Inspector verifies fixes
- Approval granted

**IF INSPECTION PASSES:**

**Inspector Actions:**
- Signs off on permit card
- Updates online system (if applicable)
- Provides inspection report
- Notes "Approved for framing"

**Your Actions:**
- Take photo of signed permit card
- Save inspection report
- Post approval on site
- Order framing lumber
- Schedule framing crew (if hiring)

**NEXT STEPS AFTER APPROVAL:**

**Immediate (Within Days):**
- Order sill plate (pressure-treated 2x6 or 2x8)
- Get sill gasket/seal (foam or sealant)
- Order termite treatment (if required)
- Schedule framing material delivery

**Within 1-2 Weeks:**
- Install sill plate (treated lumber on anchor bolts)
- Begin framing (floor system or walls)

**Moving Forward:**
- Frame house (walls, roof, windows)
- Rough-in mechanicals (plumbing, electric, HVAC)
- Insulation
- Drywall
- Finish work

**You're 8-10% done with house!**

**COST:**
- **Inspection fee:** Usually $0 (included in permit)
- **Re-inspection fee:** $0-100 (if needed)

**TIMELINE:**
- Schedule: 2-3 days ahead
- Inspection: 15-30 minutes
- Results: Immediate (on-site)
- **Total: 3-5 days** (scheduling + inspection)

**PRO TIPS:**
💡 Have inspector's card/number handy
💡 Be present for inspection (answer questions)
💡 Have permit and plans visible
💡 Clean site beforehand (shows professionalism)
💡 Take photos after approval (documentation)
💡 Don't start framing until approved (illegal, risky)
💡 Ask questions if something unclear
💡 Inspector is resource, not enemy

**QUESTIONS TO ASK INSPECTOR:**

If you're unsure about anything:
- "What's required for sill plate installation?"
- "Do I need termite treatment before framing?"
- "When should I call for framing inspection?"
- "Any concerns about the foundation?"
- "What's the next inspection milestone?"

Inspectors are usually helpful - they want your project to succeed!

**RED FLAGS (Call Inspector Before They Come):**

If you notice:
- Major cracks (>1/4")
- Walls out of plumb (>1" in 8 feet)
- Missing anchor bolts
- Wet/damp interior (drainage issue)

Better to ask how to fix than fail inspection.

**CELEBRATION TIME!** 🎉

**Foundation is done!** This is huge milestone:
- Hardest part structurally
- Most expensive component ($15K-40K)
- Most critical for longevity
- Everything builds on this

**Take a moment to appreciate:**
- You excavated correctly
- Footings are level and strong
- Walls are plumb and reinforced
- Waterproofing protecting investment
- Drainage working properly
- Code-compliant and inspected

**What's Next:**
- Foundation: ✅ COMPLETE
- Framing: Starting soon
- Roof: 4-8 weeks away
- Dried-in: 8-12 weeks away
- Move-in: 8-12 months away

**You're on your way to building your dream home!**

**SUCCESS CRITERIA:**
✅ All foundation work complete
✅ Inspection scheduled and completed
✅ Inspector approved/signed off
✅ Permit card updated (inspection passed)
✅ Inspection report received
✅ Site cleaned up
✅ Photos documented
✅ Ready to order framing materials
✅ Ready to begin framing phase
✅ **FOUNDATION COMPLETE! 🎉**
""",
        why_now="Final inspection required by law before framing. Ensures foundation meets code and is structurally sound.",
        estimated_cost=0.0,  # Included in permit fee
        estimated_duration_days=3,
        requires_professional=False,
        professional_type=None,
        requires_permit=True,
        permit_type="Foundation inspection (continuation of building permit)",
        safety_warnings=[
            "Do not begin framing until inspection passes - illegal and risky",
            "Ensure safe access for inspector - clear walkways, no hazards"
        ],
        material_list=[],
        tool_list=[
            "Permit card and plans (for inspector)",
            "Camera/phone (document approval)",
            "Tape measure (if inspector asks for measurements)",
            "Level (if inspector wants to verify)",
            "Broom (clean site beforehand)"
        ],
        reference_documents=[
            "Foundation plans (have on site)",
            "Previous inspection approvals",
            "Engineering letters (if applicable)",
            "Permit card"
        ],
        video_tutorials=[
            "What to expect during foundation inspection",
            "Preparing for building inspections"
        ],
        success_criteria=[
            "All foundation work complete per plans",
            "Inspection scheduled 2-3 days ahead",
            "Site cleaned and organized",
            "Permit and plans visible",
            "Inspector visited site",
            "All items on checklist approved",
            "Permit card signed off",
            "Inspection report received",
            "Approval to begin framing granted",
            "Photos of completed foundation taken",
            "FOUNDATION PHASE COMPLETE!"
        ]
    )


# FOUNDATION PHASE COMPLETE - All 11 Steps Implemented
# Map step numbers to functions
FOUNDATION_STEPS = {
    1: foundation_step_1_excavation,
    2: foundation_step_2_footing_layout,
    3: foundation_step_3_footing_forms,
    4: foundation_step_4_rebar,
    5: foundation_step_5_inspection,
    6: foundation_step_6_concrete_pour,
    7: foundation_step_7_strip_forms,
    8: foundation_step_8_walls,
    9: foundation_step_9_waterproofing,
    10: foundation_step_10_backfill,
    11: foundation_step_11_final_inspection
}


def get_foundation_step(step_number: int, project_state) -> NextStep:
    """Get a specific foundation step by number."""
    if step_number not in FOUNDATION_STEPS:
        raise ValueError(f"Invalid foundation step number: {step_number}. Must be 1-11.")
    
    step_func = FOUNDATION_STEPS[step_number]
    return step_func(project_state)


def get_all_foundation_steps(project_state) -> list:
    """Get all 11 foundation steps."""
    return [get_foundation_step(i, project_state) for i in range(1, 12)]
