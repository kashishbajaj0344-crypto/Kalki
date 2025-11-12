# 📚 KALKI v2.5 PDF Download Checklist

## Your Mission: Download Construction Knowledge Base

**Goal:** Collect 50-100 high-quality PDFs to train KALKI v2.5 to deliver 10/10 professional-grade construction deliverables.

**Timeline:** Download these over the next 1-2 weeks while I continue building the system.

---

## 🎯 **CRITICAL PRIORITY - Download First (This Week)**

### **1. BC Building Code Part 9** ✅ MUST HAVE
- **Source:** https://www.bccodes.ca/ (FREE official download)
- **File:** BC Building Code 2018 - Division B Part 9
- **Pages:** ~500 pages
- **Contains:**
  - Span tables for joists, beams, rafters
  - Load requirements (live load, dead load, snow load)
  - Prescriptive construction paths
  - Foundation requirements
  - Fire safety requirements

### **2. Structural Engineering Handbooks** (academia.edu)
**Search Terms:** "structural engineering handbook", "wood design handbook", "concrete design handbook"

Required PDFs:
- [ ] "Timber Construction Manual" by American Institute of Timber Construction
- [ ] "Design of Wood Structures" by Donald Breyer (6th or 7th edition)
- [ ] "Reinforced Concrete Design" by Salmon & Wang
- [ ] "Steel Structures Design and Behavior" by Salmon & Johnson
- [ ] "Structural Engineering Handbook" by Chen & Lui

**Why Critical:** These contain the span tables, formulas, and member sizing tables that KALKI needs for accurate structural calculations.

### **3. Construction Methods Textbooks** (academia.edu)
**Search Terms:** "construction methods management", "residential construction", "building construction illustrated"

Required PDFs:
- [ ] "Construction Methods and Management" by Nunnally
- [ ] "Residential Construction Academy: Carpentry" 
- [ ] "Building Construction Illustrated" by Francis Ching
- [ ] "Fundamentals of Building Construction" by Edward Allen
- [ ] "Construction Technology" by Roy Chudley

**Why Critical:** These have the step-by-step procedures KALKI needs to generate construction sequences.

---

## 💰 **HIGH PRIORITY - Download Week 2**

### **4. Cost Estimating Resources**

**Option A: Purchase RSMeans (Recommended)**
- [ ] "RSMeans Building Construction Cost Data 2024" ($429 on Amazon)
- **Worth it because:** This is THE industry standard for construction costs
- KALKI will use this to generate accurate project budgets

**Option B: Free Alternatives (academia.edu)**
- [ ] "Construction Estimating Using Excel" by Steven Peterson  
- [ ] "Estimating Building Costs" by Wayne Del Pico
- [ ] "Construction Costs Analysis" textbooks

### **5. Building Inspection Manuals** (academia.edu)
**Search Terms:** "building inspector handbook", "code check", "construction inspection"

Required PDFs:
- [ ] "Code Check Complete" by Redwood Kardon & Hansen
- [ ] "Building Inspector's Field Guide" (ICC)
- [ ] "Residential Building Inspector Practice Manual"
- [ ] "Field Guide to Residential Construction" by Steven Bliss
- [ ] "Construction Defects Prevention Manual"

**Why Critical:** These contain the inspection criteria and QC checklists KALKI needs.

### **6. Municipal Building Bylaws** (FREE from municipal websites)

Download these official PDFs:
- [ ] City of Vancouver Building Bylaw
- [ ] City of Victoria Building Regulations Bylaw  
- [ ] District of Sechelt Building Bylaw
- [ ] Resort Municipality of Whistler Building Regulations
- [ ] City of Kelowna Building Standards

**How to find:** Google "[city name] building bylaw PDF"

---

## 📖 **MEDIUM PRIORITY - Download Week 3-4**

### **7. Mechanical & Electrical Standards** (academia.edu)
**Search Terms:** "HVAC handbook", "electrical code handbook", "plumbing code"

Required PDFs:
- [ ] "ASHRAE Handbook - Fundamentals" (2021 or newer)
- [ ] "National Electrical Code Handbook" 
- [ ] "Plumbing Engineering Design Handbook" 
- [ ] "HVAC Design Manual"

### **8. Materials Science** (academia.edu)
**Search Terms:** "construction materials", "materials science civil engineering", "wood handbook"

Required PDFs:
- [ ] "Wood Handbook: Wood as an Engineering Material" (USDA - FREE)
- [ ] "Materials Science for Civil Engineers" 
- [ ] "Construction Materials" by Mindess & Young
- [ ] "Concrete Technology" by M.S. Shetty
- [ ] "Steel Designers Manual"

### **9. Energy Efficiency & Sustainability** (Free + academia.edu)
**Search Terms:** "passive house", "energy step code", "green building"

Required PDFs:
- [ ] "BC Energy Step Code Guide" (FREE from BC Govt)
- [ ] "Passive House Design Manual" 
- [ ] "Green Building Handbook" by Tom Woolley
- [ ] "Builder's Guide to Cold Climates"

---

## 🔬 **FUTURE - Month 2-3**

### **10. Advanced Topics** (academia.edu)
- [ ] "Seismic Design Manual" by AISC
- [ ] "Wind Effects on Structures" by Emil Simiu  
- [ ] "Fire Protection Handbook" (NFPA)
- [ ] "Accessible Design Guide" (CSA)

### **11. Foundation & Geotechnical** (academia.edu)
- [ ] "Foundation Design: Principles and Practices" by Coduto
- [ ] "Geotechnical Engineering Handbook" by Braja Das
- [ ] "Residential Foundations" by NAHB

---

## 📥 **Download Strategy Tips**

### **Academia.edu Search Tips:**
1. Create free account at academia.edu
2. Search exact titles from checklist above
3. Look for PDFs with 100+ pages (full textbooks, not excerpts)
4. Download highest quality version available
5. Save with descriptive filenames: "Breyer_Wood_Design_7th_Ed.pdf"

### **Filename Convention:**
```
[Author]_[Title]_[Edition/Year].pdf

Examples:
Breyer_Wood_Design_7th_Ed.pdf
BC_Building_Code_Part9_2018.pdf
RSMeans_Cost_Data_2024.pdf
Ching_Building_Construction_Illustrated_6th_Ed.pdf
```

### **Organize Your Downloads:**
```
~/Desktop/KALKI_PDFs/
  ├── 01_Building_Codes/
  ├── 02_Structural_Engineering/
  ├── 03_Construction_Methods/
  ├── 04_Cost_Estimating/
  ├── 05_Inspection_QC/
  ├── 06_MEP_Systems/
  ├── 07_Materials_Science/
  ├── 08_Energy_Efficiency/
  └── 09_Advanced_Topics/
```

---

## ✅ **As You Download, Run This:**

After downloading each PDF:

```bash
# Navigate to KALKI directory
cd ~/Desktop/Kalki

# Ingest the PDF
python3 kalki_cli.py learn ingest "/path/to/downloaded.pdf"

# Check progress
python3 kalki_cli.py learn stats
```

---

## 📊 **Target Knowledge Base (v2.5 Goals)**

| Knowledge Type | Current | Target | Status |
|---------------|---------|---------|--------|
| **Formulas** | 4,896 | 6,000+ | ✅ Good |
| **Materials** | 2 | 500+ | ❌ **PRIORITY** |
| **Design Rules** | 9 | 200+ | ❌ **PRIORITY** |
| **Code Requirements** | 2 | 1,000+ | ❌ **PRIORITY** |
| **Span Tables** | 0 | 500+ | 🆕 v2.5 |
| **Procedures** | 0 | 200+ | 🆕 v2.5 |
| **Inspection Criteria** | 0 | 150+ | 🆕 v2.5 |
| **Cost Data** | 0 | 1,000+ | 🆕 v2.5 |
| **Load Parameters** | 0 | 100+ | 🆕 v2.5 |
| **Decision Trees** | 0 | 200+ | 🆕 v2.5 |

**Total Target:** 9,000+ knowledge items

---

## 🎯 **Week-by-Week Download Plan**

### **Week 1 (Now):**
- [ ] BC Building Code Part 9
- [ ] 5 structural engineering handbooks
- [ ] 3 construction methods textbooks
- **Goal:** 8-10 PDFs, ~5,000 pages

### **Week 2:**
- [ ] RSMeans cost data (purchase)
- [ ] 5 inspection manuals
- [ ] 5 municipal bylaws
- **Goal:** 10-12 PDFs, ~3,000 pages

### **Week 3-4:**
- [ ] MEP standards
- [ ] Materials science
- [ ] Energy efficiency
- **Goal:** 15-20 PDFs, ~4,000 pages

### **Month 2:**
- [ ] Advanced topics
- [ ] Geotechnical
- [ ] Specialized domains
- **Goal:** 20+ PDFs, ~5,000 pages

---

## 🚀 **Once You Have First 10 PDFs:**

We can start testing KALKI v2.5 on real construction projects!

**Test Projects:**
1. Single-family home (2-story, 2,000 sq ft)
2. Garage addition (400 sq ft)
3. Deck construction (300 sq ft)
4. Kitchen renovation
5. Basement finishing

KALKI will generate:
- ✅ Full construction drawings
- ✅ Bill of materials with costs
- ✅ Construction schedule (step-by-step)
- ✅ Inspection checklists
- ✅ Code compliance verification
- ✅ Professional-grade deliverables

---

## 📞 **Questions While Downloading?**

Track your progress in this checklist and let me know:
- Which PDFs you've successfully downloaded
- Any you can't find (I'll suggest alternatives)
- Which categories you want to prioritize

**Remember:** Quality > Quantity. Better to have 50 high-quality construction manuals than 200 random PDFs.

---

**Status:** Download in progress...  
**Last Updated:** November 7, 2025  
**Target Completion:** November 21, 2025
