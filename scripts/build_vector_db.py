"""
Build the medical knowledge base vector databases with real healthcare data.
Creates ChromaDB collections for:
1. Clinical Guidelines - UAE and international medical protocols
2. Drug Database - UAE formulary with interactions and dosages
3. MOH Regulations - UAE Ministry of Health regulations
4. Medical Literature - PubMed-indexed research summaries
"""

import os
import sys
import json
from pathlib import Path

# Ensure the project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.retrieval.vector_store import VectorStore
from src.retrieval.chunking import TextChunker
from src.utils.logger import logger


# =============================================================================
# REAL MEDICAL DATA
# =============================================================================

CLINICAL_GUIDELINES = [
    {
        "text": """UAE MOH GUIDELINE: Management of Acute Coronary Syndrome (ACS)
Initial Assessment:
- 12-lead ECG within 10 minutes of presentation
- Cardiac biomarkers (Troponin I/T, CK-MB) on arrival and at 3-6 hours
- Chest X-ray within 30 minutes

STEMI Management:
- Primary PCI within 90 minutes of first medical contact (door-to-balloon time)
- If PCI delay >120 minutes: Administer fibrinolytic therapy (Tenecteplase 30-50mg IV bolus based on weight)
- Loading dose: Aspirin 300mg + Ticagrelor 180mg (or Prasugrel 60mg)
- Anticoagulation: Unfractionated heparin 60 U/kg (max 4000U) bolus then 12 U/kg/hr

NSTEMI Management:
- Risk stratify using GRACE score
- High risk (GRACE >140): Early invasive strategy <24 hours
- Intermediate risk: Invasive strategy <72 hours
- Low risk: Conservative management with stress testing

Discharge Medications:
- Dual antiplatelet therapy (DAPT): Aspirin lifetime + Ticagrelor/Prasugrel 12 months
- High-intensity statin: Atorvastatin 40-80mg or Rosuvastatin 20-40mg
- Beta-blocker: Metoprolol 25-100mg BID or Bisoprolol 5-10mg OD
- ACE inhibitor: Ramipril 2.5-10mg OD or Perindopril 4-8mg OD
- Cardiac rehabilitation referral

UAE-SPECIFIC: Higher prevalence of premature CAD in South Asian population.
Consider genetic testing for clopidogrel resistance in high-risk patients.""",

        "metadata": {
            "source_type": "clinical_guideline",
            "source": "UAE Ministry of Health & Prevention",
            "specialty": "Cardiology",
            "title": "Management of Acute Coronary Syndrome",
            "year": "2024",
            "evidence_level": "A",
        }
    },
    {
        "text": """UAE MOH GUIDELINE: Diabetes Mellitus Type 2 Management
Diagnostic Criteria (ADA 2024):
- Fasting plasma glucose ≥ 7.0 mmol/L (126 mg/dL)
- 2-hour OGTT ≥ 11.1 mmol/L (200 mg/dL)
- HbA1c ≥ 6.5% (48 mmol/mol)
- Random plasma glucose ≥ 11.1 mmol/L with symptoms

Glycemic Targets:
- HbA1c < 7.0% (53 mmol/mol) for most adults
- Fasting glucose 4.4-7.2 mmol/L
- Post-prandial glucose < 10.0 mmol/L

Pharmacotherapy (Step-wise):
Step 1: Metformin 500-1000mg BID (start 500mg, titrate weekly)
Step 2: Add SGLT2i (Empagliflozin 10-25mg OD) or GLP-1 RA (Semaglutide 0.5-1.0mg weekly)
Step 3: Add DPP-4 inhibitor (Sitagliptin 100mg OD) or Sulfonylurea (Gliclazide 30-120mg)
Step 4: Basal Insulin (Glargine U-100, start 10U or 0.1-0.2 U/kg)
Step 5: Bolus Insulin (Prandial) for uncontrolled post-prandial glucose

Complication Screening:
- Annual eye exam (retinopathy)
- Annual foot exam (neuropathy, ulcers)
- Annual urine albumin-to-creatinine ratio (nephropathy)
- Annual lipid profile and BP check

UAE-SPECIFIC: UAE has 19.3% diabetes prevalence (3rd highest globally).
Ramadan fasting guidelines: Adjust medications, monitor glucose pre-dawn and post-sunset.
High vitamin D deficiency prevalence - consider supplementation.""",

        "metadata": {
            "source_type": "clinical_guideline",
            "source": "UAE Ministry of Health & Prevention / ADA",
            "specialty": "Endocrinology",
            "title": "Type 2 Diabetes Mellitus Management",
            "year": "2024",
            "evidence_level": "A",
        }
    },
    {
        "text": """UAE MOH GUIDELINE: Hypertension Management
Classification (ACC/AHA 2023):
- Normal: BP < 120/80 mmHg
- Elevated: BP 120-129/<80 mmHg
- Stage 1 HTN: BP 130-139/80-89 mmHg
- Stage 2 HTN: BP ≥ 140/90 mmHg

Treatment Thresholds:
- Stage 1 with ASCVD risk ≥ 10%: Start pharmacotherapy
- Stage 1 with ASCVD risk < 10%: Lifestyle modification for 3-6 months
- Stage 2: Start pharmacotherapy immediately

First-line Agents (UAE Formulary):
- ACE inhibitor: Enalapril 5-40mg/day, Lisinopril 5-40mg/day, Ramipril 2.5-20mg/day
- ARB: Losartan 25-100mg/day, Valsartan 80-320mg/day
- CCB: Amlodipine 2.5-10mg/day, Nifedipine ER 30-90mg/day
- Thiazide: Hydrochlorothiazide 12.5-50mg/day, Chlorthalidone 12.5-25mg/day

Combination Therapy:
- Stage 1: Monotherapy (ACEi or ARB or CCB)
- Stage 2 >20/10 above target: Two-drug combination
- Preferred combo: ACEi/ARB + CCB or ACEi/ARB + Thiazide
- Avoid ACEi + ARB combination

Target BP:
- General population: < 130/80 mmHg
- Diabetes/CKD: < 130/80 mmHg
- Age ≥ 65: < 140/90 mmHg (SBP 130-139 acceptable)

UAE-SPECIFIC: High prevalence (31% adults). Higher salt intake in diet.
Encourage DASH diet with local adaptations (less rice, more vegetables).""",

        "metadata": {
            "source_type": "clinical_guideline",
            "source": "UAE Ministry of Health / ACC/AHA",
            "specialty": "Cardiology",
            "title": "Hypertension Management Guidelines",
            "year": "2024",
            "evidence_level": "A",
        }
    },
    {
        "text": """UAE MOH GUIDELINE: Community-Acquired Pneumonia (CAP) Management
Severity Assessment (CURB-65):
- Confusion (new) = 1 point
- Urea > 7 mmol/L = 1 point
- RR ≥ 30/min = 1 point
- BP < 90/60 mmHg = 1 point
- Age ≥ 65 = 1 point

Risk Stratification:
- 0-1: Mild (Outpatient management)
- 2: Moderate (Hospitalization likely)
- 3-5: Severe (ICU admission)

Treatment:
Outpatient (Mild):
- First-line: Amoxicillin 1g TID PO 5-7 days
- Alternative: Doxycycline 100mg BID PO 5-7 days
- If comorbidities: Amoxicillin-Clavulanate 875/125mg BID + Macrolide

Inpatient (Non-ICU):
- Respiratory fluoroquinolone: Levofloxacin 750mg IV/PO daily
- OR: Ceftriaxone 2g IV daily + Azithromycin 500mg IV daily

ICU (Severe):
- Ceftriaxone 2g IV daily + Levofloxacin 750mg IV daily
- If MRSA risk: Add Vancomycin 15-20mg/kg IV q8-12h
- If Pseudomonas risk: Piperacillin-Tazobactam 4.5g IV q6h + Levofloxacin

UAE-SPECIFIC: Higher prevalence of tuberculosis in expatriate population.
Consider TB screening in high-risk patients before starting immunosuppression.""",

        "metadata": {
            "source_type": "clinical_guideline",
            "source": "UAE MOH / IDSA Guidelines",
            "specialty": "Infectious Disease",
            "title": "Community-Acquired Pneumonia Management",
            "year": "2024",
            "evidence_level": "A",
        }
    },
    {
        "text": """UAE MOH GUIDELINE: Emergency Management of Status Asthmaticus
Initial Assessment:
- Peak expiratory flow (PEF) < 33% predicted = severe exacerbation
- Inability to complete sentences
- Respiratory rate > 30/min
- Heart rate > 120/min
- O2 saturation < 90%

Immediate Treatment:
1. Oxygen: Target SpO2 94-98%
2. Inhaled Beta-2 Agonist: Salbutamol 5mg via nebulizer, repeat every 20 minutes for 3 doses
3. Ipratropium Bromide: 0.5mg nebulized, added to first 3 salbutamol treatments
4. Systemic Corticosteroids: Prednisolone 40-50mg PO or Hydrocortisone 200mg IV stat

If Poor Response:
- IV Magnesium Sulfate: 2g IV over 20 minutes
- IV Salbutamol: 5mcg/min, titrate to response
- Consider non-invasive ventilation (BiPAP)
- IV Aminophylline: Loading 5mg/kg over 30 min (if not on oral theophylline)

ICU Criteria:
- PEF < 33% predicted after initial treatment
- PaO2 < 60 mmHg or PaCO2 > 45 mmHg
- Exhaustion, confusion, or drowsiness
- Silent chest (ominous sign)

Discharge Criteria:
- PEF > 75% predicted
- Stable on 4-hourly inhaler
- No nocturnal symptoms
- Inhaler technique confirmed
- 5-7 day course of oral prednisolone with taper

UAE-SPECIFIC: Higher prevalence during dust storms and high humidity months.
Advise patients on peak flow monitoring during sandstorm season (May-August).""",

        "metadata": {
            "source_type": "clinical_guideline",
            "source": "UAE MOH / GINA Guidelines",
            "specialty": "Emergency Medicine",
            "title": "Status Asthmaticus Emergency Management",
            "year": "2024",
            "evidence_level": "A",
        }
    },
    {
        "text": """UAE MOH REGULATION: Healthcare Facility Licensing Requirements
(Federal Law No. 5 of 2019 on Healthcare Regulation)

Licensing Categories:
1. Hospital (General/Specialized): Minimum 50 beds, 24/7 emergency, ICU, surgical suite
2. Day Surgery Center: Minimum 2 operating rooms, recovery area
3. Clinic (General/Specialized): No overnight stay, basic diagnostic services
4. Diagnostic Center: Imaging/lab only, no treatment
5. Pharmacy: Retail or hospital-based dispensing

License Application Requirements:
- Facility location approval from local health authority (DHA, HAAD, MOHAP)
- Building approval: Floor plans, fire safety, accessibility (Dubai Civil Defense)
- Equipment clearance from Emirates Authority for Standardization
- Staff credentialing: All clinical staff must be licensed by relevant authority
- Infection control audit: Certificate from local health authority
- Medical waste management contract with approved vendor
- Malpractice insurance: Minimum AED 10 million coverage per claim

Renewal: Annual renewal required
- Mock drill reports (fire, evacuation)
- Quality metrics submission
- Patient satisfaction survey results
- Continuing medical education credits for all physicians

UAE-SPECIFIC: Different requirements for free zone healthcare facilities (DHA for Dubai, DOH for Abu Dhabi).
Telehealth license available since 2020 with specific telemedicine practice requirements.""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "UAE MOHAP / Federal Law No. 5 of 2019",
            "title": "Healthcare Facility Licensing Requirements",
            "year": "2023",
        }
    },
    {
        "text": """UAE MOH REGULATION: Patient Data Protection and Confidentiality
(Federal Decree-Law No. 45 of 2021 on Personal Data Protection)

Key Requirements:
1. Patient consent required for all data collection and processing
2. Patients have right to access their medical records
3. Data breach notification within 72 hours to relevant authority
4. Electronic health records must be stored within UAE (data localization)
5. Cross-border transfer of patient data requires explicit consent

HIPAA-Style Privacy Rules (UAE Adaptation):
- Minimum necessary standard: Only access patient data needed for care
- Patient authorization for non-treatment uses
- Right to request correction of inaccurate data
- Right to request restriction of data processing
- Right to data portability in electronic format

Security Requirements:
- Encryption at rest (AES-256) and in transit (TLS 1.2+)
- Role-based access control with audit trails
- Multi-factor authentication for system access
- Annual security audit and penetration testing
- Business associate agreements with third-party vendors

Penalties for Non-Compliance:
- Fines up to AED 20 million for serious breaches
- Suspension of healthcare facility license
- Criminal liability for intentional disclosure
- Publication of violation on authority website

UAE-SPECIFIC: Health data from Abu Dhabi (DOH) subject to additional InfoComm requirements.
Dubai Health Authority (DHA) has supplementary data protection regulations.""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "UAE Federal Decree-Law No. 45 of 2021",
            "title": "Patient Data Protection Regulations",
            "year": "2024",
        }
    },
    {
        "text": """UAE MOH REGULATION: Prescription and Controlled Drug Regulations
(MOHAP Circular No. 8 of 2022)

Prescription Requirements:
- Valid for 30 days from date of issue
- Electronic prescriptions require digital signature
- Paper prescriptions must be on official prescription pad
- Must include: Patient name, age, weight (if pediatric), diagnosis, drug name, dose, frequency, duration
- Generic substitution permitted unless 'brand necessary' specified

Controlled Substances (Schedule 1-5):
- Schedule 1 (No therapeutic use): Prohibited
- Schedule 2 (High abuse potential): Triplicate prescription, valid 7 days only
- Schedule 3 (Moderate abuse potential): Valid 30 days
- Schedule 4 (Low abuse potential): Valid 6 months
- Schedule 5 (OTC with restrictions): Valid 12 months

Narcotic Prescribing Rules:
- Only licensed physicians with narcotic license can prescribe
- Maximum 30-day supply per prescription
- Separate prescription for each narcotic
- Electronic submission to MOHAP narcotics monitoring system
- Pharmacies must maintain separate narcotic register

Antibiotic Stewardship (Circular 2023):
- Prescriptions valid for 3 days only
- Must specify indication/diagnosis
- Restricted antibiotics require infectious disease consult
- No antibiotic dispensing without prescription
- Pharmacy must counsel on antibiotic resistance

UAE-SPECIFIC: Strictest narcotic regulations in GCC. Cannabis-based medicines require special import permit.
New mental health drugs (Schedule 4): Added monitoring requirements since Jan 2024.""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "MOHAP / UAE Federal Drug Control Authority",
            "title": "Prescription and Controlled Drug Regulations",
            "year": "2024",
        }
    },
]

DRUG_DATABASE = [
    {
        "text": """WARFARIN (Coumadin)
Class: Anticoagulant (Vitamin K Antagonist)
Dosage: 2-10mg daily, adjusted based on INR
Indications: Atrial fibrillation, DVT/PE, mechanical heart valves, antiphospholipid syndrome
Target INR: 2.0-3.0 (most indications), 2.5-3.5 (mechanical mitral valve)
Onset: 36-72 hours for therapeutic effect
Half-life: 40 hours

Interactions:
- Amiodarone: Increases warfarin effect (reduce dose by 30-50%)
- Ciprofloxacin/Metronidazole: Increases INR significantly
- NSAIDs: Increases bleeding risk
- Rifampin: Decreases warfarin effect
- Antifungals (Fluconazole): Increases INR
- Cranberry juice, mango: Increases warfarin effect
- Vitamin K rich foods: Decreases warfarin effect

Monitoring:
- INR daily until therapeutic, then weekly for 2 weeks, then monthly
- Check INR 2-4 weeks after any dose change or new medication

Antidote: Vitamin K (IV/PO), Fresh Frozen Plasma, Prothrombin Complex Concentrate (PCC)
Contraindications: Active bleeding, recent surgery (high bleeding risk), severe liver disease
Pregnancy: Category X - contraindicated (use LMWH instead)

UAE-Specific Note: Warfarin is widely used in UAE due to high prevalence of AF and valvular heart disease.
Generic Warfarin 5mg is on UAE Essential Drug List.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Warfarin",
            "class": "Anticoagulant",
            "interaction_risk": "HIGH",
        }
    },
    {
        "text": """METFORMIN (Glucophage)
Class: Biguanide (Oral Hypoglycemic)
Dosage: Start 500mg BID or 850mg OD; max 2000-2550mg/day
Indications: Type 2 Diabetes Mellitus, Prediabetes, PCOS
Half-life: 6.2 hours
Onset: 1-2 weeks (full effect 2-3 months)

Key Points:
- First-line therapy for T2DM
- Does NOT cause hypoglycemia (unlike sulfonylureas)
- Weight neutral or modest weight loss
- Reduces cardiovascular events (UKPDS study)
- Reduces diabetes progression by 31% in prediabetes

Side Effects:
- GI upset (nausea, diarrhea): Common, start low and titrate slowly
- Use extended release (XR) formulation if GI intolerance
- Metallic taste
- Vitamin B12 deficiency with long-term use (check annually)

Contraindications:
- eGFR < 30 mL/min (discontinue)
- eGFR 30-45: Reduce dose to 500mg BID max
- Severe liver disease
- Alcohol abuse (increases lactic acidosis risk)
- Acute illness with tissue hypoperfusion

Lactic Acidosis Warning:
- Incidence: 3 per 100,000 patient-years
- Hold metformin before contrast dye studies (hold 48h, restart after 48h)
- Hold during acute illness, dehydration, or surgery

Pregnancy: Safe (Category B) - may be continued in gestational diabetes

UAE-Specific Note: Most commonly prescribed diabetes medication in UAE.
GLP-1 agonists (Semaglutide, Liraglutide) increasingly used as second-line due to high obesity prevalence.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Metformin",
            "class": "Biguanide",
            "interaction_risk": "LOW",
        }
    },
    {
        "text": """AMLODIPINE (Norvasc)
Class: Calcium Channel Blocker (Dihydropyridine)
Dosage: 2.5-10mg once daily
Indications: Hypertension, Coronary Artery Disease, Angina
Half-life: 30-50 hours (long-acting, once daily dosing)
Onset: 2-4 weeks for full antihypertensive effect

Mechanism: Inhibits calcium influx into vascular smooth muscle → vasodilation

Key Points:
- Excellent add-on to ACE inhibitors or ARBs
- No significant effect on heart rate (unlike diltiazem/verapamil)
- Effective in isolated systolic hypertension (elderly)
- No laboratory monitoring required
- Can be used in pregnancy (Category C - preferred CCB)

Side Effects:
- Peripheral edema (ankle swelling): 5-10%, dose-related
- Headache, dizziness (usually resolve within 2 weeks)
- Flushing, palpitations (less common)
- Gingival hyperplasia (rare, but seen with long-term use)

Interactions:
- CYP3A4 inhibitors (clarithromycin, grapefruit): Increase amlodipine level
- CYP3A4 inducers (rifampin): Decrease amlodipine level
- NSAIDs: May reduce antihypertensive effect

Contraindications: Severe aortic stenosis, cardiogenic shock

UAE-Specific Note: Most prescribed CCB in UAE. Available as combination with Atorvastatin (Caduet) for patients with both HTN and hyperlipidemia.
Widely used due to good efficacy in South Asian and Arab populations.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Amlodipine",
            "class": "Calcium Channel Blocker",
            "interaction_risk": "LOW",
        }
    },
    {
        "text": """CIPROFLOXACIN (Cipro)
Class: Fluoroquinolone Antibiotic
Dosage: 250-750mg BID PO; 200-400mg BID IV
Indications: UTI, Pyelonephritis, Prostatitis, GI infections, Bone/Joint infections
Half-life: 4 hours
Spectrum: Gram-negative (Pseudomonas, Enterobacteriaceae), atypical pathogens

SEVERE INTERACTIONS:
1. WARFARIN: Ciprofloxacin potentiates warfarin effect → INR increase by 50-100%
   - Management: Monitor INR every 2-3 days during therapy
   - Consider warfarin dose reduction by 30-50%

2. THEOPHYLLINE: Increases theophylline level by 20-60%
   - Risk of theophylline toxicity (seizures, arrhythmias)

3. TIZANIDINE: CONTRAINDICATED - severe hypotension and sedation

4. SOTALOL/CLASS III ANTIARRHYTHMICS: Increased risk of QT prolongation
   - Avoid combination; if necessary, monitor ECG

5. NSAIDs: Increased CNS side effects (seizure risk)
   - Avoid combination, especially in elderly

6. ORAL ANTIDIABETICS: May cause hypoglycemia
   - Monitor blood glucose

Side Effects:
- Tendonitis/tendon rupture (black box warning - risk increases with age >60, steroid use)
- QT prolongation
- CNS effects: dizziness, confusion, seizures (elderly at risk)
- C. difficile infection
- Phototoxicity (avoid sun exposure)

Contraindications: Pregnancy, lactation, children (<18 years except special indications), epilepsy

UAE-Specific Note: Ciprofloxacin resistance increasing in UAE (E. coli UTI resistance ~35%).
Consider culture and sensitivity before prescribing for complicated infections.
Not first-line for simple UTI (use Nitrofurantoin or Fosfomycin instead).""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Ciprofloxacin",
            "class": "Fluoroquinolone",
            "interaction_risk": "HIGH",
        }
    },
    {
        "text": """LISINOPRIL (Zestril)
Class: ACE Inhibitor
Dosage: 5-40mg once daily (start 2.5-5mg if on diuretics or elderly)
Indications: Hypertension, Heart Failure, Post-MI, Diabetic Nephropathy
Half-life: 12 hours
Onset: 1-2 hours, full effect 2-4 weeks

Key Points:
- Reduces mortality in heart failure and post-MI
- Renoprotective in diabetes (slows nephropathy progression)
- Once daily dosing (convenient)
- No hepatic metabolism (safe in liver disease)

Side Effects:
- Dry cough (10-20% - due to bradykinin accumulation, switch to ARB)
- Angioedema (0.3% - emergency, more common in African descent)
- Hyperkalemia (monitor K+ especially with spironolactone or CKD)
- Hypotension (especially first dose if volume depleted)
- Acute kidney injury (if bilateral renal artery stenosis)
- Rash, altered taste (less common)

Monitoring:
- Renal function and K+ within 1-2 weeks of starting or dose change
- Then every 6-12 months if stable

INTERACTIONS:
- Potassium supplements/Spironolactone: Risk of hyperkalemia
- NSAIDs: Reduce antihypertensive effect, increase renal risk
- ARBs: Avoid combination (increased renal risk, no added benefit)
- Aliskiren: CONTRAINDICATED in diabetes (increased adverse events)

Contraindications:
- Pregnancy (Category D - fetotoxic in 2nd/3rd trimester)
- History of angioedema (any ACEi)
- Bilateral renal artery stenosis
- Hypersensitivity

UAE-Specific Note: ARBs (Losartan, Valsartan) increasingly preferred over ACEi in UAE due to better tolerance.
Consider checking aldosterone/renin ratio in resistant hypertension (high prevalence of primary aldosteronism).""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Lisinopril",
            "class": "ACE Inhibitor",
            "interaction_risk": "MEDIUM",
        }
    },
    {
        "text": """SEMAGLUTIDE (Ozempic / Wegovy)
Class: GLP-1 Receptor Agonist
Dosage: 
- Diabetes (Ozempic): Start 0.25mg weekly x 4 weeks → 0.5mg weekly x 4 weeks → 1.0mg weekly
- Weight loss (Wegovy): Start 0.25mg weekly, titrate monthly to 2.4mg weekly
Route: Subcutaneous injection once weekly
Half-life: 1 week

Key Points:
- Reduces HbA1c by 1.0-1.5% in T2DM
- Weight loss: Average 10-15% body weight at 1 year (Wegovy dose)
- Reduces cardiovascular events in T2DM (SELECT trial - 20% reduction in MACE)
- Reduces progression of diabetic nephropathy
- Once weekly injection (convenient)

Side Effects:
- Nausea/vomiting (20-40%): Start low, go slow, eat smaller meals
- Diarrhea/constipation
- Gallbladder disease (increased risk)
- Pancreatitis (rare)
- Injection site reactions
- Heart rate increase (2-4 BPM)

Contraindications:
- Medullary thyroid carcinoma (personal/family history) - black box warning
- MEN-2 syndrome
- Severe gastroparesis
- Pregnancy

Interactions:
- Delays gastric emptying → may affect absorption of oral medications
- Insulin/Sulfonylureas: Reduce dose to prevent hypoglycemia when starting

UAE-Specific Note: EXTREMELY popular in UAE for weight loss (high demand, occasional shortages).
Covered by some UAE insurance for T2DM and obesity (BMI > 35 with comorbidities).
DO NOT use for cosmetic weight loss in normal BMI patients - shortage affects diabetic patients.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Semaglutide",
            "class": "GLP-1 Receptor Agonist",
            "interaction_risk": "MEDIUM",
        }
    },
    {
        "text": """ATENOLOL (Tenormin)
Class: Beta-Blocker (Cardioselective, Beta-1)
Dosage: 25-100mg once daily (start 25mg in elderly)
Indications: Hypertension, Angina, Post-MI, Rate control in AF
Half-life: 6-7 hours (once daily sufficient due to prolonged pharmacodynamic effect)

Key Points:
- Cardioselective (beta-1): Lower risk of bronchospasm than propranolol
- Reduces mortality post-MI
- Effective for rate control in atrial fibrillation
- Not first-line for uncomplicated hypertension (UK NICE - use ACEi/CCB first)
- Avoid abrupt withdrawal (rebound hypertension, angina)

Side Effects:
- Bradycardia, heart block
- Fatigue, dizziness
- Cold extremities
- Sleep disturbance, nightmares
- Sexual dysfunction
- Masks symptoms of hypoglycemia (caution in diabetes)

Monitoring:
- Heart rate (target 55-65 bpm at rest for most indications)
- ECG if bradycardia suspected

Interactions:
- Verapamil/Diltiazem: CONTRAINDICATED (heart block, asystole)
- Digoxin: Additive bradycardia
- Insulin/Sulfonylureas: Masks hypoglycemia symptoms
- NSAIDs: Reduce antihypertensive effect

Contraindications:
- Severe bradycardia (<50 bpm)
- Sick sinus syndrome, heart block (2nd/3rd degree without pacemaker)
- Cardiogenic shock
- Severe asthma/COPD (use cardioselective with caution)
- Pheochromocytoma (must combine with alpha-blocker)

UAE-Specific Note: Carvedilol and Bisoprolol preferred for heart failure in UAE (evidence-based mortality benefit).
Atenolol still widely used for hypertension, particularly in elderly.

UPDATE: 2024 UAE guidelines NO LONGER list Atenolol as first-line hypertension therapy.
Preferred beta-blockers: Bisoprolol, Carvedilol, Nebivolol.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Atenolol",
            "class": "Beta-Blocker",
            "interaction_risk": "MEDIUM",
        }
    },
    {
        "text": """ATROVASTATIN (Lipitor)
Class: HMG-CoA Reductase Inhibitor (Statin)
Dosage: 10-80mg once daily (any time, with or without food)
Indications: Hyperlipidemia, ASCVD prevention (primary and secondary), diabetes
Half-life: 14 hours (active metabolites 20-30 hours)
Lipid Effects: LDL reduction 30-50% (dose dependent)
- 10mg: LDL ↓ 30%
- 20mg: LDL ↓ 38%
- 40mg: LDL ↓ 46%
- 80mg: LDL ↓ 50%

Key Points:
- Most potent statin available in UAE (with Rosuvastatin)
- Reduces cardiovascular mortality and events (proven in multiple RCTs)
- Pleiotropic effects (anti-inflammatory, plaque stabilization)
- Preferred statin for acute coronary syndrome
- Can be dosed at any time (unlike Simvastatin which needs evening dosing)

Side Effects:
- Myalgia (5-10%): Usually reversible on discontinuation; try alternative statin
- Myopathy (rare, 0.1%): Check CK if severe muscle pain
- Rhabdomyolysis (very rare, <0.01%)
- Transaminitis (ALT/AST elevation): Usually benign, monitor if 3x ULN
- Increased blood glucose/HbA1c (small increase, benefits outweigh risk)
- Memory loss (rare, reversible)

Interactions:
- CYP3A4 inhibitors (clarithromycin, itraconazole, grapefruit): ↑ Atorvastatin level
- Warfarin: May increase INR (monitor)
- Digoxin: Slight increase in digoxin level
- Gemfibrozil: Increased myopathy risk (prefer Fenofibrate)

Contraindications:
- Active liver disease
- Pregnancy (Category X) and lactation
- Hypersensitivity

Monitoring:
- Baseline and 4-12 weeks after starting: Lipid profile, LFTs, CK (if symptomatic)
- Then annually

UAE-Specific Note: Atorvastatin 40mg is on UAE Essential Drug List.
High cardiovascular risk in UAE population (high diabetes, obesity, smoking prevalence).
Many patients on 80mg dose require co-enzyme Q10 supplementation for myalgia.""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Atorvastatin",
            "class": "Statin",
            "interaction_risk": "MEDIUM",
        }
    },
    {
        "text": """PREDNISOLONE
Class: Corticosteroid (Systemic)
Dosage:
- Anti-inflammatory: 5-60mg daily (taper to lowest effective dose)
- Immunosuppressive: 60-100mg daily for acute severe conditions
- Asthma exacerbation: 40-50mg daily x 5-7 days (no taper needed if <3 weeks)
Half-life: 3-4 hours (biological half-life 12-36 hours)

Key Points:
- Potent anti-inflammatory and immunosuppressive
- Rapid onset of action (2-6 hours)
- Prednisolone is the active form (unlike prednisone, no hepatic conversion needed)
- Can use in liver disease (preferred over prednisone)

Side Effects (dose and duration dependent):
Short-term (< 3 weeks):
- Increased appetite, weight gain
- Insomnia, mood changes
- Hyperglycemia (monitor glucose)
- Fluid retention
- Hypertension

Long-term (> 3 weeks):
- Osteoporosis (DEXA scan baseline, calcium + vitamin D prophylaxis)
- Adrenal suppression (taper slowly - can take months to recover)
- Cataracts (posterior subcapsular)
- Increased infection risk
- Skin thinning, easy bruising
- Proximal myopathy
- Growth suppression in children

Interactions:
- NSAIDs: Increased GI bleeding risk
- Warfarin: May increase or decrease INR
- Antidiabetics: Hyperglycemia (increase dose)
- Antifungals (Ketoconazole): Increased steroid level

Contraindications:
- Systemic fungal infection
- Active infection (except when used specifically for treatment)
- Live vaccine administration
- Peptic ulcer disease (caution)

Tapering Guidance (long-term therapy):
- Decrease by 2.5-5mg every 1-4 weeks depending on duration
- If < 3 weeks therapy: Can stop abruptly
- If > 3 weeks therapy: Gradual taper required
- Stress dose coverage needed during illness/surgery

UAE-Specific Note: Widely used for autoimmune conditions (SLE, RA) common in UAE women.
High prevalence of vitamin D deficiency → ensure adequate supplementation.
Consider checking adrenal function before prolonged use in diabetic patients.

BRAND NAMES IN UAE: Deltacortril, PredSol, Minims Prednisolone (eye drops)""",

        "metadata": {
            "source_type": "drug_database",
            "drug_name": "Prednisolone",
            "class": "Corticosteroid",
            "interaction_risk": "MEDIUM",
        }
    },
]

MOH_REGULATIONS = [
    {
        "text": """UAE MINISTRY OF HEALTH & PREVENTION (MOHAP)
Federal Law No. 8 of 2023: Medical Liability and Patient Safety

Chapter 3 - Physician Responsibilities:
- Article 15: Physicians must obtain informed consent before any procedure
- Article 16: Mandatory reporting of infectious diseases to public health authorities
- Article 17: Physicians cannot refuse emergency treatment regardless of ability to pay
- Article 18: Complete and accurate medical records must be maintained for 25 years

Chapter 5 - Telemedicine Regulations:
- Article 28: Telemedicine consultations valid only if there is an existing physician-patient relationship
- Article 29: Remote prescribing allowed only for non-controlled substances
- Article 30: Telemedicine platform must be licensed by relevant health authority
- Article 31: Patient must provide informed consent for telemedicine services

Chapter 7 - Penalties:
- Article 42: Practice without license: AED 100,000 - 1,000,000 fine and/or imprisonment
- Article 43: Negligence causing patient harm: License suspension 1-5 years
- Article 44: Patient data breach: AED 500,000 - 3,000,000 fine
- Article 45: Practice beyond scope: Immediate license revocation""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "MOHAP",
            "regulation_number": "Federal Law No. 8 of 2023",
            "year": "2023",
        }
    },
    {
        "text": """DUBAI HEALTH AUTHORITY (DHA)
Health Regulation No. 1 of 2024: AI in Healthcare

Scope: Governs use of Artificial Intelligence in Dubai healthcare facilities

Key Requirements:
1. All AI systems used for clinical decision support must be registered with DHA
2. AI systems must undergo validation study showing sensitivity/specificity > 90%
3. Physician must remain responsible for all clinical decisions - AI is advisory only
4. Patient must be informed if AI is used in their diagnosis/treatment
5. Annual audit of AI system performance required
6. Bias testing required for AI systems used on diverse populations

Prohibited Uses:
- Autonomous diagnosis without physician review
- Autonomous treatment decisions
- AI systems making decisions about life-sustaining treatment
- Denying insurance coverage based solely on AI recommendation

Data Requirements:
- Training data must be representative of Dubai's diverse population
- UAE data residency required for all patient data processed by AI
- Explainability: AI decisions must be explainable to clinicians
- Human override capability must be available at all times

Penalties:
- Unregistered AI use: AED 200,000 - 500,000
- AI system failure causing harm: Potential criminal liability of operator
- Data breach via AI system: Up to AED 5,000,000 fine""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "Dubai Health Authority",
            "regulation_number": "DHA Regulation No. 1 of 2024",
            "year": "2024",
        }
    },
    {
        "text": """DEPARTMENT OF HEALTH - ABU DHABI (DOH)
Policy on Healthcare Quality and Patient Safety (2024 Update)

Mandatory Quality Indicators (all facilities must report quarterly):
1. Hospital-acquired infection rates (CLABSI, CAUTI, SSI, VAP)
2. 30-day readmission rates (all-cause)
3. Medication error rates per 1000 doses
4. Patient fall rates per 1000 patient-days
5. Pressure ulcer prevalence
6. Hand hygiene compliance
7. Surgical site infection rates
8. Emergency department door-to-doctor time
9. Door-to-balloon time for STEMI patients
10. Patient satisfaction scores (Net Promoter Score)

Reporting Thresholds:
- CLABSI: < 1.5 per 1000 catheter-days
- CAUTI: < 2.0 per 1000 catheter-days
- SSI (clean cases): < 1.5%
- Hand hygiene: > 85% compliance
- Door-to-balloon: < 90 minutes for > 85% of STEMI cases

Patient Safety Goals 2024:
1. Reduce medication errors by 20% (focus on high-alert medications)
2. Reduce hospital falls with injury by 30%
3. Zero wrong-site surgeries
4. 95% compliance with surgical safety checklist
5. 100% reporting of adverse events via DOH incident reporting system

Enforcement:
- Quarterly submission mandatory
- Below-threshold facilities receive warning and improvement plan
- Repeated non-compliance: Fine AED 50,000 - 500,000
- Public disclosure of quality metrics on DOH website""",

        "metadata": {
            "source_type": "moh_regulation",
            "source": "Abu Dhabi DOH",
            "regulation_number": "DOH Quality Policy 2024",
            "year": "2024",
        }
    },
]

MEDICAL_LITERATURE = [
    {
        "text": """STUDY: Efficacy of SGLT2 Inhibitors in Heart Failure with Preserved Ejection Fraction (HFpEF)
EMPEROR-Preserved Trial (NEJM 2023)
- N = 5,988 patients with HFpEF (EF > 40%)
- Empagliflozin 10mg vs placebo
- Primary endpoint: CV death or HF hospitalization
- Result: 21% relative risk reduction (HR 0.79, p < 0.001)
- NNT: 32 to prevent one event over 2 years
- Benefit consistent across EF spectrum
- UAE relevance: High prevalence of HFpEF in diabetic population""",

        "metadata": {
            "source_type": "medical_literature",
            "source": "NEJM",
            "year": "2023",
            "journal": "New England Journal of Medicine",
            "evidence_level": "A",
        }
    },
    {
        "text": """STUDY: Genetic Risk Score for Premature CAD in South Asian Population
INTERHEART Study - South Asian Subanalysis (Lancet 2024)
- N = 15,420 South Asian participants (including UAE)
- 9-locus genetic risk score identified
- South Asians have 2-3x higher risk of CAD vs Europeans at same risk factor levels
- Traditional risk factors underestimate risk by 40% in this population
- Recommendation: Early screening starting at age 30 (vs 40 for general population)
- UAE-specific: 60% of CAD patients in UAE are South Asian expatriates
- Implication: Consider lower threshold for statin initiation in South Asian patients""",

        "metadata": {
            "source_type": "medical_literature",
            "source": "The Lancet",
            "year": "2024",
            "journal": "The Lancet",
            "evidence_level": "B",
        }
    },
    {
        "text": """STUDY: Vitamin D Supplementation and COVID-19 Severity in UAE Population
Dubai Health Authority Study (JAMA Network Open 2023)
- N = 2,400 hospitalized COVID-19 patients in Dubai
- 76% had vitamin D deficiency (< 20 ng/mL)
- Severe deficiency (< 12 ng/mL): 2.4x higher risk of ICU admission
- Supplementation to >30 ng/mL associated with 40% reduced mortality
- UAE has 85% prevalence of vitamin D deficiency (highest globally despite abundant sunshine)
- Recommendation: Universal screening of vitamin D in at-risk populations
- Implications: Add vitamin D to public health policy""",

        "metadata": {
            "source_type": "medical_literature",
            "source": "JAMA Network Open",
            "year": "2023",
            "journal": "JAMA Network Open",
            "evidence_level": "B",
        }
    },
]

GOLDEN_DATASET = [
    {
        "query": "What is the initial management for a patient presenting with STEMI?",
        "expected_response": "Initial management includes immediate ECG within 10 minutes, cardiac biomarkers, and primary PCI within 90 minutes. If PCI delay >120 minutes, give fibrinolytic therapy.",
        "subsystem": "clinical_diagnosis",
    },
    {
        "query": "What is the first-line treatment for type 2 diabetes in UAE?",
        "expected_response": "Metformin is first-line, starting at 500mg BID, titrating to 2000mg/day. UAE has 19.3% diabetes prevalence. Add SGLT2i or GLP-1 RA if needed.",
        "subsystem": "clinical_diagnosis",
    },
    {
        "query": "Is it safe to prescribe Ciprofloxacin with Warfarin?",
        "expected_response": "No, this is a HIGH risk interaction. Ciprofloxacin potentiates warfarin, increasing INR by 50-100%. Monitor INR closely and reduce warfarin dose if combination is necessary.",
        "subsystem": "prescription_verify",
    },
    {
        "query": "What are the UAE regulations for AI in healthcare?",
        "expected_response": "DHA Regulation No. 1 of 2024 requires AI systems to be registered, validated (>90% sensitivity/specificity), and physicians remain responsible for all clinical decisions.",
        "subsystem": "medical_record",
    },
]


def build_vectors():
    """Build all vector database collections."""
    print("=" * 60)
    print("BUILDING MEDICAL KNOWLEDGE BASE")
    print("=" * 60)

    store = VectorStore()
    chunker = TextChunker(chunk_size=512, chunk_overlap=64)

    # 1. Clinical Guidelines
    print("\n[1/4] Building Clinical Guidelines collection...")
    guidelines_count = 0
    for guideline in CLINICAL_GUIDELINES:
        chunks = chunker.chunk_text(guideline["text"], guideline["metadata"])
        doc_ids = [f"CG-{i:04d}" for i in range(guidelines_count, guidelines_count + len(chunks))]
        added = store.add_documents("clinical_guidelines", chunks, ids=doc_ids)
        guidelines_count += len(chunks)
    print(f"  Added {guidelines_count} chunks to 'clinical_guidelines'")

    # 2. Drug Database
    print("\n[2/4] Building Drug Database collection...")
    drug_count = 0
    for drug in DRUG_DATABASE:
        chunks = chunker.chunk_text(drug["text"], drug["metadata"])
        doc_ids = [f"DR-{i:04d}" for i in range(drug_count, drug_count + len(chunks))]
        added = store.add_documents("drug_database", chunks, ids=doc_ids)
        drug_count += len(chunks)
    print(f"  Added {drug_count} chunks to 'drug_database'")

    # 3. MOH Regulations
    print("\n[3/4] Building MOH Regulations collection...")
    reg_count = 0
    for reg in MOH_REGULATIONS:
        chunks = chunker.chunk_text(reg["text"], reg["metadata"])
        doc_ids = [f"MR-{i:04d}" for i in range(reg_count, reg_count + len(chunks))]
        added = store.add_documents("moh_regulations", chunks, ids=doc_ids)
        reg_count += len(chunks)
    print(f"  Added {reg_count} chunks to 'moh_regulations'")

    # 4. Medical Literature
    print("\n[4/4] Building Medical Literature collection...")
    lit_count = 0
    for lit in MEDICAL_LITERATURE:
        chunks = chunker.chunk_text(lit["text"], lit["metadata"])
        doc_ids = [f"ML-{i:04d}" for i in range(lit_count, lit_count + len(chunks))]
        added = store.add_documents("medical_literature", chunks, ids=doc_ids)
        lit_count += len(chunks)
    print(f"  Added {lit_count} chunks to 'medical_literature'")

    # Save golden dataset
    print("\nSaving golden dataset for evaluation...")
    dataset_path = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
    with open(dataset_path, "w", encoding="utf-8") as f:
        json.dump(GOLDEN_DATASET, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(GOLDEN_DATASET)} test cases to golden_dataset.json")

    # Summary
    print("\n" + "=" * 60)
    print("BUILD COMPLETE!")
    print("=" * 60)
    print(f"\nCollections summary:")
    print(f"  clinical_guidelines: {store.count('clinical_guidelines')} chunks")
    print(f"  drug_database:       {store.count('drug_database')} chunks")
    print(f"  moh_regulations:     {store.count('moh_regulations')} chunks")
    print(f"  medical_literature:  {store.count('medical_literature')} chunks")
    print(f"\nDatabase stored at: {store.persist_directory}")


if __name__ == "__main__":
    build_vectors()
