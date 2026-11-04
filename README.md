# Postoperative urinary catheter duration, nursing care processes and CAUTI (MIMIC-IV)

Analysis code for the study *"Catheter Duration, Nursing Care Processes and Catheter-Associated Urinary Tract Infection in Postoperative Intensive Care Patients"* (Volkan Gökmen, PhD, RN; Ağrı İbrahim Çeçen University, Faculty of Health Sciences, Department of Nursing).

## Data
The code uses **MIMIC-IV v3.1** (https://doi.org/10.13026/kpb9-mt58). The data are **not** included in this repository. Access requires PhysioNet credentialing, CITI "Data or Specimens Only Research" training and signing the data use agreement. All scripts produce aggregate results only; cells with fewer than 10 patients are suppressed.

## Files
| File | Purpose |
|---|---|
| `mimic_fizibilite.py`, `mimic_fizibilite2.py` | Feasibility checks (cohort size, item availability) |
| `kateter_analiz.py` | Main analysis: cohort, CAUTI definition, incidence, Kaplan–Meier, Cox and logistic models |
| `kateter_analiz2.py` | Extended analysis: Aalen–Johansen competing-risk incidence, daily hazard (Poisson spline), E-values, nursing process indicators (early mobilisation, bladder scanning, re-catheterisation, catheter-days beyond 48 h), sensitivity analysis, Figures 1–3 |

## Usage
```
pip install -r requirements.txt
python kateter_analiz.py  --root <path-to-MIMIC-IV>
python kateter_analiz2.py --root <path-to-MIMIC-IV>
```
`kateter_analiz2.py` imports `kateter_analiz.py`; keep both in the same folder.

## Key definitions
- Cohort: adults, surgical service at ICU admission, first ICU stay ≥24 h, indwelling (Foley) catheter (procedureevents itemid 229351); episodes <24 h apart merged.
- CAUTI: positive urine culture (excluding yeast/Candida, mixed flora, contamination) >48 h after insertion and ≤24 h after removal; patients with a positive culture from 48 h before to 48 h after insertion excluded.

## License
MIT License (code only). MIMIC-IV data remain subject to the PhysioNet Credentialed Health Data License.
