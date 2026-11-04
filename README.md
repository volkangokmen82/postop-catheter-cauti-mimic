# Postoperative urinary catheter duration, nursing care processes and culture-defined catheter-associated bacteriuria (MIMIC-IV)

Analysis code for the study *"Catheter Duration, Nursing Care Processes and Culture-Defined Catheter-Associated Bacteriuria in Postoperative Intensive Care Patients"* (Volkan Gökmen, PhD, RN; Eva Kajti, MD(c)).

## Data
The code uses **MIMIC-IV v3.1** (https://doi.org/10.13026/kpb9-mt58). The data are **not** included in this repository. Access requires PhysioNet credentialing, CITI "Data or Specimens Only Research" training and signing the data use agreement. The script produces aggregate results only; cells with fewer than 10 patients or events are suppressed.

## Files
| File | Purpose |
|---|---|
| `catheter_cab_analysis.py` | Complete analysis reported in the manuscript (single self-contained script) |
| `requirements.txt` | Python dependencies |

## Usage
```
pip install -r requirements.txt
python catheter_cab_analysis.py --root <path-to-MIMIC-IV>
```
Results are written to `results/` next to the script: a summary text file, Excel tables, the flow diagram (Figure 1) and Figures 2–4 (PNG and TIFF, 600 dpi).

## Analyses
- **Cohort:** adults; surgical service at ICU admission; first ICU stay ≥24 h; indwelling (Foley) catheter (procedureevents itemid 229351); episodes <24 h apart merged; positive urine culture from 48 h before to 48 h after insertion excluded.
- **Verified postoperative cohort (primary population):** operating-room procedure code (ICD-9-CM 01–86 excluding 38.9x; ICD-10-PCS section 0 excluding change, drainage, insertion, inspection and removal) dated on the ICU admission day or the day before, or transfer from the post-anaesthesia care unit within 24 h before ICU admission. The full surgical-service ICU cohort and the PACU subgroup are analysed as sensitivity populations.
- **Outcomes:** culture-defined catheter-associated bacteriuria (CAB; urine culture with ≥1 bacterial organism, excluding yeast/Candida, mixed flora and contamination, >48 h after insertion and ≤24 h after removal in the ICU); NHSN-concordant CAUTI (calendar-day rule, ≥100,000 CFU/mL, ≤2 organisms, fever >38.0 °C within ±3 days) where colony counts are recorded.
- **Event coding:** CAB; catheter removal in the ICU (with a 24-h post-removal window) and death on catheter as competing events; ICU discharge with the catheter in place as censoring.
- **Methods:** incidence per 1000 catheter-days, calendar catheter-days and days at risk; Kaplan–Meier and Aalen–Johansen cumulative incidence; cause-specific Cox model with modified first-24-h SOFA and calendar period, E-values and stability checks (bootstrap, optimism-corrected C-index, reduced, ridge-penalised and SOFA-free models); logistic model for catheter duration >48 h; mobilisation–removal temporal order, time-dependent Cox and 24-h landmark analyses; re-catheterisation with complete 48-h follow-up and a censored Cox model; nursing process indicators.

## License
MIT License (code only). MIMIC-IV data remain subject to the PhysioNet Credentialed Health Data License.
