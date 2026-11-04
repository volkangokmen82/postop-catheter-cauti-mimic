# -*- coding: utf-8 -*-
"""
Catheter duration, nursing care processes and culture-defined catheter-associated
bacteriuria (CAB) in postoperative intensive care patients - MIMIC-IV v3.1 analysis.

Produces aggregate results only; cells with fewer than 10 patients/events are suppressed.

Analyses
 - Cohort: adults, surgical service at ICU admission, first ICU stay >=24 h, Foley catheter;
   postoperative status verified by procedure codes (ICU admission day or day before) or PACU transfer
 - Outcomes: culture-defined CAB (primary); NHSN-concordant CAUTI (>=100,000 CFU/mL, <=2 organisms,
   fever >38.0 C, calendar-day rule) where colony counts are recorded
 - Event coding: 1 CAB, 2 removal in ICU (+24 h window), 3 death on catheter, 0 censored at ICU discharge
 - Kaplan-Meier and Aalen-Johansen cumulative incidence with CIs and numbers at risk
 - Incidence by grouped catheter-time intervals (no interval with <10 events)
 - Cause-specific Cox model with modified first-24-h SOFA and calendar period, E-values;
   stability checks (events per variable, bootstrap, optimism-corrected C-index,
   reduced and ridge-penalised models, model without SOFA)
 - Logistic model for catheter duration >48 h
 - Mobilisation and removal: temporal order, time-dependent Cox, 24-h landmark analysis
 - Re-catheterisation within 48 h: complete follow-up subset and censored Cox model
 - Nursing process indicators; Figures 1-3 (600 dpi)

Installation: pip install -r requirements.txt
Usage:        python catheter_cab_analysis.py --root <path-to-MIMIC-IV>
Output:       ./results/  (summary text, Excel tables, figures)
"""
import argparse
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


warnings.filterwarnings("ignore")
LOG = []
SURG = {"SURG", "CSURG", "NSURG", "ORTHO", "PSURG", "TSURG", "VSURG", "TRAUM", "GU", "ENT"}


def sup(n):
    n = int(n)
    return "<10" if 0 < n < 10 else str(n)


def find(root, name):
    hits = list(Path(root).rglob(name))
    if not hits:
        sys.exit(f"Not found: {name}")
    return hits[0]


def npct(k, n):
    return f"{sup(k)} ({100 * k / n:.1f})" if n else "-"


def med_iqr(x):
    return f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"

SVC = {"SURG": "General surgery", "CSURG": "Cardiac surgery", "NSURG": "Neurosurgery", "TRAUM": "Trauma surgery"}
SERVICES = ["General surgery", "Cardiac surgery", "Neurosurgery", "Trauma surgery", "Other surgery"]
SDUM = [s_.replace(" ", "_") for s_ in SERVICES[1:]]   # kukla degisken adlari (bosluksuz)


def add_dummies(df):
    for s_ in SERVICES[1:]:
        df[s_.replace(" ", "_")] = (df.svc == s_).astype(int)
    return df
FOLEY, VENT = 229351, 225792
BRADEN = [224054, 224055, 224056, 224057, 224058, 224059]
TEMP_C, TEMP_F = 223762, 223761
GCS = [220739, 223900, 223901]
MAP = [220052, 220181, 225312]
VASO_HIGH = [221906, 221289, 221749, 222315]   # norepinefrin, epinefrin, fenilefrin, vazopressin
VASO_LOW = [221662, 221653]                    # dopamin, dobutamin
LAB = {"plt": 51265, "bili": 50885, "crea": 50912, "pao2": 50821, "fio2": 50816}
NON_PATH = r"yeast|candida|cancelled|no growth|mixed|flora|contamina"
CFU_RE = r">100,000|>=100,000|≥100,000|GREATERTHAN100,000|>100000|>10\^5|>10E5|>100K|100,000ORGANISMS|100,000CFU"
OOB_RE = re.compile(r"chair|ambul|walk|stand|commode|step|march|oob|out of bed", re.I)
BLUE, ORANGE, GREEN, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]


def out(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def poisson_ci(k, t):
    lo = stats.chi2.ppf(.025, 2 * k) / 2 if k else 0.0
    hi = stats.chi2.ppf(.975, 2 * (k + 1)) / 2
    return 1000 * k / t, 1000 * lo / t, 1000 * hi / t


def rate_str(k, t):
    r, lo, hi = poisson_ci(k, t)
    return f"{r:.2f} ({lo:.2f}–{hi:.2f})"


def evalue(est, lo, hi):
    def e(x):
        x = 1 / x if x < 1 else x
        return x + np.sqrt(x * (x - 1))
    return e(est), (1.0 if lo <= 1 <= hi else e(lo if est > 1 else hi))


def scan(path, usecols, keep, chunksize=5_000_000, label=""):
    parts, t0 = [], time.time()
    for i, ch in enumerate(pd.read_csv(path, usecols=usecols, chunksize=chunksize,
                                       dtype={"value": "string"} if "value" in usecols else None), 1):
        ch = keep(ch)
        if len(ch):
            parts.append(ch)
        print(f"   {label} parca {i} ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=usecols)


# =====================================================================
def build_cohort(root):
    icu = pd.read_csv(find(root, "icustays.csv.gz"), parse_dates=["intime", "outtime"])
    pat = pd.read_csv(find(root, "patients.csv.gz"),
                      usecols=["subject_id", "gender", "anchor_age", "anchor_year_group"])
    adm = pd.read_csv(find(root, "admissions.csv.gz"),
                      usecols=["hadm_id", "admission_type", "deathtime", "hospital_expire_flag"],
                      parse_dates=["deathtime"])
    svc = pd.read_csv(find(root, "services.csv.gz"), parse_dates=["transfertime"])
    icu = icu.merge(pat, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
    icu["deathtime"] = pd.to_datetime(icu.deathtime, errors="coerce")
    icu["key"] = icu.intime + pd.Timedelta("1h")
    svc = svc.dropna(subset=["hadm_id"])
    svc["hadm_id"] = svc.hadm_id.astype("int64")
    icu["hadm_id"] = icu.hadm_id.astype("int64")
    icu = pd.merge_asof(icu.sort_values("key"),
                        svc[["hadm_id", "transfertime", "curr_service"]].sort_values("transfertime"),
                        left_on="key", right_on="transfertime", by="hadm_id", direction="backward")
    icu = icu.sort_values("intime")
    icu["first_stay"] = icu.groupby("subject_id").cumcount() == 0
    flow = {"All ICU stays": len(icu)}
    icu = icu[icu.anchor_age >= 18]; flow["Adult"] = len(icu)
    icu = icu[icu.curr_service.isin(SURG)]; flow["Surgical service at ICU admission"] = len(icu)
    icu = icu[icu.first_stay]; flow["First ICU stay"] = len(icu)
    icu = icu[icu.los >= 1]; flow["ICU stay >=24 h"] = len(icu)
    return icu, flow


def or_procedure_flag(root, coh):
    pr = pd.read_csv(find(root, "procedures_icd.csv.gz"), usecols=["hadm_id", "chartdate", "icd_code", "icd_version"],
                     dtype={"icd_code": "string"}, parse_dates=["chartdate"])
    pr = pr[pr.hadm_id.isin(coh.hadm_id)].copy()
    c = pr.icd_code.fillna("")
    v9 = (pr.icd_version == 9) & c.str[:2].str.isdigit()
    two = pd.to_numeric(c.str[:2], errors="coerce")
    is9 = v9 & two.between(1, 86) & ~c.str.startswith("389")
    is10 = (pr.icd_version == 10) & c.str.startswith("0") & ~c.str[2].isin(list("29HJP"))
    pr = pr[is9 | is10].merge(coh[["hadm_id", "stay_id", "intime"]], on="hadm_id")
    d0 = pr.intime.dt.normalize()
    pr = pr[(pr.chartdate >= d0 - pd.Timedelta("1D")) & (pr.chartdate <= d0)]
    return set(pr.stay_id)


def pacu_flag(root, coh):
    trf = pd.read_csv(find(root, "transfers.csv.gz"), usecols=["hadm_id", "careunit", "outtime"],
                      parse_dates=["outtime"])
    p = trf[trf.careunit.fillna("").str.contains("PACU", case=False)].dropna(subset=["hadm_id"])
    p["hadm_id"] = p.hadm_id.astype("int64")
    y = coh[["stay_id", "hadm_id", "intime"]].merge(p, on="hadm_id")
    y = y[(y.outtime >= y.intime - pd.Timedelta("24h")) & (y.outtime <= y.intime + pd.Timedelta("1h"))]
    return set(y.stay_id)


def catheters(root, coh):
    pe = pd.read_csv(find(root, "procedureevents.csv.gz"), usecols=["stay_id", "starttime", "endtime", "itemid"],
                     parse_dates=["starttime", "endtime"])
    pe = pe[pe.stay_id.isin(coh.stay_id)]
    vent = pe[pe.itemid == VENT].merge(coh[["stay_id", "intime"]], on="stay_id")
    vent24 = set(vent[vent.starttime <= vent.intime + pd.Timedelta("24h")].stay_id)
    raw = pe[pe.itemid == FOLEY].dropna(subset=["starttime", "endtime"])
    raw = raw[raw.endtime > raw.starttime].sort_values(["stay_id", "starttime"])
    rows = []
    for sid, g in raw.groupby("stay_id"):
        s, e = g.iloc[0].starttime, g.iloc[0].endtime
        raw_end = e
        for _, r in g.iloc[1:].iterrows():
            if r.starttime - e <= pd.Timedelta("24h"):
                e = max(e, r.endtime)
            else:
                break
        rows.append((sid, s, e, raw_end))
    return pd.DataFrame(rows, columns=["stay_id", "cath_start", "cath_end", "raw_end"]), vent24, raw


def cultures(root, subjects):
    cols = ["subject_id", "micro_specimen_id", "chartdate", "charttime", "spec_type_desc", "org_name",
            "quantity", "comments"]
    mb = pd.read_csv(find(root, "microbiologyevents.csv.gz"), usecols=cols, dtype={"quantity": "string",
                                                                                    "comments": "string"})
    mb = mb[mb.subject_id.isin(subjects) & mb.spec_type_desc.fillna("").str.contains("URINE", case=False)]
    mb["t"] = pd.to_datetime(mb.charttime).fillna(pd.to_datetime(mb.chartdate))
    mb["date"] = pd.to_datetime(mb.chartdate).fillna(mb.t.dt.normalize())
    pos = mb[mb.org_name.notna() & ~mb.org_name.fillna("").str.contains(NON_PATH, case=False)].copy()
    txt = (pos.quantity.fillna("") + " " + pos.comments.fillna("")).str.upper().str.replace(" ", "")
    pos["cfu100k"] = txt.str.contains(CFU_RE)
    nspec = pos.groupby("micro_specimen_id").org_name.nunique()
    pos["n_org"] = pos.micro_specimen_id.map(nspec)
    # tanimlayici: miktar alanlarinin en sik degerleri
    q = (mb.quantity.fillna("") + " | " + mb.comments.fillna("").str[:60]).value_counts().head(12)
    pos["cfu_iso"] = pos.groupby(["micro_specimen_id", "org_name"]).cfu100k.transform("any")
    allt = (mb.quantity.fillna("") + " " + mb.comments.fillna("")).str.upper().str.replace(" ", "")
    mb["cfu_row"] = allt.str.contains(CFU_RE)
    pos["cfu_spec"] = pos.micro_specimen_id.map(mb.groupby("micro_specimen_id").cfu_row.any()).fillna(False).astype(bool)
    pos["cfu100k"] = pos.cfu_iso | pos.cfu_spec
    qp = (pos.quantity.fillna("") + " | " + pos.comments.fillna("").str[:60]).value_counts().head(12)
    return pos.drop_duplicates(["micro_specimen_id", "org_name"]), q, qp


def chart_scan(root, d, act_ids):
    ids = set(BRADEN) | set(act_ids) | {TEMP_C, TEMP_F} | set(GCS) | set(MAP)
    stays = set(d.stay_id)
    ce = scan(find(root, "chartevents.csv.gz"), ["stay_id", "charttime", "itemid", "value", "valuenum"],
              lambda ch: ch[ch.itemid.isin(ids) & ch.stay_id.isin(stays)], label="chartevents")
    ce["stay_id"] = ce.stay_id.astype("int64")
    ce["charttime"] = pd.to_datetime(ce.charttime)
    return ce.merge(d[["stay_id", "subject_id", "intime", "outtime"]], on="stay_id")


def lab_scan(root, d):
    hadms = set(d.hadm_id)
    lb = scan(find(root, "labevents.csv.gz"), ["hadm_id", "specimen_id", "itemid", "charttime", "valuenum"],
              lambda ch: ch[ch.itemid.isin(LAB.values()) & ch.hadm_id.isin(hadms)], label="labevents")
    lb["hadm_id"] = lb.hadm_id.astype("int64")
    lb["charttime"] = pd.to_datetime(lb.charttime)
    return lb


def sofa(d, ce, lb, root, vent24):
    w = d[["stay_id", "hadm_id", "intime"]].copy()
    lb = lb.merge(w, on="hadm_id")
    lb = lb[(lb.charttime >= lb.intime - pd.Timedelta("6h")) & (lb.charttime <= lb.intime + pd.Timedelta("24h"))]
    inv = {v: k for k, v in LAB.items()}
    lb["name"] = lb.itemid.map(inv)
    plt_ = lb[lb.name == "plt"].groupby("stay_id").valuenum.min()
    bil = lb[lb.name == "bili"].groupby("stay_id").valuenum.max()
    cre = lb[lb.name == "crea"].groupby("stay_id").valuenum.max()
    bg = lb[lb.name.isin(["pao2", "fio2"])].pivot_table(index=["stay_id", "specimen_id"], columns="name",
                                                         values="valuenum", aggfunc="first").dropna()
    if len(bg):
        bg["fio2"] = np.where(bg.fio2 > 1, bg.fio2 / 100, bg.fio2)
        bg = bg[(bg.fio2 >= 0.21) & (bg.fio2 <= 1)]
        pf = (bg.pao2 / bg.fio2).groupby(level=0).min()
    else:
        pf = pd.Series(dtype=float)
    c24 = ce[(ce.charttime >= ce.intime) & (ce.charttime <= ce.intime + pd.Timedelta("24h"))]
    mapm = c24[c24.itemid.isin(MAP) & c24.valuenum.between(10, 200)].groupby("stay_id").valuenum.min()
    g = c24[c24.itemid.isin(GCS)].pivot_table(index=["stay_id", "charttime"], columns="itemid", values="valuenum",
                                               aggfunc="first").dropna()
    gcs = g.sum(axis=1).groupby(level=0).min() if len(g) else pd.Series(dtype=float)
    ie = pd.read_csv(find(root, "inputevents.csv.gz"), usecols=["stay_id", "starttime", "itemid"],
                     parse_dates=["starttime"])
    ie = ie[ie.itemid.isin(VASO_HIGH + VASO_LOW) & ie.stay_id.isin(d.stay_id)].merge(w, on="stay_id")
    ie = ie[ie.starttime <= ie.intime + pd.Timedelta("24h")]
    hi = set(ie[ie.itemid.isin(VASO_HIGH)].stay_id); lo = set(ie[ie.itemid.isin(VASO_LOW)].stay_id)

    def band(x, cuts):  # cuts: esikler, artan puan
        if pd.isna(x):
            return 0
        return int(sum(x >= c for c in cuts))
    s = pd.DataFrame(index=d.stay_id.values)
    s["coag"] = [0 if pd.isna(v) else (4 if v < 20 else 3 if v < 50 else 2 if v < 100 else 1 if v < 150 else 0)
                 for v in plt_.reindex(s.index)]
    s["liver"] = [band(v, [1.2, 2.0, 6.0, 12.0]) for v in bil.reindex(s.index)]
    s["renal"] = [band(v, [1.2, 2.0, 3.5, 5.0]) for v in cre.reindex(s.index)]
    s["cns"] = [0 if pd.isna(v) else (0 if v >= 15 else 1 if v >= 13 else 2 if v >= 10 else 3 if v >= 6 else 4)
                for v in gcs.reindex(s.index)]
    cv = []
    for sid in s.index:
        if sid in hi:
            cv.append(3)
        elif sid in lo:
            cv.append(2)
        else:
            m = mapm.get(sid, np.nan)
            cv.append(1 if (not pd.isna(m) and m < 70) else 0)
    s["cardio"] = cv
    rs = []
    for sid, v in zip(s.index, pf.reindex(s.index)):
        if pd.isna(v):
            rs.append(0)
        elif v < 100 and sid in vent24:
            rs.append(4)
        elif v < 200 and sid in vent24:
            rs.append(3)
        elif v < 300:
            rs.append(2)
        elif v < 400:
            rs.append(1)
        else:
            rs.append(0)
    s["resp"] = rs
    s["sofa"] = s[["coag", "liver", "renal", "cns", "cardio", "resp"]].sum(axis=1)
    avail = pd.DataFrame({"Platelets": plt_.reindex(s.index).notna(), "Bilirubin": bil.reindex(s.index).notna(),
                          "Creatinine": cre.reindex(s.index).notna(), "GCS": gcs.reindex(s.index).notna(),
                          "MAP": mapm.reindex(s.index).notna(), "PaO2/FiO2": pf.reindex(s.index).notna()})
    return s.sofa, avail.mean() * 100


# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    odir = Path(__file__).with_name("results")
    odir.mkdir(exist_ok=True)
    t0 = time.time()
    out("CATHETER DURATION AND CAB - MIMIC-IV v3.1")
    out("=" * 64)
    coh, flow = build_cohort(root)
    orp, pacu = or_procedure_flag(root, coh), pacu_flag(root, coh)
    ep, vent24, raw = catheters(root, coh)
    d = coh.merge(ep, on="stay_id")
    flow["Foley catheter"] = len(d)

    # ---------- izlem sonu nedenleri (olay kodlamasi)
    d["died_cath"] = d.deathtime.notna() & (d.deathtime <= d.cath_end)
    d["removed_icu"] = (~d.died_cath) & (d.cath_end < d.outtime - pd.Timedelta("1h"))
    d["cath_stop"] = np.where(d.died_cath, d.deathtime, d.cath_end)
    d["cath_stop"] = pd.to_datetime(d.cath_stop)
    d = d[d.cath_stop > d.cath_start]
    d["dur_d"] = (d.cath_stop - d.cath_start).dt.total_seconds() / 86400
    # takvim gunu kateter-gunleri (NHSN: takildigi gun = 1. gun)
    d["cal_days"] = (d.cath_stop.dt.normalize() - d.cath_start.dt.normalize()).dt.days + 1
    # izlem sonu: ICU'da cekildiyse cekilme + 24 saat (o arada olum olursa olum zamani); aksi halde cath_stop
    fu = d.cath_stop + pd.Timedelta("24h")
    fu = fu.where(~(d.deathtime.notna() & (d.deathtime < fu)), d.deathtime)
    d["fu_end"] = pd.to_datetime(np.where(d.removed_icu, fu, d.cath_stop))

    pos, qtab, qpos = cultures(root, set(d.subject_id))
    out("\nIdrar kulturu miktar/yorum alanlarinin en sik degerleri (dogrulama icin):")
    for k, v in qtab.items():
        out(f"  {k[:90]}: {sup(v)}")
    out("Pozitif izolat satirlarinda miktar/yorum alanlarinin en sik degerleri:")
    for k, v in qpos.items():
        out(f"  {k[:90]}: {sup(v)}")
    x = d[["stay_id", "subject_id", "cath_start", "cath_stop"]].merge(pos, on="subject_id")
    pre = x[(x.t >= x.cath_start - pd.Timedelta("48h")) & (x.t < x.cath_start + pd.Timedelta("48h"))]
    d = d[~d.stay_id.isin(pre.stay_id)].copy()
    flow["After excluding positive culture 48 h before to 48 h after insertion"] = len(d)

    # ---------- cerrahi dogrulama
    d["or_proc"] = d.stay_id.isin(orp).astype(int)
    d["pacu"] = d.stay_id.isin(pacu).astype(int)
    d["postop_confirmed"] = ((d.or_proc == 1) | (d.pacu == 1)).astype(int)
    out(f"\nPOSTOPERATIF DOGRULAMA (n={sup(len(d))}):")
    out(f"  ICU kabul gunu/bir gun once ameliyat kodu: {npct(d.or_proc.sum(), len(d))}")
    out(f"  PACU'dan gelis: {npct(d.pacu.sum(), len(d))}")
    out(f"  Ameliyat kodu VEYA PACU (dogrulanmis postoperatif): {npct(d.postop_confirmed.sum(), len(d))}")
    out(f"  Her iki kriter (kod VE PACU): {npct(((d.or_proc == 1) & (d.pacu == 1)).sum(), len(d))}; "
        f"yalniz kod: {sup(((d.or_proc == 1) & (d.pacu == 0)).sum())}; yalniz PACU: {sup(((d.or_proc == 0) & (d.pacu == 1)).sum())}")
    for s_ in sorted(d.curr_service.unique()):
        g = d[d.curr_service == s_]
        out(f"    {s_}: {npct(g.postop_confirmed.sum(), len(g))}")

    # ---------- kovaryatlar
    dx = pd.read_csv(find(root, "diagnoses_icd.csv.gz"), dtype={"icd_code": "string"})
    dm = dx[((dx.icd_version == 10) & dx.icd_code.str.match(r"E1[0-4]")) |
            ((dx.icd_version == 9) & dx.icd_code.str.startswith("250"))].hadm_id
    d["diabetes"] = d.hadm_id.isin(dm).astype(int)
    d["female"] = (d.gender == "F").astype(int)
    d["age10"] = d.anchor_age / 10
    d["vent24"] = d.stay_id.isin(vent24).astype(int)
    d["emergency"] = (~d.admission_type.fillna("").str.contains("ELECTIVE|SAME DAY", case=False)).astype(int)
    d["svc"] = d.curr_service.map(SVC).fillna("Other surgery")
    d["periop"] = ((d.cath_start - d.intime) <= pd.Timedelta("6h")).astype(int)
    d["prolonged"] = (d.dur_d > 2).astype(int)
    yg = d.anchor_year_group.fillna("")
    d["period_2014_2016"] = (yg == "2014 - 2016").astype(int)
    d["period_2017_2022"] = yg.isin(["2017 - 2019", "2020 - 2022"]).astype(int)
    cu = d.first_careunit.fillna("")
    d["icu_type"] = np.select([cu.str.contains("CVICU"), cu.str.contains("Trauma"), cu.str.contains("Neuro"),
                               cu.str.contains(r"^Surgical Intensive", regex=True)],
                              ["CVICU", "Trauma SICU", "Neuro ICU/stepdown", "SICU"], "Other ICU")

    di = pd.read_csv(find(root, "d_items.csv.gz"))
    lab = di[di.linksto == "chartevents"].label.fillna("")
    act = di[(di.linksto == "chartevents") & lab.str.contains(r"^activity$|activity / mobility \(jh-hlm\)",
                                                             case=False, regex=True)]
    bsc = di[(di.linksto == "chartevents") & lab.str.contains("bladder scan", case=False)]
    ce = chart_scan(root, d, list(act.itemid) + list(bsc.itemid))
    lb = lab_scan(root, d)
    sofa_s, avail = sofa(d, ce, lb, root, vent24)
    d["sofa"] = d.stay_id.map(sofa_s)
    out("\nMODIFIYE SOFA (ilk 24 saat): medyan (IQR) " + med_iqr(d.sofa))
    out("  Bilesen verisi olan hasta (%): " + ", ".join(f"{k} {v:.1f}" for k, v in avail.items()))
    br = ce[ce.itemid.isin(BRADEN) & (ce.charttime <= ce.intime + pd.Timedelta("24h"))]
    bw = br.pivot_table(index=["stay_id", "charttime"], columns="itemid", values="valuenum", aggfunc="first").dropna()
    d["braden"] = d.stay_id.map(bw.sum(axis=1).reset_index().sort_values("charttime").groupby("stay_id")[0].first())
    d["braden_high"] = (d.braden <= 12).astype(float).where(d.braden.notna())

    # ---------- sonlanimlar
    # (a) kultur tanimli bakteriuri (CAB): >48 saat sonra, cekilmeden sonraki 24 saate kadar
    x = d[["stay_id", "subject_id", "cath_start", "cath_stop", "removed_icu"]].merge(pos, on="subject_id")
    x = x.merge(d[["stay_id", "fu_end"]], on="stay_id")
    win_end = x.fu_end
    cab = x[(x.t >= x.cath_start + pd.Timedelta("48h")) & (x.t <= pd.to_datetime(win_end))]
    cab1 = cab.sort_values("t").groupby("stay_id").first()
    d["cab"] = d.stay_id.isin(cab1.index).astype(int)
    d["cab_t"] = d.stay_id.map(cab1.t)
    # (b) NHSN ile uyumlu CAUTI: takvim gunu kurali + >=1e5 CFU/mL + <=2 organizma + ates (pencere +-3 gun)
    xn = x.copy()
    ins = xn.cath_start.dt.normalize()
    rem = xn.cath_stop.dt.normalize()
    ok_day = ((xn.date - ins).dt.days >= 2) & (xn.date <= rem + pd.Timedelta("1D"))
    xn = xn[ok_day & xn.cfu100k & (xn.n_org <= 2)]
    tmp = ce[ce.itemid.isin([TEMP_C, TEMP_F])].copy()
    tmp["tc"] = np.where(tmp.itemid == TEMP_F, (tmp.valuenum - 32) * 5 / 9, tmp.valuenum)
    fever = tmp[tmp.tc.between(38.05, 43)][["subject_id", "charttime"]].rename(columns={"charttime": "ft"})
    xf = xn.merge(fever, on="subject_id")
    xf = xf[(xf.ft.dt.normalize() >= xf.date - pd.Timedelta("3D")) &
            (xf.ft.dt.normalize() <= xf.date + pd.Timedelta("3D"))]
    nh1 = xf.sort_values("t").groupby("stay_id").t.first()
    d["nhsn"] = d.stay_id.isin(nh1.index).astype(int)
    d["nhsn_t"] = d.stay_id.map(nh1)
    out("\nSONLANIM BILESENLERI (kultur tanimli bakteriuri olan hastalarda):")
    cabp = cab[cab.stay_id.isin(d.stay_id)]
    out(f"  CAB: {npct(d.cab.sum(), len(d))}")
    out(f"  >=100.000 CFU/mL kaydi olan (izolat veya numune duzeyi): {sup(cabp[cabp.cfu100k].stay_id.nunique())}"
        f" [izolat: {sup(cabp[cabp.cfu_iso].stay_id.nunique())}, numune: {sup(cabp[cabp.cfu_spec].stay_id.nunique())}]")
    cabpo = cabp[cabp.stay_id.isin(d[d.postop_confirmed == 1].stay_id)]
    out(f"  Dogrulanmis postop kohortta CAB: {sup(cabpo.stay_id.nunique())}; >=100.000 CFU/mL: {sup(cabpo[cabpo.cfu100k].stay_id.nunique())}")
    out(f"  <=2 mikroorganizma: {sup(cabp[cabp.n_org <= 2].stay_id.nunique())}")
    out(f"  NHSN ile uyumlu CAUTI (miktar + <=2 org + ates + takvim gunu): {npct(d.nhsn.sum(), len(d))}")

    # ---------- olay kodlamasi (KM/AJ/Cox)
    def code(outc, tcol):
        t_ev = d[tcol]
        end = d.fu_end
        has = d[outc] == 1
        tt = np.where(has, (t_ev - d.cath_start).dt.total_seconds() / 86400,
                      (end - d.cath_start).dt.total_seconds() / 86400)
        ev = np.where(has, 1, np.where(d.removed_icu, 2, np.where(d.died_cath, 3, 0)))
        return tt, ev
    d["time_cab"], d["ev_cab"] = code("cab", "cab_t")
    d["time_nh"], d["ev_nh"] = code("nhsn", "nhsn_t")
    N = len(d)
    out("\nIZLEM SONU NEDENI (CAB analizi): " + ", ".join(
        f"{lbl} {npct((d.ev_cab == k).sum(), N)}" for k, lbl in [(1, "CAB"), (2, "cekilme"), (3, "olum"),
                                                                  (0, "ICU cikisi-sansur")]))
    sheets = {}

    def analyse(pop, label, tcol, ecol, outc_name):
        from lifelines import KaplanMeierFitter, AalenJohansenFitter, CoxPHFitter
        res = {}
        p = pop.copy()
        out(f"\n================ {label} | sonlanim: {outc_name} | n={sup(len(p))}, olay={sup((p[ecol] == 1).sum())}")
        cd = p.dur_d.sum(); cal = p.cal_days.sum(); ar = p[tcol].sum()
        k = int((p[ecol] == 1).sum())
        out(f"  1000 kateter-gunu (saat bazli, {cd:.0f} gun): {rate_str(k, cd)}")
        out(f"  1000 takvim kateter-gunu (NHSN, {cal:.0f} gun): {rate_str(k, cal)}")
        out(f"  1000 risk altindaki gun ({ar:.0f}): {rate_str(k, ar)}")
        rows = []
        for lab_, g in [("Total", p)] + [(s, p[p.svc == s]) for s in SERVICES] + \
                [(f"Catheter {c}", p[pd.cut(p.dur_d, [0, 2, 4, 7, 1e9], labels=["≤2 days", ">2–4 days", ">4–7 days", ">7 days"]) == c])
                 for c in ["≤2 days", ">2–4 days", ">4–7 days", ">7 days"]]:
            kk = int((g[ecol] == 1).sum())
            rows.append({"Group": lab_, "Patients": sup(len(g)), "Events, n (%)": npct(kk, len(g)),
                         "Calendar catheter-days": f"{g.cal_days.sum():.0f}",
                         "Per 1000 calendar catheter-days (95% CI)": rate_str(kk, g.cal_days.sum()),
                         "Events <10 (imprecise)": "yes" if 0 < kk < 10 else ""})
        res["rates"] = pd.DataFrame(rows)
        kmf = KaplanMeierFitter().fit(p[tcol].values, (p[ecol] == 1).astype(int).values)
        ajf = AalenJohansenFitter(calculate_variance=True, seed=1).fit(pd.Series(p[tcol].values),
                                                                         pd.Series(p[ecol].values), event_of_interest=1)
        rr = []
        ci = kmf.confidence_interval_survival_function_
        aci = ajf.confidence_interval_cumulative_density_ if hasattr(ajf, "confidence_interval_cumulative_density_") else None
        for day in [3, 5, 7, 10, 14]:
            km = 1 - kmf.survival_function_at_times(day).iloc[0]
            idx = ci.index[ci.index <= day].max()
            aj = ajf.cumulative_density_.loc[:day].iloc[-1, 0]
            aj_ci = ""
            if aci is not None:
                ai = aci.ffill().loc[:day].iloc[-1]
                if ai.notna().all():
                    aj_ci = f" ({100 * ai.min():.2f}–{100 * ai.max():.2f})"
            rr.append({"Day": day, "KM %, 95% CI": f"{100 * km:.1f} ({100 * (1 - ci.loc[idx].iloc[1]):.1f}–{100 * (1 - ci.loc[idx].iloc[0]):.1f})",
                       "AJ %, 95% CI": f"{100 * aj:.2f}{aj_ci}", "At risk": sup((p[tcol] >= day).sum()),
                       "Events up to day": sup(((p[ecol] == 1) & (p[tcol] <= day)).sum())})
        res["cum"] = pd.DataFrame(rr)
        out(res["rates"].to_string(index=False)); out(res["cum"].to_string(index=False))
        # gunluk hazard
        hz = []
        for kd in range(0, 15):
            hi_ = kd + 1 if kd < 14 else 1e9
            T = np.clip(p[tcol] - kd, 0, hi_ - kd).sum()
            e = int(((p[ecol] == 1) & (p[tcol] >= kd) & (p[tcol] < hi_)).sum())
            if T > 0:
                r, lo, up = poisson_ci(e, T)
                hz.append({"day": kd, "events": e, "T": T, "rate": r, "lo": lo, "hi": up, "at_risk": int((p[tcol] > kd).sum())})
        res["hz"] = pd.DataFrame(hz)
        # Cox
        m = p.copy()
        m = add_dummies(m)
        covs = ["age10", "female", "diabetes", "emergency", "vent24", "braden_high", "periop", "sofa",
                "period_2014_2016", "period_2017_2022"] + SDUM
        mc = m[[tcol, ecol] + covs].dropna().copy()
        mc["E"] = (mc[ecol] == 1).astype(int)
        covs = [c for c in covs if mc[c].nunique() > 1 and mc.loc[mc.E == 1, c].nunique() > 0]
        if mc.E.sum() >= 20:
            f = CoxPHFitter(penalizer=0.0).fit(mc[[tcol, "E"] + covs], tcol, "E")
            s = f.summary
            ev = [evalue(h, l, u) for h, l, u in zip(s["exp(coef)"], s["exp(coef) lower 95%"], s["exp(coef) upper 95%"])]
            res["cox"] = pd.DataFrame({"Variable": s.index, "HR": s["exp(coef)"].values,
                                       "lo": s["exp(coef) lower 95%"].values, "hi": s["exp(coef) upper 95%"].values,
                                       "p": s.p.values, "E-value": [f"{a:.2f} / {b:.2f}" for a, b in ev]})
            out(f"  Cox (n={sup(len(mc))}, olay={sup(mc.E.sum())}, C={f.concordance_index_:.3f})")
            out(res["cox"].assign(HR=lambda z: z.HR.round(2), lo=lambda z: z.lo.round(2), hi=lambda z: z.hi.round(2),
                                  p=lambda z: z.p.map(fmt_p)).to_string(index=False))
            try:
                from lifelines.statistics import proportional_hazard_test
                out(f"  PH testi en kucuk p: {proportional_hazard_test(f, mc[[tcol, 'E'] + covs], time_transform='rank').summary.p.min():.3f}")
            except Exception:
                pass
        res["kmf"], res["ajf"] = kmf, ajf
        return res

    main_pop = d[d.postop_confirmed == 1]
    R = {}
    R["all_cab"] = analyse(d, "TUM CERRAHI SERVIS KOHORTU", "time_cab", "ev_cab", "CAB")
    R["po_cab"] = analyse(main_pop, "DOGRULANMIS POSTOPERATIF KOHORT", "time_cab", "ev_cab", "CAB")
    R["po_nh"] = analyse(main_pop, "DOGRULANMIS POSTOPERATIF KOHORT", "time_nh", "ev_nh", "NHSN-uyumlu CAUTI")
    R["all_nh"] = analyse(d, "TUM CERRAHI SERVIS KOHORTU", "time_nh", "ev_nh", "NHSN-uyumlu CAUTI")
    R["pacu_cab"] = analyse(d[d.pacu == 1], "PACU ALT GRUBU", "time_cab", "ev_cab", "CAB")

    # ---------- COX MODEL KARARLILIGI (dogrulanmis postop kohort, CAB)
    from lifelines import CoxPHFitter
    from lifelines.utils import concordance_index
    COVS = ["age10", "female", "diabetes", "emergency", "vent24", "braden_high", "periop", "sofa",
            "period_2014_2016", "period_2017_2022"] + SDUM
    mc = add_dummies(main_pop.copy())[["time_cab", "ev_cab"] + COVS].dropna().copy()
    mc["E"] = (mc.ev_cab == 1).astype(int)
    mc = mc.drop(columns="ev_cab")
    COVS = [c for c in COVS if mc[c].nunique() > 1 and mc.loc[mc.E == 1, c].nunique() > 0]
    nev = int(mc.E.sum())
    out(f"\nCOX MODEL KARARLILIGI: n={sup(len(mc))}, olay={sup(nev)}, parametre={len(COVS)}, EPV={nev / len(COVS):.1f}")
    stab = []
    def fit(df, covs, pen=0.0):
        return CoxPHFitter(penalizer=pen).fit(df[["time_cab", "E"] + covs], "time_cab", "E")
    full = fit(mc, COVS)
    mc = mc.reset_index(drop=True)
    capp = concordance_index(mc.time_cab, -full.predict_partial_hazard(mc[COVS]), mc.E)
    # bootstrap: yuzdelik GA, yon tutarliligi, iyimserlik
    rng = np.random.default_rng(2026)
    B = 500
    bh, opt, fail = [], [], 0
    for b in range(B):
        bs = mc.iloc[rng.integers(0, len(mc), len(mc))].reset_index(drop=True)
        cv = [c for c in COVS if bs[c].nunique() > 1 and bs.loc[bs.E == 1, c].nunique() > 0]
        try:
            fb = fit(bs, cv)
        except Exception:
            fail += 1
            continue
        hr = np.exp(fb.params_).reindex(COVS)
        if (hr > 50).any() or (hr < 0.02).any():
            fail += 1
        bh.append(hr)
        c_b = concordance_index(bs.time_cab, -fb.predict_partial_hazard(bs[cv]), bs.E)
        c_o = concordance_index(mc.time_cab, -fb.predict_partial_hazard(mc[cv]), mc.E)
        opt.append(c_b - c_o)
        if (b + 1) % 100 == 0:
            print(f"   bootstrap {b + 1}/{B}", flush=True)
    bh = pd.DataFrame(bh)
    optim = float(np.mean(opt))
    out(f"  Gorunur C={capp:.3f}; ortalama iyimserlik={optim:.3f}; iyimserlik duzeltilmis C={capp - optim:.3f}; "
        f"yakinsamayan/asiri tahminli bootstrap={fail}/{B}")
    for c in COVS:
        hr0 = np.exp(full.params_[c]); col = bh[c].dropna()
        same = (np.sign(np.log(col)) == np.sign(np.log(hr0))).mean() * 100
        stab.append({"Variable": c, "HR (full model)": round(hr0, 2),
                     "Bootstrap 2.5%": round(col.quantile(.025), 2), "Bootstrap 97.5%": round(col.quantile(.975), 2),
                     "Same direction, %": round(same, 1)})
    st = pd.DataFrame(stab)
    out(st.to_string(index=False))
    # cezali model (ridge) ve indirgenmis model
    rows_ = []
    for pen in [0.05, 0.1]:
        fp = fit(mc, COVS, pen)
        for c in [c_ for c_ in ["female", "age10", "periop", "Neurosurgery", "Trauma_surgery", "braden_high", "sofa"] if c_ in COVS]:
            rows_.append({"Model": f"Ridge penalizer {pen}", "Variable": c, "HR": round(np.exp(fp.params_[c]), 2)})
    mc["neuro_trauma"] = ((mc.Neurosurgery == 1) | (mc.Trauma_surgery == 1)).astype(int)
    RED = [c for c in ["age10", "female", "periop", "sofa", "neuro_trauma"] if mc[c].nunique() > 1]
    fr = fit(mc, RED)
    sr = fr.summary
    out(f"  Indirgenmis model (5 parametre, EPV={nev / 5:.1f}; C={fr.concordance_index_:.3f}):")
    for c in RED:
        out(f"    {c}: HR {sr.loc[c, 'exp(coef)']:.2f} ({sr.loc[c, 'exp(coef) lower 95%']:.2f}–{sr.loc[c, 'exp(coef) upper 95%']:.2f}), p={fmt_p(sr.loc[c, 'p'])}")
        rows_.append({"Model": "Reduced (5 parameters)", "Variable": c,
                      "HR": f"{sr.loc[c, 'exp(coef)']:.2f} ({sr.loc[c, 'exp(coef) lower 95%']:.2f}–{sr.loc[c, 'exp(coef) upper 95%']:.2f})"})
    for r_ in rows_:
        if r_["Model"].startswith("Ridge"):
            pass
    pr = pd.DataFrame(rows_)
    out(pr[pr.Model.str.startswith("Ridge")].pivot(index="Variable", columns="Model", values="HR").to_string())
    # SOFA'siz model
    COV_NS = [c for c in COVS if c != "sofa"]
    mns = add_dummies(main_pop.copy())[["time_cab", "ev_cab"] + COV_NS].dropna().copy()
    mns["E"] = (mns.ev_cab == 1).astype(int)
    fns = fit(mns, COV_NS); sn = fns.summary
    out(f"  SOFA'siz Cox (n={sup(len(mns))}, C={fns.concordance_index_:.3f}):")
    for c in [c_ for c_ in ["female", "age10", "periop", "Neurosurgery", "Trauma_surgery", "braden_high"] if c_ in COV_NS]:
        out(f"    {c}: HR {sn.loc[c, 'exp(coef)']:.2f} ({sn.loc[c, 'exp(coef) lower 95%']:.2f}–{sn.loc[c, 'exp(coef) upper 95%']:.2f})")
        rows_.append({"Model": "Without SOFA", "Variable": c,
                      "HR": f"{sn.loc[c, 'exp(coef)']:.2f} ({sn.loc[c, 'exp(coef) lower 95%']:.2f}–{sn.loc[c, 'exp(coef) upper 95%']:.2f})"})
    sheets["Cox_bootstrap"] = st
    sheets["Cox_alt_models"] = pd.DataFrame(rows_)

    # ---------- BIRLESTIRILMIS GUNLUK ARALIKLAR (her aralikta 0 ya da >=10 olay)
    def merged_hz(p, tcol, ecol, cuts):
        rr = []
        for lo_, hi_ in zip(cuts[:-1], cuts[1:]):
            T = np.clip(p[tcol] - lo_, 0, hi_ - lo_).sum()
            e = int(((p[ecol] == 1) & (p[tcol] >= lo_) & (p[tcol] < hi_)).sum())
            r, l, u = poisson_ci(e, T)
            rr.append({"interval": f"{lo_:g}–{hi_:g}" if hi_ < 1e8 else f"≥{lo_:g}", "lo_d": lo_, "hi_d": hi_,
                       "at_risk": int((p[tcol] > lo_).sum()), "days": round(T), "events": sup(e),
                       "rate": round(r, 2), "ci": f"{l:.2f}–{u:.2f}", "l": l, "u": u})
        return pd.DataFrame(rr)
    CUTS = [0, 1, 2, 3, 5, 6, 8, 1e9]
    mh = merged_hz(main_pop, "time_cab", "ev_cab", CUTS)
    out("\nBIRLESTIRILMIS GUNLUK INSIDANS (dogrulanmis postop):")
    out(mh[["interval", "at_risk", "days", "events", "rate", "ci"]].to_string(index=False))
    small = [r_["interval"] for _, r_ in mh.iterrows() if r_["events"] == "<10"]
    out(f"  <10 olayli aralik: {small if small else 'yok'}")
    sheets["Daily_merged"] = mh.drop(columns=["l", "u"])

    # ---------- Tablo 1 (dogrulanmis postoperatif kohort; CAB'ye gore)
    def tcol_(g):
        n = len(g)
        return {"n": sup(n), "Age, mean ± SD": f"{g.anchor_age.mean():.1f} ± {g.anchor_age.std():.1f}",
                "Female": npct(g.female.sum(), n), "Emergency admission": npct(g.emergency.sum(), n),
                "Diabetes": npct(g.diabetes.sum(), n), "Invasive MV within 24 h": npct(g.vent24.sum(), n),
                "Modified SOFA, median (IQR)": med_iqr(g.sofa), "Braden ≤12": npct((g.braden <= 12).sum(), g.braden.notna().sum()),
                "Perioperative insertion": npct(g.periop.sum(), n), "Catheter days, median (IQR)": med_iqr(g.dur_d),
                "Catheter >48 h": npct(g.prolonged.sum(), n), "ICU LOS, median (IQR)": med_iqr(g.los),
                "In-hospital mortality": npct(g.hospital_expire_flag.sum(), n),
                **{f"Service: {s}": npct((g.svc == s).sum(), n) for s in SERVICES},
                **{f"ICU: {t}": npct((g.icu_type == t).sum(), n) for t in ["CVICU", "SICU", "Trauma SICU", "Neuro ICU/stepdown", "Other ICU"]},
                **{f"Period: {y}": npct((g.anchor_year_group == y).sum(), n) for y in sorted(d.anchor_year_group.dropna().unique())}}
    mp = main_pop
    t1 = pd.DataFrame({"Total": tcol_(mp), "No CAB": tcol_(mp[mp.cab == 0]), "CAB": tcol_(mp[mp.cab == 1])})
    pv = {}
    for nm, v in [("Female", "female"), ("Emergency admission", "emergency"), ("Diabetes", "diabetes"),
                  ("Invasive MV within 24 h", "vent24"), ("Perioperative insertion", "periop"),
                  ("Catheter >48 h", "prolonged"), ("In-hospital mortality", "hospital_expire_flag")]:
        pv[nm] = fmt_p(stats.chi2_contingency(pd.crosstab(mp[v], mp.cab))[1])
    for nm, v in [("Modified SOFA, median (IQR)", "sofa"), ("Catheter days, median (IQR)", "dur_d"),
                  ("ICU LOS, median (IQR)", "los")]:
        pv[nm] = fmt_p(stats.mannwhitneyu(mp.loc[mp.cab == 0, v].dropna(), mp.loc[mp.cab == 1, v].dropna()).pvalue)
    pv["Age, mean ± SD"] = fmt_p(stats.ttest_ind(mp.loc[mp.cab == 0, "anchor_age"], mp.loc[mp.cab == 1, "anchor_age"], equal_var=False).pvalue)
    t1["p"] = pd.Series(pv); t1["p"] = t1.p.fillna("")
    out("\nTABLO 1 (dogrulanmis postoperatif kohort)"); out(t1.to_string())
    sheets["Table1"] = t1.reset_index()

    # ---------- Uzamis kateterizasyon (mobilizasyonsuz) lojistik
    import statsmodels.api as sm
    m = mp.copy()
    m = add_dummies(m)
    cv4 = ["age10", "female", "diabetes", "emergency", "vent24", "braden_high", "sofa", "period_2014_2016",
           "period_2017_2022"] + SDUM
    ml = m[["prolonged"] + cv4].dropna()
    lg = sm.Logit(ml.prolonged, sm.add_constant(ml[cv4])).fit(disp=0)
    ci4 = np.exp(lg.conf_int())
    t4 = pd.DataFrame({"Variable": cv4, "OR": np.exp(lg.params[cv4]).round(2).values,
                       "lo": ci4.loc[cv4, 0].round(2).values, "hi": ci4.loc[cv4, 1].round(2).values,
                       "p": [fmt_p(v) for v in lg.pvalues[cv4]]})
    out(f"\nUZAMIS KATETERIZASYON LOJISTIK (n={sup(len(ml))}, McFadden={lg.prsquared:.3f})"); out(t4.to_string(index=False))
    sheets["Logistic_prolonged"] = t4

    # ---------- Mobilizasyon: zamansal siralama + zaman-bagimli Cox + landmark
    a = ce[ce.itemid.isin(act.itemid)].copy()
    num = pd.to_numeric(a.value.str.extract(r"^\s*(\d)")[0], errors="coerce")
    is_hlm = a.itemid.map(dict(zip(act.itemid, act.label))).str.contains("JH-HLM", case=False)
    a["oob"] = np.where(is_hlm, num >= 4, a.value.fillna("").str.contains(OOB_RE))
    first_oob = a[a.oob & (a.charttime >= a.intime)].groupby("stay_id").charttime.min()
    mp = mp.copy()
    mp["oob_t"] = mp.stay_id.map(first_oob)
    mm = mp[mp.removed_icu].copy()
    seq_before = (mm.oob_t.notna() & (mm.oob_t < mm.cath_stop)).sum()
    seq_after = (mm.oob_t.notna() & (mm.oob_t >= mm.cath_stop)).sum()
    out("\nMOBILIZASYON - ZAMANSAL SIRALAMA (ICU'da kateteri cekilenler, n=" + sup(len(mm)) + ")")
    out(f"  Ilk yataktan cikis kateter cekilmeden ONCE: {npct(seq_before, len(mm))}")
    out(f"  Ilk yataktan cikis kateter cekildikten SONRA: {npct(seq_after, len(mm))}")
    out(f"  ICU'da yataktan cikis kaydi yok: {npct(mm.oob_t.isna().sum(), len(mm))}")
    gap = ((mm.cath_stop - mm.oob_t).dt.total_seconds() / 3600)[mm.oob_t.notna() & (mm.oob_t < mm.cath_stop)]
    if len(gap):
        out(f"  Mobilizasyondan cekilmeye sure, saat, medyan (IQR): {med_iqr(gap)}")
    # zaman-bagimli Cox: sonlanim = kateterin cekilmesi; sansur = olum / ICU cikisi kateterle
    from lifelines import CoxTimeVaryingFitter
    base = ["age10", "female", "diabetes", "emergency", "vent24", "sofa", "period_2014_2016", "period_2017_2022"]
    mp = add_dummies(mp)
    rows = []
    for r in mp[["stay_id", "cath_start", "cath_stop", "oob_t", "removed_icu"] + base + SDUM].dropna(subset=base).itertuples(index=False):
        T = (r.cath_stop - r.cath_start).total_seconds() / 86400
        ev = int(r.removed_icu)
        cov = [getattr(r, c) for c in base + SDUM]
        tm = (r.oob_t - r.cath_start).total_seconds() / 86400 if pd.notna(r.oob_t) else None
        if tm is not None and 0 < tm < T:
            rows.append([r.stay_id, 0, tm, 0, 0] + cov)
            rows.append([r.stay_id, tm, T, 1, ev] + cov)
        else:
            rows.append([r.stay_id, 0, T, 1 if (tm is not None and tm <= 0) else 0, ev] + cov)
    cols = ["id", "start", "stop", "mobilised", "removed"] + base + SDUM
    tv = pd.DataFrame(rows, columns=cols).dropna()
    tv = tv[tv.stop > tv.start]
    tv = tv.drop(columns=[c for c in base + SDUM if tv[c].nunique() < 2])
    ctv = CoxTimeVaryingFitter().fit(tv, id_col="id", start_col="start", stop_col="stop", event_col="removed")
    sm_ = ctv.summary.loc["mobilised"]
    out(f"  Zaman-bagimli Cox (kateter cekilmesi; mobilizasyon zamanla degisen): HR {sm_['exp(coef)']:.2f} "
        f"({sm_['exp(coef) lower 95%']:.2f}–{sm_['exp(coef) upper 95%']:.2f}), p={fmt_p(sm_['p'])}; "
        f"n={sup(tv.id.nunique())}, cekilme={sup(tv.groupby('id').removed.max().sum())}")
    # landmark 24 saat: 24. saatte kateter yerinde ve canli; 24 saate kadar mobilize vs degil -> 72. saate kadar cekilme
    L = mp[(mp.cath_stop > mp.cath_start + pd.Timedelta("24h"))].copy()
    L["mob24"] = (L.oob_t.notna() & (L.oob_t <= L.cath_start + pd.Timedelta("24h"))).astype(int)
    L["rem72"] = (L.removed_icu & (L.cath_stop <= L.cath_start + pd.Timedelta("72h"))).astype(int)
    lm = L[["rem72", "mob24"] + base + SDUM].dropna()
    xL = [c for c in ["mob24"] + base + SDUM if lm[c].nunique() > 1]
    lgL = sm.Logit(lm.rem72, sm.add_constant(lm[xL])).fit(disp=0)
    o, (l_, u_) = np.exp(lgL.params["mob24"]), np.exp(lgL.conf_int().loc["mob24"])
    out(f"  Landmark (24 saat; n={sup(len(lm))}): 24 saatte mobilize olan {npct(lm.mob24.sum(), len(lm))}; "
        f"72. saate kadar cekilme mobilize {npct(lm[lm.mob24 == 1].rem72.sum(), lm.mob24.sum())} vs "
        f"degil {npct(lm[lm.mob24 == 0].rem72.sum(), (lm.mob24 == 0).sum())}; duzeltilmis OR {o:.2f} ({l_:.2f}–{u_:.2f}), p={fmt_p(lgL.pvalues['mob24'])}")
    sheets["Mobilisation"] = pd.DataFrame([
        {"Analysis": "First out-of-bed before removal", "Result": npct(seq_before, len(mm))},
        {"Analysis": "First out-of-bed after removal", "Result": npct(seq_after, len(mm))},
        {"Analysis": "No out-of-bed documented", "Result": npct(mm.oob_t.isna().sum(), len(mm))},
        {"Analysis": "Time-dependent Cox HR for removal", "Result": f"{sm_['exp(coef)']:.2f} ({sm_['exp(coef) lower 95%']:.2f}–{sm_['exp(coef) upper 95%']:.2f})"},
        {"Analysis": "Landmark 24 h OR for removal by 72 h", "Result": f"{o:.2f} ({l_:.2f}–{u_:.2f})"}])

    # ---------- Yeniden kateterizasyon: tam 48 saat izlem + sansurlu Cox
    di_o = di[(di.linksto == "outputevents") & di.label.fillna("").str.contains("straight cath", case=False)]
    oe = pd.read_csv(find(root, "outputevents.csv.gz"), usecols=["stay_id", "charttime", "itemid"], parse_dates=["charttime"])
    oe = oe[oe.itemid.isin(di_o.itemid) & oe.stay_id.isin(mp.stay_id)][["stay_id", "charttime"]].rename(columns={"charttime": "t"})
    # ilk ham kateter epizodunun ICU icinde cekilmesi (sonradan yeniden takilsa da)
    rm = mp[(mp.raw_end < mp.outtime - pd.Timedelta("1h")) &
            ~(mp.deathtime.notna() & (mp.deathtime <= mp.raw_end))].copy()
    rm["rem_t"] = rm.raw_end
    extra = rm[~rm.removed_icu]
    out(f"\nCEKILME POPULASYONU UZLASTIRMASI: ilk ham epizod ICU'da cekilen={sup(len(rm))}; "
        f"birlestirilmis epizod ICU'da cekilen={sup(mp.removed_icu.sum())}; ikisi de={sup(rm.removed_icu.sum())}")
    reins = extra.stay_id.isin(raw[raw.stay_id.isin(extra.stay_id)].groupby("stay_id").size().loc[lambda z: z > 1].index)
    out(f"  Yalniz ilk epizodu cekilen (24 saat icinde yeniden takilip ICU cikisi/olume kadar kalan): {sup(len(extra))}; "
        f"bunlardan ikinci epizodu olan: {sup(reins.sum())}")
    only_merged = mp[mp.removed_icu & ~mp.stay_id.isin(rm.stay_id)]
    out(f"  Yalniz birlestirilmis epizodu cekilen: {sup(len(only_merged))}")
    rm["death48"] = (rm.deathtime.notna() & (rm.deathtime <= rm.rem_t + pd.Timedelta("48h"))).astype(int)
    rf = raw[raw.stay_id.isin(rm.stay_id)][["stay_id", "starttime"]].rename(columns={"starttime": "t"})
    rc = pd.concat([rf, oe]).merge(rm[["stay_id", "rem_t", "outtime"]], on="stay_id")
    rc = rc[(rc.t >= rc.rem_t + pd.Timedelta("1h")) & (rc.t <= rc.rem_t + pd.Timedelta("48h")) & (rc.t <= rc.outtime)]
    first_rc = rc.groupby("stay_id").t.min()
    rm["recath"] = rm.stay_id.isin(first_rc.index).astype(int)
    rm["early_rm"] = (((rm.rem_t - rm.cath_start).dt.total_seconds() / 86400) <= 2).astype(int)
    rm["fu_end"] = rm[["outtime"]].assign(h48=rm.rem_t + pd.Timedelta("48h"),
                                          dth=rm.deathtime.fillna(rm.outtime)).min(axis=1)
    rm["fu_h"] = np.where(rm.recath == 1, (rm.stay_id.map(first_rc) - rm.rem_t).dt.total_seconds() / 3600,
                          (rm.fu_end - rm.rem_t).dt.total_seconds() / 3600)
    rm["complete48"] = ((rm.outtime >= rm.rem_t + pd.Timedelta("48h")) & (rm.death48 == 0)).astype(int)
    out("\nYENIDEN KATETERIZASYON (dogrulanmis postoperatif kohort)")
    for lab_, g in [("<=48 h", rm[rm.early_rm == 1]), (">48 h", rm[rm.early_rm == 0])]:
        out(f"  Cekim {lab_}: n={sup(len(g))}; tam 48 saat ICU izlemi {npct(g.complete48.sum(), len(g))}; "
            f"izlem saat medyan (IQR) {med_iqr(g.fu_h)}; yeniden kateterizasyon {npct(g.recath.sum(), len(g))}")
    cvr = ["early_rm", "age10", "female", "emergency", "vent24", "sofa"] + SDUM
    rows = []
    c48 = rm[rm.complete48 == 1]
    mr = c48[["recath"] + cvr].dropna()
    try:
        lr = sm.Logit(mr.recath, sm.add_constant(mr[cvr])).fit(disp=0)
        o2, (l2, u2) = np.exp(lr.params["early_rm"]), np.exp(lr.conf_int().loc["early_rm"])
        out(f"  Tam 48 saat izlemli alt grup (n={sup(len(mr))}): <=48 h {npct(c48[c48.early_rm == 1].recath.sum(), (c48.early_rm == 1).sum())} vs "
            f">48 h {npct(c48[c48.early_rm == 0].recath.sum(), (c48.early_rm == 0).sum())}; duzeltilmis OR {o2:.2f} ({l2:.2f}–{u2:.2f}), p={fmt_p(lr.pvalues['early_rm'])}")
        rows.append({"Analysis": "Complete 48-h follow-up, adjusted OR (≤48 h vs >48 h)", "Result": f"{o2:.2f} ({l2:.2f}–{u2:.2f}); p={fmt_p(lr.pvalues['early_rm'])}", "n": sup(len(mr))})
    except Exception as ex:
        out(f"  lojistik kurulamadi: {ex}")
    from lifelines import CoxPHFitter
    mcx = rm[["fu_h", "recath"] + cvr].dropna()
    mcx = mcx[mcx.fu_h > 0]
    fx = CoxPHFitter().fit(mcx, "fu_h", "recath")
    sx = fx.summary.loc["early_rm"]
    out(f"  Sansurlu Cox (ICU cikisi/48 saatte sansur; n={sup(len(mcx))}): HR {sx['exp(coef)']:.2f} ({sx['exp(coef) lower 95%']:.2f}–{sx['exp(coef) upper 95%']:.2f}), p={fmt_p(sx['p'])}")
    rows.append({"Analysis": "Censored Cox HR (≤48 h vs >48 h)", "Result": f"{sx['exp(coef)']:.2f} ({sx['exp(coef) lower 95%']:.2f}–{sx['exp(coef) upper 95%']:.2f}); p={fmt_p(sx['p'])}", "n": sup(len(mcx))})
    sheets["Recatheterisation"] = pd.DataFrame(rows)

    # ---------- surec gostergeleri (dogrulanmis kohort)
    bsc_ids = set(bsc.itemid)
    b = ce[ce.itemid.isin(bsc_ids)].merge(rm[["stay_id", "rem_t"]], on="stay_id")
    b = b[(b.charttime > b.rem_t) & (b.charttime <= b.rem_t + pd.Timedelta("24h"))]
    out(f"\n  Cekim sonrasi 24 saatte mesane taramasi: {npct(rm.stay_id.isin(b.stay_id).sum(), len(rm))}")
    mp["excess"] = np.clip(mp.dur_d - 2, 0, None)
    out(f"  48 saati asan kateter-gunu: {mp.excess.sum():.0f} / {mp.dur_d.sum():.0f} ({100 * mp.excess.sum() / mp.dur_d.sum():.1f}%)")
    out(f"  Kateteri ICU'da cekilen: {npct(mp.removed_icu.sum(), len(mp))}; 48 saat icinde cekilen: {npct((mp.dur_d <= 2).sum(), len(mp))}")

    # ---------- Sekiller (dogrulanmis kohort, CAB)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False})
    res = R["po_cab"]
    kmf, ajf, hz = res["kmf"], res["ajf"], res["hz"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=600)
    # egriler, risk altindaki hasta sayisi >=10 oldugu son zamana kadar cizilir
    tt_ = main_pop.time_cab.values
    tmax = max([u for u in np.sort(tt_) if (tt_ >= u).sum() >= 10] or [14])
    sf = kmf.survival_function_.loc[:tmax]
    ci = kmf.confidence_interval_survival_function_.loc[:tmax]
    t = sf.index
    a1.fill_between(t, 100 * (1 - ci.iloc[:, 0]), 100 * (1 - ci.iloc[:, 1]), step="post", color=BLUE, alpha=.15, lw=0)
    a1.step(t, 100 * (1 - sf.iloc[:, 0]), where="post", color=BLUE, lw=1.6,
            label="Kaplan–Meier (removal and death censored)")
    cdn = ajf.cumulative_density_.loc[:tmax]
    if hasattr(ajf, "confidence_interval_cumulative_density_"):
        aci = ajf.confidence_interval_cumulative_density_.ffill().fillna(0).loc[:tmax]
        a1.fill_between(aci.index, 100 * aci.min(axis=1), 100 * aci.max(axis=1), step="post", color=ORANGE, alpha=.15, lw=0)
    a1.step(cdn.index, 100 * cdn.iloc[:, 0], where="post", color=ORANGE, lw=1.6,
            label="Aalen–Johansen (removal and death as competing events)")
    ytop = max(8, np.ceil(100 * (1 - ci.loc[:14].iloc[:, 0].min()) * 1.35))
    a1.set_xlim(0, 14); a1.set_ylim(0, ytop); a1.set_xticks(range(0, 15, 2))
    a1.set_xlabel("Catheter day"); a1.set_ylabel("Cumulative incidence of CAB (%)")
    a1.legend(loc="upper left", fontsize=6.5, frameon=False, handlelength=1.6)
    out(f"  Sekil 1A egrileri {tmax:.1f}. gune kadar cizildi (risk altinda >=10 hasta)")
    a1.grid(axis="y", color=GRID, lw=.6); a1.set_axisbelow(True)
    a1.set_title("A", loc="left", fontweight="bold", fontsize=10)
    for kd in range(0, 15, 2):
        a1.annotate(sup((main_pop.time_cab >= kd).sum()), (kd, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=6, color=MUTED)
    a1.annotate("At risk", (0, 0), xycoords=("data", "axes fraction"), xytext=(-14, -34), textcoords="offset points",
                ha="right", fontsize=6, color=MUTED)
    for _, r_ in mh.iterrows():
        x0, x1 = r_.lo_d, min(r_.hi_d, 14)
        a2.fill_between([x0, x1], r_.l, r_.u, color=BLUE, alpha=.15, lw=0)
        a2.plot([x0, x1], [r_.rate, r_.rate], color=BLUE, lw=1.8)
    hzp = mh.rename(columns={"u": "hi"})
    a2.axvline(2, color=MUTED, lw=.8, ls=(0, (3, 3)))
    ytop2 = hzp.hi.max() * 1.08
    a2.set_xlim(0, 14); a2.set_xticks(range(0, 15, 2)); a2.set_ylim(0, ytop2)
    a2.text(2.15, ytop2 * .97, "48 h", color=MUTED, fontsize=7, va="top")
    a2.set_xlabel("Catheter day"); a2.set_ylabel("CAB per 1000 days at risk")
    a2.text(14, ytop2 * .97, "last interval: day ≥8", color=MUTED, fontsize=6.5, ha="right", va="top")
    a2.grid(axis="y", color=GRID, lw=.6); a2.set_axisbelow(True)
    a2.set_title("B", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure1.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    names = {"age10": "Age (per 10 years)", "female": "Female sex", "diabetes": "Diabetes mellitus",
             "emergency": "Emergency admission", "vent24": "Invasive MV within 24 h", "braden_high": "Braden score ≤12",
             "periop": "Perioperative catheter insertion", "sofa": "Modified SOFA (per point)",
             "period_2014_2016": "Period 2014–2016*", "period_2017_2022": "Period 2017–2022*",
             "Cardiac_surgery": "Cardiac surgery†", "Neurosurgery": "Neurosurgery†", "Trauma_surgery": "Trauma surgery†",
             "Other_surgery": "Other surgery†"}

    def forest(ax, df, col, xlabel, title):
        df = df.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(df))
        ax.axvline(1, color=MUTED, lw=.8)
        ax.hlines(y, df.lo, df.hi, color=BLUE, lw=1.4)
        ax.plot(df[col], y, "s", color=BLUE, ms=4.5)
        ax.set_yticks(y); ax.set_yticklabels([names.get(v, v) for v in df.Variable], fontsize=7, color=INK)
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
        tk = [t_ for t_ in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16] if df.lo.min() * .7 <= t_ <= df.hi.max() * 1.6]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(xlabel); ax.tick_params(axis="y", length=0); ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
        xmax = df.hi.max() * 1.6
        ax.set_xlim(min(df.lo.min() * .8, .5), xmax * 3.2)
        for yi, r in df.iterrows():
            ax.text(xmax * 1.05, yi, f"{r[col]:.2f} ({r.lo:.2f}–{r.hi:.2f})", va="center", fontsize=6.5, color=INK)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.6, 4.4), dpi=600)
    forest(b1, res["cox"], "HR", "Hazard ratio (log scale)", "A  CAB")
    forest(b2, t4, "OR", "Odds ratio (log scale)", "B  Catheter >48 h")
    fig.text(0.01, -0.01, "*Reference: 2008–2013. †Reference: general surgery. MV: mechanical ventilation; SOFA: Sequential Organ Failure Assessment.",
             fontsize=6.5, color=MUTED)
    fig.tight_layout(w_pad=3.5)
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure2.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    order = ["Total"] + SERVICES
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=600)
    cats = ["≤2 days", ">2–4 days", ">4–7 days", ">7 days"]
    dc = pd.cut(main_pop.dur_d, [0, 2, 4, 7, 1e9], labels=cats)
    for i, grp in enumerate(order):
        mask = np.ones(len(main_pop), bool) if grp == "Total" else (main_pop.svc == grp).values
        share = dc[mask].value_counts(normalize=True).reindex(cats).fillna(0).values * 100
        left = 0
        for j, v in enumerate(share):
            ax.barh(i, v, left=left, color=RAMP[j], edgecolor="white", linewidth=1.2, height=.62, label=cats[j] if i == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5, color="white" if j >= 1 else INK)
            left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{o_} (n={sup(mask_n)})" for o_, mask_n in
                        zip(order, [len(main_pop)] + [(main_pop.svc == s).sum() for s in SERVICES])], fontsize=7, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Patients (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(title="Catheter duration", ncol=4, fontsize=7, title_fontsize=7, frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure3.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    out("\nAKIS:")
    for k, v in flow.items():
        out(f"  {k}: {sup(v)}")
    out(f"  Dogrulanmis postoperatif kohort: {sup(len(main_pop))}")
    with pd.ExcelWriter(odir / "catheter_cab_tables.xlsx") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, k[:31], index=False)
        for k, r in R.items():
            r["rates"].to_excel(xw, f"{k}_rates"[:31], index=False)
            r["cum"].to_excel(xw, f"{k}_cum"[:31], index=False)
            if "cox" in r:
                r["cox"].to_excel(xw, f"{k}_cox"[:31], index=False)
            r["hz"].assign(events=r["hz"].events.map(sup)).to_excel(xw, f"{k}_hz"[:31], index=False)
    out(f"\nToplam sure: {(time.time() - t0) / 60:.1f} dk")
    (odir / "catheter_cab_summary.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nSonuclar: {odir}")


if __name__ == "__main__":
    main()
