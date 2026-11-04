# -*- coding: utf-8 -*-
"""
MIMIC-IV v3.1 - Ameliyat sonrasi uriner kateter ve KAUTI - GENISLETILMIS ANALIZ (v2)
Ilk betikle (kateter_analiz.py) ayni kohort ve tanimlar; ustune:
  - Rekabet eden risk (Aalen-Johansen) kumulatif insidans  vs  Kaplan-Meier
  - Kateter gunune gore gunluk KAUTI hazardi (Poisson + kubik spline)
  - Hemsirelik sureci gostergeleri (Donabedian): erken mobilizasyon, kateterin
    cekildigi saat, cekim sonrasi mesane taramasi, yeniden kateterizasyon,
    "fazla kateter-gunu"
  - Cox ve lojistik modellere erken mobilizasyon; E-degerleri; duyarlilik (PACU kohortu)
  - Yayin kalitesinde Sekil 1-3 (300 dpi PNG + TIFF)
Yalnizca TOPLU sonuc uretir; <10 hucreler gizlenir.

Kurulum:  py -m pip install pandas numpy scipy statsmodels lifelines matplotlib openpyxl patsy
Calistirma (kateter_analiz.py ile ayni klasorde olmali):
    py "D:\\MIMICIV\\kateter_analiz2.py" --root D:\\MIMICIV
Cikti: D:\\MIMICIV\\kateter_sonuclar2\\
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

sys.path.insert(0, str(Path(__file__).parent))
import kateter_analiz as K  # noqa: E402

warnings.filterwarnings("ignore")
LOG = []
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
SERVICES = ["General surgery", "Cardiac surgery", "Neurosurgery", "Trauma surgery", "Other surgery"]
TR2EN = {"Genel cerrahi": "General surgery", "Kalp cerrahisi": "Cardiac surgery", "Norosirurji": "Neurosurgery",
         "Travma cerrahisi": "Trauma surgery", "Diger cerrahi": "Other surgery"}
OOB_RE = re.compile(r"chair|ambul|walk|stand|commode|step|march|oob|out of bed", re.I)


def out(s=""):
    print(s, flush=True)
    LOG.append(str(s))


sup, npct, fmt_p, find = K.sup, K.npct, K.fmt_p, K.find


def evalue(est, lo, hi):
    def e(x):
        x = 1 / x if x < 1 else x
        return x + np.sqrt(x * (x - 1))
    ev = e(est)
    if lo <= 1 <= hi:
        evc = 1.0
    else:
        evc = e(lo if est > 1 else hi)
    return ev, evc


def poisson_ci(k, t):
    lo = stats.chi2.ppf(.025, 2 * k) / 2 if k else 0.0
    hi = stats.chi2.ppf(.975, 2 * (k + 1)) / 2
    return 1000 * k / t, 1000 * lo / t, 1000 * hi / t


# ---------------------------------------------------------------- veri hazirlama
def build(root):
    coh, flow = K.load_cohort(root)
    pe = pd.read_csv(find(root, "procedureevents.csv.gz"),
                     usecols=["stay_id", "starttime", "endtime", "itemid"],
                     parse_dates=["starttime", "endtime"])
    pe = pe[pe.stay_id.isin(coh.stay_id)]
    vent = pe[pe.itemid == K.VENT_INV].merge(coh[["stay_id", "intime"]], on="stay_id")
    vent24 = set(vent[vent.starttime <= vent.intime + pd.Timedelta("24h")].stay_id)
    raw = pe[pe.itemid == K.FOLEY].dropna(subset=["starttime", "endtime"])
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
    ep = pd.DataFrame(rows, columns=["stay_id", "cath_start", "cath_end", "raw_end"])
    d = coh.merge(ep, on="stay_id")
    flow["Foley kateteri olan"] = len(d)
    d["cath_end"] = d[["cath_end", "deathtime"]].min(axis=1)
    d = d[d.cath_end > d.cath_start]
    d["dur_d"] = (d.cath_end - d.cath_start).dt.total_seconds() / 86400

    pos = K.urine_cultures(root, set(d.subject_id))
    x = d[["stay_id", "subject_id", "cath_start", "cath_end"]].merge(pos, on="subject_id")
    pre = x[(x.t >= x.cath_start - pd.Timedelta("48h")) & (x.t < x.cath_start + pd.Timedelta("48h"))]
    d = d[~d.stay_id.isin(pre.stay_id)]
    flow["Onceden pozitif kultur cikarildiktan sonra"] = len(d)
    ev = x[x.stay_id.isin(d.stay_id) & (x.t >= x.cath_start + pd.Timedelta("48h")) &
           (x.t <= x.cath_end + pd.Timedelta("24h"))]
    fe = ev.sort_values("t").groupby("stay_id").first()
    d["cauti"] = d.stay_id.isin(fe.index).astype(int)
    d = d.merge(fe[["t"]], left_on="stay_id", right_index=True, how="left")
    stop = d.cath_end + pd.Timedelta("24h")
    d["time_d"] = np.where(d.cauti == 1, (d.t - d.cath_start).dt.total_seconds() / 86400,
                           (stop - d.cath_start).dt.total_seconds() / 86400)
    d["died_on_cath"] = (d.deathtime.notna() & (d.deathtime <= d.cath_end)).astype(int)

    dx = pd.read_csv(find(root, "diagnoses_icd.csv.gz"), dtype={"icd_code": "string"})
    dm = dx[((dx.icd_version == 10) & dx.icd_code.str.match(r"E1[0-4]")) |
            ((dx.icd_version == 9) & dx.icd_code.str.startswith("250"))].hadm_id
    d["diabetes"] = d.hadm_id.isin(dm).astype(int)
    d["female"] = (d.gender == "F").astype(int)
    d["age10"] = d.anchor_age / 10
    d["vent24"] = d.stay_id.isin(vent24).astype(int)
    d["emergency"] = (~d.admission_type.fillna("").str.contains("ELECTIVE|SAME DAY", case=False)).astype(int)
    d["svc"] = d.curr_service.map(K.SVC_GRP).map(TR2EN)
    d["periop"] = ((d.cath_start - d.intime) <= pd.Timedelta("6h")).astype(int)
    d["prolonged"] = (d.dur_d > 2).astype(int)
    d["dur_cat"] = pd.cut(d.dur_d, [0, 2, 4, 7, 1e9], labels=["≤2 days", ">2–4 days", ">4–7 days", ">7 days"])

    # PACU (duyarlilik kohortu)
    trf = pd.read_csv(find(root, "transfers.csv.gz"), usecols=["hadm_id", "careunit", "outtime"],
                      parse_dates=["outtime"])
    pacu = trf[trf.careunit.fillna("").str.contains("PACU", case=False)].dropna(subset=["hadm_id"])
    pacu["hadm_id"] = pacu.hadm_id.astype("int64")
    y = d[["stay_id", "hadm_id", "intime"]].merge(pacu, on="hadm_id")
    y = y[(y.outtime >= y.intime - pd.Timedelta("24h")) & (y.outtime <= y.intime + pd.Timedelta("1h"))]
    d["pacu"] = d.stay_id.isin(y.stay_id).astype(int)
    return d, flow, raw, dx


def chart_scan(root, d):
    di = pd.read_csv(find(root, "d_items.csv.gz"))
    ce_items = di[di.linksto == "chartevents"]
    lab = ce_items.label.fillna("")
    act = ce_items[lab.str.contains(r"^activity$|activity / mobility|jh-hlm|highest level of mobility",
                                    case=False, regex=True)]
    bsc = ce_items[lab.str.contains(r"bladder scan", case=False)]
    out("\nHemsirelik kalemleri (chartevents):")
    for _, r in pd.concat([act, bsc]).iterrows():
        out(f"  {r.itemid} | {r.label}")
    ids = set(K.BRADEN) | set(act.itemid) | set(bsc.itemid)
    stays = set(d.stay_id)
    parts, t0 = [], time.time()
    for i, ch in enumerate(pd.read_csv(find(root, "chartevents.csv.gz"),
                                       usecols=["stay_id", "charttime", "itemid", "value", "valuenum"],
                                       chunksize=5_000_000, dtype={"value": "string"}), 1):
        ch = ch[ch.itemid.isin(ids) & ch.stay_id.isin(stays)]
        if len(ch):
            parts.append(ch)
        print(f"   chartevents parca {i} ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    ce = pd.concat(parts, ignore_index=True)
    ce["stay_id"] = ce.stay_id.astype("int64")
    ce["charttime"] = pd.to_datetime(ce.charttime)
    ce = ce.merge(d[["stay_id", "intime", "cath_start", "raw_end", "outtime"]], on="stay_id")

    # Braden (ilk 24 saatte ilk tam puan)
    br = ce[ce.itemid.isin(K.BRADEN) & (ce.charttime >= ce.intime) &
            (ce.charttime <= ce.intime + pd.Timedelta("24h"))]
    w = br.pivot_table(index=["stay_id", "charttime"], columns="itemid", values="valuenum",
                       aggfunc="first").dropna()
    w["total"] = w.sum(axis=1)
    braden = w.reset_index().sort_values("charttime").groupby("stay_id").total.first()

    # erken mobilizasyon: yogun bakima giristen sonraki 48 saatte yataktan cikis kaydi
    a = ce[ce.itemid.isin(act.itemid)].copy()
    if len(a):
        vc = a.value.value_counts().head(15)
        out("  Mobilite kalemlerinin en sik degerleri:")
        for k, v in vc.items():
            out(f"    {k}: {sup(v)}")
        num = pd.to_numeric(a.value.str.extract(r"^\s*(\d)")[0], errors="coerce")
        is_hlm = a.itemid.map(dict(zip(act.itemid, act.label))).str.contains("JH-HLM|Mobility", case=False)
        # JH-HLM kalemlerinde yalnizca puan >=4 (sandalyeye gecis ve uzeri); "2c - Lift to chair" pasif, sayilmaz
        a["oob"] = np.where(is_hlm, num >= 4, a.value.fillna("").str.contains(OOB_RE))
        a48 = a[(a.charttime >= a.intime) & (a.charttime <= a.intime + pd.Timedelta("48h"))]
        mob = a48.groupby("stay_id").oob.any()
        has_act = set(a48.stay_id)
    else:
        mob, has_act = pd.Series(dtype=bool), set()

    # cekim sonrasi 24 saatte mesane taramasi
    b = ce[ce.itemid.isin(bsc.itemid)]
    b = b[(b.charttime > b.raw_end) & (b.charttime <= b.raw_end + pd.Timedelta("24h"))]
    bscan = set(b.stay_id)
    return braden, mob, has_act, bscan


def straight_caths(root, d):
    di = pd.read_csv(find(root, "d_items.csv.gz"))
    oi = di[(di.linksto == "outputevents") & di.label.fillna("").str.contains("straight cath", case=False)]
    out("\nOutputevents straight cath kalemleri:")
    for _, r in oi.iterrows():
        out(f"  {r.itemid} | {r.label}")
    if oi.empty:
        return pd.DataFrame(columns=["stay_id", "charttime"])
    oe = pd.read_csv(find(root, "outputevents.csv.gz"), usecols=["stay_id", "charttime", "itemid"],
                     parse_dates=["charttime"])
    return oe[oe.itemid.isin(oi.itemid) & oe.stay_id.isin(d.stay_id)][["stay_id", "charttime"]]


# ---------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    root = ap.parse_args().root
    odir = Path(__file__).with_name("kateter_sonuclar2")
    odir.mkdir(exist_ok=True)
    t0 = time.time()
    out("KATETER CALISMASI - GENISLETILMIS ANALIZ (v2)")
    out("=" * 64)
    d, flow, raw, dx = build(root)
    braden, mob, has_act, bscan = chart_scan(root, d)
    d["braden"] = d.stay_id.map(braden)
    d["braden_high"] = (d.braden <= 12).astype(float).where(d.braden.notna())
    d["early_mob"] = d.stay_id.map(mob).astype(float)
    d.loc[~d.stay_id.isin(has_act), "early_mob"] = np.nan
    N = len(d)
    out(f"\nAnaliz kohortu: {sup(N)}  (KAUTI: {sup(d.cauti.sum())})")
    out(f"Mobilite kaydi olan (ilk 48 saat): {npct(d.early_mob.notna().sum(), N)}; "
        f"erken mobilize: {npct((d.early_mob == 1).sum(), d.early_mob.notna().sum())}")
    sheets = {}

    # ======================= 1) Aalen-Johansen vs Kaplan-Meier
    from lifelines import KaplanMeierFitter, AalenJohansenFitter, CoxPHFitter
    kmf = KaplanMeierFitter().fit(d.time_d, d.cauti)
    evt = np.where(d.cauti == 1, 1, 2)   # 2 = kateterin cekilmesi/olum (rekabet eden)
    ajf = AalenJohansenFitter(calculate_variance=True, seed=1).fit(
        pd.Series(d.time_d.values), pd.Series(evt), event_of_interest=1)
    rows = []
    for day in [3, 5, 7, 10, 14]:
        km = 1 - kmf.survival_function_at_times(day).iloc[0]
        cik = ajf.cumulative_density_.loc[:day].iloc[-1, 0]
        rows.append({"Catheter day": day, "KM: risk if catheter retained, %": f"{100 * km:.1f}",
                     "Aalen–Johansen: observed cumulative incidence, %": f"{100 * cik:.2f}",
                     "Number at risk": sup((d.time_d >= day).sum())})
    t_aj = pd.DataFrame(rows)
    sheets["KM_vs_AJ"] = t_aj
    out("\n1) KUMULATIF INSIDANS: KM vs Aalen-Johansen")
    out(t_aj.to_string(index=False))

    # ======================= 2) Gunluk hazard
    edges = list(range(0, 15))
    hz = []
    for k in edges:
        hi = k + 1 if k < 14 else 1e9
        expo = np.clip(d.time_d - k, 0, hi - k)
        evk = ((d.cauti == 1) & (d.time_d >= k) & (d.time_d < hi)).sum()
        T = expo.sum()
        if T <= 0:
            continue
        r, lo, up = poisson_ci(evk, T)
        hz.append({"day": k, "label": f"{k}–{k + 1}" if k < 14 else "≥14", "events": int(evk),
                   "catheter_days": T, "rate": r, "lo": lo, "hi": up,
                   "at_risk": int((d.time_d > k).sum())})
    hz = pd.DataFrame(hz)
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    fit_df = hz[(hz.day >= 2) & (hz.day < 14)].copy()
    fit_df["mid"] = fit_df.day + 0.5
    pm = smf.glm("events ~ cr(mid, df=3)", data=fit_df, family=sm.families.Poisson(),
                 offset=np.log(fit_df.catheter_days)).fit()
    grid = pd.DataFrame({"mid": np.linspace(2.5, 13.5, 100)})
    pr = pm.get_prediction(grid, offset=np.zeros(len(grid))).summary_frame()
    grid["rate"], grid["lo"], grid["hi"] = 1000 * pr["mean"], 1000 * pr["mean_ci_lower"], 1000 * pr["mean_ci_upper"]
    t_hz = hz.assign(**{"CAUTI per 1,000 catheter-days (95% CI)":
                        [f"{r:.2f} ({lo:.2f}–{h:.2f})" for r, lo, h in zip(hz.rate, hz.lo, hz.hi)],
                        "Events": [sup(e) for e in hz.events],
                        "Catheter-days": [f"{t:.0f}" for t in hz.catheter_days]})[
        ["label", "Events", "Catheter-days", "CAUTI per 1,000 catheter-days (95% CI)"]].rename(
        columns={"label": "Catheter day"})
    sheets["Daily_hazard"] = t_hz
    out("\n2) KATETER GUNUNE GORE GUNLUK HAZARD")
    out(t_hz.to_string(index=False))
    # ilk 4 gun vs sonrasi oran orani
    e1, T1 = hz[(hz.day >= 2) & (hz.day < 4)].events.sum(), hz[(hz.day >= 2) & (hz.day < 4)].catheter_days.sum()
    e2, T2 = hz[hz.day >= 4].events.sum(), hz[hz.day >= 4].catheter_days.sum()
    irr = (e2 / T2) / (e1 / T1) if e1 else np.nan
    se = np.sqrt(1 / e1 + 1 / e2) if e1 and e2 else np.nan
    out(f"  Gun >=4 vs gun 2-4 oran orani (IRR): {irr:.2f} "
        f"({np.exp(np.log(irr) - 1.96 * se):.2f}–{np.exp(np.log(irr) + 1.96 * se):.2f})")

    # ======================= 3) Cox + E-degerleri + duyarlilik
    m = d.copy()
    for g in SERVICES[1:]:
        m[g] = (m.svc == g).astype(int)
    names = {"age10": "Age (per 10 years)", "female": "Female sex", "diabetes": "Diabetes mellitus",
             "emergency": "Emergency admission", "vent24": "Invasive MV within 24 h",
             "braden_high": "Braden score ≤12", "periop": "Perioperative catheter insertion",
             "early_mob": "Out-of-bed by 48 h",
             "Cardiac surgery": "Cardiac surgery*", "Neurosurgery": "Neurosurgery*",
             "Trauma surgery": "Trauma surgery*", "Other surgery": "Other surgery*"}
    base = ["age10", "female", "diabetes", "emergency", "vent24", "braden_high", "periop"]
    use_mob = d.early_mob.notna().mean() > 0.8 and d.early_mob.nunique() == 2
    covs = base + (["early_mob"] if use_mob else []) + SERVICES[1:]

    def cox(df, cv, label):
        mc = df[["time_d", "cauti"] + cv].dropna()
        cv = [c for c in cv if mc[c].nunique() > 1]
        f = CoxPHFitter().fit(mc[["time_d", "cauti"] + cv], "time_d", "cauti")
        s = f.summary
        res = pd.DataFrame({"Variable": [names[c] for c in s.index],
                            "HR": s["exp(coef)"], "lo": s["exp(coef) lower 95%"], "hi": s["exp(coef) upper 95%"],
                            "p": s.p})
        out(f"\n{label} (n={sup(len(mc))}, olay={sup(mc.cauti.sum())}, C={f.concordance_index_:.3f})")
        return res, f, mc

    cres, cph, mc = cox(m, covs, "3) COX MODELI (KAUTI)")
    evs = [evalue(h, l, u) for h, l, u in zip(cres.HR, cres.lo, cres.hi)]
    cres["E-value (point / CI)"] = [f"{a:.2f} / {b:.2f}" for a, b in evs]
    t_cox = cres.assign(**{"HR (95% CI)": [f"{h:.2f} ({l:.2f}–{u:.2f})" for h, l, u in zip(cres.HR, cres.lo, cres.hi)],
                           "p ": [fmt_p(p) for p in cres.p]})[["Variable", "HR (95% CI)", "p ", "E-value (point / CI)"]]
    out(t_cox.to_string(index=False))
    sheets["Cox"] = t_cox
    try:
        from lifelines.statistics import proportional_hazard_test
        out(f"  PH testi en kucuk p: {proportional_hazard_test(cph, mc, time_transform='rank').summary.p.min():.3f}")
    except Exception:
        pass
    sres, _, _ = cox(m[m.pacu == 1], covs, "  Duyarlilik: yalnizca PACU'dan gelenler")
    t_sens = sres.assign(**{"HR (95% CI)": [f"{h:.2f} ({l:.2f}–{u:.2f})" for h, l, u in zip(sres.HR, sres.lo, sres.hi)],
                            "p ": [fmt_p(p) for p in sres.p]})[["Variable", "HR (95% CI)", "p "]]
    out(t_sens.to_string(index=False))
    sheets["Cox_PACU"] = t_sens

    # ======================= 4) Uzamis kateterizasyon (lojistik)
    cv4 = [c for c in covs if c != "periop"]
    ml = m[["prolonged"] + cv4].dropna()
    lg = sm.Logit(ml.prolonged, sm.add_constant(ml[cv4])).fit(disp=0)
    ci4 = np.exp(lg.conf_int())
    lres = pd.DataFrame({"Variable": [names[c] for c in cv4], "OR": np.exp(lg.params[cv4]).values,
                         "lo": ci4.loc[cv4, 0].values, "hi": ci4.loc[cv4, 1].values, "p": lg.pvalues[cv4].values})
    t_lg = lres.assign(**{"OR (95% CI)": [f"{o:.2f} ({l:.2f}–{u:.2f})" for o, l, u in zip(lres.OR, lres.lo, lres.hi)],
                          "p ": [fmt_p(p) for p in lres.p]})[["Variable", "OR (95% CI)", "p "]]
    out(f"\n4) KATETER >48 SAAT - LOJISTIK (n={sup(len(ml))}, McFadden R2={lg.prsquared:.3f})")
    try:
        from sklearn.metrics import roc_auc_score
        out(f"  AUC: {roc_auc_score(ml.prolonged, lg.predict(sm.add_constant(ml[cv4]))):.3f}")
    except Exception:
        pass
    out(t_lg.to_string(index=False))
    sheets["Logistic_prolonged"] = t_lg

    # ======================= 5) Hemsirelik sureci gostergeleri
    out("\n5) HEMSIRELIK SURECI GOSTERGELERI (Donabedian: surec)")
    removed = d[(d.raw_end < d.outtime - pd.Timedelta("2h")) &
                ~(d.deathtime.notna() & (d.deathtime <= d.raw_end + pd.Timedelta("48h")))].copy()
    out(f"  Kateteri yogun bakimda cekilen (canli): {npct(len(removed), N)}")
    hr_ = removed.raw_end.dt.hour
    night = ((hr_ >= 22) | (hr_ < 6))
    out(f"  Gece (22:00-06:00) cekilen: {npct(night.sum(), len(removed))}; "
        f"sabah 06:00-10:00: {npct(((hr_ >= 6) & (hr_ < 10)).sum(), len(removed))}")
    out(f"  Cekim sonrasi 24 saatte mesane taramasi: {npct(removed.stay_id.isin(bscan).sum(), len(removed))}")
    sc = straight_caths(root, d)
    rf = raw[raw.stay_id.isin(removed.stay_id)][["stay_id", "starttime"]].rename(columns={"starttime": "t"})
    sc = sc.rename(columns={"charttime": "t"})
    rc = pd.concat([rf, sc]).merge(removed[["stay_id", "raw_end", "outtime"]], on="stay_id")
    rc = rc[(rc.t >= rc.raw_end + pd.Timedelta("1h")) & (rc.t <= rc.raw_end + pd.Timedelta("48h")) &
            (rc.t <= rc.outtime)]
    removed["recath"] = removed.stay_id.isin(rc.stay_id).astype(int)
    removed["raw_dur"] = (removed.raw_end - removed.cath_start).dt.total_seconds() / 86400
    removed["early_rm"] = (removed.raw_dur <= 2).astype(int)
    out(f"  48 saat icinde yeniden kateterizasyon (Foley ya da aralikli): {npct(removed.recath.sum(), len(removed))}")
    rr = []
    for lab_, g in [("Removed ≤48 h", removed[removed.early_rm == 1]), ("Removed >48 h", removed[removed.early_rm == 0])]:
        rr.append({"Group": lab_, "Patients": sup(len(g)), "Re-catheterisation within 48 h, n (%)": npct(g.recath.sum(), len(g))})
    cvr = ["early_rm", "age10", "female", "emergency", "vent24"] + SERVICES[1:]
    for g in SERVICES[1:]:
        removed[g] = (removed.svc == g).astype(int)
    mr = removed[["recath"] + cvr].dropna()
    try:
        lr = sm.Logit(mr.recath, sm.add_constant(mr[cvr])).fit(disp=0)
        o, (l, u) = np.exp(lr.params["early_rm"]), np.exp(lr.conf_int().loc["early_rm"])
        rr.append({"Group": "Adjusted OR, ≤48 h vs >48 h", "Patients": "",
                   "Re-catheterisation within 48 h, n (%)": f"{o:.2f} ({l:.2f}–{u:.2f}); p={fmt_p(lr.pvalues['early_rm'])}"})
    except Exception as ex:
        out(f"  (yeniden kateterizasyon modeli kurulamadi: {ex})")
    t_rc = pd.DataFrame(rr)
    out(t_rc.to_string(index=False))
    sheets["Recatheterisation"] = t_rc

    # fazla kateter-gunu
    d["excess"] = np.clip(d.dur_d - 2, 0, None)
    rows = []
    for lab_, g in [("Total", d)] + [(s, d[d.svc == s]) for s in SERVICES]:
        rows.append({"Group": lab_, "Patients": sup(len(g)),
                     "Removed ≤48 h, n (%)": npct((g.dur_d <= 2).sum(), len(g)),
                     "Total catheter-days": f"{g.dur_d.sum():.0f}",
                     "Catheter-days beyond 48 h, n (%)": f"{g.excess.sum():.0f} ({100 * g.excess.sum() / g.dur_d.sum():.1f})",
                     "Early mobilisation, n (%)": npct((g.early_mob == 1).sum(), g.early_mob.notna().sum())})
    t_ex = pd.DataFrame(rows)
    out("\n  Fazla kateter-gunu (48 saati asan) ve surec gostergeleri:")
    out(t_ex.to_string(index=False))
    sheets["Process_indicators"] = t_ex

    # ======================= SEKILLER
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                         "axes.spines.top": False, "axes.spines.right": False})

    # ---- Sekil 1: A) KM vs AJ  B) gunluk hazard
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 3.3), dpi=300)
    t = kmf.survival_function_.index
    ci = kmf.confidence_interval_survival_function_
    a1.fill_between(t, 100 * (1 - ci.iloc[:, 0]), 100 * (1 - ci.iloc[:, 1]), step="post", color=BLUE, alpha=.15, lw=0)
    a1.step(t, 100 * (1 - kmf.survival_function_.iloc[:, 0]), where="post", color=BLUE, lw=1.6)
    cd = ajf.cumulative_density_
    a1.step(cd.index, 100 * cd.iloc[:, 0], where="post", color=ORANGE, lw=1.6)
    km14 = 100 * (1 - kmf.survival_function_at_times(14).iloc[0])
    aj14 = 100 * cd.loc[:14].iloc[-1, 0]
    ytop = max(8, np.ceil(km14 * 1.45))
    a1.set_xlim(0, 14); a1.set_ylim(0, ytop); a1.set_xticks(range(0, 15, 2))
    a1.set_xlabel("Catheter day"); a1.set_ylabel("Cumulative incidence of CAUTI (%)")
    km8 = 100 * (1 - kmf.survival_function_at_times(8).iloc[0])
    aj8 = 100 * cd.loc[:8].iloc[-1, 0]
    a1.text(4.4, ytop * .56, "Catheter retained\n(Kaplan–Meier)", color=INK,
            ha="left", va="bottom", fontsize=7)
    a1.text(8.3, aj8 + ytop * .03, "Observed, removal as competing\nevent (Aalen–Johansen)", color=INK,
            ha="left", va="bottom", fontsize=7)
    a1.grid(axis="y", color=GRID, lw=.6); a1.set_axisbelow(True)
    a1.set_title("A", loc="left", fontweight="bold", fontsize=10)
    for k in range(0, 15, 2):
        a1.annotate(sup((d.time_d >= k).sum()), (k, 0), xycoords=("data", "axes fraction"), xytext=(0, -34),
                    textcoords="offset points", ha="center", fontsize=6, color=MUTED)
    a1.annotate("At risk", (0, 0), xycoords=("data", "axes fraction"), xytext=(-14, -34),
                textcoords="offset points", ha="right", fontsize=6, color=MUTED)
    hzp = hz[(hz.day < 14) & (hz.at_risk >= 50)]
    gmax = hzp.day.max() + 1
    grid = grid[grid.mid <= gmax]
    a2.errorbar(hzp.day + .5, hzp.rate, yerr=[hzp.rate - hzp.lo, hzp.hi - hzp.rate], fmt="o", ms=3.5,
                color=BLUE, ecolor=BLUE, elinewidth=.8, capsize=0, zorder=3)
    a2.fill_between(grid.mid, grid.lo, grid.hi, color=ORANGE, alpha=.15, lw=0)
    a2.plot(grid.mid, grid.rate, color=ORANGE, lw=1.6)
    a2.axvline(2, color=MUTED, lw=.8, ls=(0, (3, 3)))
    ytop2 = max(hzp.hi.max(), grid.hi.max()) * 1.08
    a2.set_xlim(0, 14); a2.set_xticks(range(0, 15, 2)); a2.set_ylim(0, ytop2)
    a2.text(2.15, ytop2 * .97, "48 h", color=MUTED, fontsize=7, va="top")
    a2.set_xlabel("Catheter day"); a2.set_ylabel("CAUTI per 1,000 catheter-days")
    a2.grid(axis="y", color=GRID, lw=.6); a2.set_axisbelow(True)
    a2.set_title("B", loc="left", fontweight="bold", fontsize=10)
    fig.tight_layout(w_pad=2.5)
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure1.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    # ---- Sekil 2: orman grafigi (A Cox, B lojistik)
    def forest(ax, res, col, xlabel, title):
        res = res.iloc[::-1].reset_index(drop=True)
        y = np.arange(len(res))
        ax.axvline(1, color=MUTED, lw=.8)
        ax.hlines(y, res.lo, res.hi, color=BLUE, lw=1.4)
        ax.plot(res[col], y, "s", color=BLUE, ms=4.5)
        ax.set_yticks(y); ax.set_yticklabels(res.Variable, fontsize=7, color=INK)
        ax.set_xscale("log")
        from matplotlib.ticker import FixedLocator, NullLocator, FuncFormatter
        tk = [t for t in [0.1, 0.25, 0.5, 1, 2, 4, 8, 16] if res.lo.min() * .7 <= t <= res.hi.max() * 1.6]
        ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.set_xlabel(xlabel)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.grid(axis="x", color=GRID, lw=.6); ax.set_axisbelow(True)
        xmax = res.hi.max() * 1.6
        ax.set_xlim(min(res.lo.min() * .8, .5), xmax * 3.2)
        for yi, r in res.iterrows():
            ax.text(xmax * 1.05, yi, f"{r[col]:.2f} ({r.lo:.2f}–{r.hi:.2f})", va="center", fontsize=6.5, color=INK)
        ax.set_title(title, loc="left", fontweight="bold", fontsize=10)
    fig, (b1, b2) = plt.subplots(1, 2, figsize=(8.6, 3.9), dpi=300)
    forest(b1, cres, "HR", "Hazard ratio (log scale)", "A  CAUTI")
    forest(b2, lres, "OR", "Odds ratio (log scale)", "B  Catheter >48 h")
    fig.text(0.01, -0.01, "*Reference: general surgery. MV: mechanical ventilation.", fontsize=6.5, color=MUTED)
    fig.tight_layout(w_pad=3.5)
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure2.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    # ---- Sekil 3: servis bazinda kateter suresi dagilimi (100% yigilmis cubuk)
    order = ["Total"] + SERVICES
    fig, ax = plt.subplots(figsize=(7.0, 2.9), dpi=300)
    cats = list(d.dur_cat.cat.categories)
    for i, grp in enumerate(order):
        g = d if grp == "Total" else d[d.svc == grp]
        share = g.dur_cat.value_counts(normalize=True).reindex(cats).fillna(0).values * 100
        left = 0
        for j, v in enumerate(share):
            ax.barh(i, v, left=left, color=RAMP[j], edgecolor="white", linewidth=1.2, height=.62,
                    label=cats[j] if i == 0 else None)
            if v >= 7:
                ax.text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=6.5,
                        color="white" if j >= 1 else INK)
            left += v
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{o} (n={sup(len(d) if o == 'Total' else (d.svc == o).sum())})" for o in order],
                       fontsize=7, color=INK)
    ax.invert_yaxis(); ax.set_xlim(0, 100); ax.set_xlabel("Patients (%)")
    ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(title="Catheter duration", ncol=4, fontsize=7, title_fontsize=7, frameon=False,
              loc="lower left", bbox_to_anchor=(0, 1.0))
    fig.tight_layout()
    for ext in ("png", "tiff"):
        fig.savefig(odir / f"Figure3.{ext}", bbox_inches="tight", dpi=600)
    plt.close(fig)

    with pd.ExcelWriter(odir / "kateter_tablolar_v2.xlsx") as xw:
        for k, v in sheets.items():
            v.to_excel(xw, k[:31], index=False)
    out(f"\nToplam sure: {(time.time() - t0) / 60:.1f} dk")
    (odir / "kateter_ozet_v2.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nSonuclar: {odir}")


if __name__ == "__main__":
    main()
