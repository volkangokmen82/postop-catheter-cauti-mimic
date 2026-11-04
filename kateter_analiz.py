# -*- coding: utf-8 -*-
"""
MIMIC-IV v3.1 - Ameliyat sonrasi yogun bakimda uriner kateter kalis suresi ve
kateterle iliskili uriner sistem enfeksiyonu (KAUTI) - ANALIZ BETIGI

Yalnizca TOPLU sonuclar uretir (tablolar, model katsayilari, grafik).
10'dan kucuk hucreler "<10" olarak gizlenir. Hasta duzeyinde veri yazilmaz.

Kurulum (bir kez):
    pip install pandas numpy scipy statsmodels lifelines matplotlib openpyxl
Calistirma:
    python "D:\\MIMICIV\\kateter_analiz.py" --root D:\\MIMICIV
Ciktilar (betigin klasorunde, "kateter_sonuclar" alt klasoru):
    kateter_tablolar.xlsx, sekil1_kumulatif_insidans.png, kateter_ozet.txt
"""
import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

SURG = {"SURG", "CSURG", "NSURG", "ORTHO", "PSURG", "TSURG", "VSURG", "TRAUM", "GU", "ENT"}
SVC_GRP = {"SURG": "Genel cerrahi", "CSURG": "Kalp cerrahisi", "NSURG": "Norosirurji",
           "TRAUM": "Travma cerrahisi", "VSURG": "Diger cerrahi", "TSURG": "Diger cerrahi",
           "ORTHO": "Diger cerrahi", "ENT": "Diger cerrahi", "GU": "Diger cerrahi",
           "PSURG": "Diger cerrahi"}
FOLEY = 229351
VENT_INV = 225792
BRADEN = [224054, 224055, 224056, 224057, 224058, 224059]
NON_PATH = r"yeast|candida|cancelled|no growth|mixed|flora|contamina"

LOG = []


def out(s=""):
    print(s, flush=True)
    LOG.append(str(s))


def sup(n):
    n = int(n)
    return "<10" if 0 < n < 10 else str(n)


def find(root, name):
    hits = list(Path(root).rglob(name))
    if not hits:
        sys.exit(f"Bulunamadi: {name}")
    return hits[0]


def npct(k, n):
    return f"{sup(k)} ({100 * k / n:.1f})" if n else "-"


def mean_sd(x):
    return f"{x.mean():.1f} ± {x.std():.1f}"


def med_iqr(x):
    return f"{x.median():.1f} ({x.quantile(.25):.1f}–{x.quantile(.75):.1f})"


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


# ---------------------------------------------------------------- kohort
def load_cohort(root):
    icu = pd.read_csv(find(root, "icustays.csv.gz"), parse_dates=["intime", "outtime"])
    pat = pd.read_csv(find(root, "patients.csv.gz"), usecols=["subject_id", "gender", "anchor_age"])
    adm = pd.read_csv(find(root, "admissions.csv.gz"),
                      usecols=["hadm_id", "admission_type", "deathtime", "hospital_expire_flag"],
                      parse_dates=["deathtime"])
    svc = pd.read_csv(find(root, "services.csv.gz"), parse_dates=["transfertime"])
    icu = icu.merge(pat, on="subject_id", how="left").merge(adm, on="hadm_id", how="left")
    icu["key"] = icu.intime + pd.Timedelta("1h")
    svc = svc.dropna(subset=["hadm_id"])
    svc["hadm_id"] = svc.hadm_id.astype("int64")
    icu["hadm_id"] = icu.hadm_id.astype("int64")
    icu = pd.merge_asof(icu.sort_values("key"),
                        svc[["hadm_id", "transfertime", "curr_service"]].sort_values("transfertime"),
                        left_on="key", right_on="transfertime", by="hadm_id", direction="backward")
    icu = icu.sort_values("intime")
    icu["first_stay"] = icu.groupby("subject_id").cumcount() == 0
    flow = {"Tum yogun bakim yatislari": len(icu)}
    icu = icu[icu.anchor_age >= 18]; flow["Eriskin"] = len(icu)
    icu = icu[icu.curr_service.isin(SURG)]; flow["Yogun bakima giriste cerrahi servis"] = len(icu)
    icu = icu[icu.first_stay]; flow["Ilk yogun bakim yatisi"] = len(icu)
    icu = icu[icu.los >= 1]; flow["Yogun bakimda >=24 saat"] = len(icu)
    return icu, flow


def catheter_episodes(root, coh):
    pe = pd.read_csv(find(root, "procedureevents.csv.gz"),
                     usecols=["stay_id", "starttime", "endtime", "itemid"],
                     parse_dates=["starttime", "endtime"])
    pe = pe[pe.stay_id.isin(coh.stay_id)]
    vent = pe[pe.itemid == VENT_INV].merge(coh[["stay_id", "intime"]], on="stay_id")
    vent24 = set(vent[vent.starttime <= vent.intime + pd.Timedelta("24h")].stay_id)
    fo = pe[pe.itemid == FOLEY].dropna(subset=["starttime", "endtime"])
    fo = fo[fo.endtime > fo.starttime].sort_values(["stay_id", "starttime"])
    # 24 saatten kisa aralikla yeniden takilan kateterler tek epizot sayilir (NHSN yaklasimi)
    rows = []
    for sid, g in fo.groupby("stay_id"):
        s, e = g.iloc[0].starttime, g.iloc[0].endtime
        for _, r in g.iloc[1:].iterrows():
            if r.starttime - e <= pd.Timedelta("24h"):
                e = max(e, r.endtime)
            else:
                break   # yalnizca ilk (indeks) epizot
        rows.append((sid, s, e))
    ep = pd.DataFrame(rows, columns=["stay_id", "cath_start", "cath_end"])
    return ep, vent24


def braden_first24(root, coh):
    stays = set(coh.stay_id)
    parts = []
    t0 = time.time()
    for i, ch in enumerate(pd.read_csv(find(root, "chartevents.csv.gz"),
                                       usecols=["stay_id", "charttime", "itemid", "valuenum"],
                                       chunksize=5_000_000), 1):
        ch = ch[ch.itemid.isin(BRADEN) & ch.stay_id.isin(stays)]
        if len(ch):
            parts.append(ch)
        print(f"   chartevents parca {i} ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    br = pd.concat(parts)
    br["stay_id"] = br.stay_id.astype("int64")
    br["charttime"] = pd.to_datetime(br.charttime)
    br = br.merge(coh[["stay_id", "intime"]], on="stay_id")
    br = br[(br.charttime >= br.intime) & (br.charttime <= br.intime + pd.Timedelta("24h"))]
    w = br.pivot_table(index=["stay_id", "charttime"], columns="itemid", values="valuenum", aggfunc="first")
    w = w.dropna()
    w["total"] = w.sum(axis=1)
    return w.reset_index().sort_values("charttime").groupby("stay_id").total.first()


def urine_cultures(root, subjects):
    mb = pd.read_csv(find(root, "microbiologyevents.csv.gz"),
                     usecols=["subject_id", "micro_specimen_id", "chartdate", "charttime",
                              "spec_type_desc", "org_name"])
    mb = mb[mb.subject_id.isin(subjects) &
            mb.spec_type_desc.fillna("").str.contains("URINE", case=False)]
    mb["t"] = pd.to_datetime(mb.charttime).fillna(pd.to_datetime(mb.chartdate))
    pos = mb[mb.org_name.notna() & ~mb.org_name.fillna("").str.contains(NON_PATH, case=False)]
    pos = pos.drop_duplicates(["micro_specimen_id", "org_name"])
    return pos[["subject_id", "t", "org_name"]]


# ---------------------------------------------------------------- ana akis
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root = a.root
    odir = Path(__file__).with_name("kateter_sonuclar")
    odir.mkdir(exist_ok=True)
    t0 = time.time()

    out("ABSTRACT/YONTEM ICIN OZET - KATETER CALISMASI")
    out("=" * 64)
    coh, flow = load_cohort(root)
    ep, vent24 = catheter_episodes(root, coh)
    d = coh.merge(ep, on="stay_id")
    flow["Foley kateteri olan"] = len(d)

    # olum zamani ile sansurleme
    d["cath_end"] = d[["cath_end", "deathtime"]].min(axis=1)
    d = d[d.cath_end > d.cath_start]
    d["dur_d"] = (d.cath_end - d.cath_start).dt.total_seconds() / 86400

    pos = urine_cultures(root, set(d.subject_id))
    x = d[["stay_id", "subject_id", "cath_start", "cath_end"]].merge(pos, on="subject_id")
    # onceden var olan enfeksiyon: takilmadan 48 saat once ile takildiktan sonraki 48 saat arasi pozitif kultur
    pre = x[(x.t >= x.cath_start - pd.Timedelta("48h")) & (x.t < x.cath_start + pd.Timedelta("48h"))]
    d = d[~d.stay_id.isin(pre.stay_id)]
    flow["Kateter oncesi/ilk 48 saatte pozitif idrar kulturu olanlar cikarildiktan sonra"] = len(d)

    ev = x[x.stay_id.isin(d.stay_id) & (x.t >= x.cath_start + pd.Timedelta("48h")) &
           (x.t <= x.cath_end + pd.Timedelta("24h"))]
    first_ev = ev.sort_values("t").groupby("stay_id").first()
    d["cauti"] = d.stay_id.isin(first_ev.index).astype(int)
    d = d.merge(first_ev[["t", "org_name"]], left_on="stay_id", right_index=True, how="left")
    # risk altindaki sure (gun): takilmadan olay ya da cekilme+24 saate kadar
    stop = d.cath_end + pd.Timedelta("24h")
    d["time_d"] = np.where(d.cauti == 1, (d.t - d.cath_start).dt.total_seconds() / 86400,
                           (stop - d.cath_start).dt.total_seconds() / 86400)

    # kovaryatlar
    dx = pd.read_csv(find(root, "diagnoses_icd.csv.gz"), dtype={"icd_code": "string"})
    dm = dx[((dx.icd_version == 10) & dx.icd_code.str.match(r"E1[0-4]")) |
            ((dx.icd_version == 9) & dx.icd_code.str.startswith("250"))].hadm_id
    d["diabetes"] = d.hadm_id.isin(dm).astype(int)
    d["female"] = (d.gender == "F").astype(int)
    d["age10"] = d.anchor_age / 10
    d["vent24"] = d.stay_id.isin(vent24).astype(int)
    d["emergency"] = (~d.admission_type.fillna("").str.contains("ELECTIVE|SAME DAY", case=False)).astype(int)
    d["svc"] = d.curr_service.map(SVC_GRP)
    d["periop"] = ((d.cath_start - d.intime) <= pd.Timedelta("6h")).astype(int)
    d["prolonged"] = (d.dur_d > 2).astype(int)
    br = braden_first24(root, d)
    d["braden"] = d.stay_id.map(br)
    d["braden_high"] = (d.braden <= 12).astype(float).where(d.braden.notna())
    d["dur_cat"] = pd.cut(d.dur_d, [0, 2, 4, 7, 1e9], labels=["≤2 gün", ">2–4 gün", ">4–7 gün", ">7 gün"])
    N = len(d)

    out("\nAkis semasi (STROBE):")
    for k, v in flow.items():
        out(f"  {k}: {sup(v)}")
    out(f"  Analiz kohortu: {sup(N)}")

    # ------------------------------------------------ Tablo 1
    def col(sub):
        n = len(sub)
        return {
            "n": sup(n),
            "Yas, yil (ort ± SS)": mean_sd(sub.anchor_age),
            "Kadin, n (%)": npct(sub.female.sum(), n),
            **{f"{g}, n (%)": npct((sub.svc == g).sum(), n) for g in
               ["Genel cerrahi", "Kalp cerrahisi", "Norosirurji", "Travma cerrahisi", "Diger cerrahi"]},
            "Acil yatis, n (%)": npct(sub.emergency.sum(), n),
            "Diyabet, n (%)": npct(sub.diabetes.sum(), n),
            "Ilk 24 saatte invaziv mekanik ventilasyon, n (%)": npct(sub.vent24.sum(), n),
            "Braden puani, medyan (IQR)": med_iqr(sub.braden.dropna()),
            "Braden ≤12, n (%)": npct((sub.braden <= 12).sum(), sub.braden.notna().sum()),
            "Kateter perioperatif takilmis (≤6 saat), n (%)": npct(sub.periop.sum(), n),
            "Kateter suresi, gun, medyan (IQR)": med_iqr(sub.dur_d),
            "Kateter >48 saat, n (%)": npct(sub.prolonged.sum(), n),
            "Yogun bakim yatis suresi, gun, medyan (IQR)": med_iqr(sub.los),
            "Hastane ici olum, n (%)": npct(sub.hospital_expire_flag.sum(), n),
        }
    c0, c1 = d[d.cauti == 0], d[d.cauti == 1]
    t1 = pd.DataFrame({"Toplam": col(d), "KAUTI yok": col(c0), "KAUTI var": col(c1)})
    pv = {}
    pv["Yas, yil (ort ± SS)"] = stats.ttest_ind(c0.anchor_age, c1.anchor_age, equal_var=False).pvalue
    for nm, v in [("Kadin, n (%)", "female"), ("Acil yatis, n (%)", "emergency"),
                  ("Diyabet, n (%)", "diabetes"), ("Ilk 24 saatte invaziv mekanik ventilasyon, n (%)", "vent24"),
                  ("Kateter perioperatif takilmis (≤6 saat), n (%)", "periop"),
                  ("Kateter >48 saat, n (%)", "prolonged"), ("Hastane ici olum, n (%)", "hospital_expire_flag")]:
        pv[nm] = stats.chi2_contingency(pd.crosstab(d[v], d.cauti))[1]
    pv["Genel cerrahi, n (%)"] = stats.chi2_contingency(pd.crosstab(d.svc, d.cauti))[1]
    for nm, v in [("Braden puani, medyan (IQR)", "braden"), ("Kateter suresi, gun, medyan (IQR)", "dur_d"),
                  ("Yogun bakim yatis suresi, gun, medyan (IQR)", "los")]:
        pv[nm] = stats.mannwhitneyu(c0[v].dropna(), c1[v].dropna()).pvalue
    t1["p"] = pd.Series({k: fmt_p(v) for k, v in pv.items()})
    t1["p"] = t1["p"].fillna("")

    # ------------------------------------------------ Tablo 2: hizlar
    def rate_row(sub, lab):
        k, cd = sub.cauti.sum(), sub.time_d.sum()
        lo, hi = stats.chi2.ppf(.025, 2 * k) / 2 if k else 0, stats.chi2.ppf(.975, 2 * (k + 1)) / 2
        return {"Grup": lab, "Hasta, n": sup(len(sub)), "KAUTI, n (%)": npct(k, len(sub)),
                "Kateter-gun": f"{cd:.0f}",
                "1000 kateter-gunune KAUTI (%95 GA)": f"{1000 * k / cd:.2f} ({1000 * lo / cd:.2f}–{1000 * hi / cd:.2f})"}
    rows = [rate_row(d, "Toplam")]
    rows += [rate_row(d[d.svc == g], g) for g in
             ["Genel cerrahi", "Kalp cerrahisi", "Norosirurji", "Travma cerrahisi", "Diger cerrahi"]]
    rows += [rate_row(d[d.dur_cat == g], f"Kateter suresi {g}") for g in d.dur_cat.cat.categories]
    t2 = pd.DataFrame(rows)

    # ------------------------------------------------ Kaplan-Meier (kumulatif insidans)
    from lifelines import KaplanMeierFitter, CoxPHFitter
    kmf = KaplanMeierFitter().fit(d.time_d, d.cauti)
    km_rows = []
    for day in [3, 5, 7, 10, 14]:
        s = kmf.survival_function_at_times(day).iloc[0]
        ci = kmf.confidence_interval_survival_function_
        idx = ci.index[ci.index <= day].max()
        km_rows.append({"Kateter gunu": day, "Kumulatif KAUTI, %": f"{100 * (1 - s):.1f}",
                        "%95 GA": f"{100 * (1 - ci.loc[idx].iloc[1]):.1f}–{100 * (1 - ci.loc[idx].iloc[0]):.1f}",
                        "Risk altinda kalan hasta": sup((d.time_d >= day).sum())})
    tkm = pd.DataFrame(km_rows)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    t = kmf.survival_function_.index
    ax.step(t, 100 * (1 - kmf.survival_function_.iloc[:, 0]), where="post", color="#1f4e79", lw=1.8)
    ci = kmf.confidence_interval_survival_function_
    ax.fill_between(t, 100 * (1 - ci.iloc[:, 0]), 100 * (1 - ci.iloc[:, 1]), step="post",
                    color="#1f4e79", alpha=.15, lw=0)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 15)
    ax.set_xlabel("Catheter day")
    ax.set_ylabel("Cumulative incidence of CAUTI (%)")
    ax.spines[["top", "right"]].set_visible(False)
    ticks = [0, 2, 4, 6, 8, 10, 12, 14]
    ax.set_xticks(ticks)
    risk = [sup((d.time_d >= k).sum()) for k in ticks]
    tr = ("data", "axes fraction")
    for k, r in zip(ticks, risk):
        ax.annotate(r, (k, 0), xycoords=tr, xytext=(0, -52), textcoords="offset points",
                    ha="center", fontsize=7)
    ax.annotate("Number at risk", (0, 0), xycoords=tr, xytext=(-30, -52), textcoords="offset points",
                ha="right", fontsize=7)
    fig.tight_layout()
    fig.savefig(odir / "sekil1_kumulatif_insidans.png", bbox_inches="tight")
    fig.savefig(odir / "sekil1_kumulatif_insidans.tiff", bbox_inches="tight")

    # ------------------------------------------------ Tablo 3: Cox (KAUTI risk etmenleri)
    m = d.copy()
    for g in ["Kalp cerrahisi", "Norosirurji", "Travma cerrahisi", "Diger cerrahi"]:
        m[g] = (m.svc == g).astype(int)
    covs = ["age10", "female", "diabetes", "emergency", "vent24", "braden_high", "periop",
            "Kalp cerrahisi", "Norosirurji", "Travma cerrahisi", "Diger cerrahi"]
    names = {"age10": "Yas (10 yillik artis)", "female": "Kadin cinsiyet", "diabetes": "Diyabet",
             "emergency": "Acil yatis", "vent24": "Ilk 24 saatte invaziv MV",
             "braden_high": "Braden ≤12 (yuksek risk)", "periop": "Perioperatif takilan kateter",
             "Kalp cerrahisi": "Kalp cerrahisi (ref: genel)", "Norosirurji": "Norosirurji (ref: genel)",
             "Travma cerrahisi": "Travma cerrahisi (ref: genel)", "Diger cerrahi": "Diger cerrahi (ref: genel)"}
    mc = m[["time_d", "cauti"] + covs].dropna()
    dropped = [c for c in covs if mc[c].nunique() < 2]
    if dropped:
        out(f"  Uyari: degiskenligi olmayan degisken(ler) modelden cikarildi: {dropped}")
    covs = [c for c in covs if c not in dropped]
    mc = mc[["time_d", "cauti"] + covs]
    cph = CoxPHFitter().fit(mc, "time_d", "cauti")
    s = cph.summary
    t3 = pd.DataFrame({"Degisken": [names[c] for c in s.index],
                       "HR (%95 GA)": [f"{r['exp(coef)']:.2f} ({r['exp(coef) lower 95%']:.2f}–{r['exp(coef) upper 95%']:.2f})"
                                       for _, r in s.iterrows()],
                       "p": [fmt_p(p) for p in s.p]})
    ph = None
    try:
        from lifelines.statistics import proportional_hazard_test
        ph = proportional_hazard_test(cph, mc, time_transform="rank")
        ph_global = ph.summary.p.min()
    except Exception:
        ph_global = np.nan

    # ------------------------------------------------ Tablo 4: uzamis kateterizasyon (>48 saat) belirleyicileri
    import statsmodels.api as sm
    covs4 = [c for c in covs if c != "periop"]
    ml = m[["prolonged"] + covs4].dropna()
    lg = sm.Logit(ml.prolonged, sm.add_constant(ml[covs4])).fit(disp=0)
    ci4 = np.exp(lg.conf_int())
    t4 = pd.DataFrame({"Degisken": [names[c] for c in covs4],
                       "OR (%95 GA)": [f"{np.exp(lg.params[c]):.2f} ({ci4.loc[c, 0]:.2f}–{ci4.loc[c, 1]:.2f})" for c in covs4],
                       "p": [fmt_p(lg.pvalues[c]) for c in covs4]})

    # ------------------------------------------------ Etkenler
    org = d[d.cauti == 1].org_name.str.upper().value_counts()
    t5 = pd.DataFrame({"Etken": org.index, "n (%)": [npct(v, d.cauti.sum()) for v in org.values]}).head(10)

    # ------------------------------------------------ duyarlilik: ICD T83.51 / N39.0
    cicd = dx[((dx.icd_version == 10) & (dx.icd_code.str.startswith("T8351") | (dx.icd_code == "N390"))) |
              ((dx.icd_version == 9) & dx.icd_code.isin(["99664", "5990"]))].hadm_id
    d["icd_uti"] = d.hadm_id.isin(cicd).astype(int)

    # ------------------------------------------------ yazdir & kaydet
    out("\nTABLO 1 - Hasta ozellikleri")
    out(t1.to_string())
    out("\nTABLO 2 - KAUTI sikligi ve hizi")
    out(t2.to_string(index=False))
    out("\nKumulatif insidans (Kaplan-Meier)")
    out(tkm.to_string(index=False))
    out(f"\nTABLO 3 - Cox modeli (n={sup(len(mc))}, olay={sup(mc.cauti.sum())})")
    out(t3.to_string(index=False))
    out(f"  Orantili hazard varsayimi testi, en kucuk p: {ph_global:.3f}" if not np.isnan(ph_global) else "")
    out(f"  Concordance (C-indeksi): {cph.concordance_index_:.3f}")
    out(f"\nTABLO 4 - Kateterin 48 saatten uzun kalmasinin belirleyicileri (lojistik, n={sup(len(ml))})")
    out(t4.to_string(index=False))
    out(f"  Pseudo R2 (McFadden): {lg.prsquared:.3f}")
    out("\nTABLO 5 - En sik etkenler")
    out(t5.to_string(index=False))
    out("\nDuyarlilik: ICD idrar yolu enfeksiyonu/KAUTI kodu olan hasta: "
        f"{npct(d.icd_uti.sum(), N)}; kultur tanimiyla uyum (ikisi de var): "
        f"{sup(((d.icd_uti == 1) & (d.cauti == 1)).sum())}")
    out(f"Eksik Braden: {sup(d.braden.isna().sum())}")
    out(f"\nToplam sure: {(time.time() - t0) / 60:.1f} dk")

    with pd.ExcelWriter(odir / "kateter_tablolar.xlsx") as xw:
        pd.DataFrame(list(flow.items()) + [("Analiz kohortu", N)],
                     columns=["Adim", "n"]).to_excel(xw, "Akis", index=False)
        t1.to_excel(xw, "Tablo1")
        t2.to_excel(xw, "Tablo2_hiz", index=False)
        tkm.to_excel(xw, "KM", index=False)
        t3.to_excel(xw, "Tablo3_Cox", index=False)
        t4.to_excel(xw, "Tablo4_Lojistik", index=False)
        t5.to_excel(xw, "Tablo5_Etken", index=False)
    (odir / "kateter_ozet.txt").write_text("\n".join(LOG), encoding="utf-8")
    print(f"\nSonuclar: {odir}")


if __name__ == "__main__":
    main()
