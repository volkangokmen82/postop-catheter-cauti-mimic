# -*- coding: utf-8 -*-
"""
MIMIC-IV v3.1 - FIZIBILITE 2
 A) Basinc yarasi tanimini netlestirme (SCIE calismasi icin)
 B) Ameliyat sonrasi uriner kateter kalis suresi ve KAUTI (Nuh'un Gemisi calismasi icin)

- Yalnizca TOPLU sayilar; 10'dan kucuk sayilar "<10".
- Cikti: betigin bulundugu klasorde  mimic_fizibilite2_ozet.txt
Kullanim:
    python "D:\\MIMICIV\\mimic_fizibilite2.py" --root D:\\MIMICIV
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

SURG = {"SURG", "CSURG", "NSURG", "ORTHO", "PSURG", "TSURG", "VSURG", "TRAUM", "GU", "ENT"}
LINES = []


def out(s=""):
    print(s, flush=True)
    LINES.append(str(s))


def sup(n):
    n = int(n)
    return "<10" if 0 < n < 10 else str(n)


def find(root, name):
    hits = list(Path(root).rglob(name))
    if not hits:
        sys.exit(f"Bulunamadi: {name}")
    return hits[0]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def build_cohort(root):
    icu = pd.read_csv(find(root, "icustays.csv.gz"), parse_dates=["intime", "outtime"])
    pat = pd.read_csv(find(root, "patients.csv.gz"), usecols=["subject_id", "anchor_age"])
    svc = pd.read_csv(find(root, "services.csv.gz"), parse_dates=["transfertime"])
    icu = icu.merge(pat, on="subject_id", how="left")
    icu["key"] = icu["intime"] + pd.Timedelta("1h")
    svc = svc.dropna(subset=["hadm_id"])
    svc["hadm_id"] = svc["hadm_id"].astype("int64")
    icu["hadm_id"] = icu["hadm_id"].astype("int64")
    icu = pd.merge_asof(icu.sort_values("key"),
                        svc[["hadm_id", "transfertime", "curr_service"]].sort_values("transfertime"),
                        left_on="key", right_on="transfertime", by="hadm_id", direction="backward")
    icu = icu.sort_values("intime")
    icu["first_stay"] = icu.groupby("subject_id").cumcount() == 0
    c = icu[(icu.anchor_age >= 18) & icu.curr_service.isin(SURG) & icu.first_stay & (icu.los >= 1)]
    return c[["subject_id", "hadm_id", "stay_id", "intime", "outtime", "los", "curr_service"]].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    a = ap.parse_args()
    root, t0 = a.root, time.time()
    out("MIMIC-IV FIZIBILITE 2 OZETI")
    out("=" * 64)
    coh = build_cohort(root)
    N = len(coh)
    out(f"Ana kohort (cerrahi servis, eriskin, ilk yatis, >=24 saat): {sup(N)}")

    di = pd.read_csv(find(root, "d_items.csv.gz"))

    # ===================== A) BASINC YARASI =====================
    out("\nA) BASINC YARASI - NETLESTIRILMIS TANIM")
    ce_items = di[di.linksto == "chartevents"]
    lab = ce_items.label.fillna("")
    type_ids = set(ce_items[lab.str.match(r"Impaired Skin Type #\d+|Impaired Skin #\d+- Type", case=False)].itemid)
    putype_ids = set(ce_items[lab.str.match(r"Pressure Ulcer #\d+- Type", case=False)].itemid)
    stage_ids = set(ce_items[lab.str.match(r"Pressure Ulcer Stage #\d+|Pressure ulcer #\d+- Stage", case=False)].itemid)
    present_id = set(ce_items[lab.str.fullmatch("Pressure Ulcer Present", case=False)].itemid)
    turn_ids = set(ce_items[lab.str.fullmatch("Turn", case=False)].itemid)
    all_ids = type_ids | putype_ids | stage_ids | present_id | turn_ids

    stays = set(coh.stay_id)
    parts = []
    for i, ch in enumerate(pd.read_csv(find(root, "chartevents.csv.gz"),
                                       usecols=["stay_id", "charttime", "itemid", "value"],
                                       chunksize=5_000_000, dtype={"value": "string"}), 1):
        ch = ch[ch.itemid.isin(all_ids) & ch.stay_id.isin(stays)]
        if len(ch):
            parts.append(ch)
        print(f"   parca {i} ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    ce = pd.concat(parts, ignore_index=True)
    ce["stay_id"] = ce.stay_id.astype("int64")
    ce["charttime"] = pd.to_datetime(ce.charttime)
    ce = ce.merge(coh[["stay_id", "intime", "outtime"]], on="stay_id")
    ce = ce[(ce.charttime >= ce.intime) & (ce.charttime <= ce.outtime)]
    v = ce.value.fillna("").str.strip()

    real_stage = re.compile(r"stage|deep tissue|unstageable|unable to stage|red; unbroken|"
                            r"partial thickness|full thickness", re.I)
    pi_row = ((ce.itemid.isin(type_ids) & v.str.contains("pressure", case=False)) |
              ce.itemid.isin(putype_ids) |
              (ce.itemid.isin(stage_ids) & v.str.contains(real_stage) &
               ~v.str.contains("not applicable|unable to assess|unknown", case=False)) |
              (ce.itemid.isin(present_id) & (v == "1")))
    pi = ce[pi_row]
    first = pi.groupby("stay_id").agg(t=("charttime", "min"), intime=("intime", "first"))
    h = (first.t - first.intime).dt.total_seconds() / 3600
    out(f"  Basinc yarasi kaydi olan yatis: {sup(len(first))} ({pct(len(first), N)})")
    out(f"    ilk 24 saatte (mevcut kabul): {sup((h <= 24).sum())}")
    out(f"    24-48 saat arasi: {sup(((h > 24) & (h <= 48)).sum())}")
    out(f"    48 saatten sonra ilk kez (YENI - yogun bakim kaynakli): {sup((h > 48).sum())} "
        f"({pct((h > 48).sum(), N)})")
    out(f"    72 saatten sonra ilk kez: {sup((h > 72).sum())}")

    # en ileri evre dagilimi (yeni yaralarda)
    new_ids = set(first[h > 48].index)
    st = ce[ce.itemid.isin(stage_ids) & ce.stay_id.isin(new_ids)].copy()
    sv = st.value.fillna("")
    grade = pd.Series("Diger/Belirsiz", index=st.index)
    grade[sv.str.contains("Stage I$|Red; unbroken", case=False, regex=True)] = "Evre 1"
    grade[sv.str.contains("Stage II$|Partial thickness", case=False, regex=True)] = "Evre 2"
    grade[sv.str.contains("Stage III|fascia", case=False, regex=True)] = "Evre 3"
    grade[sv.str.contains("Stage IV|muscle, bone", case=False, regex=True)] = "Evre 4"
    grade[sv.str.contains("deep tissue", case=False)] = "Derin doku hasari"
    grade[sv.str.contains("unstageable|unable to stage", case=False)] = "Evrelendirilemeyen"
    st["grade"] = grade
    out("  Yeni yaralarda kaydedilen evreler (yatis sayisi, bir yatis birden fazla evrede olabilir):")
    for k, n in st.groupby("grade").stay_id.nunique().sort_values(ascending=False).items():
        out(f"    {k}: {sup(n)}")

    # maruziyet: ilk 48 saatte pozisyon sikligi -> sonra yeni yara (ters nedensellik onlemi)
    tr = ce[ce.itemid.isin(turn_ids)]
    tr48 = tr[(tr.charttime - tr.intime) <= pd.Timedelta("48h")]
    rate = tr48.groupby("stay_id").size() / 2.0   # gunde
    land = coh[coh.los >= 2].set_index("stay_id")
    # 48. saatte zaten yara olanlari dislama
    prior = set(first[h <= 48].index)
    land = land[~land.index.isin(prior)]
    land["rate"] = rate.reindex(land.index).fillna(0)
    land["new_pi"] = land.index.isin(new_ids)
    land["grp"] = pd.cut(land.rate, [-0.1, 6, 12, 1e9], right=False,
                         labels=[">4 saatte bir (<6/gun)", "2-4 saatte bir (6-11/gun)",
                                 "<=2 saatte bir (>=12/gun)"])
    out("\n  Landmark analizi (yogun bakimda >=48 saat, ilk 48 saatte yarasi olmayan):")
    out(f"    n = {sup(len(land))}, yeni yara = {sup(land.new_pi.sum())} ({pct(land.new_pi.sum(), len(land))})")
    for g, d in land.groupby("grp", observed=False):
        out(f"    {g}: n={sup(len(d))}, yeni yara={sup(d.new_pi.sum())} ({pct(d.new_pi.sum(), len(d))})")

    # ===================== B) URINER KATETER =====================
    out("\nB) URINER KATETER VE KAUTI")
    pe_items = di[di.linksto == "procedureevents"]
    cath = pe_items[pe_items.label.fillna("").str.contains(r"foley|urinary|indwelling|suprapubic|"
                                                           r"straight cath", case=False)]
    out("  procedureevents kateter kalemleri:")
    for _, r in cath.iterrows():
        out(f"    {r.itemid} | {r.label}")
    pe = pd.read_csv(find(root, "procedureevents.csv.gz"),
                     usecols=["stay_id", "starttime", "endtime", "itemid"],
                     parse_dates=["starttime", "endtime"])
    pe = pe[pe.itemid.isin(cath.itemid) & pe.stay_id.isin(stays)]
    out("  Kalem bazinda kohortta yatis sayisi:")
    for iid, n in pe.groupby("itemid").stay_id.nunique().items():
        out(f"    {iid}: {sup(n)}")

    foley_ids = set(cath[cath.label.str.contains("foley|indwelling", case=False)].itemid)
    fo = pe[pe.itemid.isin(foley_ids)].merge(coh[["stay_id", "subject_id", "hadm_id", "intime"]], on="stay_id")
    fo["dur_h"] = (fo.endtime - fo.starttime).dt.total_seconds() / 3600
    per = fo.groupby("stay_id").agg(start=("starttime", "min"), end=("endtime", "max"),
                                    dur_h=("dur_h", "sum"), subject_id=("subject_id", "first"),
                                    intime=("intime", "first"))
    out(f"  Foley/kalici kateteri olan yatis: {sup(len(per))} ({pct(len(per), N)})")
    if len(per):
        d = per.dur_h / 24
        out(f"    Kateter suresi medyan (IQR), gun: {d.median():.1f} ({d.quantile(.25):.1f}-{d.quantile(.75):.1f})")
        for lo, hi, lbl in [(0, 2, "<=48 saat"), (2, 4, "2-4 gun"), (4, 7, "4-7 gun"), (7, 1e9, ">7 gun")]:
            out(f"    {lbl}: {sup(((d > lo) & (d <= hi)).sum() if lo else (d <= hi).sum())}")
        st0 = (per.start - per.intime).dt.total_seconds() / 3600
        out(f"    Kateter yogun bakima giristen once/ilk 6 saatte basliyor: {sup((st0 <= 6).sum())}")

        mb = pd.read_csv(find(root, "microbiologyevents.csv.gz"),
                         usecols=["subject_id", "chartdate", "charttime", "spec_type_desc", "org_name"])
        mb = mb[mb.spec_type_desc.fillna("").str.contains("URINE", case=False) &
                mb.subject_id.isin(per.subject_id)]
        mb["t"] = pd.to_datetime(mb.charttime).fillna(pd.to_datetime(mb.chartdate))
        out(f"  Kateterli hastalarda idrar kulturu kaydi (tum hastane): {sup(len(mb))}")
        pos = mb[mb.org_name.notna() & ~mb.org_name.fillna("").str.contains(
            r"candida|yeast|cancelled|no growth", case=False)]
        x = per.reset_index().merge(pos[["subject_id", "t"]], on="subject_id")
        # NHSN benzeri: kateter >2 gun yerindeyken veya cekildikten sonraki 1 gun icinde pozitif kultur
        x = x[(x.t >= x.start + pd.Timedelta("48h")) & (x.t <= x.end + pd.Timedelta("24h"))]
        kauti = set(x.stay_id)
        out(f"  Olasi KAUTI (kateter >48 saat + pozitif idrar kulturu, maya haric): "
            f"{sup(len(kauti))} ({pct(len(kauti), len(per))})")
        dd = per.dur_h / 24
        for lo, hi, lbl in [(0, 2, "<=48 saat"), (2, 4, "2-4 gun"), (4, 7, "4-7 gun"), (7, 1e9, ">7 gun")]:
            m = (dd <= hi) if lo == 0 else ((dd > lo) & (dd <= hi))
            k = per.index[m].isin(list(kauti)).sum()
            out(f"    {lbl}: n={sup(m.sum())}, KAUTI={sup(k)} ({pct(k, m.sum())})")

    dx = pd.read_csv(find(root, "diagnoses_icd.csv.gz"), dtype={"icd_code": "string"})
    cauti_icd = dx[((dx.icd_version == 10) & dx.icd_code.str.startswith("T8351")) |
                   ((dx.icd_version == 9) & (dx.icd_code == "99664"))]
    uti_icd = dx[((dx.icd_version == 10) & (dx.icd_code == "N390")) |
                 ((dx.icd_version == 9) & (dx.icd_code == "5990"))]
    out(f"  ICD KAUTI kodu (T83.51x / 996.64) olan yatis: {sup(coh.hadm_id.isin(cauti_icd.hadm_id).sum())}")
    out(f"  ICD idrar yolu enfeksiyonu (N39.0 / 599.0) olan yatis: {sup(coh.hadm_id.isin(uti_icd.hadm_id).sum())}")

    out(f"\nToplam sure: {(time.time() - t0) / 60:.1f} dk")
    dest = Path(__file__).with_name("mimic_fizibilite2_ozet.txt")
    dest.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nOzet kaydedildi: {dest}")


if __name__ == "__main__":
    main()
