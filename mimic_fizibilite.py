# -*- coding: utf-8 -*-
"""
MIMIC-IV v3.1 - Ameliyat sonrasi yogun bakimda pozisyon verme sikligi ve basinc yarasi
FIZIBILITE (on kontrol) betigi.

- Yalnizca TOPLU sayilar uretir; hasta duzeyinde hicbir veri dosyaya yazilmaz.
- 10'dan kucuk sayilar "<10" olarak gizlenir.
- Cikti: betigin bulundugu klasorde  mimic_fizibilite_ozet.txt

Kullanim (Windows komut istemi):
    pip install pandas
    python mimic_fizibilite.py --root D:\\MIMICIV
"""
import argparse
import re
import sys
import time
from pathlib import Path

import pandas as pd

SURG_SERVICES = {"SURG", "CSURG", "NSURG", "ORTHO", "PSURG", "TSURG",
                 "VSURG", "TRAUM", "GU", "ENT"}

# chartevents kalemlerini etiketinden bulmak icin desenler (oncelik sirasiyla)
GROUP_PATTERNS = [
    ("braden",   re.compile(r"^braden", re.I)),
    ("skin",     re.compile(r"pressure (ulcer|injur|sore)|impaired skin|ulcer.*stage|"
                            r"skin integrity|wound.*stage", re.I)),
    ("turn",     re.compile(r"\bturn|reposition", re.I)),
    ("position", re.compile(r"^position\b|position change|patient position", re.I)),
]
PI_VALUE_RE = re.compile(r"pressure|stage|deep tissue|unstageable", re.I)

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
        sys.exit(f"Bulunamadi: {name}  ({root} altinda). --root yolunu kontrol edin.")
    return hits[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="MIMIC-IV klasoru, or. D:\\MIMICIV")
    ap.add_argument("--chunksize", type=int, default=5_000_000)
    a = ap.parse_args()
    root = a.root
    t0 = time.time()

    out("MIMIC-IV FIZIBILITE OZETI - pozisyon verme ve basinc yarasi")
    out("=" * 64)

    # ---------- 1. Kohort ----------
    icu = pd.read_csv(find(root, "icustays.csv.gz"), parse_dates=["intime", "outtime"])
    pat = pd.read_csv(find(root, "patients.csv.gz"), usecols=["subject_id", "anchor_age"])
    svc = pd.read_csv(find(root, "services.csv.gz"), parse_dates=["transfertime"])
    trf = pd.read_csv(find(root, "transfers.csv.gz"), parse_dates=["intime", "outtime"])

    icu = icu.merge(pat, on="subject_id", how="left")
    icu["key"] = icu["intime"] + pd.Timedelta("1h")
    svc = svc.dropna(subset=["hadm_id"]).copy()
    svc["hadm_id"] = svc["hadm_id"].astype("int64")
    icu["hadm_id"] = icu["hadm_id"].astype("int64")
    icu = pd.merge_asof(icu.sort_values("key"),
                        svc[["hadm_id", "transfertime", "curr_service"]].sort_values("transfertime"),
                        left_on="key", right_on="transfertime", by="hadm_id",
                        direction="backward")

    # PACU'dan (ameliyat sonrasi bakim unitesi) son 24 saatte gelis
    cu = trf["careunit"].fillna("")
    pacu = trf[cu.str.contains("PACU", case=False)][["hadm_id", "outtime"]].dropna()
    pacu["hadm_id"] = pacu["hadm_id"].astype("int64")
    x = icu[["stay_id", "hadm_id", "intime"]].merge(pacu, on="hadm_id")
    x = x[(x["outtime"] >= x["intime"] - pd.Timedelta("24h")) &
          (x["outtime"] <= x["intime"] + pd.Timedelta("1h"))]
    icu["pacu24"] = icu["stay_id"].isin(x["stay_id"])

    icu = icu.sort_values("intime")
    icu["first_stay"] = icu.groupby("subject_id").cumcount() == 0
    icu["adult"] = icu["anchor_age"] >= 18
    icu["surg"] = icu["curr_service"].isin(SURG_SERVICES)
    icu["los24"] = icu["los"] >= 1

    out("\n1) KOHORT")
    out(f"Tum yogun bakim yatislari: {sup(len(icu))}")
    out(f"  Eriskin (>=18): {sup(icu.adult.sum())}")
    d1 = icu.adult & icu.surg
    out(f"  T1 - Yogun bakima giriste cerrahi servis: {sup(d1.sum())}")
    d2 = d1 & icu.pacu24
    out(f"  T2 - T1 + son 24 saatte PACU'dan gelis: {sup(d2.sum())}")
    for lab, m in [("T1", d1), ("T2", d2)]:
        mm = m & icu.first_stay & icu.los24
        out(f"  {lab} + ilk yogun bakim yatisi + yatis >=24 saat: {sup(mm.sum())}")
    out("\nCerrahi servis dagilimi (T1, ilk yatis, >=24 saat):")
    vc = icu[d1 & icu.first_stay & icu.los24]["curr_service"].value_counts()
    for k, v in vc.items():
        out(f"  {k}: {sup(v)}")
    out("\nIlk yogun bakim unitesi dagilimi (T1, ilk yatis, >=24 saat):")
    vc = icu[d1 & icu.first_stay & icu.los24]["first_careunit"].value_counts()
    for k, v in vc.items():
        out(f"  {k}: {sup(v)}")
    out("\nTransfers tablosunda PACU/ameliyathane benzeri uniteler (kayit sayisi):")
    vc = trf[cu.str.contains(r"PACU|operating|\bOR\b", case=False, regex=True)]["careunit"].value_counts()
    for k, v in vc.items():
        out(f"  {k}: {sup(v)}")

    cohort = icu[d1 & icu.los24].copy()          # tarama icin genis kume
    main_c = cohort[cohort.first_stay]           # ana kohort (T1, ilk yatis, >=24 saat)
    stays = set(cohort["stay_id"].astype("int64"))

    # ---------- 2. Kalem (item) secimi ----------
    di = pd.read_csv(find(root, "d_items.csv.gz"))
    di = di[di["linksto"] == "chartevents"].copy()

    def grp(label):
        for g, rx in GROUP_PATTERNS:
            if rx.search(str(label)):
                return g
        return None

    di["group"] = di["label"].apply(grp)
    di = di.dropna(subset=["group"])
    item2group = dict(zip(di["itemid"], di["group"]))
    item2label = dict(zip(di["itemid"], di["label"]))

    # ---------- 3. chartevents taramasi ----------
    out("\n2) CHARTEVENTS TARAMASI (uzun surebilir: ~30-60 dk)")
    ce_path = find(root, "chartevents.csv.gz")
    parts = []
    reader = pd.read_csv(ce_path, usecols=["stay_id", "charttime", "itemid", "value"],
                         chunksize=a.chunksize, dtype={"value": "string"})
    for i, ch in enumerate(reader, 1):
        ch = ch[ch["itemid"].isin(item2group.keys()) & ch["stay_id"].isin(stays)]
        if len(ch):
            parts.append(ch)
        print(f"   parca {i} ({(time.time() - t0) / 60:.1f} dk)", flush=True)
    ce = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
        columns=["stay_id", "charttime", "itemid", "value"])
    ce["stay_id"] = ce["stay_id"].astype("int64")
    ce["charttime"] = pd.to_datetime(ce["charttime"])
    ce["group"] = ce["itemid"].map(item2group)
    ce = ce.merge(main_c[["stay_id", "intime", "outtime", "los"]], on="stay_id")
    ce = ce[(ce.charttime >= ce.intime) & (ce.charttime <= ce.outtime)]

    out(f"\nAna kohort (T1, ilk yatis, >=24 saat): {sup(len(main_c))} yatis")
    out("\nEslesen kalemler (ana kohortta):  grup | itemid | etiket | yatis | kayit")
    agg = ce.groupby("itemid").agg(n_stay=("stay_id", "nunique"), n_row=("stay_id", "size"))
    for iid, r in agg.sort_values("n_stay", ascending=False).iterrows():
        out(f"  {item2group[iid]:8s} | {iid} | {item2label[iid]} | {sup(r.n_stay)} | {sup(r.n_row)}")

    out("\nPozisyon/deri kalemlerinde en sik degerler (ilk 10):")
    for iid in agg.index:
        if item2group[iid] in ("skin", "position", "turn"):
            vc = ce.loc[ce.itemid == iid, "value"].value_counts().head(10)
            vc = vc[vc >= 10]
            if len(vc):
                out(f"  [{iid}] {item2label[iid]}")
                for k, v in vc.items():
                    out(f"      {k}: {v}")

    # ---------- 4. Pozisyon verme sikligi ----------
    out("\n3) POZISYON VERME SIKLIGI (ana kohort)")
    for g in ("turn", "position"):
        sub = ce[ce.group == g]
        if sub.empty:
            out(f"  {g}: kayit yok")
            continue
        n = sub.groupby("stay_id").size()
        los = main_c.set_index("stay_id")["los"].reindex(n.index)
        per_day = n / los
        cat = pd.cut(per_day, [0, 6, 12, float("inf")], right=False,
                     labels=["<6/gun (>4 saatte bir)", "6-11/gun (2-4 saatte bir)",
                             ">=12/gun (<=2 saatte bir)"])
        out(f"  [{g}] kaydi olan yatis: {sup(len(n))} / {sup(len(main_c))}")
        out(f"     gunluk kayit medyan (IQR): {per_day.median():.1f} "
            f"({per_day.quantile(.25):.1f}-{per_day.quantile(.75):.1f})")
        for k, v in cat.value_counts().sort_index().items():
            out(f"     {k}: {sup(v)}")
        gap = (sub.sort_values("charttime").groupby("stay_id")["charttime"]
               .diff().dt.total_seconds() / 3600).dropna()
        if len(gap):
            out(f"     ardisik kayitlar arasi sure medyan (IQR), saat: {gap.median():.1f} "
                f"({gap.quantile(.25):.1f}-{gap.quantile(.75):.1f})")

    # ---------- 5. Basinc yarasi ----------
    out("\n4) BASINC YARASI (ana kohort)")
    skin = ce[ce.group == "skin"].copy()
    lbl = skin["itemid"].map(item2label).fillna("")
    skin["pi"] = lbl.str.contains("pressure", case=False) | \
        skin["value"].fillna("").str.contains(PI_VALUE_RE)
    pi = skin[skin.pi]
    out(f"  Deri kalemi kaydi olan yatis: {sup(skin.stay_id.nunique())}")
    out(f"  Herhangi bir basinc yarasi kaydi olan yatis: {sup(pi.stay_id.nunique())}")
    first = pi.groupby("stay_id").agg(t=("charttime", "min"), intime=("intime", "first"))
    early = (first.t - first.intime) <= pd.Timedelta("24h")
    out(f"    ilk 24 saatte var (olasi mevcut yara): {sup(early.sum())}")
    out(f"    ilk 24 saatte yok, sonra ortaya cikan (olasi YENI yara): {sup((~early).sum())}")

    dx = pd.read_csv(find(root, "diagnoses_icd.csv.gz"), dtype={"icd_code": "string"})
    l89 = dx[((dx.icd_version == 10) & dx.icd_code.str.startswith("L89")) |
             ((dx.icd_version == 9) & dx.icd_code.str.startswith("7070"))]
    out(f"  ICD basinc yarasi tanisi (L89 / 707.0x) olan yatis: "
        f"{sup(main_c.hadm_id.isin(l89.hadm_id).sum())}")

    # ---------- 6. Braden ----------
    br = ce[ce.group == "braden"]
    br24 = br[(br.charttime - br.intime) <= pd.Timedelta("24h")]
    full = br24.groupby("stay_id")["itemid"].nunique()
    out("\n5) BRADEN")
    out(f"  Braden kaydi olan yatis: {sup(br.stay_id.nunique())}")
    out(f"  Ilk 24 saatte >=6 Braden alt boyutu kayitli yatis: {sup((full >= 6).sum())}")

    out(f"\nToplam sure: {(time.time() - t0) / 60:.1f} dk")
    dest = Path(__file__).with_name("mimic_fizibilite_ozet.txt")
    dest.write_text("\n".join(LINES), encoding="utf-8")
    print(f"\nOzet kaydedildi: {dest}")


if __name__ == "__main__":
    main()
