#!/usr/bin/env python3
"""Build CNES 2025 hospital points and cholecystectomy hospital set (G0)."""
import os, json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(os.environ.get("URBAN_SURG_DATA_DIR", Path(__file__).resolve().parents[3] / "data"))
RAW = BASE / "01_data/raw/cnes/cnes_estabelecimentos.csv"
INTERIM = BASE / "01_data/interim"
OUT = BASE / "03_outputs"
os.makedirs(INTERIM, exist_ok=True)
os.makedirs(OUT, exist_ok=True)

usecols = [
    "CO_CNES", "CO_UF", "CO_IBGE", "NO_RAZAO_SOCIAL", "NO_FANTASIA",
    "TP_UNIDADE", "NU_LATITUDE", "NU_LONGITUDE", "CO_CEP", "NO_BAIRRO",
    "ST_CENTRO_CIRURGICO", "ST_CENTRO_OBSTETRICO", "ST_CENTRO_NEONATAL",
    "ST_ATEND_HOSPITALAR", "ST_ATEND_AMBULATORIAL", "CO_NIVEL_HIERARQUIA",
]
df = pd.read_csv(RAW, sep=";", dtype=str, encoding="utf-8", usecols=usecols, low_memory=False)
print("raw rows", len(df), flush=True)

for c in ["NU_LATITUDE", "NU_LONGITUDE"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["cnes"] = df["CO_CNES"].str.strip().str.zfill(7)
df["municipio6"] = df["CO_IBGE"].str.strip().str[:6]
df["uf"] = df["CO_UF"].str.strip()
df["name"] = df["NO_FANTASIA"].fillna(df["NO_RAZAO_SOCIAL"]).str.strip()

# hospital type codes (CNES TP_UNIDADE): 05 geral, 07 especializado, 15 misto,
# 20 pronto socorro geral, 21 pronto socorro especializado
HOSP_CODES = {"05", "07", "15", "20", "21"}
df["is_hospital"] = df["TP_UNIDADE"].str.strip().isin(HOSP_CODES) | (df["ST_ATEND_HOSPITALAR"].str.strip() == "1.0")
df["has_surg_center"] = df["ST_CENTRO_CIRURGICO"].str.strip() == "1.0"
df["has_obst_center"] = df["ST_CENTRO_OBSTETRICO"].str.strip() == "1.0"
df["has_neo_center"] = df["ST_CENTRO_NEONATAL"].str.strip() == "1.0"
df["coord_ok"] = (
    df["NU_LATITUDE"].between(-34.0, 5.5)
    & df["NU_LONGITUDE"].between(-74.5, -32.0)
)

print("coord_ok", int(df["coord_ok"].sum()), flush=True)
print("is_hospital", int(df["is_hospital"].sum()), flush=True)
print("is_hospital & coord_ok", int((df["is_hospital"] & df["coord_ok"]).sum()), flush=True)
print("has_surg_center", int(df["has_surg_center"].sum()), flush=True)
print("has_obst_center", int(df["has_obst_center"].sum()), flush=True)
print("has_surg & coord", int((df["has_surg_center"] & df["coord_ok"]).sum()), flush=True)
print("has_obst & coord", int((df["has_obst_center"] & df["coord_ok"]).sum()), flush=True)

keep = df[df["is_hospital"] | df["has_surg_center"] | df["has_obst_center"]].copy()
keep = keep[[
    "cnes", "uf", "municipio6", "name", "TP_UNIDADE", "NU_LATITUDE", "NU_LONGITUDE",
    "coord_ok", "is_hospital", "has_surg_center", "has_obst_center", "has_neo_center",
    "ST_ATEND_HOSPITALAR", "CO_NIVEL_HIERARQUIA",
]].rename(columns={"NU_LATITUDE": "lat", "NU_LONGITUDE": "lon"})
keep.to_parquet(f"{INTERIM}/cnes_2025_points.parquet", index=False)
keep[keep["coord_ok"]].to_csv(f"{OUT}/hospital_points.csv", index=False)
print("saved points", len(keep), "with coords", int(keep["coord_ok"].sum()), flush=True)

# --- cholecystectomy hospitals from SIH 2024 ---
chole_path = Path(os.environ.get("SIH_CHOLE_PARQUET", str(BASE / "01_data/raw/sih/sih_2024_cholec.parquet")))
ch = pd.read_parquet(chole_path, columns=["CNES", "MUNIC_RES", "MUNIC_MOV", "PROC_REA", "MES_CMPT"])
ch["cnes"] = ch["CNES"].astype(str).str.zfill(7)
g = ch.groupby("cnes").size().rename("n_chole").reset_index()
g = g.merge(keep[["cnes", "uf", "municipio6", "name", "lat", "lon", "coord_ok",
                  "is_hospital", "has_surg_center", "has_obst_center"]], on="cnes", how="left")
g["in_cnes"] = g["lat"].notna()
g.to_parquet(f"{INTERIM}/hosp_chole_2024.parquet", index=False)

vol_total = int(g["n_chole"].sum())
vol_coord = int(g.loc[g["coord_ok"] == True, "n_chole"].sum())
vol_hosp = int(g.loc[g["is_hospital"] == True, "n_chole"].sum())
vol_hosp_coord = int(g.loc[(g["is_hospital"] == True) & (g["coord_ok"] == True), "n_chole"].sum())
summary = {
    "cnes_rows": int(len(df)),
    "coord_ok": int(df["coord_ok"].sum()),
    "is_hospital": int(df["is_hospital"].sum()),
    "is_hospital_coord": int((df["is_hospital"] & df["coord_ok"]).sum()),
    "has_surg_center": int(df["has_surg_center"].sum()),
    "has_obst_center": int(df["has_obst_center"].sum()),
    "chole_rows": int(len(ch)),
    "chole_hospitals": int(len(g)),
    "chole_hospitals_matched_cnes": int(g["in_cnes"].sum()),
    "chole_volume": vol_total,
    "chole_volume_coord": vol_coord,
    "chole_volume_coord_share": round(vol_coord / vol_total, 4),
    "chole_volume_hospital": vol_hosp,
    "chole_volume_hospital_coord": vol_hosp_coord,
    "chole_volume_hospital_coord_share": round(vol_hosp_coord / vol_total, 4),
}
with open(f"{BASE}/02_analysis/g0_hospital_summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
