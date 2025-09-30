import argparse
import os
import glob
import time
import duckdb
import numpy as np
import pandas as pd


CUTOFF = pd.Timestamp("1900-01-01")  # treat pre-1900 dates as dummy -> NaT

NEEDED_COLS = [
    "PSEUDO_HESID", "EPIKEY",
    "ADMIDATE", "DISDATE", "ADMIMETH", "ADMISORC",
    "DIAG_4_01", "ETHNOS", "SEX", "STARTAGE", "LSOA11",
    "SPELBGIN", "SPELEND", "FYEAR", "PARTYEAR"
]


def mode_or_nan(s: pd.Series):
    s = s.dropna()
    return s.mode().iloc[0] if not s.mode().empty else np.nan


def list_year_csvs(path: str, n_years: int | None):
    files = sorted(glob.glob(os.path.join(path, "*.csv")))
    if n_years is not None:
        files = files[-n_years:]
    if not files:
        raise FileNotFoundError(f"No CSV files found under: {path}")
    return files


def load_df(inp: str, n_years: int | None) -> pd.DataFrame:
    # Only read columns we actually need
    usecols = [c for c in NEEDED_COLS]
    # Dtypes to reduce memory (dates parsed separately)
    dtypes = {
        "EPIKEY": "Int64",
        "ADMIMETH": "category",
        "ADMISORC": "category",
        "DIAG_4_01": "category",
        "ETHNOS": "category",
        "SEX": "category",
        "STARTAGE": "Int16",
        "LSOA11": "category",
        "SPELBGIN": "Int8",
        "SPELEND": "category",
        "FYEAR": "Int16",
        "PARTYEAR": "Int32",
        "PSEUDO_HESID": "category",
    }
    parse_dates = ["ADMIDATE", "DISDATE"]

    def read_one(path):
        df = pd.read_csv(
            path,
            usecols=lambda c: c in usecols,
            dtype={k: v for k, v in dtypes.items(
            ) if k in usecols and k not in parse_dates},
            parse_dates=[c for c in parse_dates if c in usecols],
            low_memory=False
        )
        return df

    if os.path.isdir(inp):
        files = list_year_csvs(inp, n_years)
        print(f"📂 Reading {len(files)} file(s):")
        dfs = []
        for f in files:
            print("   -", os.path.basename(f))
            dfs.append(read_one(f))
        out = pd.concat(dfs, ignore_index=True)
    else:
        print(f"📄 Reading file: {inp}")
        out = read_one(inp)

    if out.empty:
        raise ValueError("Loaded an empty dataframe after column filtering.")
    return out


def attach_imd(df: pd.DataFrame, imd_parquet: str) -> pd.DataFrame:
    imd = pd.read_parquet(imd_parquet)
    if "lsoa11_code" in imd.columns and "LSOA11" in df.columns:
        imd = imd.rename(columns={"lsoa11_code": "LSOA11"})
    if not {"LSOA11", "imd_quintile"}.issubset(imd.columns):
        raise KeyError(
            "IMD parquet must have columns: LSOA11, imd_quintile (or lsoa11_code + imd_quintile).")
    return df.merge(imd[["LSOA11", "imd_quintile"]].drop_duplicates("LSOA11"), on="LSOA11", how="left")


def clean_and_collapse(df: pd.DataFrame) -> pd.DataFrame:
    t0 = time.perf_counter()
    n0 = len(df)

    # Dates & LOS in pandas (cheap, deterministic)
    for c in ("ADMIDATE", "DISDATE"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
            df.loc[df[c] < CUTOFF, c] = pd.NaT
    df["los_days"] = (df["DISDATE"] - df["ADMIDATE"]).dt.days + 1
    df.loc[df["los_days"] <= 0, "los_days"] = pd.NA

    # Keep IMD quintile nullable-int
    if "imd_quintile" in df.columns:
        df["imd_quintile"] = df["imd_quintile"].astype("Int64")

    # Register for SQL
    duckdb.register("episodes", df)

    # Collapse with true mode for DIAG_4_01, ETHNOS, SEX (ties => alphabetical)
    spell = duckdb.query("""
        WITH base AS (
            SELECT
                PSEUDO_HESID,
                ADMIDATE AS adm,
                DISDATE  AS dis,
                COUNT(DISTINCT EPIKEY) AS n_episodes,
                MAX(los_days)          AS los_days,
                /* emergency if any ADMIMETH starts with '2' */
                BOOL_OR(LEFT(COALESCE(CAST(ADMIMETH AS VARCHAR),''),1)='2') AS any_emerg,
                ANY_VALUE(imd_quintile) AS imd_quintile
            FROM episodes
            GROUP BY 1,2,3
        ),
        diag_counts AS (
            SELECT PSEUDO_HESID, ADMIDATE AS adm, DISDATE AS dis, DIAG_4_01 AS v, COUNT(*) AS cnt
            FROM episodes
            WHERE DIAG_4_01 IS NOT NULL
            GROUP BY 1,2,3,4
        ),
        diag_mode AS (
            SELECT PSEUDO_HESID, adm, dis, v AS primary_diag
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY PSEUDO_HESID, adm, dis
                    ORDER BY cnt DESC, v
                ) rn
                FROM diag_counts
            ) t
            WHERE rn = 1
        ),
        ethn_counts AS (
            SELECT PSEUDO_HESID, ADMIDATE AS adm, DISDATE AS dis, ETHNOS AS v, COUNT(*) AS cnt
            FROM episodes
            WHERE ETHNOS IS NOT NULL
            GROUP BY 1,2,3,4
        ),
        ethn_mode AS (
            SELECT PSEUDO_HESID, adm, dis, v AS ethnicity
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY PSEUDO_HESID, adm, dis
                    ORDER BY cnt DESC, v
                ) rn
                FROM ethn_counts
            ) t
            WHERE rn = 1
        ),
        sex_counts AS (
            SELECT PSEUDO_HESID, ADMIDATE AS adm, DISDATE AS dis, SEX AS v, COUNT(*) AS cnt
            FROM episodes
            WHERE SEX IS NOT NULL
            GROUP BY 1,2,3,4
        ),
        sex_mode AS (
            SELECT PSEUDO_HESID, adm, dis, v AS sex
            FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY PSEUDO_HESID, adm, dis
                    ORDER BY cnt DESC, v
                ) rn
                FROM sex_counts
            ) t
            WHERE rn = 1
        )
        SELECT
            b.*,
            d.primary_diag,
            e.ethnicity,
            s.sex
        FROM base b
        LEFT JOIN diag_mode d ON b.PSEUDO_HESID=d.PSEUDO_HESID AND b.adm=d.adm AND b.dis=d.dis
        LEFT JOIN ethn_mode e ON b.PSEUDO_HESID=e.PSEUDO_HESID AND b.adm=e.adm AND b.dis=e.dis
        LEFT JOIN sex_mode  s ON b.PSEUDO_HESID=s.PSEUDO_HESID AND b.adm=s.adm AND b.dis=s.dis
    """).to_df()

    # Final LOS & filters back in pandas
    spell["los_days"] = (spell["dis"] - spell["adm"]).dt.days + 1
    ok = spell["los_days"].isna() | ((spell["los_days"] >= 1)
                                     & (spell["los_days"] <= 365*2))
    spell = spell[ok].reset_index(drop=True)

    # Readable id (cheap, post-agg)
    spell["spell_id"] = (
        spell["PSEUDO_HESID"].astype("string") + "|" +
        spell["adm"].astype("string") + "|" +
        spell["dis"].astype("string")
    )

    print(
        f"DuckDB collapse: episodes {n0:,} → spells {len(spell):,} in {time.perf_counter()-t0:.1f}s")
    return spell


def main(inp: str, imd_parquet: str, outp: str, n_years: int | None):
    # 0) load
    df = load_df(inp, n_years=n_years)
    print(f"✅ Loaded raw dataframe: {df.shape[0]:,} rows, {df.shape[1]} cols")

    # 1) attach IMD (required)
    df = attach_imd(df, imd_parquet)
    imd_missing = df["imd_quintile"].isna().mean() * 100
    print(f"ℹ️  IMD missing after join: {imd_missing:.1f}%")

    # 2) clean & collapse
    spell = clean_and_collapse(df)

    # 3) save
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    spell.to_parquet(outp, index=False)
    print(f"💾 Wrote: {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Repair artificial HES APC and collapse to spell-level Parquet.")
    ap.add_argument("--in",  dest="inp",  required=True,
                    help="Single CSV file OR folder of yearly CSVs.")
    ap.add_argument("--imd", dest="imd_parquet", required=True,
                    help="IMD parquet (LSOA11 + imd_quintile).")
    ap.add_argument("--out", dest="outp", required=True,
                    help="Output Parquet (spell-level).")
    ap.add_argument("--n_years", dest="n_years", type=int, default=3,
                    help="If --in is a folder, include only the last N CSV files (default: 3).")
    args = ap.parse_args()
    main(args.inp, args.imd_parquet, args.outp, args.n_years)
