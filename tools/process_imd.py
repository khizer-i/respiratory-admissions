import pandas as pd
import argparse


def main(inp, outp):
    imd = pd.read_excel(inp, sheet_name="IMD2019")
    imd.columns = imd.columns.str.strip().str.lower().str.replace(" ", "_")
    imd = imd.rename(columns={
        "lsoa_code_(2011)": "lsoa11_code",
        "index_of_multiple_deprivation_(imd)_decile": "imd_decile"
    })[["lsoa11_code", "imd_decile"]]
    imd["imd_quintile"] = ((imd["imd_decile"] - 1) // 2 + 1).astype("Int8")
    imd.to_parquet(outp, index=False)
    print(f"✅ Wrote {outp}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--in",  dest="inp",  required=True)
    ap.add_argument("--out", dest="outp", required=True)
    args = ap.parse_args()
    main(args.inp, args.outp)
