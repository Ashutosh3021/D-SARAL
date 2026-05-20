"""
backend_processing.py  —  D-SARAL Production Data Cleaning Pipeline
=====================================================================
Output is fully ML / EDA ready.  A data scientist or ML engineer can
take the cleaned CSV and go straight to feature engineering.

Pipeline steps (apply_cleaning_techniques)
-------------------------------------------
 1. Encoding & whitespace normalisation
 2. Placeholder  →  NaN  (20+ variants)
 3. Per-column type inference & auto-cast
 4. Smart missing-value imputation
     • numeric  : median (<5 % missing) / mean (<30 %) / median (≥30 %)
     • datetime : median date
     • object   : mode  → fallback "UNKNOWN"
     • drop col if >60 % missing
 5. Date standardisation  →  ISO 8601 (YYYY-MM-DD)
 6. Numeric-string normalisation  (strips $, commas; ALL object cols tried)
 7. Text-case standardisation  (title / lower by column semantics)
 8. Outlier capping  (IQR × 1.5 fences, skips ID/zip/phone/year cols)
 9. Domain validation & nullification  (age 0-120, email regex, phone basic)
10. Exact duplicate removal
11. Fuzzy near-duplicate flagging  (token_sort_ratio ≥ 92)
12. Constant / near-constant column removal  (≥99.5 % single value)
13. Column-name normalisation  (lowercase, spaces → underscores)
14. Final type enforcement  (float→Int64 where safe)
"""

import glob
import json
import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from thefuzz import fuzz

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# Tuneable constants
# ──────────────────────────────────────────────────────────────────────────────
PLACEHOLDER_VALUES: set = {
    "n/a", "na", "null", "none", "nan", "empty", "missing", "unknown",
    "undefined", "nil", "not available", "not applicable", "not disclosed",
    "#n/a", "-", "--", "---", "?", ".", "0000-00-00", "1900-01-01",
}

DATE_OUTPUT_FORMAT       = "%Y-%m-%d"   # ISO 8601
OUTLIER_IQR_MULTIPLIER   = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0
FUZZY_DEDUP_THRESHOLD    = 92           # 0-100; lower = more aggressive
NEAR_CONSTANT_THRESHOLD  = 0.995        # drop col if top value ≥ this fraction
MISSING_DROP_THRESHOLD   = 0.60         # drop col if missing fraction > this

_SKIP_OUTLIER_KW = ("id", "code", "zip", "postal", "phone", "mobile", "year",
                    "flag", "bool", "index")
_DATE_KW         = ("date", "time", "dob", "birth", "created", "updated",
                    "modified", "timestamp", "at")
_NUMERIC_KW      = ("salary", "income", "price", "amount", "cost", "revenue",
                    "fee", "wage", "rate", "balance", "total", "count",
                    "score", "mark", "grade", "rating", "quantity", "qty",
                    "weight", "height", "distance", "age", "year")
_TITLE_KW        = ("name", "department", "dept", "category", "type",
                    "status", "gender", "sex", "city", "country", "state",
                    "region", "district", "title", "role", "position",
                    "occupation", "subject")
_LOWER_KW        = ("email", "url", "website", "domain", "username", "user",
                    "login", "handle", "slug")
_PHONE_PATTERN   = re.compile(r"^\+?[\d\s\-().]{7,15}$")
_EMAIL_PATTERN   = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ──────────────────────────────────────────────────────────────────────────────
class DataProcessingPipeline:
    """Production-grade data cleaning pipeline for D-SARAL."""

    def __init__(self, sample_size: int = 10_000):
        self.sample_size         = sample_size
        self.current_dataframes: Dict[str, pd.DataFrame] = {}
        self.analysis_report:    Dict[str, Any]          = {}
        self.cleaning_log:       List[Dict]              = []
        self.processed_data:     Optional[pd.DataFrame]  = None

    # ──────────────────────────────────────────────────────────────────────────
    # 1. File loading
    # ──────────────────────────────────────────────────────────────────────────
    def load_files_from_directory(
        self,
        data_directory: str,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load every supported file under *data_directory*."""
        if file_types is None:
            file_types = ["csv", "json", "txt"]

        self.current_dataframes = {}

        for ext in file_types:
            for fp in glob.glob(
                os.path.join(data_directory, "**", f"*.{ext}"), recursive=True
            ):
                try:
                    large  = os.path.getsize(fp) > 1_000_000
                    nrows  = self.sample_size if large else None
                    ext_lc = ext.lower()

                    if ext_lc == "csv":
                        df = pd.read_csv(
                            fp, nrows=nrows, encoding="utf-8",
                            on_bad_lines="skip", low_memory=False,
                        )
                    elif ext_lc == "json":
                        df = pd.read_json(fp)
                        if large:
                            df = df.head(self.sample_size)
                    elif ext_lc == "txt":
                        df = pd.read_csv(
                            fp, sep=None, engine="python",
                            on_bad_lines="skip", nrows=nrows,
                        )
                    else:
                        continue

                    self.current_dataframes[fp] = df
                except Exception as exc:
                    print(f"[LOAD ERROR] {fp}: {exc}")

        return self.current_dataframes

    # ──────────────────────────────────────────────────────────────────────────
    # 2. Analysis helpers
    # ──────────────────────────────────────────────────────────────────────────
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        ph_mask  = df.apply(
            lambda c: c.astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
        )
        nan_mask = df.isnull()
        emp_mask = df.apply(lambda c: c.astype(str).str.strip().eq(""))
        combined = nan_mask | emp_mask | ph_mask

        total = combined.sum()
        pct   = (total / len(df) * 100).round(2)
        return {
            "total_missing_per_column": total[total > 0].to_dict(),
            "pct_missing_per_column":   pct[pct > 0].to_dict(),
            "standard_nan_counts":      nan_mask.sum()[nan_mask.sum() > 0].to_dict(),
            "empty_string_counts":      emp_mask.sum()[emp_mask.sum() > 0].to_dict(),
            "placeholder_counts":       ph_mask.sum()[ph_mask.sum() > 0].to_dict(),
        }

    def analyze_format_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        sample  = df.sample(min(5_000, len(df)), random_state=42)
        result: Dict[str, Any] = {}

        for col in df.columns:
            s = sample[col].dropna()

            # mixed Python types
            utypes = {type(v).__name__ for v in s}
            if len(utypes) > 1:
                result[f"{col}_mixed_types"] = list(utypes)

            col_l = col.lower()

            # date-format variety
            if any(kw in col_l for kw in _DATE_KW):
                fmts: set = set()
                for v in s.astype(str):
                    if re.match(r"\d{4}-\d{2}-\d{2}", v):     fmts.add("YYYY-MM-DD")
                    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", v): fmts.add("MM/DD/YYYY")
                    elif re.match(r"\d{1,2}-\d{1,2}-\d{4}", v): fmts.add("MM-DD-YYYY")
                    elif re.match(r"\d{1,2} \w+ \d{4}", v):     fmts.add("DD Month YYYY")
                if len(fmts) > 1:
                    result[f"{col}_multiple_date_formats"] = list(fmts)

            # numeric-string format variety
            if s.dtype == object:
                pats: List[str] = []
                for v in s.astype(str):
                    if re.match(r"^\$?\d{1,3}(,\d{3})*(\.\d+)?$", v):
                        pats.append("comma_formatted")
                    elif re.match(r"^\$\d+(\.\d+)?$", v):
                        pats.append("currency_symbol")
                    elif re.match(r"^\d+$", v):
                        pats.append("plain_integer")
                    elif re.match(r"^\d+\.\d+$", v):
                        pats.append("decimal")
                if len(set(pats)) > 1:
                    result[f"{col}_multiple_numeric_formats"] = list(set(pats))

        return result

    def analyze_broken_entries(self, df: pd.DataFrame) -> Dict[str, Any]:
        broken: Dict[str, Any] = {}

        for col in df.columns:
            col_l  = col.lower()
            series = df[col]

            if "age" in col_l:
                num = pd.to_numeric(series, errors="coerce")
                bad = num[(num < 0) | (num > 150)].dropna()
                if not bad.empty:
                    broken[f"{col}_invalid_ages"] = {
                        "count": len(bad), "examples": bad.head(5).tolist()
                    }

            if "email" in col_l:
                bad = series[~series.astype(str).str.match(_EMAIL_PATTERN, na=False)]
                bad = bad.dropna()
                if not bad.empty:
                    broken[f"{col}_invalid_emails"] = {
                        "count": len(bad), "examples": bad.head(5).tolist()
                    }

            if any(kw in col_l for kw in ("phone", "mobile", "tel", "contact")):
                bad = series[~series.astype(str).str.match(_PHONE_PATTERN, na=False)]
                bad = bad.dropna()
                if not bad.empty:
                    broken[f"{col}_invalid_phones"] = {
                        "count": len(bad), "examples": bad.head(5).tolist()
                    }

            if any(kw in col_l for kw in ("date", "birth", "dob")):
                parsed    = pd.to_datetime(series, errors="coerce")
                bad_dates = series[parsed.isna() & series.notna()]
                if not bad_dates.empty:
                    broken[f"{col}_unparseable_dates"] = {
                        "count": len(bad_dates), "examples": bad_dates.head(5).tolist()
                    }
                if "birth" in col_l or "dob" in col_l:
                    future = parsed[parsed > pd.Timestamp.now()]
                    if not future.empty:
                        broken[f"{col}_future_birth_dates"] = {
                            "count": len(future),
                            "examples": future.dropna()
                                                .dt.strftime(DATE_OUTPUT_FORMAT)
                                                .head(5).tolist(),
                        }

        return broken

    def analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        exact = int(df.duplicated().sum())
        result: Dict[str, Any] = {
            "exact_duplicates":  exact,
            "duplicate_indices": df[df.duplicated(keep=False)].index.tolist()[:20],
        }

        near_pairs: List[Dict] = []
        str_cols = [
            c for c in df.columns
            if df[c].dtype == object
            and any(kw in c.lower() for kw in ("name", "title", "description", "label"))
        ]
        for col in str_cols[:2]:
            vals = df[col].dropna().astype(str).head(500).tolist()
            for i in range(len(vals)):
                for j in range(i + 1, len(vals)):
                    score = fuzz.token_sort_ratio(vals[i], vals[j])
                    if FUZZY_DEDUP_THRESHOLD <= score < 100:
                        near_pairs.append(
                            {"col": col, "a": vals[i], "b": vals[j], "score": score}
                        )
                    if len(near_pairs) >= 30:
                        break
                if len(near_pairs) >= 30:
                    break

        result["near_duplicate_pairs"] = near_pairs
        return result

    def analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        report: Dict[str, Any] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) < 10:
                continue
            z      = np.abs(scipy_stats.zscore(s))
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - OUTLIER_IQR_MULTIPLIER * iqr, q3 + OUTLIER_IQR_MULTIPLIER * iqr
            z_out  = s[z > OUTLIER_ZSCORE_THRESHOLD]
            i_out  = s[(s < lo) | (s > hi)]
            if not z_out.empty or not i_out.empty:
                report[col] = {
                    "z_score_outliers": {"count": len(z_out), "examples": z_out.head(5).tolist()},
                    "iqr_outliers":     {
                        "count": len(i_out), "examples": i_out.head(5).tolist(),
                        "lower_fence": round(lo, 4), "upper_fence": round(hi, 4),
                    },
                }
        return report

    def analyze_column_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for col in df.select_dtypes(include="object").columns:
            lower_map: Dict[str, List[str]] = {}
            for v in df[col].dropna().unique():
                k = str(v).strip().lower()
                lower_map.setdefault(k, []).append(str(v))
            variants = {k: vs for k, vs in lower_map.items() if len(vs) > 1}
            if variants:
                result[f"{col}_case_inconsistencies"] = variants
        return result

    def comprehensive_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        analysis = {
            "dataset_overview": {
                "shape":        df.shape,
                "columns":      df.columns.tolist(),
                "dtypes":       {c: str(t) for c, t in df.dtypes.items()},
                "memory_bytes": int(df.memory_usage(deep=True).sum()),
            },
            "missing_values":         self.analyze_missing_values(df),
            "format_inconsistencies": self.analyze_format_inconsistencies(df),
            "broken_entries":         self.analyze_broken_entries(df),
            "duplicates":             self.analyze_duplicates(df),
            "outliers":               self.analyze_outliers(df),
            "column_inconsistencies": self.analyze_column_inconsistencies(df),
        }
        self.analysis_report = analysis
        return analysis

    # ──────────────────────────────────────────────────────────────────────────
    # 3. Human-readable report
    # ──────────────────────────────────────────────────────────────────────────
    def document_issues_with_examples(self, df: pd.DataFrame) -> str:
        self.comprehensive_data_analysis(df)
        r     = self.analysis_report
        lines: List[str] = []

        lines += [
            "# D-SARAL Data Quality Report",
            f"Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Dataset Overview",
            f"- Rows    : {r['dataset_overview']['shape'][0]}",
            f"- Columns : {r['dataset_overview']['shape'][1]}",
            f"- Memory  : {r['dataset_overview']['memory_bytes'] / 1024 / 1024:.2f} MB",
            f"- Columns : {', '.join(r['dataset_overview']['columns'])}",
            "",
        ]

        mv = r["missing_values"]["total_missing_per_column"]
        if mv:
            lines.append("## Missing Values")
            for col, cnt in mv.items():
                pct = r["missing_values"]["pct_missing_per_column"].get(col, 0)
                lines.append(f"- **{col}**: {cnt} missing ({pct} %)")
            lines.append("")

        fi = r["format_inconsistencies"]
        if fi:
            lines.append("## Format Inconsistencies")
            for k, v in fi.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        be = r["broken_entries"]
        if be:
            lines.append("## Broken Entries")
            for k, v in be.items():
                lines.append(
                    f"- **{k}**: {v['count']} entries  —  examples: {v.get('examples', [])[:3]}"
                )
            lines.append("")

        dup = r["duplicates"]
        lines.append("## Duplicates")
        lines.append(f"- Exact duplicates : {dup['exact_duplicates']}")
        if dup.get("near_duplicate_pairs"):
            lines.append(f"- Near-duplicate pairs : {len(dup['near_duplicate_pairs'])}")
            for p in dup["near_duplicate_pairs"][:5]:
                lines.append(
                    f"  - [{p['score']} %]  \"{p['a']}\"  ≈  \"{p['b']}\"  (col: {p['col']})"
                )
        lines.append("")

        ol = r["outliers"]
        if ol:
            lines.append("## Outliers")
            for col, info in ol.items():
                z  = info["z_score_outliers"]["count"]
                iq = info["iqr_outliers"]["count"]
                lo = info["iqr_outliers"]["lower_fence"]
                hi = info["iqr_outliers"]["upper_fence"]
                lines.append(
                    f"- **{col}**: {z} z-score | {iq} IQR  (fences: [{lo}, {hi}])"
                )
            lines.append("")

        ci = r["column_inconsistencies"]
        if ci:
            lines.append("## Case / Spelling Inconsistencies")
            for k, v in ci.items():
                lines.append(f"- **{k}**: {v}")
            lines.append("")

        return "\n".join(lines)

    # ──────────────────────────────────────────────────────────────────────────
    # 4. THE CLEANING ENGINE
    # ──────────────────────────────────────────────────────────────────────────
    def apply_cleaning_techniques(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        14-step production-grade cleaning pipeline.
        Input  : raw DataFrame (any mix of types / quality)
        Output : ML / EDA-ready DataFrame — no further cleaning needed.
        """
        original_shape = df.shape
        df             = df.copy()
        self._log("START", original_shape, df.shape, "Pipeline started")

        # ── Step 1  Encoding & whitespace ────────────────────────────────────
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.encode("ascii", errors="ignore")
                .str.decode("ascii")
                .str.strip()
                .replace("nan", np.nan)
            )
        self._log("step_01_whitespace", original_shape, df.shape,
                  "Encoding & whitespace normalised")

        # ── Step 2  Placeholder → NaN ────────────────────────────────────────
        for col in df.columns:
            df[col] = df[col].replace("", np.nan)
            mask = df[col].astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
            df.loc[mask, col] = np.nan
        self._log("step_02_placeholders", original_shape, df.shape,
                  "Placeholders → NaN")

        # ── Step 3  Type inference & auto-cast ───────────────────────────────
        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()

            # date columns first (before numeric strips the separators)
            if any(kw in col_l for kw in _DATE_KW):
                parsed = pd.to_datetime(df[col], errors="coerce")
                hit_rate = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                if hit_rate > 0.7:
                    df[col] = parsed
                    continue

            # numeric
            cleaned_num = (
                df[col].astype(str)
                       .str.replace(r"[^\d.\-]", "", regex=True)
                       .replace("", np.nan)
            )
            converted = pd.to_numeric(cleaned_num, errors="coerce")
            hit_rate  = converted.notna().sum() / max(df[col].notna().sum(), 1)
            if hit_rate > 0.85:
                df[col] = converted

        self._log("step_03_type_inference", original_shape, df.shape,
                  "Type inference applied")

        # ── Step 4  Smart missing-value imputation ───────────────────────────
        imp_log: Dict[str, str] = {}
        for col in list(df.columns):          # list() because we may drop cols
            n_miss = df[col].isna().sum()
            if n_miss == 0:
                continue
            pct = n_miss / len(df)

            if pct > MISSING_DROP_THRESHOLD:
                df.drop(columns=[col], inplace=True)
                imp_log[col] = f"DROPPED (>{int(pct*100)} % missing)"
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                if pct < 0.05:
                    fv = df[col].median(); tag = f"median ({fv:.4g})"
                elif pct < 0.30:
                    fv = df[col].mean();   tag = f"mean ({fv:.4g})"
                else:
                    fv = df[col].median(); tag = f"median-high ({fv:.4g})"
                df[col].fillna(fv, inplace=True)

            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                fv  = df[col].dropna().sort_values().iloc[len(df[col].dropna()) // 2]
                tag = f"median_date ({fv})"
                df[col].fillna(fv, inplace=True)

            else:
                modes = df[col].mode()
                fv    = modes[0] if not modes.empty else "UNKNOWN"
                tag   = f"mode ('{fv}')"
                df[col].fillna(fv, inplace=True)

            imp_log[col] = tag

        self.cleaning_log.append({"step": "step_04_imputation", "details": imp_log})
        self._log("step_04_imputation", original_shape, df.shape,
                  f"Imputed {len(imp_log)} columns")

        # ── Step 5  Date standardisation → ISO 8601 ──────────────────────────
        for col in df.select_dtypes(
            include=["datetime64[ns]", "datetime64[ns, UTC]"]
        ).columns:
            df[col] = df[col].dt.strftime(DATE_OUTPUT_FORMAT)

        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()
            if any(kw in col_l for kw in _DATE_KW):
                parsed   = pd.to_datetime(df[col], errors="coerce")
                hit_rate = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                if hit_rate > 0.5:
                    df[col] = parsed.dt.strftime(DATE_OUTPUT_FORMAT)

        self._log("step_05_dates", original_shape, df.shape,
                  "Dates standardised → ISO 8601")

        # ── Step 6  Numeric-string normalisation (ALL object cols) ────────────
        #   Tries every object column; only replaces if ≥70 % parse cleanly.
        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()
            # skip columns that are clearly not numeric
            if any(kw in col_l for kw in ("name", "email", "url", "address",
                                          "description", "comment", "note",
                                          "text", "label", "tag")):
                continue
            cleaned = (
                df[col].astype(str)
                       .str.replace(r"[^\d.\-]", "", regex=True)
                       .replace("", np.nan)
            )
            as_num   = pd.to_numeric(cleaned, errors="coerce")
            hit_rate = as_num.notna().sum() / max(df[col].notna().sum(), 1)
            if hit_rate >= 0.70:
                df[col] = as_num

        self._log("step_06_numeric_normalise", original_shape, df.shape,
                  "Numeric-string columns normalised")

        # ── Step 7  Text-case standardisation ────────────────────────────────
        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()
            if any(kw in col_l for kw in _TITLE_KW):
                df[col] = df[col].str.title()
            elif any(kw in col_l for kw in _LOWER_KW):
                df[col] = df[col].str.lower()

        self._log("step_07_text_case", original_shape, df.shape,
                  "Text case standardised")

        # ── Step 8  Outlier capping (IQR) ─────────────────────────────────────
        caps: Dict[str, Dict] = {}
        for col in df.select_dtypes(include=[np.number]).columns:
            if any(kw in col.lower() for kw in _SKIP_OUTLIER_KW):
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr     = q3 - q1
            if iqr == 0:
                continue
            lo = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            hi = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            n  = int(((df[col] < lo) | (df[col] > hi)).sum())
            if n:
                df[col] = df[col].clip(lower=lo, upper=hi)
                caps[col] = {"capped": n, "lo": round(lo, 4), "hi": round(hi, 4)}

        self.cleaning_log.append({"step": "step_08_outlier_caps", "details": caps})
        self._log("step_08_outlier_cap", original_shape, df.shape,
                  f"Outliers capped in {len(caps)} columns")

        # ── Step 9  Domain validation & nullification ─────────────────────────
        for col in df.columns:
            col_l = col.lower()

            if "age" in col_l and pd.api.types.is_numeric_dtype(df[col]):
                df.loc[(df[col] < 0) | (df[col] > 120), col] = np.nan

            if "email" in col_l:
                invalid = ~df[col].astype(str).str.match(_EMAIL_PATTERN, na=False)
                df.loc[invalid, col] = np.nan

            if any(kw in col_l for kw in ("phone", "mobile", "tel")):
                invalid = ~df[col].astype(str).str.match(_PHONE_PATTERN, na=False)
                df.loc[invalid, col] = np.nan

        self._log("step_09_domain_validation", original_shape, df.shape,
                  "Invalid ages / emails / phones nullified")

        # ── Step 10  Exact deduplication ──────────────────────────────────────
        before = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        self._log("step_10_exact_dedup", original_shape, df.shape,
                  f"Removed {before - len(df)} exact duplicates")

        # ── Step 11  Fuzzy near-duplicate flagging ────────────────────────────
        str_cols = [
            c for c in df.columns
            if df[c].dtype == object
            and any(kw in c.lower() for kw in ("name", "title", "description", "label"))
        ]
        n_flagged = 0
        if str_cols:
            primary = str_cols[0]
            vals    = df[primary].fillna("").astype(str).tolist()
            flags   = [False] * len(vals)
            for i in range(len(vals)):
                if flags[i]:
                    continue
                for j in range(i + 1, min(i + 300, len(vals))):
                    if fuzz.token_sort_ratio(vals[i], vals[j]) >= FUZZY_DEDUP_THRESHOLD:
                        flags[j] = True
            n_flagged = sum(flags)
            if n_flagged:
                df["_dsaral_near_duplicate"] = flags

        self._log("step_11_fuzzy_dedup", original_shape, df.shape,
                  f"Flagged {n_flagged} near-duplicates")

        # ── Step 12  Constant / near-constant column removal ──────────────────
        dropped: List[str] = []
        for col in list(df.columns):
            if col.startswith("_dsaral_"):
                continue
            top = df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top >= NEAR_CONSTANT_THRESHOLD:
                df.drop(columns=[col], inplace=True)
                dropped.append(col)

        self._log("step_12_const_cols", original_shape, df.shape,
                  f"Dropped {len(dropped)} near-constant columns: {dropped}")

        # ── Step 13  Column-name normalisation ───────────────────────────────
        df.columns = [
            re.sub(r"\s+", "_", c.strip().lower())
               .replace("-", "_")
               .replace(".", "_")
            for c in df.columns
        ]
        self._log("step_13_col_names", original_shape, df.shape,
                  "Column names normalised (snake_case)")

        # ── Step 14  Final type enforcement ───────────────────────────────────
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) and s.apply(lambda x: float(x).is_integer()).all():
                try:
                    df[col] = df[col].astype("Int64")
                except Exception:
                    pass

        self._log("step_14_final_types", original_shape, df.shape,
                  "Final integer types enforced")

        # ── Summary ──────────────────────────────────────────────────────────
        self._log(
            "COMPLETE", original_shape, df.shape,
            f"Done. Rows: {original_shape[0]}→{df.shape[0]}  "
            f"Cols: {original_shape[1]}→{df.shape[1]}"
        )
        self.processed_data = df
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────
    def _log(self, step: str, orig, cur, msg: str) -> None:
        self.cleaning_log.append({
            "step":           step,
            "original_shape": list(orig),
            "current_shape":  list(cur),
            "message":        msg,
            "timestamp":      datetime.now().isoformat(),
        })
        print(f"[{step}] {msg}  →  shape: {cur}")