"""
backend_processing.py  —  D-SARAL  |  Messy Data Autopsy Workflow
==================================================================
Follows the exact workflow diagram:

  Dataset Selection
       ↓
  Load & Observe  ──→  observe_only()   [NO fixing, just looking]
       ↓                 • load
       ↓                 • shape_info
       ↓                 • head_tail
       ↓                 • columns
       ↓
  Identify Mess Types
       ↓  missing / formats / broken / duplicates / contradictory cols
       ↓
  Decision Making
       ↓  establish rules / analyze trade-offs / document rationale / NO GUESSING
       ↓
  Incremental Cleaning          ← NO "clean everything" function
       ↓  fix_missing()         one operation at a time
       ↓  fix_formats()         re-check stats after each
       ↓  fix_broken_entries()  verify shape unchanged where expected
       ↓  fix_duplicates()
       ↓  fix_column_names()
       ↓
  Validation  (loop back if issues found)
       ↓  re-run .info()
       ↓  check missing values
       ↓  look for new anomalies
       ↓  before/after comparison
       ↓
  Final Narrative
       • complete story
       • clear assumptions
       • document remaining risks
       • professional documentation
"""

import glob
import json
import os
import re
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

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

DATE_OUTPUT_FORMAT       = "%Y-%m-%d"
OUTLIER_IQR_MULTIPLIER   = 1.5
OUTLIER_ZSCORE_THRESHOLD = 3.0
FUZZY_DEDUP_THRESHOLD    = 92
NEAR_CONSTANT_THRESHOLD  = 0.995
MISSING_DROP_THRESHOLD   = 0.60
MAX_VALIDATION_LOOPS     = 3        # safety cap on the validation feedback loop

_SKIP_OUTLIER_KW = ("id", "code", "zip", "postal", "phone", "mobile",
                    "year", "flag", "bool", "index")
_DATE_KW         = ("date", "time", "dob", "birth", "created", "updated",
                    "modified", "timestamp", "at")
_TITLE_KW        = ("name", "department", "dept", "category", "type",
                    "status", "gender", "sex", "city", "country", "state",
                    "region", "district", "title", "role", "position",
                    "occupation", "subject")
_LOWER_KW        = ("email", "url", "website", "domain", "username",
                    "user", "login", "handle", "slug")
_PHONE_PATTERN   = re.compile(r"^\+?[\d\s\-().]{7,15}$")
_EMAIL_PATTERN   = re.compile(
    r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
)


# ══════════════════════════════════════════════════════════════════════════════
class DataProcessingPipeline:
    """
    D-SARAL pipeline — follows the Messy Data Autopsy Workflow diagram exactly.
    """

    def __init__(self, sample_size: int = 10_000):
        self.sample_size          = sample_size
        self.current_dataframes:  Dict[str, pd.DataFrame] = {}
        self.analysis_report:     Dict[str, Any]          = {}
        self.cleaning_log:        List[Dict]              = []
        self.decision_log:        List[Dict]              = []   # NO GUESSING log
        self.assumptions:         List[str]               = []   # Final Narrative
        self.remaining_risks:     List[str]               = []   # Final Narrative
        self.processed_data:      Optional[pd.DataFrame]  = None
        self._df_before:          Optional[pd.DataFrame]  = None # for before/after

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 1 — LOAD & OBSERVE
    # "No Fixing — Only Looking"
    # ══════════════════════════════════════════════════════════════════════════

    def load_files_from_directory(
        self,
        data_directory: str,
        file_types: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Load every supported file. No modifications — pure ingestion."""
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
                    print(f"[LOAD] {os.path.basename(fp)}  shape={df.shape}")
                except Exception as exc:
                    print(f"[LOAD ERROR] {fp}: {exc}")

        return self.current_dataframes

    def observe_only(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        DIAGRAM NODE: observe_only sub-flow
        load → shape_info → head_tail → columns → observe_only

        Pure observation pass. Absolutely NO modifications to data.
        Returns a structured snapshot a human or frontend can read.
        """
        observation: Dict[str, Any] = {}

        # ── shape_info ───────────────────────────────────────────────────────
        observation["shape_info"] = {
            "rows":          int(df.shape[0]),
            "cols":          int(df.shape[1]),
            "total_cells":   int(df.shape[0] * df.shape[1]),
            "memory_mb":     round(df.memory_usage(deep=True).sum() / 1024 / 1024, 3),
            "dtypes":        {c: str(t) for c, t in df.dtypes.items()},
            "dtype_summary": df.dtypes.value_counts().to_dict(),
        }

        # ── head_tail ────────────────────────────────────────────────────────
        observation["head_tail"] = {
            "head": df.head(5).to_dict(orient="records"),
            "tail": df.tail(5).to_dict(orient="records"),
        }

        # ── columns ──────────────────────────────────────────────────────────
        col_profiles: Dict[str, Any] = {}
        for col in df.columns:
            s    = df[col]
            prof: Dict[str, Any] = {
                "dtype":        str(s.dtype),
                "non_null":     int(s.notna().sum()),
                "null_count":   int(s.isna().sum()),
                "null_pct":     round(s.isna().mean() * 100, 2),
                "unique_count": int(s.nunique(dropna=True)),
                "sample_values": s.dropna().head(5).tolist(),
            }
            if pd.api.types.is_numeric_dtype(s):
                prof["stats"] = {
                    "min":    round(float(s.min()), 4) if s.notna().any() else None,
                    "max":    round(float(s.max()), 4) if s.notna().any() else None,
                    "mean":   round(float(s.mean()), 4) if s.notna().any() else None,
                    "median": round(float(s.median()), 4) if s.notna().any() else None,
                    "std":    round(float(s.std()), 4) if s.notna().any() else None,
                }
            else:
                top = s.value_counts(dropna=True).head(3)
                prof["top_values"] = top.to_dict()

            col_profiles[col] = prof

        observation["columns"] = col_profiles

        # ── observe_only flag ─────────────────────────────────────────────────
        observation["observe_only"] = True
        observation["note"]         = (
            "This is a READ-ONLY observation pass. "
            "No data has been modified. "
            "Proceed to identify_mess_types() for issue detection."
        )
        observation["observed_at"]  = datetime.now().isoformat()

        print(f"[OBSERVE] shape={df.shape}  cols={df.columns.tolist()}")
        return observation

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 2 — IDENTIFY MESS TYPES
    # ══════════════════════════════════════════════════════════════════════════

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
        sample = df.sample(min(5_000, len(df)), random_state=42)
        result: Dict[str, Any] = {}

        for col in df.columns:
            s     = sample[col].dropna()
            col_l = col.lower()

            utypes = {type(v).__name__ for v in s}
            if len(utypes) > 1:
                result[f"{col}_mixed_types"] = list(utypes)

            if any(kw in col_l for kw in _DATE_KW):
                fmts: set = set()
                for v in s.astype(str):
                    if re.match(r"\d{4}-\d{2}-\d{2}", v):        fmts.add("YYYY-MM-DD")
                    elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", v):  fmts.add("MM/DD/YYYY")
                    elif re.match(r"\d{1,2}-\d{1,2}-\d{4}", v):  fmts.add("MM-DD-YYYY")
                    elif re.match(r"\d{1,2} \w+ \d{4}", v):       fmts.add("DD Month YYYY")
                if len(fmts) > 1:
                    result[f"{col}_multiple_date_formats"] = list(fmts)

            if s.dtype == object:
                pats: List[str] = []
                for v in s.astype(str):
                    if re.match(r"^\$?\d{1,3}(,\d{3})*(\.\d+)?$", v): pats.append("comma_formatted")
                    elif re.match(r"^\$\d+(\.\d+)?$", v):              pats.append("currency_symbol")
                    elif re.match(r"^\d+$", v):                         pats.append("plain_integer")
                    elif re.match(r"^\d+\.\d+$", v):                    pats.append("decimal")
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
                bad = series[~series.astype(str).str.match(_EMAIL_PATTERN, na=False)].dropna()
                if not bad.empty:
                    broken[f"{col}_invalid_emails"] = {
                        "count": len(bad), "examples": bad.head(5).tolist()
                    }

            if any(kw in col_l for kw in ("phone", "mobile", "tel", "contact")):
                bad = series[~series.astype(str).str.match(_PHONE_PATTERN, na=False)].dropna()
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
            lo     = q1 - OUTLIER_IQR_MULTIPLIER * iqr
            hi     = q3 + OUTLIER_IQR_MULTIPLIER * iqr
            z_out  = s[z > OUTLIER_ZSCORE_THRESHOLD]
            i_out  = s[(s < lo) | (s > hi)]
            if not z_out.empty or not i_out.empty:
                report[col] = {
                    "z_score_outliers": {
                        "count": len(z_out), "examples": z_out.head(5).tolist()
                    },
                    "iqr_outliers": {
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

    def identify_mess_types(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        DIAGRAM NODE: Identify Mess Types
        Runs all 5 detectors and returns unified mess report.
        Nothing is fixed here — identification only.
        """
        print("[IDENTIFY] Scanning for all mess types…")
        mess = {
            "missing_values":         self.analyze_missing_values(df),
            "format_inconsistencies": self.analyze_format_inconsistencies(df),
            "broken_entries":         self.analyze_broken_entries(df),
            "duplicates":             self.analyze_duplicates(df),
            "contradictory_columns":  self.analyze_column_inconsistencies(df),
            "outliers":               self.analyze_outliers(df),
        }
        total_issues = (
            len(mess["missing_values"].get("total_missing_per_column", {})) +
            len(mess["format_inconsistencies"]) +
            len(mess["broken_entries"]) +
            (1 if mess["duplicates"].get("exact_duplicates", 0) > 0 else 0) +
            (1 if mess["duplicates"].get("near_duplicate_pairs") else 0) +
            len(mess["contradictory_columns"]) +
            len(mess["outliers"])
        )
        mess["total_issues_detected"] = total_issues
        print(f"[IDENTIFY] Found {total_issues} issue categories")
        self.analysis_report = mess
        return mess

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 3 — DECISION MAKING
    # "Establish Rules / Analyze Trade-offs / Document Rationale / NO GUESSING"
    # ══════════════════════════════════════════════════════════════════════════

    def _decide(self, issue: str, rule: str, rationale: str, tradeoff: str) -> None:
        """
        DIAGRAM NODE: No Guessing
        Every cleaning decision is explicitly logged before execution.
        Nothing is applied silently.
        """
        entry = {
            "issue":     issue,
            "rule":      rule,
            "rationale": rationale,
            "tradeoff":  tradeoff,
            "decided_at": datetime.now().isoformat(),
        }
        self.decision_log.append(entry)
        print(f"[DECISION] {issue}  →  {rule}")

    def make_decisions(self, df: pd.DataFrame, mess: Dict[str, Any]) -> List[Dict]:
        """
        DIAGRAM NODE: Decision Making
        Analyses the mess report and produces an explicit decision for every issue.
        Returns the decision log so the frontend can show it.
        """
        self.decision_log = []

        # ── Missing values ────────────────────────────────────────────────────
        for col, pct in mess.get("missing_values", {}).get("pct_missing_per_column", {}).items():
            if pct > MISSING_DROP_THRESHOLD * 100:
                self._decide(
                    issue     = f"missing::{col}",
                    rule      = f"DROP column '{col}'",
                    rationale = f"{pct}% missing — retaining would introduce more noise than signal",
                    tradeoff  = "Lose the column entirely; acceptable when >60% gone",
                )
                self.assumptions.append(
                    f"Column '{col}' dropped: {pct}% missing exceeded {int(MISSING_DROP_THRESHOLD*100)}% threshold."
                )
            elif col in df.select_dtypes(include=[np.number]).columns:
                if pct < 5:
                    strat = "median"
                    why   = "< 5% missing — median is robust, preserves distribution"
                elif pct < 30:
                    strat = "mean"
                    why   = "5–30% missing — mean acceptable, distribution likely symmetric"
                else:
                    strat = "median"
                    why   = "> 30% missing — median safer than mean for skewed data"
                self._decide(
                    issue     = f"missing::{col}",
                    rule      = f"IMPUTE '{col}' with {strat}",
                    rationale = why,
                    tradeoff  = f"Imputation reduces variance; {pct}% of values are synthetic",
                )
                self.assumptions.append(
                    f"Column '{col}': {pct}% missing values filled with {strat}."
                )
                if pct > 20:
                    self.remaining_risks.append(
                        f"'{col}' has {pct}% imputed values — treat model features from this column with caution."
                    )
            else:
                self._decide(
                    issue     = f"missing::{col}",
                    rule      = f"IMPUTE '{col}' with mode",
                    rationale = "Categorical column — mode preserves most-common category",
                    tradeoff  = f"Mode imputation may over-represent dominant category; {pct}% synthetic",
                )
                self.assumptions.append(
                    f"Column '{col}': {pct}% missing categorical values filled with mode."
                )

        # ── Format inconsistencies ────────────────────────────────────────────
        for key in mess.get("format_inconsistencies", {}):
            if "date_format" in key:
                self._decide(
                    issue     = f"format::{key}",
                    rule      = f"STANDARDISE all dates → ISO 8601 (YYYY-MM-DD)",
                    rationale = "Multiple date formats cause silent parse errors downstream",
                    tradeoff  = "Ambiguous dates (e.g. 01/02/03) resolved by pandas heuristic",
                )
                self.assumptions.append("All date columns standardised to ISO 8601.")
                self.remaining_risks.append(
                    f"Ambiguous date format in '{key}' — verify pandas parsed them correctly."
                )
            elif "numeric_format" in key:
                self._decide(
                    issue     = f"format::{key}",
                    rule      = "STRIP currency symbols and commas → plain float",
                    rationale = "Mixed numeric formats block type casting",
                    tradeoff  = "Currency context lost (USD vs GBP not preserved)",
                )

        # ── Broken entries ────────────────────────────────────────────────────
        for key in mess.get("broken_entries", {}):
            self._decide(
                issue     = f"broken::{key}",
                rule      = f"NULLIFY impossible values in '{key}'",
                rationale = "Impossible values (age > 120, invalid emails) are data errors, not outliers",
                tradeoff  = "Creates new NaN values — will be imputed in fix_missing step",
            )

        # ── Duplicates ────────────────────────────────────────────────────────
        if mess.get("duplicates", {}).get("exact_duplicates", 0) > 0:
            self._decide(
                issue     = "duplicates::exact",
                rule      = "DROP exact duplicate rows (keep first occurrence)",
                rationale = "Exact duplicates inflate counts and bias model training",
                tradeoff  = "If duplicates were legitimate repeated events, data is lost",
            )
            self.assumptions.append("Exact duplicate rows removed (kept first occurrence).")

        if mess.get("duplicates", {}).get("near_duplicate_pairs"):
            self._decide(
                issue     = "duplicates::near",
                rule      = f"FLAG near-duplicates (score ≥ {FUZZY_DEDUP_THRESHOLD}) in '_dsaral_near_duplicate' column",
                rationale = "Near-duplicates need human review — automated removal risks false positives",
                tradeoff  = "Flagged rows retained; human must decide removal",
            )
            self.remaining_risks.append(
                "Near-duplicate rows flagged in '_dsaral_near_duplicate' — requires human review."
            )

        # ── Outliers ──────────────────────────────────────────────────────────
        for col in mess.get("outliers", {}):
            if any(kw in col.lower() for kw in _SKIP_OUTLIER_KW):
                continue
            self._decide(
                issue     = f"outlier::{col}",
                rule      = f"CAP outliers in '{col}' at IQR × {OUTLIER_IQR_MULTIPLIER} fences",
                rationale = "Capping preserves row count while neutralising extreme influence",
                tradeoff  = "True extreme values are altered — log transformation may be better for some use cases",
            )
            self.remaining_risks.append(
                f"'{col}' outliers capped — if extremes are valid (e.g. CEO salary), consider reviewing."
            )

        # ── Contradictory columns ─────────────────────────────────────────────
        for key in mess.get("contradictory_columns", {}):
            self._decide(
                issue     = f"case::{key}",
                rule      = "STANDARDISE text case (Title / lower by column type)",
                rationale = "Case variants create phantom categories in groupby / encoding",
                tradeoff  = "Title-casing proper nouns is correct; may be wrong for abbreviations",
            )

        print(f"[DECISIONS] {len(self.decision_log)} decisions documented")
        return self.decision_log

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 4 — INCREMENTAL CLEANING
    # "One Operation at a Time — No Clean Everything Function"
    # Each method: applies ONE fix → re-checks stats → verifies shape
    # ══════════════════════════════════════════════════════════════════════════

    def fix_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 1 — Fix missing values only.
        Re-checks stats after. Verifies no unexpected row loss.
        """
        before_shape = df.shape
        df = df.copy()

        # Step 1a: placeholder → NaN
        for col in df.columns:
            df[col] = df[col].replace("", np.nan)
            mask = df[col].astype(str).str.strip().str.lower().isin(PLACEHOLDER_VALUES)
            df.loc[mask, col] = np.nan

        # Step 1b: impute per decision log
        imp_log: Dict[str, str] = {}
        for col in list(df.columns):
            n_miss = df[col].isna().sum()
            if n_miss == 0:
                continue
            pct = n_miss / len(df)

            if pct > MISSING_DROP_THRESHOLD:
                df.drop(columns=[col], inplace=True)
                imp_log[col] = f"DROPPED ({pct*100:.1f}% missing)"
                continue

            if pd.api.types.is_numeric_dtype(df[col]):
                fv  = df[col].median() if (pct < 0.05 or pct >= 0.30) else df[col].mean()
                tag = f"{"median" if (pct < 0.05 or pct >= 0.30) else "mean"} ({fv:.4g})"
                try:
                    df[col] = df[col].astype("float64")
                except Exception:
                    pass
                df[col].fillna(fv, inplace=True)
            elif pd.api.types.is_datetime64_any_dtype(df[col]):
                s   = df[col].dropna().sort_values()
                fv  = s.iloc[len(s) // 2] if len(s) else pd.NaT
                tag = f"median_date ({fv})"
                df[col].fillna(fv, inplace=True)
            else:
                modes = df[col].mode()
                fv    = modes[0] if not modes.empty else "UNKNOWN"
                tag   = f"mode ('{fv}')"
                df[col].fillna(fv, inplace=True)

            imp_log[col] = tag

        # Re-check stats after fix
        remaining_missing = df.isnull().sum().sum()
        self._recheck(df, "fix_missing", before_shape,
                      f"Imputed {len(imp_log)} cols. Remaining NaN cells: {remaining_missing}")
        return df

    def fix_formats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 2 — Fix format inconsistencies only.
        Covers: type inference, date standardisation, numeric normalisation.
        Re-checks stats after. Verifies shape unchanged.
        """
        before_shape = df.shape
        df = df.copy()

        # Encoding & whitespace first
        for col in df.select_dtypes(include="object").columns:
            df[col] = (
                df[col].astype(str)
                .str.encode("ascii", errors="ignore")
                .str.decode("ascii")
                .str.strip()
                .replace("nan", np.nan)
            )

        # Type inference: date columns first, then numeric
        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()
            if any(kw in col_l for kw in _DATE_KW):
                parsed   = pd.to_datetime(df[col], errors="coerce")
                hit_rate = parsed.notna().sum() / max(df[col].notna().sum(), 1)
                if hit_rate > 0.7:
                    df[col] = parsed
                    continue
            # Numeric normalisation: strip $, commas
            if not any(kw in col_l for kw in
                       ("name", "email", "url", "address", "description",
                        "comment", "note", "text", "label", "tag")):
                cleaned  = df[col].astype(str).str.replace(r"[^\d.\-]", "", regex=True).replace("", np.nan)
                as_num   = pd.to_numeric(cleaned, errors="coerce")
                hit_rate = as_num.notna().sum() / max(df[col].notna().sum(), 1)
                if hit_rate >= 0.70:
                    df[col] = as_num

        # Date → ISO 8601
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

        self._recheck(df, "fix_formats", before_shape,
                      "Type inference + date ISO 8601 + numeric normalisation applied")
        assert df.shape[0] == before_shape[0], \
            f"[fix_formats] Row count changed! {before_shape[0]} → {df.shape[0]}"
        return df

    def fix_broken_entries(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 3 — Nullify impossible / domain-invalid values only.
        Covers: age range, email regex, phone regex, outlier capping.
        Re-checks stats after. Verifies shape unchanged.
        """
        before_shape = df.shape
        df = df.copy()

        # Domain validation → NaN
        for col in df.columns:
            col_l = col.lower()
            if "age" in col_l and pd.api.types.is_numeric_dtype(df[col]):
                df.loc[(df[col] < 0) | (df[col] > 120), col] = np.nan
            if "email" in col_l:
                df.loc[~df[col].astype(str).str.match(_EMAIL_PATTERN, na=False), col] = np.nan
            if any(kw in col_l for kw in ("phone", "mobile", "tel")):
                df.loc[~df[col].astype(str).str.match(_PHONE_PATTERN, na=False), col] = np.nan

        # Outlier capping (IQR)
        caps: Dict[str, Any] = {}
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

        self.cleaning_log.append({"step": "fix_broken_entries::outlier_caps", "details": caps})
        self._recheck(df, "fix_broken_entries", before_shape,
                      f"Domain validation done. Outliers capped in {len(caps)} cols.")
        assert df.shape[0] == before_shape[0], \
            f"[fix_broken_entries] Row count changed! {before_shape[0]} → {df.shape[0]}"
        return df

    def fix_text_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 4 — Standardise text case only.
        Covers: Title case for names/depts, lowercase for emails/urls.
        Re-checks stats after. Verifies shape unchanged.
        """
        before_shape = df.shape
        df = df.copy()

        for col in df.select_dtypes(include="object").columns:
            col_l = col.lower()
            if any(kw in col_l for kw in _TITLE_KW):
                df[col] = df[col].str.title()
            elif any(kw in col_l for kw in _LOWER_KW):
                df[col] = df[col].str.lower()

        self._recheck(df, "fix_text_consistency", before_shape,
                      "Text case standardised (Title / lower)")
        assert df.shape[0] == before_shape[0], \
            f"[fix_text_consistency] Row count changed!"
        return df

    def fix_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 5 — Remove exact duplicates. Flag near-duplicates.
        Re-checks stats after.
        """
        before_shape = df.shape
        df = df.copy()

        # Exact
        before = len(df)
        df.drop_duplicates(inplace=True)
        df.reset_index(drop=True, inplace=True)
        removed = before - len(df)

        # Fuzzy near-dup flagging
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

        self._recheck(df, "fix_duplicates", before_shape,
                      f"Removed {removed} exact dups. Flagged {n_flagged} near-dups.")
        return df

    def fix_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        INCREMENTAL STEP 6 — Normalise column names to snake_case.
        Re-checks stats after. Verifies shape fully unchanged.
        """
        before_shape = df.shape
        df = df.copy()

        df.columns = [
            re.sub(r"\s+", "_", c.strip().lower())
               .replace("-", "_")
               .replace(".", "_")
            for c in df.columns
        ]

        # Remove near-constant columns
        dropped: List[str] = []
        for col in list(df.columns):
            if col.startswith("_dsaral_"):
                continue
            vc = df[col].value_counts(normalize=True, dropna=False)
            if len(vc) and vc.iloc[0] >= NEAR_CONSTANT_THRESHOLD:
                df.drop(columns=[col], inplace=True)
                dropped.append(col)

        # Final type enforcement
        for col in df.select_dtypes(include=[np.number]).columns:
            s = df[col].dropna()
            if len(s) and s.apply(lambda x: float(x).is_integer()).all():
                try:
                    df[col] = df[col].astype("Int64")
                except Exception:
                    pass

        self._recheck(df, "fix_column_names", before_shape,
                      f"snake_case names. Dropped {len(dropped)} near-constant cols: {dropped}. Types enforced.")
        return df

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 5 — VALIDATION (with feedback loop)
    # "Re-run .info() → Check Missing → Look for New Anomalies
    #  → Before/After Comparison → if Issues Found: loop back"
    # ══════════════════════════════════════════════════════════════════════════

    def validate(
        self,
        df_before: pd.DataFrame,
        df_after:  pd.DataFrame,
        loop_number: int = 1,
    ) -> Dict[str, Any]:
        """
        DIAGRAM NODE: Validation
        Runs full re-analysis on cleaned df. If issues remain, flags them.
        Returns validation report + 'issues_remain' bool for loop control.
        """
        print(f"[VALIDATE] Loop {loop_number} — re-running .info() + checks…")

        # Re-run info (equivalent to df.info())
        info: Dict[str, Any] = {
            "shape":      df_after.shape,
            "dtypes":     {c: str(t) for c, t in df_after.dtypes.items()},
            "memory_mb":  round(df_after.memory_usage(deep=True).sum() / 1024 / 1024, 3),
            "non_null":   {c: int(df_after[c].notna().sum()) for c in df_after.columns},
        }

        # Check missing values
        remaining_missing = self.analyze_missing_values(df_after)

        # Look for new anomalies
        remaining_broken  = self.analyze_broken_entries(df_after)
        remaining_outliers = self.analyze_outliers(df_after)

        # Before / after comparison
        comparison: Dict[str, Any] = {
            "rows_before":    int(df_before.shape[0]),
            "rows_after":     int(df_after.shape[0]),
            "rows_removed":   int(df_before.shape[0] - df_after.shape[0]),
            "cols_before":    int(df_before.shape[1]),
            "cols_after":     int(df_after.shape[1]),
            "cols_removed":   int(df_before.shape[1] - df_after.shape[1]),
            "missing_before": int(df_before.isnull().sum().sum()),
            "missing_after":  int(df_after.isnull().sum().sum()),
            "duplicates_before": int(df_before.duplicated().sum()),
            "duplicates_after":  int(df_after.duplicated().sum()),
            "pct_data_retained": round(df_after.shape[0] / max(df_before.shape[0], 1) * 100, 2),
        }

        # Decide if issues remain (triggers loop-back)
        issues_remain = bool(
            remaining_missing.get("total_missing_per_column") or
            remaining_broken or
            (loop_number < MAX_VALIDATION_LOOPS and
             any(v["iqr_outliers"]["count"] > 0
                 for v in remaining_outliers.values()))
        )

        validation_report = {
            "loop_number":       loop_number,
            "info":              info,
            "remaining_missing": remaining_missing,
            "remaining_broken":  remaining_broken,
            "remaining_outliers": remaining_outliers,
            "before_after":      comparison,
            "issues_remain":     issues_remain,
            "validated_at":      datetime.now().isoformat(),
        }

        status = "⚠ Issues remain — looping back" if issues_remain else "✓ Clean"
        print(f"[VALIDATE] Loop {loop_number} complete. Status: {status}")
        return validation_report

    # ══════════════════════════════════════════════════════════════════════════
    # STAGE 6 — FINAL NARRATIVE
    # "Single story / Clear assumptions / Remaining risks / Professional docs"
    # ══════════════════════════════════════════════════════════════════════════

    def generate_final_narrative(
        self,
        df_before:         pd.DataFrame,
        df_after:          pd.DataFrame,
        validation_report: Dict[str, Any],
    ) -> str:
        """
        DIAGRAM NODE: Final Narrative
        Produces the complete professional documentation:
        - complete story of what was done
        - every assumption made explicit
        - remaining risks documented
        - before/after comparison
        """
        lines: List[str] = []
        ba = validation_report["before_after"]
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines += [
            "# D-SARAL  —  Messy Data Autopsy Report",
            f"Generated : {ts}",
            "",
            "---",
            "",
            "## 1. Complete Story",
            "",
            "This dataset was processed through D-SARAL's Messy Data Autopsy Workflow:",
            "",
            "| Stage | What happened |",
            "|---|---|",
            "| Load & Observe | Data loaded and profiled without any modification |",
            "| Identify Mess Types | All issue categories scanned: missing, formats, broken entries, duplicates, contradictions |",
            f"| Decision Making | {len(self.decision_log)} explicit decisions made — nothing applied silently |",
            "| Incremental Cleaning | 6 independent fix steps applied one at a time, each validated before the next |",
            f"| Validation | {validation_report['loop_number']} validation loop(s) run; before/after compared |",
            "",
        ]

        # Before/After
        lines += [
            "## 2. Before / After Comparison",
            "",
            f"| Metric | Before | After |",
            f"|---|---|---|",
            f"| Rows | {ba['rows_before']} | {ba['rows_after']} (−{ba['rows_removed']}) |",
            f"| Columns | {ba['cols_before']} | {ba['cols_after']} (−{ba['cols_removed']}) |",
            f"| Missing cells | {ba['missing_before']} | {ba['missing_after']} |",
            f"| Exact duplicates | {ba['duplicates_before']} | {ba['duplicates_after']} |",
            f"| Data retained | — | {ba['pct_data_retained']} % |",
            "",
        ]

        # Decisions made
        lines += [
            "## 3. Decisions Made  (No Guessing)",
            "",
            "Every cleaning action below was decided explicitly before execution:",
            "",
        ]
        for d in self.decision_log:
            lines.append(f"**Issue:** {d['issue']}")
            lines.append(f"- Rule      : {d['rule']}")
            lines.append(f"- Rationale : {d['rationale']}")
            lines.append(f"- Trade-off : {d['tradeoff']}")
            lines.append("")

        # Assumptions
        lines += ["## 4. Assumptions Made", ""]
        if self.assumptions:
            for a in self.assumptions:
                lines.append(f"- {a}")
        else:
            lines.append("- No assumptions were required.")
        lines.append("")

        # Remaining risks
        lines += ["## 5. Remaining Risks", ""]
        if self.remaining_risks:
            for r in self.remaining_risks:
                lines.append(f"- ⚠ {r}")
        else:
            lines.append("- No significant risks identified.")
        lines.append("")

        # Remaining issues after validation
        rm = validation_report.get("remaining_missing", {}).get("total_missing_per_column", {})
        rb = validation_report.get("remaining_broken", {})
        if rm or rb:
            lines += ["## 6. Issues Not Fully Resolved", ""]
            for col, cnt in rm.items():
                lines.append(f"- Missing: '{col}' still has {cnt} null values")
            for k, v in rb.items():
                lines.append(f"- Broken:  '{k}' — {v.get('count', '?')} entries remain invalid")
            lines.append("")
        else:
            lines += ["## 6. Issues Not Fully Resolved", "", "- None. Dataset is fully clean.", ""]

        lines += [
            "---",
            "",
            "*Generated by D-SARAL  —  Messy Data Autopsy Workflow*",
        ]

        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════════════════
    # ORCHESTRATOR  —  runs the full workflow in sequence with validation loop
    # ══════════════════════════════════════════════════════════════════════════

    def comprehensive_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Backward-compatible wrapper — returns full analysis dict."""
        return self.identify_mess_types(df)

    def document_issues_with_examples(self, df: pd.DataFrame) -> str:
        """Backward-compatible wrapper — returns text report of issues found."""
        mess = self.identify_mess_types(df)
        lines: List[str] = [
            "# D-SARAL Data Quality Issues Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total issue categories found: {mess['total_issues_detected']}",
            "",
        ]
        for section, content in mess.items():
            if section == "total_issues_detected":
                continue
            if isinstance(content, dict) and content:
                lines.append(f"## {section.replace('_', ' ').title()}")
                for k, v in content.items():
                    lines.append(f"- **{k}**: {v}")
                lines.append("")
        return "\n".join(lines)

    def apply_cleaning_techniques(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        FULL WORKFLOW ORCHESTRATOR
        Follows the diagram exactly:
          observe → identify → decide → incremental clean → validate (loop) → narrative
        """
        self._df_before  = df.copy()
        self.assumptions = []
        self.remaining_risks = []

        # Stage 1: Observe (no touching)
        self.observe_only(df)

        # Stage 2: Identify
        mess = self.identify_mess_types(df)

        # Stage 3: Decide
        self.make_decisions(df, mess)

        # Stage 4: Incremental cleaning (one operation at a time)
        df = self.fix_missing(df)
        df = self.fix_formats(df)
        df = self.fix_broken_entries(df)
        df = self.fix_text_consistency(df)
        df = self.fix_duplicates(df)
        df = self.fix_column_names(df)

        # Stage 5: Validation loop
        loop      = 1
        val_report = self.validate(self._df_before, df, loop)

        while val_report["issues_remain"] and loop < MAX_VALIDATION_LOOPS:
            print(f"[LOOP] Issues remain — re-running incremental cleaning (loop {loop + 1})")
            df = self.fix_missing(df)
            df = self.fix_broken_entries(df)
            loop      += 1
            val_report = self.validate(self._df_before, df, loop)

        # Stage 6: Final narrative
        self.final_narrative = self.generate_final_narrative(
            self._df_before, df, val_report
        )

        self.processed_data   = df
        self.analysis_report  = mess
        print(f"[COMPLETE] Final shape: {df.shape}")
        return df

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _recheck(self, df: pd.DataFrame, step: str, orig_shape, msg: str) -> None:
        """
        DIAGRAM NODE: Re-check Statistics after each incremental step.
        Logs shape, missing count, dtype summary.
        """
        n_missing = int(df.isnull().sum().sum())
        entry = {
            "step":          step,
            "original_shape": list(orig_shape),
            "current_shape":  list(df.shape),
            "missing_cells":  n_missing,
            "dtype_counts":   df.dtypes.value_counts().to_dict(),
            "message":        msg,
            "timestamp":      datetime.now().isoformat(),
        }
        self.cleaning_log.append(entry)
        print(f"[{step}] {msg}  |  shape: {df.shape}  |  missing_cells: {n_missing}")

    def _log(self, step: str, orig, cur, msg: str) -> None:
        """Legacy log helper — kept for backward compatibility."""
        self._recheck(
            pd.DataFrame(index=range(cur[0]), columns=range(cur[1])),
            step, orig, msg
        )