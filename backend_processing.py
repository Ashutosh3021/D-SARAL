"""
Backend Processing Module for D-SARAL Data Cleaning & Analysis Tool
Contains all data processing logic, file handling, and cleaning pipeline implementation
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings
import json
import tempfile
from werkzeug.utils import secure_filename

warnings.filterwarnings("ignore")

class DataProcessingPipeline:
    """
    Main class for data processing pipeline in D-SARAL
    Handles loading, analyzing, cleaning, and validating data
    """
    
    def __init__(self, sample_size: int = 10000):
        """
        Initialize the data processing pipeline
        
        Args:
            sample_size: Number of rows to sample from large files for analysis
        """
        self.sample_size = sample_size
        self.current_dataframes = {}
        self.analysis_report = {}
        self.cleaning_log = []
        self.processed_data = None
        
    def load_files_from_directory(self, data_directory: str, file_types: list = None) -> Dict[str, pd.DataFrame]:
        """
        Load all supported files from the specified directory
        
        Args:
            data_directory: Path to the directory containing files
            file_types: List of file extensions to load (e.g., ['csv', 'json'])
            
        Returns:
            Dictionary mapping file paths to loaded DataFrames
        """
        if file_types is None:
            file_types = ['csv', 'json', 'txt']
        
        self.current_dataframes = {}
        
        # Find all supported files recursively
        for ext in file_types:
            pattern = f"*.{ext}"
            files = glob.glob(os.path.join(data_directory, "**", pattern), recursive=True)
            
            for file_path in files:
                try:
                    if ext.lower() == 'csv':
                        df = pd.read_csv(file_path, nrows=self.sample_size if os.path.getsize(file_path) > 1000000 else None)
                    elif ext.lower() == 'json':
                        df = pd.read_json(file_path, nrows=self.sample_size if os.path.getsize(file_path) > 1000000 else None)
                    elif ext.lower() == 'txt':
                        # Try to infer delimiter
                        if os.path.getsize(file_path) > 1000000:
                            # For large files, read a sample first to infer the delimiter, then apply nrows
                            sample_df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip', nrows=self.sample_size)
                            df = sample_df  # Just use the sample for now
                        else:
                            df = pd.read_csv(file_path, sep=None, engine='python', on_bad_lines='skip')
                    else:
                        continue
                        
                    self.current_dataframes[file_path] = df
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
        
        return self.current_dataframes
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in the dataframe efficiently
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value analysis
        """
        missing_analysis = {}
        
        # Standard NaN values
        standard_nans = df.isnull().sum()
        missing_analysis["standard_nan_counts"] = standard_nans[standard_nans > 0].to_dict()
        
        # Empty strings
        empty_strings = df.apply(lambda col: col.astype(str).str.strip().eq("").sum())
        missing_analysis["empty_string_counts"] = empty_strings[empty_strings > 0].to_dict()
        
        # Placeholder values that indicate missing data
        placeholder_patterns = ["N/A", "NA", "NULL", "null", "nan", "NaN", "None", "empty", "missing", "Unknown", "unknown"]
        placeholder_counts = {}
        
        for col in df.columns:
            placeholder_count = 0
            # Process in chunks to avoid memory issues with large datasets
            chunk_size = min(5000, len(df))
            for i in range(0, len(df), chunk_size):
                chunk = df[col].iloc[i:i+chunk_size]
                for pattern in placeholder_patterns:
                    mask = chunk.astype(str).str.lower() == pattern.lower()
                    placeholder_count += mask.sum()
            
            if placeholder_count > 0:
                placeholder_counts[col] = placeholder_count
        
        missing_analysis["placeholder_counts"] = placeholder_counts
        
        # Total missing values per column
        total_missing = {}
        for col in df.columns:
            total = 0
            # Count standard NaNs
            total += df[col].isnull().sum()
            # Count empty strings
            total += df[col].astype(str).str.strip().eq("").sum()
            # Count placeholders in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df[col].iloc[i:i+chunk_size]
                for pattern in placeholder_patterns:
                    total += (chunk.astype(str).str.lower() == pattern.lower()).sum()
            
            if total > 0:
                total_missing[col] = total
        
        missing_analysis["total_missing_per_column"] = total_missing
        
        return missing_analysis
    
    def analyze_format_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze inconsistent formats in the dataframe efficiently
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with format inconsistency analysis
        """
        format_analysis = {}
        chunk_size = min(5000, len(df))  # Process in chunks
        
        for col in df.columns:
            series = df[col]
            
            # Check for mixed data types (sample for large datasets)
            if len(series) > chunk_size:
                sampled_series = series.sample(min(chunk_size, len(series)), random_state=42)
            else:
                sampled_series = series
            
            unique_types = set()
            for val in sampled_series.dropna():
                unique_types.add(type(val).__name__)
            
            if len(unique_types) > 1:
                format_analysis[f"{col}_mixed_types"] = list(unique_types)
            
            # Check for inconsistent date formats
            if "date" in col.lower() or "birth" in col.lower() or "time" in col.lower():
                date_formats = set()
                valid_dates = 0
                
                # Sample data for processing large datasets
                if len(series) > chunk_size:
                    date_sample = series.dropna().sample(min(chunk_size, len(series.dropna())), random_state=42)
                else:
                    date_sample = series.dropna()
                
                for val in date_sample:
                    try:
                        parsed_date = pd.to_datetime(val, errors="coerce")
                        if pd.notna(parsed_date):
                            valid_dates += 1
                            # Try to detect common formats
                            if isinstance(val, str):
                                if "/" in val:
                                    date_formats.add("MM/DD/YYYY or DD/MM/YYYY")
                                elif "-" in val:
                                    date_formats.add("YYYY-MM-DD or MM-DD-YYYY")
                                elif re.match(r"\d{1,2}/\d{1,2}/\d{4}", val):
                                    date_formats.add("MM/DD/YYYY")
                                elif re.match(r"\d{4}-\d{2}-\d{2}", val):
                                    date_formats.add("YYYY-MM-DD")
                                elif re.match(r"\d{1,2}-\d{1,2}-\d{4}", val):
                                    date_formats.add("MM-DD-YYYY")
                    
                    except:
                        continue
                
                if len(date_formats) > 1:
                    format_analysis[f"{col}_multiple_date_formats"] = list(date_formats)
            
            # Check for inconsistent number formats (with currency symbols, commas, etc.) - sample for large datasets
            if len(series) > chunk_size:
                numeric_sample = series.dropna().sample(min(chunk_size, len(series.dropna())), random_state=42)
            else:
                numeric_sample = series.dropna()
            
            numeric_like = numeric_sample.apply(lambda x: str(x)).astype(str)
            numeric_patterns = []
            
            for val in numeric_like:
                if pd.api.types.is_numeric_dtype(series.dtype):
                    continue
                elif re.match(r"^\$?\d{1,3}(,\d{3})*(\.\d+)?$", val):  # Matches $1,000.00 or 1,000.00
                    numeric_patterns.append("formatted_number_with_commas")
                elif re.match(r"^\$\d+(\.\d+)?$", val):  # Matches $50000
                    numeric_patterns.append("currency_format")
                elif re.match(r"^\d+$", val):  # Matches 50000
                    numeric_patterns.append("plain_number")
                elif re.match(r"^\d+\.\d+$", val):  # Matches 50000.00
                    numeric_patterns.append("decimal_number")
            
            if len(set(numeric_patterns)) > 1:
                format_analysis[f"{col}_multiple_numeric_formats"] = list(set(numeric_patterns))
        
        return format_analysis
    
    def analyze_broken_entries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze broken entries in the dataframe (impossible values, etc.) efficiently
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with broken entry analysis
        """
        broken_analysis = {}
        chunk_size = min(5000, len(df))  # Process in chunks
        
        for col in df.columns:
            series = df[col]
            
            # Check for impossible age values
            if "age" in col.lower():
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(series, errors="coerce")
                invalid_ages = numeric_series[(numeric_series < 0) | (numeric_series > 150)]
                
                if len(invalid_ages) > 0:
                    # Limit results for large datasets
                    sample_invalid = invalid_ages.dropna().head(100) if len(invalid_ages) > 100 else invalid_ages.dropna()
                    broken_analysis[f"{col}_invalid_ages"] = {
                        "count": len(invalid_ages),
                        "values": sample_invalid.tolist(),
                        "indices": sample_invalid.index.tolist()
                    }
            
            # Check for invalid email formats
            if "email" in col.lower():
                # Process in chunks for large datasets
                invalid_emails_list = []
                invalid_indices_list = []
                
                for i in range(0, len(series), chunk_size):
                    chunk = series.iloc[i:i+chunk_size]
                    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                    chunk_invalid_emails = chunk[~chunk.astype(str).str.contains(email_pattern, na=False)]
                    
                    invalid_emails_list.extend(chunk_invalid_emails.tolist())
                    invalid_indices_list.extend(chunk_invalid_emails.index.tolist())
                    
                    # Limit results for reporting
                    if len(invalid_emails_list) > 100:
                        break
                
                if len(invalid_emails_list) > 0:
                    broken_analysis[f"{col}_invalid_emails"] = {
                        "count": min(len(invalid_emails_list), 100),
                        "values": invalid_emails_list[:100],
                        "indices": invalid_indices_list[:100]
                    }
            
            # Check for impossible date values
            if "date" in col.lower() or "birth" in col.lower():
                # Try to parse dates and check for impossible values
                parsed_dates = pd.to_datetime(series, errors="coerce")
                invalid_dates = parsed_dates[pd.isna(parsed_dates) & pd.notna(series)]
                
                if len(invalid_dates) > 0:
                    # Limit results for large datasets
                    sample_invalid = invalid_dates.head(100) if len(invalid_dates) > 100 else invalid_dates
                    broken_analysis[f"{col}_invalid_dates"] = {
                        "count": len(invalid_dates),
                        "values": series[pd.isna(parsed_dates) & pd.notna(series)].head(100).tolist(),
                        "indices": sample_invalid.index.tolist()
                    }
                
                # Check for future dates in birth columns
                if "birth" in col.lower():
                    future_births = parsed_dates[parsed_dates > datetime.now()]
                    if len(future_births) > 0:
                        # Limit results for large datasets
                        sample_future = future_births.head(100) if len(future_births) > 100 else future_births
                        broken_analysis[f"{col}_future_birth_dates"] = {
                            "count": len(future_births),
                            "values": sample_future.dropna().dt.strftime("%Y-%m-%d").tolist(),
                            "indices": sample_future.dropna().index.tolist()
                        }
        
        return broken_analysis
    
    def analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze duplicate records in the dataframe efficiently
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with duplicate analysis
        """
        duplicate_analysis = {}
        
        # Overall duplicates - sample if dataset is too large
        if len(df) > 50000:  # For very large datasets, use sampling approach
            # Take a sample to estimate duplicates
            sample_size = min(10000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            total_duplicates = int((sample_df.duplicated().sum() / sample_size) * len(df))
            duplicate_analysis["total_row_duplicates"] = total_duplicates
            duplicate_analysis["estimated_from_sample"] = True
            
            # Get actual duplicate indices from a smaller sample for examples
            actual_duplicates = df.head(5000).duplicated(keep=False)
            duplicate_indices = df.head(5000)[actual_duplicates].index.tolist()
            duplicate_analysis["duplicate_row_indices"] = duplicate_indices[:10]  # Limit to first 10 for brevity
        else:
            # For smaller datasets, analyze completely
            total_duplicates = df.duplicated().sum()
            duplicate_analysis["total_row_duplicates"] = total_duplicates
            duplicate_analysis["estimated_from_sample"] = False
            
            if total_duplicates > 0:
                duplicate_indices = df[df.duplicated(keep=False)].index.tolist()
                duplicate_analysis["duplicate_row_indices"] = duplicate_indices[:10]  # Limit to first 10 for brevity
        
        # Check for near-duplicates based on specific columns that should be unique
        likely_unique_cols = []
        
        # Heuristic: columns with 'id', 'email', 'name' might be expected to be unique
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ["id", "email", "name"]):
                # For large datasets, sample the column duplicate analysis
                if len(df) > 25000:
                    sample_col_df = df[[col]].sample(n=min(25000, len(df)), random_state=42)
                    col_duplicates = sample_col_df[sample_col_df.duplicated(subset=[col], keep=False)][col].value_counts()
                else:
                    col_duplicates = df[df.duplicated(subset=[col], keep=False)][col].value_counts()
                
                if len(col_duplicates) > 0:
                    likely_unique_cols.append({
                        "column": col,
                        "duplicate_counts": col_duplicates[col_duplicates > 1].to_dict()
                    })
        
        duplicate_analysis["likely_unique_column_issues"] = likely_unique_cols
        
        return duplicate_analysis
    
    def analyze_column_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze column inconsistencies or contradictions efficiently
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with column inconsistency analysis
        """
        inconsistency_analysis = {}
        
        # Check for categorical inconsistencies (case, spelling, etc.)
        categorical_columns = df.select_dtypes(include=["object"]).columns
        
        for col in categorical_columns:
            # Sample for large datasets
            series = df[col]
            if len(series) > 25000:
                sample_series = series.sample(n=min(25000, len(series)), random_state=42)
                unique_values = sample_series.dropna().unique()
            else:
                unique_values = series.dropna().unique()
            
            # Check for case inconsistencies
            lowercase_values = [str(val).lower() for val in unique_values if pd.notna(val)]
            original_to_lowercase = {str(val): str(val).lower() for val in unique_values if pd.notna(val)}
            
            # Find values that differ only by case
            lowercase_counts = pd.Series(lowercase_values).value_counts()
            case_variants = lowercase_counts[lowercase_counts > 1]
            
            if len(case_variants) > 0:
                case_inconsistencies = {}
                for case_val in case_variants.index:
                    original_forms = [orig for orig, lower in original_to_lowercase.items() if lower == case_val]
                    if len(original_forms) > 1:
                        case_inconsistencies[case_val] = original_forms[:10]  # Limit to first 10 variants
                
                if case_inconsistencies:
                    inconsistency_analysis[f"{col}_case_inconsistencies"] = case_inconsistencies
        
        return inconsistency_analysis
    
    def comprehensive_data_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the provided dataframe
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with complete analysis report
        """
        # Perform all analyses
        analysis_results = {
            "dataset_overview": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.to_dict(),
                "memory_usage": df.memory_usage(deep=True).sum()
            },
            "missing_values": self.analyze_missing_values(df),
            "format_inconsistencies": self.analyze_format_inconsistencies(df),
            "broken_entries": self.analyze_broken_entries(df),
            "duplicates": self.analyze_duplicates(df),
            "column_inconsistencies": self.analyze_column_inconsistencies(df)
        }
        
        self.analysis_report = analysis_results
        return analysis_results
    
    def document_issues_with_examples(self, df: pd.DataFrame) -> str:
        """
        Create a detailed report documenting all issues with specific examples
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            String with detailed issue documentation
        """
        if df is None:
            raise ValueError("DataFrame cannot be None. Provide a valid DataFrame for analysis.")
        
        # Perform comprehensive analysis
        self.comprehensive_data_analysis(df)
        
        report = []
        report.append("# Data Quality Issues Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overview
        overview = self.analysis_report["dataset_overview"]
        report.append(f"## Dataset Overview\n")
        report.append(f"- Shape: {overview['shape'][0]} rows × {overview['shape'][1]} columns")
        report.append(f"- Memory usage: {overview['memory_usage']} bytes ({overview['memory_usage']/1024/1024:.2f} MB)")
        report.append(f"- Columns: {', '.join(overview['columns'])}\n")
        
        # Missing Values
        missing_vals = self.analysis_report["missing_values"]
        if missing_vals["total_missing_per_column"]:
            report.append(f"## Missing Values\n")
            for col, count in missing_vals["total_missing_per_column"].items():
                report.append(f"- **{col}**: {count} missing values")
                
                # Show examples of missing values
                missing_mask = (
                    df[col].isnull() |
                    (df[col].astype(str).str.strip() == "") |
                    (df[col].astype(str).str.lower().isin(["n/a", "na", "null", "none", "empty", "missing", "unknown"]))
                )
                examples = df[missing_mask][col].head(3).tolist()
                report.append(f"  - Examples: {examples}")
            report.append("")
        
        # Format Inconsistencies
        format_issues = self.analysis_report["format_inconsistencies"]
        if format_issues:
            report.append(f"## Format Inconsistencies\n")
            for issue_type, details in format_issues.items():
                report.append(f"- **{issue_type}**: {details}")
            report.append("")
        
        # Broken Entries
        broken_entries = self.analysis_report["broken_entries"]
        if broken_entries:
            report.append(f"## Broken Entries\n")
            for issue_type, details in broken_entries.items():
                report.append(f"- **{issue_type}**: {details.get('count', 'N/A')} problematic entries")
                if 'values' in details:
                    report.append(f"  - Examples: {details['values'][:3]}")  # Show first 3 examples
            report.append("")
        
        # Duplicates
        duplicates = self.analysis_report["duplicates"]
        if duplicates.get("total_row_duplicates", 0) > 0:
            report.append(f"## Duplicates\n")
            report.append(f"- Total row duplicates: {duplicates['total_row_duplicates']}")
            if duplicates.get("estimated_from_sample"):
                report.append("- Note: This is an estimate from sampling due to large dataset size")
            if "duplicate_row_indices" in duplicates:
                report.append(f"  - First few duplicate indices: {duplicates['duplicate_row_indices']}")
            report.append("")
        
        # Column Inconsistencies
        col_inconsistencies = self.analysis_report["column_inconsistencies"]
        if col_inconsistencies:
            report.append(f"## Column Inconsistencies\n")
            for issue_type, details in col_inconsistencies.items():
                report.append(f"- **{issue_type}**: {details}")
            report.append("")
        
        return "\n".join(report)
    
    def apply_cleaning_techniques(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply appropriate cleaning techniques to create a refined dataset
        
        Args:
            df: DataFrame to clean
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            raise ValueError("DataFrame cannot be None. Provide a valid DataFrame for cleaning.")
        
        original_shape = df.shape
        cleaned_df = df.copy()
        
        print("Applying cleaning techniques...")
        
        # For very large datasets, process in chunks
        chunk_size = min(10000, len(df))
        
        # 1. Handle missing values
        print("- Handling missing values...")
        for col in cleaned_df.columns:
            # Replace common placeholders with NaN
            cleaned_df[col] = cleaned_df[col].replace(["N/A", "NA", "NULL", "null", "nan", "NaN", "None", "empty", "missing", "Unknown", "unknown"], np.nan)
            
            # Also handle empty strings
            cleaned_df[col] = cleaned_df[col].replace("", np.nan)
        
        # 2. Fix format inconsistencies
        print("- Fixing format inconsistencies...")
        for col in cleaned_df.columns:
            # Standardize text case for categorical columns
            if cleaned_df[col].dtype == "object":
                # If the column seems categorical based on name or value patterns
                if any(keyword in col.lower() for keyword in ["name", "department", "category", "type", "status"]):
                    # Process in chunks for large datasets
                    if len(cleaned_df) > chunk_size:
                        for i in range(0, len(cleaned_df), chunk_size):
                            chunk_end = min(i + chunk_size, len(cleaned_df))
                            cleaned_df.loc[i:chunk_end-1, col] = cleaned_df.loc[i:chunk_end-1, col].str.title()
                    else:
                        cleaned_df[col] = cleaned_df[col].str.title()  # Title case for names/categories
                    
                    # Handle case variations specifically
                    case_mappings = {}
                    unique_vals = cleaned_df[col].dropna().unique()
                    for val in unique_vals:
                        if pd.notna(val):
                            lowercase_val = str(val).lower()
                            # Group similar values by lowercase
                            if lowercase_val not in case_mappings:
                                case_mappings[lowercase_val] = str(val)  # Keep first encountered format
                    
                    # Map all variations to standard format
                    if len(cleaned_df) > chunk_size:
                        # Process in chunks to avoid memory issues
                        for i in range(0, len(cleaned_df), chunk_size):
                            chunk_end = min(i + chunk_size, len(cleaned_df))
                            cleaned_df.loc[i:chunk_end-1, col] = cleaned_df.loc[i:chunk_end-1, col].apply(
                                lambda x: case_mappings.get(str(x).lower(), x) if pd.notna(x) else x
                            )
                    else:
                        cleaned_df[col] = cleaned_df[col].apply(
                            lambda x: case_mappings.get(str(x).lower(), x) if pd.notna(x) else x
                        )
            
            # Fix numeric columns stored as strings
            if "salary" in col.lower() or "income" in col.lower() or "amount" in col.lower():
                # Remove currency symbols and commas, convert to numeric
                # Process in chunks for large datasets
                if len(cleaned_df) > chunk_size:
                    for i in range(0, len(cleaned_df), chunk_size):
                        chunk_end = min(i + chunk_size, len(cleaned_df))
                        cleaned_df.loc[i:chunk_end-1, col] = cleaned_df.loc[i:chunk_end-1, col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                        cleaned_df.loc[i:chunk_end-1, col] = pd.to_numeric(cleaned_df.loc[i:chunk_end-1, col], errors="coerce")
                else:
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace(r"[^\d.-]", "", regex=True)
                    cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors="coerce")
        
        # 3. Fix broken entries
        print("- Fixing broken entries...")
        for col in cleaned_df.columns:
            if "age" in col.lower():
                # Cap age values to reasonable range
                if len(cleaned_df) > chunk_size:
                    for i in range(0, len(cleaned_df), chunk_size):
                        chunk_end = min(i + chunk_size, len(cleaned_df))
                        numeric_age = pd.to_numeric(cleaned_df.loc[i:chunk_end-1, col], errors="coerce")
                        cleaned_df.loc[i:chunk_end-1, col] = numeric_age.apply(lambda x: x if 0 <= x <= 120 else np.nan)
                else:
                    numeric_age = pd.to_numeric(cleaned_df[col], errors="coerce")
                    cleaned_df[col] = numeric_age.apply(lambda x: x if 0 <= x <= 120 else np.nan)
            
            if "email" in col.lower():
                # Keep only valid email formats
                email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                if len(cleaned_df) > chunk_size:
                    for i in range(0, len(cleaned_df), chunk_size):
                        chunk_end = min(i + chunk_size, len(cleaned_df))
                        valid_email_mask = cleaned_df.loc[i:chunk_end-1, col].astype(str).str.contains(email_pattern, na=False)
                        cleaned_df.loc[i:chunk_end-1, ~valid_email_mask, col] = np.nan
                else:
                    valid_email_mask = cleaned_df[col].astype(str).str.contains(email_pattern, na=False)
                    cleaned_df.loc[~valid_email_mask, col] = np.nan
            
            if "date" in col.lower() or "birth" in col.lower():
                # Parse dates consistently
                if len(cleaned_df) > chunk_size:
                    for i in range(0, len(cleaned_df), chunk_size):
                        chunk_end = min(i + chunk_size, len(cleaned_df))
                        cleaned_df.loc[i:chunk_end-1, col] = pd.to_datetime(cleaned_df.loc[i:chunk_end-1, col], errors="coerce")
                else:
                    cleaned_df[col] = pd.to_datetime(cleaned_df[col], errors="coerce")
        
        # 4. Handle duplicates
        print("- Handling duplicates...")
        print("  Checking for duplicates...")
        
        # For very large datasets, estimate duplicates first
        if len(cleaned_df) > 50000:
            sample_size = min(10000, len(cleaned_df))
            sample_df = cleaned_df.sample(n=sample_size, random_state=42)
            estimated_duplicates = int((sample_df.duplicated().sum() / sample_size) * len(cleaned_df))
            print(f"  Estimated duplicates: ~{estimated_duplicates}")
        
        # Actually drop duplicates
        initial_duplicate_count = cleaned_df.duplicated().sum()
        print(f"  Actual duplicates found: {initial_duplicate_count}")
        cleaned_df = cleaned_df.drop_duplicates()
        final_duplicate_count = cleaned_df.duplicated().sum()
        
        print(f"  After dropping duplicates: {cleaned_df.shape}")
        
        # Log cleaning steps
        self.cleaning_log.append({
            "step": "Handle missing values",
            "original_shape": original_shape,
            "current_shape": cleaned_df.shape,
            "rows_removed": original_shape[0] - cleaned_df.shape[0]  # Track actual rows removed
        })
        
        print(f"Cleaning completed. Shape changed from {original_shape} to {cleaned_df.shape}")
        
        self.processed_data = cleaned_df
        return cleaned_df
    
    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: str):
        """
        Save the cleaned dataset to the specified location
        
        Args:
            cleaned_df: Cleaned DataFrame to save
            output_path: Path to save the cleaned data
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as CSV
        cleaned_df.to_csv(output_path, index=False)
        
        # Also save analysis report
        report_path = output_path.replace(".csv", "_analysis_report.txt")
        with open(report_path, "w") as f:
            f.write(self.document_issues_with_examples(cleaned_df))
        
        # Save cleaning log
        log_path = output_path.replace(".csv", "_cleaning_log.json")
        with open(log_path, "w") as f:
            json.dump(self.cleaning_log, f, indent=2, default=str)
    
    def validate_file_type(self, filename: str, allowed_extensions: list) -> bool:
        """
        Validate if a file has an allowed extension
        
        Args:
            filename: Name of the file to validate
            allowed_extensions: List of allowed extensions (without dots)
            
        Returns:
            Boolean indicating if the file type is valid
        """
        if not filename:
            return False
        
        extension = filename.rsplit('.', 1)[-1].lower()
        return extension in allowed_extensions
    
    def secure_file_path(self, file_path: str, base_directory: str) -> str:
        """
        Secure a file path to prevent directory traversal attacks
        
        Args:
            file_path: Original file path
            base_directory: Base directory to validate against
            
        Returns:
            Secured file path or raises ValueError if unsafe
        """
        # Resolve the absolute paths
        abs_base = os.path.abspath(base_directory)
        abs_path = os.path.abspath(file_path)
        
        # Check if the file path is within the allowed base directory
        if not abs_path.startswith(abs_base):
            raise ValueError(f"File path '{file_path}' is outside allowed directory '{base_directory}'")
        
        return abs_path
    
    def calculate_data_quality_metrics(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate metrics comparing original and cleaned data quality
        
        Args:
            original_df: Original DataFrame before cleaning
            cleaned_df: Cleaned DataFrame after processing
            
        Returns:
            Dictionary with data quality metrics
        """
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        original_duplicates = original_df.duplicated().sum()
        cleaned_duplicates = cleaned_df.duplicated().sum()
        
        return {
            "original_missing_values": original_missing,
            "cleaned_missing_values": cleaned_missing,
            "missing_values_improvement": original_missing - cleaned_missing,
            "original_duplicate_rows": original_duplicates,
            "cleaned_duplicate_rows": cleaned_duplicates,
            "duplicate_removal_count": original_duplicates - cleaned_duplicates,
            "data_retention_rate": len(cleaned_df) / len(original_df) * 100 if len(original_df) > 0 else 0,
            "columns_changed": len(original_df.columns) - len(cleaned_df.columns)
        }


# Utility functions for file handling
def validate_and_secure_filename(filename: str) -> str:
    """
    Validate and secure a filename for safe usage
    
    Args:
        filename: Original filename
        
    Returns:
        Secured filename
    """
    return secure_filename(filename)


def create_temp_directory(prefix: str = "dsaral_") -> str:
    """
    Create a temporary directory for processing
    
    Args:
        prefix: Prefix for the temporary directory name
        
    Returns:
        Path to the created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return temp_dir


def cleanup_temp_directory(temp_dir: str):
    """
    Clean up a temporary directory and all its contents
    
    Args:
        temp_dir: Path to the temporary directory to clean up
    """
    if os.path.exists(temp_dir):
        import shutil
        shutil.rmtree(temp_dir)