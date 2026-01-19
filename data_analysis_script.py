"""
Messy Data Autopsy - Systematic Analysis and Cleaning Script

This script systematically:
1. Loads all CSV files from the 'data' directory (including subdirectories)
2. Analyzes data quality issues
3. Documents findings with specific examples
4. Applies appropriate cleaning techniques
5. Preserves cleaned data in a new format/output
"""

import pandas as pd
import numpy as np
import glob
import os
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')

class DataAutopsy:
    def __init__(self, data_directory: str = "data"):
        """
        Initialize the Data Autopsy class
        
        Args:
            data_directory: Path to the directory containing CSV files
        """
        self.data_directory = data_directory
        self.all_dataframes = {}
        self.combined_dataframe = None
        self.analysis_report = {}
        self.cleaning_log = []
        
    def load_all_csv_files(self) -> Dict[str, pd.DataFrame]:
        """
        Load all CSV files from the data directory (including subdirectories)
        
        Returns:
            Dictionary mapping file paths to DataFrames
        """
        print(f"Loading CSV files from directory: {self.data_directory}")
        
        # Find all CSV files recursively
        csv_patterns = ["*.csv", "*.CSV"]
        all_files = []
        
        for pattern in csv_patterns:
            files = glob.glob(os.path.join(self.data_directory, "**", pattern), recursive=True)
            all_files.extend(files)
        
        print(f"Found {len(all_files)} CSV files")
        
        if not all_files:
            print("No CSV files found. Creating sample data for demonstration...")
            self._create_sample_data()
            all_files = glob.glob(os.path.join(self.data_directory, "**", "*.csv"), recursive=True)
        
        # Load each CSV file into a DataFrame
        for file_path in all_files:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
                self.all_dataframes[file_path] = df
                print(f"Loaded {file_path}: Shape {df.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")
        
        # For large datasets, we'll process individual files separately
        # Only combine if total size is manageable (< 100,000 rows)
        total_rows = sum(df.shape[0] for df in self.all_dataframes.values())
        
        if total_rows <= 100000 and len(self.all_dataframes) > 1:
            # Attempt to concatenate all dataframes if small enough
            try:
                dfs_list = list(self.all_dataframes.values())
                # Reset index to avoid overlapping indices
                dfs_list = [df.reset_index(drop=True) for df in dfs_list]
                
                # Try to concatenate - this might fail if column structures differ
                self.combined_dataframe = pd.concat(dfs_list, ignore_index=True, sort=False)
                print(f"Combined all dataframes: Shape {self.combined_dataframe.shape}")
            except Exception as e:
                print(f"Could not concatenate dataframes due to structural differences: {str(e)}")
                # Just use the first dataframe for analysis
                self.combined_dataframe = list(self.all_dataframes.values())[0]
        elif len(self.all_dataframes) == 1:
            self.combined_dataframe = list(self.all_dataframes.values())[0]
        else:
            # For large datasets, we'll analyze each file separately
            print(f"Dataset too large ({total_rows} rows), analyzing first file for demonstration")
            self.combined_dataframe = list(self.all_dataframes.values())[0]
        
        return self.all_dataframes
    
    def _create_sample_data(self):
        """Create sample messy data for demonstration purposes"""
        os.makedirs(self.data_directory, exist_ok=True)
        
        # Sample messy data with various issues
        sample_data = {
            'id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'name': ['John Doe', 'jane smith', 'Bob JOHNSON', '', 'Alice Brown', 'Charlie', 'david WILSON', 'Eve NULL', 'frank', 'Grace'],
            'age': [25, 'thirty', -5, 45, 30, 999, 28, 'unknown', 35, 120],  # Invalid ages
            'email': ['john@example.com', 'jane@gmail.com', 'bob@outlook.com', 'invalid-email', 'alice@yahoo.com', 
                     'charlie@', 'david@test.org', 'eve@company.co.uk', 'frank@site', 'grace@valid.net'],
            'date_of_birth': ['1995-01-15', '02/15/1988', '1990-13-01', '1985-05-20', '01-10-1992', 
                             '1980-03-30', '15/04/1985', '1992-08-25', 'invalid_date', '1988-12-01'],
            'salary': ['$50,000', '60000', 'N/A', 75000.0, '$80,000.00', 'NULL', 90000, 'not disclosed', '$45,500', 100000],
            'department': ['Sales', 'marketing', 'SALES', 'HR', 'hr', 'Engineering', 'engineering', 'Marketing', 'Finance', 'finance'],
            'rating': [4.5, 3.2, 'N/A', 4.8, 2.1, 'five', 3.9, 4.0, 'unknown', 5.0]
        }
        
        df = pd.DataFrame(sample_data)
        df.to_csv(os.path.join(self.data_directory, 'sample_messy_data.csv'), index=False)
        print("Created sample_messy_data.csv with various data quality issues")
    
    def analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze missing values in the dataframe
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with missing value analysis
        """
        missing_analysis = {}
        
        # Standard NaN values
        standard_nans = df.isnull().sum()
        missing_analysis['standard_nan_counts'] = standard_nans[standard_nans > 0].to_dict()
        
        # Empty strings
        empty_strings = df.apply(lambda col: col.astype(str).str.strip().eq('').sum())
        missing_analysis['empty_string_counts'] = empty_strings[empty_strings > 0].to_dict()
        
        # Placeholder values that indicate missing data
        placeholder_patterns = ['N/A', 'NA', 'NULL', 'null', 'nan', 'NaN', 'None', 'empty', 'missing', 'Unknown', 'unknown']
        placeholder_counts = {}
        
        chunk_size = 10000  # Process in chunks to handle large datasets
        for col in df.columns:
            placeholder_count = 0
            for i in range(0, len(df), chunk_size):
                chunk = df[col].iloc[i:i+chunk_size]
                for pattern in placeholder_patterns:
                    mask = chunk.astype(str).str.lower() == pattern.lower()
                    placeholder_count += mask.sum()
            
            if placeholder_count > 0:
                placeholder_counts[col] = placeholder_count
        
        missing_analysis['placeholder_counts'] = placeholder_counts
        
        # Total missing values per column
        total_missing = {}
        for col in df.columns:
            total = 0
            # Count standard NaNs
            total += df[col].isnull().sum()
            # Count empty strings
            total += df[col].astype(str).str.strip().eq('').sum()
            # Count placeholders in chunks
            for i in range(0, len(df), chunk_size):
                chunk = df[col].iloc[i:i+chunk_size]
                for pattern in placeholder_patterns:
                    total += (chunk.astype(str).str.lower() == pattern.lower()).sum()
            
            if total > 0:
                total_missing[col] = total
        
        missing_analysis['total_missing_per_column'] = total_missing
        
        return missing_analysis
    
    def analyze_format_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze inconsistent formats in the dataframe
            
        Args:
            df: DataFrame to analyze
                
        Returns:
            Dictionary with format inconsistency analysis
        """
        format_analysis = {}
        chunk_size = 10000  # Process in chunks to handle large datasets
            
        for col in df.columns:
            series = df[col]
                
            # Check for mixed data types (sample a subset for large datasets)
            if len(series) > chunk_size:
                sampled_series = series.sample(min(chunk_size, len(series)), random_state=42)
            else:
                sampled_series = series
                
            unique_types = set()
            for val in sampled_series.dropna():
                unique_types.add(type(val).__name__)
                
            if len(unique_types) > 1:
                format_analysis[f'{col}_mixed_types'] = list(unique_types)
                
            # Check for inconsistent date formats
            if 'date' in col.lower() or 'birth' in col.lower() or 'time' in col.lower():
                date_formats = set()
                valid_dates = 0
                    
                # Sample data for processing large datasets
                if len(series) > chunk_size:
                    date_sample = series.dropna().sample(min(chunk_size, len(series.dropna())), random_state=42)
                else:
                    date_sample = series.dropna()
                    
                for val in date_sample:
                    try:
                        parsed_date = pd.to_datetime(val, errors='coerce')
                        if pd.notna(parsed_date):
                            valid_dates += 1
                            # Try to detect common formats
                            if isinstance(val, str):
                                if '/' in val:
                                    date_formats.add('MM/DD/YYYY or DD/MM/YYYY')
                                elif '-' in val:
                                    date_formats.add('YYYY-MM-DD or MM-DD-YYYY')
                                elif re.match(r'\d{1,2}/\d{1,2}/\d{4}', val):
                                    date_formats.add('MM/DD/YYYY')
                                elif re.match(r'\d{4}-\d{2}-\d{2}', val):
                                    date_formats.add('YYYY-MM-DD')
                                elif re.match(r'\d{1,2}-\d{1,2}-\d{4}', val):
                                    date_formats.add('MM-DD-YYYY')
                            
                    except:
                        continue
                    
                if len(date_formats) > 1:
                    format_analysis[f'{col}_multiple_date_formats'] = list(date_formats)
                
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
                elif re.match(r'^\$?\d{1,3}(,\d{3})*(\.\d+)?$', val):  # Matches $1,000.00 or 1,000.00
                    numeric_patterns.append('formatted_number_with_commas')
                elif re.match(r'^\$\d+(\.\d+)?$', val):  # Matches $50000
                    numeric_patterns.append('currency_format')
                elif re.match(r'^\d+$', val):  # Matches 50000
                    numeric_patterns.append('plain_number')
                elif re.match(r'^\d+\.\d+$', val):  # Matches 50000.00
                    numeric_patterns.append('decimal_number')
                
            if len(set(numeric_patterns)) > 1:
                format_analysis[f'{col}_multiple_numeric_formats'] = list(set(numeric_patterns))
            
        return format_analysis
    
    def analyze_broken_entries(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze broken entries in the dataframe (impossible values, etc.)
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with broken entry analysis
        """
        broken_analysis = {}
        chunk_size = 10000  # Process in chunks to handle large datasets
        
        for col in df.columns:
            series = df[col]
            
            # Check for impossible age values
            if 'age' in col.lower():
                # Convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(series, errors='coerce')
                invalid_ages = numeric_series[(numeric_series < 0) | (numeric_series > 150)]
                
                if len(invalid_ages) > 0:
                    # Limit results for large datasets
                    sample_invalid = invalid_ages.dropna().head(100) if len(invalid_ages) > 100 else invalid_ages.dropna()
                    broken_analysis[f'{col}_invalid_ages'] = {
                        'count': len(invalid_ages),
                        'values': sample_invalid.tolist(),
                        'indices': sample_invalid.index.tolist()
                    }
            
            # Check for invalid email formats
            if 'email' in col.lower():
                # Process in chunks for large datasets
                invalid_emails_list = []
                invalid_indices_list = []
                
                for i in range(0, len(series), chunk_size):
                    chunk = series.iloc[i:i+chunk_size]
                    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                    chunk_invalid_emails = chunk[~chunk.astype(str).str.contains(email_pattern, na=False)]
                    
                    invalid_emails_list.extend(chunk_invalid_emails.tolist())
                    invalid_indices_list.extend(chunk_invalid_emails.index.tolist())
                    
                    # Limit results for reporting
                    if len(invalid_emails_list) > 100:
                        break
                
                if len(invalid_emails_list) > 0:
                    broken_analysis[f'{col}_invalid_emails'] = {
                        'count': min(len(invalid_emails_list), 100),
                        'values': invalid_emails_list[:100],
                        'indices': invalid_indices_list[:100]
                    }
            
            # Check for impossible date values
            if 'date' in col.lower() or 'birth' in col.lower():
                # Try to parse dates and check for impossible values
                parsed_dates = pd.to_datetime(series, errors='coerce')
                invalid_dates = parsed_dates[pd.isna(parsed_dates) & pd.notna(series)]
                
                if len(invalid_dates) > 0:
                    # Limit results for large datasets
                    sample_invalid = invalid_dates.head(100) if len(invalid_dates) > 100 else invalid_dates
                    broken_analysis[f'{col}_invalid_dates'] = {
                        'count': len(invalid_dates),
                        'values': series[pd.isna(parsed_dates) & pd.notna(series)].head(100).tolist(),
                        'indices': sample_invalid.index.tolist()
                    }
                
                # Check for future dates in birth columns
                if 'birth' in col.lower():
                    future_births = parsed_dates[parsed_dates > datetime.now()]
                    if len(future_births) > 0:
                        # Limit results for large datasets
                        sample_future = future_births.head(100) if len(future_births) > 100 else future_births
                        broken_analysis[f'{col}_future_birth_dates'] = {
                            'count': len(future_births),
                            'values': sample_future.dropna().dt.strftime('%Y-%m-%d').tolist(),
                            'indices': sample_future.dropna().index.tolist()
                        }
        
        return broken_analysis
    
    def analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze duplicate records in the dataframe
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with duplicate analysis
        """
        duplicate_analysis = {}
        
        # Overall duplicates - sample if dataset is too large
        if len(df) > 100000:  # For very large datasets, use sampling approach
            # Take a sample to estimate duplicates
            sample_size = min(100000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            total_duplicates = int((sample_df.duplicated().sum() / sample_size) * len(df))
            duplicate_analysis['total_row_duplicates'] = total_duplicates
            duplicate_analysis['estimated_from_sample'] = True
            
            # Get actual duplicate indices from a smaller sample for examples
            actual_duplicates = df.head(10000).duplicated(keep=False)
            duplicate_indices = df.head(10000)[actual_duplicates].index.tolist()
            duplicate_analysis['duplicate_row_indices'] = duplicate_indices[:10]  # Limit to first 10 for brevity
        else:
            # For smaller datasets, analyze completely
            total_duplicates = df.duplicated().sum()
            duplicate_analysis['total_row_duplicates'] = total_duplicates
            duplicate_analysis['estimated_from_sample'] = False
            
            if total_duplicates > 0:
                duplicate_indices = df[df.duplicated(keep=False)].index.tolist()
                duplicate_analysis['duplicate_row_indices'] = duplicate_indices[:10]  # Limit to first 10 for brevity
        
        # Check for near-duplicates based on specific columns that should be unique
        likely_unique_cols = []
        
        # Heuristic: columns with 'id', 'email', 'name' might be expected to be unique
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['id', 'email', 'name']):
                # For large datasets, sample the column duplicate analysis
                if len(df) > 50000:
                    sample_col_df = df[[col]].sample(n=min(50000, len(df)), random_state=42)
                    col_duplicates = sample_col_df[sample_col_df.duplicated(subset=[col], keep=False)][col].value_counts()
                else:
                    col_duplicates = df[df.duplicated(subset=[col], keep=False)][col].value_counts()
                
                if len(col_duplicates) > 0:
                    likely_unique_cols.append({
                        'column': col,
                        'duplicate_counts': col_duplicates[col_duplicates > 1].to_dict()
                    })
        
        duplicate_analysis['likely_unique_column_issues'] = likely_unique_cols
        
        return duplicate_analysis
    
    def analyze_column_inconsistencies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze column inconsistencies or contradictions
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with column inconsistency analysis
        """
        inconsistency_analysis = {}
        
        # Check for categorical inconsistencies (case, spelling, etc.)
        categorical_columns = df.select_dtypes(include=['object']).columns
        
        for col in categorical_columns:
            # Sample for large datasets
            series = df[col]
            if len(series) > 50000:
                sample_series = series.sample(n=min(50000, len(series)), random_state=42)
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
                    inconsistency_analysis[f'{col}_case_inconsistencies'] = case_inconsistencies
        
        return inconsistency_analysis
    
    def comprehensive_data_analysis(self) -> Dict[str, Any]:
        """
        Perform comprehensive analysis of the combined dataframe
        
        Returns:
            Dictionary with complete analysis report
        """
        if self.combined_dataframe is None:
            raise ValueError("No data loaded. Call load_all_csv_files() first.")
        
        df = self.combined_dataframe
        print(f"Starting comprehensive analysis of dataframe with shape: {df.shape}")
        
        # Perform all analyses
        analysis_results = {
            'dataset_overview': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'dtypes': df.dtypes.to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum()
            },
            'missing_values': self.analyze_missing_values(df),
            'format_inconsistencies': self.analyze_format_inconsistencies(df),
            'broken_entries': self.analyze_broken_entries(df),
            'duplicates': self.analyze_duplicates(df),
            'column_inconsistencies': self.analyze_column_inconsistencies(df)
        }
        
        self.analysis_report = analysis_results
        return analysis_results
    
    def document_issues_with_examples(self) -> str:
        """
        Create a detailed report documenting all issues with specific examples
        
        Returns:
            String with detailed issue documentation
        """
        if not self.analysis_report:
            raise ValueError("Run comprehensive_data_analysis() first.")
        
        report = []
        report.append("# Data Quality Issues Report\n")
        report.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # Overview
        overview = self.analysis_report['dataset_overview']
        report.append(f"## Dataset Overview\n")
        report.append(f"- Shape: {overview['shape'][0]} rows × {overview['shape'][1]} columns")
        report.append(f"- Memory usage: {overview['memory_usage']} bytes ({overview['memory_usage']/1024/1024:.2f} MB)")
        report.append(f"- Columns: {', '.join(overview['columns'])}\n")
        
        # Missing Values
        missing_vals = self.analysis_report['missing_values']
        if missing_vals['total_missing_per_column']:
            report.append(f"## Missing Values\n")
            for col, count in missing_vals['total_missing_per_column'].items():
                report.append(f"- **{col}**: {count} missing values")
                
                # Show examples of missing values
                missing_mask = (
                    self.combined_dataframe[col].isnull() |
                    (self.combined_dataframe[col].astype(str).str.strip() == '') |
                    (self.combined_dataframe[col].astype(str).str.lower().isin(['n/a', 'na', 'null', 'none', 'empty', 'missing', 'unknown']))
                )
                examples = self.combined_dataframe[missing_mask][col].head(3).tolist()
                report.append(f"  - Examples: {examples}")
            report.append("")
        
        # Format Inconsistencies
        format_issues = self.analysis_report['format_inconsistencies']
        if format_issues:
            report.append(f"## Format Inconsistencies\n")
            for issue_type, details in format_issues.items():
                report.append(f"- **{issue_type}**: {details}")
            report.append("")
        
        # Broken Entries
        broken_entries = self.analysis_report['broken_entries']
        if broken_entries:
            report.append(f"## Broken Entries\n")
            for issue_type, details in broken_entries.items():
                report.append(f"- **{issue_type}**: {details.get('count', 'N/A')} problematic entries")
                if 'values' in details:
                    report.append(f"  - Examples: {details['values'][:3]}")  # Show first 3 examples
            report.append("")
        
        # Duplicates
        duplicates = self.analysis_report['duplicates']
        if duplicates.get('total_row_duplicates', 0) > 0:
            report.append(f"## Duplicates\n")
            report.append(f"- Total row duplicates: {duplicates['total_row_duplicates']}")
            if 'duplicate_row_indices' in duplicates:
                report.append(f"  - First few duplicate indices: {duplicates['duplicate_row_indices']}")
            report.append("")
        
        # Column Inconsistencies
        col_inconsistencies = self.analysis_report['column_inconsistencies']
        if col_inconsistencies:
            report.append(f"## Column Inconsistencies\n")
            for issue_type, details in col_inconsistencies.items():
                report.append(f"- **{issue_type}**: {details}")
            report.append("")
        
        return "\n".join(report)
    
    def apply_cleaning_techniques(self) -> pd.DataFrame:
        """
        Apply appropriate cleaning techniques to create a refined dataset
            
        Returns:
            Cleaned DataFrame
        """
        if self.combined_dataframe is None:
            raise ValueError("No data loaded. Call load_all_csv_files() first.")
            
        df = self.combined_dataframe.copy()
        original_shape = df.shape
            
        print("Applying cleaning techniques...")
            
        # For very large datasets, process in chunks
        chunk_size = 50000
            
        # 1. Handle missing values
        print("- Handling missing values...")
        for col in df.columns:
            # Replace common placeholders with NaN
            df[col] = df[col].replace(['N/A', 'NA', 'NULL', 'null', 'nan', 'NaN', 'None', 'empty', 'missing', 'Unknown', 'unknown'], np.nan)
                
            # Also handle empty strings
            df[col] = df[col].replace('', np.nan)
            
        # 2. Fix format inconsistencies
        print("- Fixing format inconsistencies...")
        for col in df.columns:
            # Standardize text case for categorical columns
            if df[col].dtype == 'object':
                # If the column seems categorical based on name or value patterns
                if any(keyword in col.lower() for keyword in ['name', 'department', 'category', 'type', 'status']):
                    # Process in chunks for large datasets
                    if len(df) > chunk_size:
                        for i in range(0, len(df), chunk_size):
                            chunk_end = min(i + chunk_size, len(df))
                            df.loc[i:chunk_end-1, col] = df.loc[i:chunk_end-1, col].str.title()
                    else:
                        df[col] = df[col].str.title()  # Title case for names/categories
                        
                    # Handle case variations specifically
                    case_mappings = {}
                    unique_vals = df[col].dropna().unique()
                    for val in unique_vals:
                        if pd.notna(val):
                            lowercase_val = str(val).lower()
                            # Group similar values by lowercase
                            if lowercase_val not in case_mappings:
                                case_mappings[lowercase_val] = str(val)  # Keep first encountered format
                        
                    # Map all variations to standard format
                    if len(df) > chunk_size:
                        # Process in chunks to avoid memory issues
                        for i in range(0, len(df), chunk_size):
                            chunk_end = min(i + chunk_size, len(df))
                            df.loc[i:chunk_end-1, col] = df.loc[i:chunk_end-1, col].apply(
                                lambda x: case_mappings.get(str(x).lower(), x) if pd.notna(x) else x
                            )
                    else:
                        df[col] = df[col].apply(
                            lambda x: case_mappings.get(str(x).lower(), x) if pd.notna(x) else x
                        )
                
            # Fix numeric columns stored as strings
            if 'salary' in col.lower() or 'income' in col.lower() or 'amount' in col.lower():
                # Remove currency symbols and commas, convert to numeric
                # Process in chunks for large datasets
                if len(df) > chunk_size:
                    for i in range(0, len(df), chunk_size):
                        chunk_end = min(i + chunk_size, len(df))
                        df.loc[i:chunk_end-1, col] = df.loc[i:chunk_end-1, col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                        df.loc[i:chunk_end-1, col] = pd.to_numeric(df.loc[i:chunk_end-1, col], errors='coerce')
                else:
                    df[col] = df[col].astype(str).str.replace(r'[^\d.-]', '', regex=True)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
        # 3. Fix broken entries
        print("- Fixing broken entries...")
        for col in df.columns:
            if 'age' in col.lower():
                # Cap age values to reasonable range
                if len(df) > chunk_size:
                    for i in range(0, len(df), chunk_size):
                        chunk_end = min(i + chunk_size, len(df))
                        numeric_age = pd.to_numeric(df.loc[i:chunk_end-1, col], errors='coerce')
                        df.loc[i:chunk_end-1, col] = numeric_age.apply(lambda x: x if 0 <= x <= 120 else np.nan)
                else:
                    numeric_age = pd.to_numeric(df[col], errors='coerce')
                    df[col] = numeric_age.apply(lambda x: x if 0 <= x <= 120 else np.nan)
                
            if 'email' in col.lower():
                # Keep only valid email formats
                email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
                if len(df) > chunk_size:
                    for i in range(0, len(df), chunk_size):
                        chunk_end = min(i + chunk_size, len(df))
                        valid_email_mask = df.loc[i:chunk_end-1, col].astype(str).str.contains(email_pattern, na=False)
                        df.loc[i:chunk_end-1, ~valid_email_mask, col] = np.nan
                else:
                    valid_email_mask = df[col].astype(str).str.contains(email_pattern, na=False)
                    df.loc[~valid_email_mask, col] = np.nan
                
            if 'date' in col.lower() or 'birth' in col.lower():
                # Parse dates consistently
                if len(df) > chunk_size:
                    for i in range(0, len(df), chunk_size):
                        chunk_end = min(i + chunk_size, len(df))
                        df.loc[i:chunk_end-1, col] = pd.to_datetime(df.loc[i:chunk_end-1, col], errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
            
        # 4. Handle duplicates
        print("- Handling duplicates...")
        print("  Checking for duplicates...")
            
        # For very large datasets, estimate duplicates first
        if len(df) > 100000:
            sample_size = min(100000, len(df))
            sample_df = df.sample(n=sample_size, random_state=42)
            estimated_duplicates = int((sample_df.duplicated().sum() / sample_size) * len(df))
            print(f"  Estimated duplicates: ~{estimated_duplicates}")
            
        # Actually drop duplicates
        initial_duplicate_count = df.duplicated().sum()
        print(f"  Actual duplicates found: {initial_duplicate_count}")
        df = df.drop_duplicates()
        final_duplicate_count = df.duplicated().sum()
            
        print(f"  After dropping duplicates: {df.shape}")
            
        # Log cleaning steps
        self.cleaning_log.append({
            'step': 'Handle missing values',
            'original_shape': original_shape,
            'current_shape': df.shape,
            'rows_removed': original_shape[0] - df.shape[0]  # Track actual rows removed
        })
            
        print(f"Cleaning completed. Shape changed from {original_shape} to {df.shape}")
        return df
    
    def save_cleaned_data(self, cleaned_df: pd.DataFrame, output_path: str = "cleaned_data/cleaned_dataset.csv"):
        """
        Save the cleaned dataset to a new location
        
        Args:
            cleaned_df: The cleaned DataFrame
            output_path: Path to save the cleaned data
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save as CSV
        cleaned_df.to_csv(output_path, index=False)
        print(f"Cleaned data saved to: {output_path}")
        
        # Also save analysis report
        report_path = output_path.replace('.csv', '_analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write(self.document_issues_with_examples())
        print(f"Analysis report saved to: {report_path}")
        
        # Save cleaning log
        log_path = output_path.replace('.csv', '_cleaning_log.json')
        import json
        with open(log_path, 'w') as f:
            json.dump(self.cleaning_log, f, indent=2, default=str)
        print(f"Cleaning log saved to: {log_path}")
    
    def run_complete_autopsy(self):
        """
        Run the complete data autopsy process:
        1. Load data
        2. Analyze quality issues
        3. Document findings
        4. Clean data
        5. Save results
        """
        print("=== Starting Complete Data Autopsy ===\n")
        
        # 1. Load all CSV files
        print("Step 1: Loading CSV files...")
        self.load_all_csv_files()
        
        # 2. Analyze data quality issues
        print("\nStep 2: Analyzing data quality issues...")
        analysis_results = self.comprehensive_data_analysis()
        
        # 3. Document issues with examples
        print("\nStep 3: Documenting issues...")
        issue_report = self.document_issues_with_examples()
        print(issue_report)
        
        # 4. Apply cleaning techniques
        print("\nStep 4: Applying cleaning techniques...")
        cleaned_df = self.apply_cleaning_techniques()
        
        # 5. Save cleaned data and documentation
        print("\nStep 5: Saving cleaned data and documentation...")
        self.save_cleaned_data(cleaned_df)
        
        print("\n=== Data Autopsy Completed Successfully ===")
        return cleaned_df


if __name__ == "__main__":
    # Initialize the data autopsy process
    autopsy = DataAutopsy(data_directory="data")
    
    # Run the complete autopsy process
    cleaned_data = autopsy.run_complete_autopsy()
    
    # Display final results
    print("\nFinal Results:")
    print(f"Original data shape: {autopsy.combined_dataframe.shape if autopsy.combined_dataframe is not None else 'N/A'}")
    print(f"Cleaned data shape: {cleaned_data.shape}")
    print(f"Total rows removed during cleaning: {autopsy.combined_dataframe.shape[0] - cleaned_data.shape[0] if autopsy.combined_dataframe is not None else 'N/A'}")
    
    print("\nFirst few rows of cleaned data:")
    print(cleaned_data.head())