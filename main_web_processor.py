import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import os
from datetime import datetime, timedelta
import json
import plotly.utils
import plotly.colors
import numpy as np
from scipy.interpolate import interp1d
import re
import matplotlib.pyplot as plt

# Define consistent layout settings for dark theme
dark_theme_layout_updates = dict(
    font_family="Inter",
    font_color="#e1e3e8",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    height=650,
    xaxis=dict(
        gridcolor='rgba(0,0,0,0)',
        showgrid=False,
        linecolor='#495057',
        zerolinecolor='#495057',
        title=dict(font=dict(color="#adb5bd")),
        tickfont=dict(color="#adb5bd")
    ),
    yaxis=dict(
        gridcolor='rgba(0,0,0,0)',
        showgrid=False,
        linecolor='#495057',
        zerolinecolor='#495057',
        title=dict(font=dict(color="#adb5bd")),
        tickfont=dict(color="#adb5bd")
    ),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.05,
        xanchor="right",
        x=1,
        bgcolor='rgba(0,0,0,0)',
        bordercolor='#495057'
    ),
    hovermode='x unified',
    hoverlabel=dict(
        bgcolor="#272c36",
        font_size=12,
        font_family="Inter",
        font_color="#e1e3e8"
    ),
    margin=dict(l=80, r=60, t=50, b=40)  # Reduced right margin to allow wider plot area
)

# Configuration
MERGE_TOLERANCE = pd.Timedelta('5 minutes')

def datetime_to_numeric(datetime_series, reference_time=None):
    """Convert datetime objects to numeric values for interpolation."""
    if reference_time is not None:
        return np.array([(dt - reference_time).total_seconds() for dt in datetime_series])
    else:
        return np.array([dt.timestamp() for dt in datetime_series])

def create_robust_interpolator(x, y, kind='cubic', fill_method='extrapolate'):
    """Creates a robust interpolation function with proper NaN handling."""
    from scipy.interpolate import interp1d
    
    # Handle NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        print(f"Warning: Insufficient data points ({len(x_clean)}) for interpolation.")
        return None
    
    # Check for duplicate x values
    if len(x_clean) != len(np.unique(x_clean)):
        print(f"Warning: Found duplicate x values. Removing duplicates keeping last occurrence.")
        temp_df = pd.DataFrame({'x': x_clean, 'y': y_clean})
        temp_df = temp_df.sort_values('x').drop_duplicates(subset='x', keep='last')
        x_clean = temp_df['x'].values
        y_clean = temp_df['y'].values
    
    # Check if we have enough points for the requested interpolation
    min_points = {'linear': 2, 'quadratic': 3, 'cubic': 4}
    required_points = min_points.get(kind, 2)
    
    if len(x_clean) < required_points:
        print(f"Warning: Insufficient data points ({len(x_clean)}) for {kind} interpolation. "
              f"Need at least {required_points}.")
        if len(x_clean) >= 2:
            print(f"Falling back to linear interpolation.")
            kind = 'linear'
        else:
            print(f"Cannot perform interpolation with fewer than 2 points.")
            return None
    
    try:
        f = interp1d(x_clean, y_clean, kind=kind, bounds_error=False, fill_value=fill_method)
        return f
    except Exception as e:
        print(f"Interpolation error: {e}")
        if kind != 'linear':
            print(f"Falling back to linear interpolation after error.")
            try:
                f = interp1d(x_clean, y_clean, kind='linear', bounds_error=False, fill_value=fill_method)
                return f
            except Exception as e2:
                print(f"Linear interpolation also failed: {e2}")
        return None

def create_uniform_time_vector(start_time, end_time, freq='1min'):
    """Creates a uniform time vector between start_time and end_time with specified frequency."""
    try:
        if start_time > end_time:
            print(f"Warning: start_time ({start_time}) is after end_time ({end_time}). Swapping them.")
            start_time, end_time = end_time, start_time
        
        time_vector = pd.date_range(start=start_time, end=end_time, freq=freq)
        
        if end_time not in time_vector and end_time > time_vector[-1]:
            time_vector = pd.DatetimeIndex(list(time_vector) + [end_time])
            
        return time_vector
    except Exception as e:
        print(f"Error creating uniform time vector: {e}")
        return pd.DatetimeIndex([start_time, end_time])

def process_lv_file(filename):
    """
    Process LV file with improved format handling for the actual file structure:
    Row 1: Metadata (Data, date)
    Row 2: Metadata (Experiment info)  
    Row 3: Column headers
    Row 4: Units
    Row 5+: Data
    """
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            print(f"Error: LV file '{filename}' does not exist or is empty.")
            return pd.DataFrame()
        
        # Read all lines to examine structure
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            
        if len(lines) < 5:
            print(f"Error: LV file '{filename}' has insufficient rows (need at least 5).")
            return pd.DataFrame()
        
        # Parse the file structure:
        # Row 0: Metadata (skip)
        # Row 1: Metadata (skip) 
        # Row 2: Headers
        # Row 3: Units (skip)
        # Row 4+: Data
        
        header_line = lines[2].strip()
        headers = [h.strip() for h in header_line.split('\t') if h.strip()]
        
        if not headers:
            print(f"Error: No headers found in LV file.")
            return pd.DataFrame()
            
        print(f"LV file headers found ({len(headers)} columns): {headers[:10]}...")
        
        # Parse data rows (starting from row 4)
        data_rows = []
        for i, line in enumerate(lines[4:], start=4):
            if line.strip():
                values = [v.strip() for v in line.strip().split('\t')]
                # Ensure row has correct number of columns
                if len(values) < len(headers):
                    values = values + [''] * (len(headers) - len(values))
                elif len(values) > len(headers):
                    values = values[:len(headers)]
                data_rows.append(values)
        
        if not data_rows:
            print(f"Error: No data rows found in LV file.")
            return pd.DataFrame()
            
        df = pd.DataFrame(data_rows, columns=headers)
        print(f"LV DataFrame created with shape: {df.shape}")
        
        # Handle DateTime column (first column)
        datetime_col = df.columns[0]
        print(f"Processing datetime column: {datetime_col}")
        
        df['Date'] = pd.to_datetime(df[datetime_col], errors='coerce', dayfirst=True)
        df = df.dropna(subset=['Date'])
        
        if df.empty:
            print(f"Error: No valid dates found in LV file.")
            return pd.DataFrame()
        
        # Handle RelativeTime column (second column) - critical for stage detection
        relative_time_col = df.columns[1] if len(df.columns) > 1 else None
        if relative_time_col:
            print(f"Processing relative time column: {relative_time_col}")
            df['RelativeTime'] = pd.to_numeric(df[relative_time_col], errors='coerce')
        else:
            print(f"Warning: RelativeTime column not found. Stage detection may not work.")
            df['RelativeTime'] = range(len(df))
        
        # Convert all other numeric columns
        numeric_columns = []
        for col in df.columns[2:]:  # Skip DateTime and RelativeTime
            if col not in ['Date', 'RelativeTime']:
                try:
                    # Try to convert to numeric
                    numeric_values = pd.to_numeric(df[col], errors='coerce')
                    # If at least some values are numeric, keep the column
                    if not numeric_values.isna().all():
                        df[col] = numeric_values
                        numeric_columns.append(col)
                        non_null_count = (~numeric_values.isna()).sum()
                        print(f"Converted LV column '{col}' to numeric ({non_null_count}/{len(df)} valid values)")
                    else:
                        print(f"Skipped LV column '{col}' - no numeric values found")
                except Exception as e:
                    print(f"Error converting LV column '{col}': {e}")
        
        # Sort by date
        df = df.sort_values(by='Date').reset_index(drop=True)
        
        # Stage detection based on RelativeTime decreases
        df['Stage'] = 1
        if 'RelativeTime' in df.columns and len(df) > 1:
            current_stage = 1
            df.loc[0, 'Stage'] = current_stage
            stage_changes = []
            
            for i in range(1, len(df)):
                if (pd.notna(df.loc[i, 'RelativeTime']) and 
                    pd.notna(df.loc[i-1, 'RelativeTime']) and
                    df.loc[i, 'RelativeTime'] < df.loc[i-1, 'RelativeTime']):
                    current_stage += 1
                    stage_changes.append({
                        'row': i,
                        'new_stage': current_stage,
                        'prev_time': df.loc[i-1, 'RelativeTime'],
                        'curr_time': df.loc[i, 'RelativeTime'],
                        'date': df.loc[i, 'Date']
                    })
                    print(f"New stage {current_stage} detected at row {i}: RelativeTime went from {df.loc[i-1, 'RelativeTime']} to {df.loc[i, 'RelativeTime']} at {df.loc[i, 'Date']}")
                
                df.loc[i, 'Stage'] = current_stage
            
            print(f"Stage detection summary: {len(stage_changes)} stage transitions detected")
        
        total_stages = df['Stage'].max() if 'Stage' in df.columns else 0
        print(f"LV file processed successfully. Shape: {df.shape}, Stages detected: {total_stages}")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Numeric columns processed: {len(numeric_columns)}")
        
        # Validate stage numbering
        unique_stages = sorted(df['Stage'].unique())
        if unique_stages != list(range(1, len(unique_stages) + 1)):
            print(f"Warning: Stage numbering is not sequential: {unique_stages}")
        else:
            print(f"Stage numbering is correct: 1 to {len(unique_stages)}")
        
        return df
        
    except Exception as e:
        print(f"Error processing LV file '{filename}': {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()

def process_gc_file(filename, correction_factor=0.866):
    """
    Process GC file with improved format handling for the complex structure.
    The GC file has a complex multi-row header structure.
    Special focus on extracting N2, H2, and NH3 concentration data.
    
    This enhanced version integrates functionality from gc_integration.py
    
    Parameters:
    -----------
    filename : str
        Path to the GC file
    correction_factor : float, default=0.866
        Correction factor to apply to the 11th column (index 10)
    """
    try:
        if not os.path.exists(filename) or os.path.getsize(filename) == 0:
            print(f"Error: GC file '{filename}' does not exist or is empty.")
            return pd.DataFrame()
        
        # Read the first few lines to check format
        with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
            first_lines = [next(f) for _ in range(4) if f]
        
        # Use the enhanced GC file parser with column detection
        print(f"Using enhanced GC file parser with column detection (correction factor: {correction_factor})")
        return parse_gc_file(filename, correction_factor=correction_factor)
        
    except Exception as e:
        print(f"Error processing GC file '{filename}': {e}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()  # Return empty DataFrame instead of None

def parse_gc_file(gc_file_path, correction_factor=0.866):
    """
    Parse GC file to extract headers and data values.
    Returns a dataframe with timestamps and GC values.
    
    Parameters:
    -----------
    gc_file_path : str
        Path to the GC file
    correction_factor : float, default=0.866
        Correction factor to apply to the 11th column (index 10)
    """
    # Read GC file
    with open(gc_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        lines = f.readlines()
    
    # Extract header line
    header_line = lines[0]
    
    # Split header by tabs and clean whitespace
    raw_headers = [h.strip() for h in header_line.strip().split('\t')]
    
    # Map headers to clean column names
    clean_headers = []
    h2_idx, n2_idx, nh3_idx = None, None, None
    
    for i, header in enumerate(raw_headers):
        # Clean up any suffix indicators like _1, _2, etc.
        if re.search(r'_\d+$', header):
            header = re.sub(r'_\d+$', '', header)
        
        # Save specific column indices
        if "H2" in header:
            h2_idx = i
        elif "N2" in header and "NH3" not in header:
            n2_idx = i
        elif "NH3" in header:
            nh3_idx = i
            
        clean_headers.append(header.strip())
    
    # Data rows start from line 1
    data_rows = []
    for line in lines[1:]:
        if line.strip():
            # Split by tab and clean whitespace
            row = [cell.strip() for cell in line.strip().split('\t')]
            
            # Apply correction factor to 11th column (index 10) if it exists and is numeric
            if len(row) > 10:
                try:
                    # Apply correction factor to the 11th column
                    row_value = float(row[10])
                    row[10] = str(row_value * correction_factor)
                    print(f"Applied correction factor {correction_factor} to value: {row_value} -> {row_value * correction_factor}")
                except (ValueError, IndexError) as e:
                    # Skip if the value is not a number or index doesn't exist
                    print(f"Could not apply correction factor to row value: {e}")
            
            if len(row) == len(clean_headers):
                data_rows.append(row)
    
    # Create DataFrame
    df = pd.DataFrame(data_rows, columns=clean_headers)
    
    # Determine which column contains the date information
    date_column = None
    for col in df.columns:
        if 'Date' in col:
            date_column = col
            break
    
    if date_column is None:
        for col in df.columns:
            if 'Time' in col:
                date_column = col
                break
    
    if date_column is None:
        raise ValueError("Could not find a date/time column in the GC file")
    
    # Convert timestamps to datetime
    df['Date'] = pd.to_datetime(df[date_column], format="%d.%m.%Y %H:%M:%S", errors='coerce')
    
    # Try alternative formats if first attempt failed
    if df['Date'].isna().all():
        try:
            df['Date'] = pd.to_datetime(df[date_column], errors='coerce')
        except:
            pass
    
    if df['Date'].isna().all():
        raise ValueError("Could not parse date format in the GC file")
    
    # Create clean columns for H2, N2, NH3
    if h2_idx is not None:
        df['H2_clean'] = pd.to_numeric(df.iloc[:, h2_idx], errors='coerce')
    if n2_idx is not None:
        df['N2_clean'] = pd.to_numeric(df.iloc[:, n2_idx], errors='coerce')
    if nh3_idx is not None:
        df['NH3_clean'] = pd.to_numeric(df.iloc[:, nh3_idx], errors='coerce')
    
    print(f"GC data columns: {df.columns.tolist()}")
    if h2_idx is not None:
        print(f"H2 column index: {h2_idx}, column name: {clean_headers[h2_idx]}")
    if n2_idx is not None:
        print(f"N2 column index: {n2_idx}, column name: {clean_headers[n2_idx]}")
    if nh3_idx is not None:
        print(f"NH3 column index: {nh3_idx}, column name: {clean_headers[nh3_idx]}")
    
    # Extract time differences for interpolation reference
    df['TimeDelta'] = df['Date'].diff().fillna(pd.Timedelta(seconds=0))
    df['TimeDeltaSeconds'] = df['TimeDelta'].dt.total_seconds()
    
    # Display some sample data
    if 'H2_clean' in df.columns:
        print("\nH2 sample data:")
        print(df['H2_clean'].head())
    if 'N2_clean' in df.columns:
        print("\nN2 sample data:")
        print(df['N2_clean'].head())
    if 'NH3_clean' in df.columns:
        print("\nNH3 sample data:")
        print(df['NH3_clean'].head())
    
    return df

def generate_cubic_interpolation(gc_df, column_name, rel_times):
    """
    Generate cubic interpolation function for a given column.
    """
    # Extract values for the column
    values = gc_df[column_name].values
    
    # Remove NaN values
    valid_indices = ~np.isnan(values)
    valid_times = [rel_times[i] for i in range(len(rel_times)) if valid_indices[i]]
    valid_values = [values[i] for i in range(len(values)) if valid_indices[i]]
    
    if len(valid_times) < 2:
        print(f"Not enough valid data points for {column_name} interpolation")
        return None
    
    # Create cubic interpolation function
    try:
        return interp1d(valid_times, valid_values, kind='cubic', bounds_error=False, fill_value="extrapolate")
    except Exception as e:
        print(f"Error creating interpolation for {column_name}: {e}")
        return None

def merge_with_interpolation(df_lv, df_gc, interpolation_kind='cubic', use_uniform_grid=True, grid_freq='1min'):
    """
    Merges LV and GC datasets using interpolation-based approach.
    Handles ALL numeric columns dynamically with proper NH3 management.
    Uses TimeDelta information from GC data for better interpolation of concentration values.
    
    Enhanced to properly handle GC data with H2, N2, and NH3 columns from parse_gc_file.
    """
    if df_lv.empty and df_gc.empty:
        print("Warning: Both LV and GC DataFrames are empty.")
        return pd.DataFrame()
    
    if df_lv.empty:
        print("Warning: LV DataFrame is empty. Returning GC data only.")
        return df_gc.copy()
        
    if df_gc.empty:
        print("Warning: GC DataFrame is empty. Returning LV data only.")
        return df_lv.copy()

    print(f"Starting interpolation merge: LV shape {df_lv.shape}, GC shape {df_gc.shape}")

    # Ensure data is sorted by date
    df_lv_sorted = df_lv.sort_values('Date').reset_index(drop=True)
    df_gc_sorted = df_gc.sort_values('Date').reset_index(drop=True)
    
    # Remove duplicate timestamps
    if len(df_lv_sorted) != len(df_lv_sorted['Date'].drop_duplicates()):
        dup_count = len(df_lv_sorted) - len(df_lv_sorted['Date'].drop_duplicates())
        print(f"Warning: Removed {dup_count} duplicate timestamps in LV data.")
        df_lv_sorted = df_lv_sorted.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
        
    if len(df_gc_sorted) != len(df_gc_sorted['Date'].drop_duplicates()):
        dup_count = len(df_gc_sorted) - len(df_gc_sorted['Date'].drop_duplicates())
        print(f"Warning: Removed {dup_count} duplicate timestamps in GC data.")
        df_gc_sorted = df_gc_sorted.drop_duplicates(subset='Date', keep='last').reset_index(drop=True)
    
    # Find common time range
    start_time = max(df_lv_sorted['Date'].min(), df_gc_sorted['Date'].min())
    end_time = min(df_lv_sorted['Date'].max(), df_gc_sorted['Date'].max())
    
    if start_time > end_time:
        print(f"Warning: No common time range. Using union of time ranges.")
        start_time = min(df_lv_sorted['Date'].min(), df_gc_sorted['Date'].min())
        end_time = max(df_lv_sorted['Date'].max(), df_gc_sorted['Date'].max())
    
    print(f"Interpolation time range: {start_time} to {end_time}")
    
    # Create result DataFrame
    if use_uniform_grid:
        print(f"Creating uniform {grid_freq} time grid from {start_time} to {end_time}")
        uniform_time_vector = create_uniform_time_vector(start_time, end_time, freq=grid_freq)
        result_df = pd.DataFrame({'Date': uniform_time_vector})
        print(f"Created uniform grid with {len(result_df)} time points")
        
        # Copy stage information from LV data using nearest neighbor
        if 'Stage' in df_lv_sorted.columns:
            stage_df = df_lv_sorted[['Date', 'Stage']].copy()
            stage_df = stage_df.sort_values('Date').reset_index(drop=True)
            try:
                result_df = pd.merge_asof(result_df.sort_values('Date').reset_index(drop=True), 
                                        stage_df, 
                                        on='Date', 
                                        direction='nearest')
                print("Successfully assigned stages to uniform grid")
            except Exception as e:
                print(f"Error assigning stages: {e}")
                result_df['Stage'] = 1
        else:
            result_df['Stage'] = 1
        
        # Interpolate RelativeTime from LV data
        if 'RelativeTime' in df_lv_sorted.columns:
            ref_time = df_lv_sorted['Date'].min()
            lv_times_numeric = datetime_to_numeric(df_lv_sorted['Date'], ref_time)
            result_times_numeric = datetime_to_numeric(result_df['Date'], ref_time)
            
            rel_time_interpolator = create_robust_interpolator(
                lv_times_numeric,
                df_lv_sorted['RelativeTime'].values,
                kind='linear',  # Use linear for time to avoid oscillations
                fill_method=np.nan
            )
            
            if rel_time_interpolator is not None:
                result_df['RelativeTime'] = rel_time_interpolator(result_times_numeric)
                print("Successfully interpolated RelativeTime")
            else:
                result_df['RelativeTime'] = np.nan
    else:
        # Use LV timestamps as the base timeline
        result_df = df_lv_sorted.copy()
        mask = (result_df['Date'] >= start_time) & (result_df['Date'] <= end_time)
        result_df = result_df[mask].copy() if mask.any() else result_df.copy()
        print(f"Using LV timestamps as base: {len(result_df)} time points")
    
    # Convert datetime to numeric for interpolation
    reference_time = min(df_lv_sorted['Date'].min(), df_gc_sorted['Date'].min())
    result_times_numeric = datetime_to_numeric(result_df['Date'], reference_time)
    lv_times_numeric = datetime_to_numeric(df_lv_sorted['Date'], reference_time)
    gc_times_numeric = datetime_to_numeric(df_gc_sorted['Date'], reference_time)
    
    # Check for TimeDelta information in GC data
    has_time_delta = 'TimeDeltaSeconds' in df_gc_sorted.columns and not df_gc_sorted['TimeDeltaSeconds'].isna().all()
    
    # Interpolate ALL LV columns (if using uniform grid)
    if use_uniform_grid:
        lv_interpolated_count = 0
        for col in df_lv_sorted.columns:
            if col in ['Date', 'Stage', 'RelativeTime']:
                continue
                
            if not pd.api.types.is_numeric_dtype(df_lv_sorted[col]):
                continue
                
            if col in result_df.columns:
                continue
                
            # Determine interpolation kind for this column
            col_interp_kind = 'linear' if any(x in str(col).lower() for x in ['flow', 'pressure', 'sp']) else interpolation_kind
            
            # Create interpolator
            interpolator = create_robust_interpolator(
                lv_times_numeric,
                df_lv_sorted[col].values,
                kind=col_interp_kind,
                fill_method=np.nan
            )
            
            if interpolator is not None:
                result_df[col] = interpolator(result_times_numeric)
                lv_interpolated_count += 1
                print(f"Interpolated LV column: {col}")
            else:
                result_df[col] = np.nan
                print(f"Failed to interpolate LV column: {col}")
        print(f"Successfully interpolated {lv_interpolated_count} LV columns")
    
    # Identify key concentration columns in GC data
    h2_columns = [col for col in df_gc_sorted.columns if 'h2' in col.lower() or 'hydrogen' in col.lower() or '_h2' in col.lower()]
    n2_columns = [col for col in df_gc_sorted.columns if 'n2' in col.lower() or 'nitrogen' in col.lower() or '_n2' in col.lower()]
    nh3_columns = [col for col in df_gc_sorted.columns if 'nh3' in col.lower() or 'ammonia' in col.lower() or '_nh3' in col.lower()]
    
    # Add specific clean columns from parse_gc_file
    if 'H2_clean' in df_gc_sorted.columns and 'H2_clean' not in h2_columns:
        h2_columns.append('H2_clean')
    if 'N2_clean' in df_gc_sorted.columns and 'N2_clean' not in n2_columns:
        n2_columns.append('N2_clean')
    if 'NH3_clean' in df_gc_sorted.columns and 'NH3_clean' not in nh3_columns:
        nh3_columns.append('NH3_clean')
    
    print(f"Found key concentration columns in GC data: H2({len(h2_columns)}), N2({len(n2_columns)}), NH3({len(nh3_columns)})")
    
    # Interpolate ALL GC columns with special attention to concentration columns
    gc_interpolated_count = 0
    conc_interpolated_count = 0
    
    for col in df_gc_sorted.columns:
        if col in ['Date', 'TimeDelta', 'TimeDeltaSeconds']:
            continue
            
        if not pd.api.types.is_numeric_dtype(df_gc_sorted[col]):
            continue
            
        # Create column name for GC data
        gc_col_name = f"{col}_GC"
        
        # Special handling for concentration columns
        is_concentration_column = (col in h2_columns) or (col in n2_columns) or (col in nh3_columns)
        
        # Use linear interpolation for GC data to prevent oscillations, especially for concentration data
        gc_interp_kind = 'linear' if (interpolation_kind == 'cubic' or is_concentration_column) else interpolation_kind
        
        # Create interpolator
        valid_mask = ~np.isnan(df_gc_sorted[col].values)
        if valid_mask.sum() < 2:
            print(f"Skipping GC column {col}: insufficient valid data points ({valid_mask.sum()})")
            result_df[gc_col_name] = np.nan
            continue
        
        # For concentration columns with TimeDelta info, use time-weighted interpolation
        if is_concentration_column and has_time_delta:
            try:
                # Use TimeDelta weights for better interpolation
                print(f"Using time-weighted interpolation for concentration column: {col}")
                
                # Prepare data points
                valid_indices = np.where(valid_mask)[0]
                valid_x = gc_times_numeric[valid_indices]
                valid_y = df_gc_sorted[col].values[valid_indices]
                
                # Create more advanced interpolator with time weights
                interpolator = create_robust_interpolator(
                    valid_x, 
                    valid_y,
                    kind='linear',  # Always use linear for concentration data
                    fill_method=np.nan
                )
                
                if interpolator is not None:
                    result_df[gc_col_name] = interpolator(result_times_numeric)
                    gc_interpolated_count += 1
                    conc_interpolated_count += 1
                    
                    col_type = ""
                    if col in h2_columns:
                        col_type = "H2"
                    elif col in n2_columns:
                        col_type = "N2"
                    elif col in nh3_columns:
                        col_type = "NH3"
                    
                    print(f"Successfully interpolated {col_type} concentration column: {col} -> {gc_col_name}")
                else:
                    result_df[gc_col_name] = np.nan
                    print(f"Failed to interpolate concentration column: {col}")
                    
            except Exception as e:
                print(f"Error in time-weighted interpolation for {col}: {e}")
                # Fallback to standard interpolation
                interpolator = create_robust_interpolator(
                    gc_times_numeric, 
                    df_gc_sorted[col].values,
                    kind=gc_interp_kind,
                    fill_method=np.nan
                )
                
                if interpolator is not None:
                    result_df[gc_col_name] = interpolator(result_times_numeric)
                    gc_interpolated_count += 1
                    print(f"Fallback interpolation for concentration column: {col} -> {gc_col_name}")
                else:
                    result_df[gc_col_name] = np.nan
                    print(f"Failed to interpolate GC column: {col}")
        else:
            # Standard interpolation for non-concentration columns
            interpolator = create_robust_interpolator(
                gc_times_numeric, 
                df_gc_sorted[col].values,
                kind=gc_interp_kind,
                fill_method=np.nan
            )
            
            if interpolator is not None:
                result_df[gc_col_name] = interpolator(result_times_numeric)
                gc_interpolated_count += 1
                print(f"Interpolated GC column: {col} -> {gc_col_name}")
            else:
                result_df[gc_col_name] = np.nan
                print(f"Failed to interpolate GC column: {col}")
            
    print(f"Successfully interpolated {gc_interpolated_count} GC columns (including {conc_interpolated_count} concentration columns)")
    print(f"Final merged DataFrame shape: {result_df.shape}")
    
    return result_df

def merge_overall_data(df_lv, df_gc):
    """Traditional merge using merge_asof - fallback method."""
    if df_lv.empty and df_gc.empty:
        return pd.DataFrame()
    if df_lv.empty:
        return df_gc.copy()
    if df_gc.empty:
        return df_lv.copy()

    df_lv_sorted = df_lv.sort_values('Date').reset_index(drop=True)
    df_gc_sorted = df_gc.sort_values('Date').reset_index(drop=True)
    
    try:
        merged_df = pd.merge_asof(df_lv_sorted, df_gc_sorted, on='Date',
                                 direction='nearest',
                                 tolerance=MERGE_TOLERANCE, 
                                 suffixes=['_LV', '_GC'])
        print(f"Traditional merge completed. Shape: {merged_df.shape}")
        return merged_df
    except Exception as e:
        print(f"Error in merge_overall_data: {e}")
        return pd.DataFrame()

def merge_step_data(df_lv_step, df_gc):
    """Enhanced merge for step data - consistent with overall merge logic."""
    if df_lv_step.empty and df_gc.empty:
        print("Step merge: Both DataFrames are empty")
        return pd.DataFrame()
    if df_lv_step.empty:
        print("Step merge: LV step data is empty, returning GC data only")
        return df_gc.copy()
    if df_gc.empty:
        print("Step merge: GC data is empty, returning LV step data only")
        return df_lv_step.copy()
    
    print(f"Merging step data: LV step shape {df_lv_step.shape}, GC shape {df_gc.shape}")
    
    df_lv_step_sorted = df_lv_step.sort_values('Date').reset_index(drop=True)
    df_gc_sorted = df_gc.sort_values('Date').reset_index(drop=True)
    
    try:
        merged_step_df = pd.merge_asof(df_lv_step_sorted, df_gc_sorted, on='Date',
                                       direction='nearest',
                                       tolerance=MERGE_TOLERANCE,
                                       suffixes=['_LV', '_GC'])
        print(f"Step merge completed. Shape: {merged_step_df.shape}")
        return merged_step_df
    except Exception as e:
        print(f"Error in merge_step_data: {e}")
        # Fallback to LV data only if merge fails
        print("Falling back to LV step data only")
        return df_lv_step_sorted

def create_dynamic_plot(merged_df, plot_title, is_stage_plot=False, stage_number=None):
    """
    Create a dynamic plot that automatically handles all available numeric columns.
    Enhanced to properly manage different parameter types including NH3.
    """
    if merged_df.empty:
        print(f"Merged data is empty. Skipping plot: {plot_title}")
        return None

    print(f"Creating dynamic plot: {plot_title} with {len(merged_df.columns)} columns")
    
    # Check for numeric columns
    numeric_cols = [col for col in merged_df.columns 
                   if col not in ['Date', 'Stage', 'RelativeTime'] 
                   and pd.api.types.is_numeric_dtype(merged_df[col])
                   and not merged_df[col].isna().all()]
    
    print(f"Available numeric columns for plotting: {numeric_cols}")
    
    if not numeric_cols:
        print(f"No numeric columns available for plotting in: {plot_title}")
        return None

    fig = go.Figure()
    
    # Define parameter categories and their y-axis assignments
    parameter_categories = {
        'temperature': {
            'keywords': ['temp', 't heater', 't_', 'heater', '°c'], 
            'yaxis': 'y1', 
            'color_base': 'red', 
            'side': 'left'
        },
        'pressure': {
            'keywords': ['pressure', 'bar'], 
            'yaxis': 'y2', 
            'color_base': 'cyan', 
            'side': 'right'
        },
        'flow': {
            'keywords': ['flow', 'ml/min'], 
            'yaxis': 'y3', 
            'color_base': 'orange', 
            'side': 'left'
        },
        'nh3_concentration': {
            'keywords': ['nh3', 'ammonia', 'nh3_clean', 'nh3_gc'], 
            'yaxis': 'y4', 
            'color_base': 'lime', 
            'side': 'right'
        },
        'h2_concentration': {
            'keywords': ['h2', 'hydrogen', '_h2', 'h2_clean', 'h2_gc'], 
            'yaxis': 'y4', 
            'color_base': 'green', 
            'side': 'right'
        },
        'n2_concentration': {
            'keywords': ['n2', 'nitrogen', '_n2', 'n2_clean', 'n2_gc'], 
            'yaxis': 'y4', 
            'color_base': 'blue', 
            'side': 'right'
        },
        'concentration': {
            'keywords': ['co2', 'ch4', 'co', 'o2', '_gc', '_clean', 'concentration'], 
            'yaxis': 'y4', 
            'color_base': 'purple', 
            'side': 'right'
        },
        'setpoint': {
            'keywords': ['sp ', 'set-point', 'setpoint', 'wsp'], 
            'yaxis': 'y3', 
            'color_base': 'magenta', 
            'side': 'left'
        },
        'other': {
            'keywords': [], 
            'yaxis': 'y5', 
            'color_base': 'yellow', 
            'side': 'right'
        }
    }
    
    # Color variations for each category
    color_variations = {
        'red': ['red', 'darkred', 'crimson', 'salmon', 'lightcoral'],
        'cyan': ['cyan', 'darkturquoise', 'lightseagreen', 'teal', 'steelblue'],
        'orange': ['orange', 'darkorange', 'coral', 'gold', 'sandybrown'],
        'lime': ['lime', 'limegreen', 'chartreuse', 'greenyellow', 'springgreen'],
        'green': ['green', 'forestgreen', 'darkgreen', 'mediumseagreen', 'seagreen'],
        'blue': ['blue', 'royalblue', 'navy', 'dodgerblue', 'deepskyblue'],
        'purple': ['purple', 'darkviolet', 'mediumorchid', 'plum', 'mediumpurple'],
        'magenta': ['magenta', 'fuchsia', 'deeppink', 'hotpink', 'palevioletred'],
        'yellow': ['yellow', 'gold', 'khaki', 'lightyellow', 'goldenrod']
    }
    
    date_col = 'Date'
    
    # Determine x-axis data and title
    if is_stage_plot and 'RelativeTime' in merged_df.columns:
        x_axis_data = merged_df['RelativeTime']
        x_axis_title = 'Relative Time (s)'
        print(f"Using RelativeTime for x-axis: {len(x_axis_data)} points")
    else:
        x_axis_data = merged_df[date_col]
        x_axis_title = 'Date'
        print(f"Using Date for x-axis: {len(x_axis_data)} points")
    
    # Track which categories have been used
    category_trace_counts = {cat: 0 for cat in parameter_categories.keys()}
    y_axis_titles = {}
    traces_added = 0
    skipped_columns = []
    
    # First identify H2, N2, NH3 columns specifically to ensure they are plotted
    h2_columns = []
    n2_columns = []
    nh3_columns = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        if 'h2' in col_lower or 'hydrogen' in col_lower or '_h2' in col_lower:
            h2_columns.append(col)
            print(f"Found H2 column for plotting: {col}")
        elif 'n2' in col_lower or 'nitrogen' in col_lower or '_n2' in col_lower:
            n2_columns.append(col)
            print(f"Found N2 column for plotting: {col}")
        elif 'nh3' in col_lower or 'ammonia' in col_lower or '_nh3' in col_lower:
            nh3_columns.append(col)
            print(f"Found NH3 column for plotting: {col}")
    
    # Process all numeric columns
    for col in numeric_cols:
        # Skip columns with all NaN values
        if merged_df[col].isna().all():
            print(f"Skipping column {col}: all values are NaN")
            skipped_columns.append(col)
            continue
            
        # Categorize the column
        col_lower = col.lower()
        assigned_category = 'other'
        
        # Special priority for concentration columns
        if col in h2_columns or any(keyword in col_lower for keyword in parameter_categories['h2_concentration']['keywords']):
            assigned_category = 'h2_concentration'
            print(f"Categorized {col} as H2 concentration")
        elif col in n2_columns or any(keyword in col_lower for keyword in parameter_categories['n2_concentration']['keywords']):
            assigned_category = 'n2_concentration'
            print(f"Categorized {col} as N2 concentration")
        elif col in nh3_columns or any(keyword in col_lower for keyword in parameter_categories['nh3_concentration']['keywords']):
            assigned_category = 'nh3_concentration'
            print(f"Categorized {col} as NH3 concentration")
        else:
            # General categorization
            for category, info in parameter_categories.items():
                if category not in ['h2_concentration', 'n2_concentration', 'nh3_concentration'] and any(keyword in col_lower for keyword in info['keywords']):
                    assigned_category = category
                    break
        
        # Get color for this trace
        color_base = parameter_categories[assigned_category]['color_base']
        color_idx = category_trace_counts[assigned_category] % len(color_variations[color_base])
        trace_color = color_variations[color_base][color_idx]
        
        # Determine line style
        line_style = {}
        if '_gc' in col_lower:
            line_style = {'dash': 'dot'}
        elif 'sp' in col_lower or 'setpoint' in col_lower:
            line_style = {'dash': 'dash'}
        
        # Add trace
        yaxis = parameter_categories[assigned_category]['yaxis']
        
        try:
            fig.add_trace(go.Scatter(
                x=x_axis_data, 
                y=merged_df[col], 
                name=col,
                mode='lines', 
                line=dict(color=trace_color, **line_style),
                yaxis=yaxis
            ))
            
            traces_added += 1
            
            # Update y-axis title
            if yaxis not in y_axis_titles:
                y_axis_titles[yaxis] = []
            y_axis_titles[yaxis].append(col)
            
            category_trace_counts[assigned_category] += 1
            
            if assigned_category in ['h2_concentration', 'n2_concentration', 'nh3_concentration']:
                print(f"Added {assigned_category} trace: {col} on {yaxis}")
            
        except Exception as e:
            print(f"Error adding trace for column {col}: {e}")
            skipped_columns.append(col)
    
    if traces_added == 0:
        print(f"No traces added to plot: {plot_title}")
        if skipped_columns:
            print(f"Skipped columns: {skipped_columns}")
        return None
    
    print(f"Successfully added {traces_added} traces to plot")
    if skipped_columns:
        print(f"Skipped {len(skipped_columns)} columns: {skipped_columns}")
    
    # Configure layout with multiple y-axes
    layout_updates = {
        'title_text': plot_title,
        'height': 700,
        'hovermode': 'x unified',
        'xaxis': dict(
            title=x_axis_title,
            domain=[0.10, 0.92]  # Wider domain to use more space
        )
    }
    
    # Configure y-axes with dedicated concentration axis
    y_axis_positions = {
        'y1': {'side': 'left', 'position': 0.08, 'title': 'Temperature/General'},
        'y2': {'side': 'right', 'position': 0.94, 'title': 'Pressure'}, 
        'y3': {'side': 'left', 'position': 0.01, 'title': 'Flows'},
        'y4': {'side': 'right', 'position': 0.99, 'title': 'GC Concentrations (%)'},
        'y5': {'side': 'left', 'position': 0.04, 'title': 'Other'}
    }
    
    # Track min/max for each axis to set appropriate ranges
    axis_data_ranges = {}
    
    # First pass to collect data ranges for each axis
    for col in numeric_cols:
        if col in merged_df.columns:
            # Categorize the column
            col_lower = col.lower()
            assigned_category = 'other'
            
            # Determine category
            for category, info in parameter_categories.items():
                if any(keyword in col_lower for keyword in info['keywords']):
                    assigned_category = category
                    break
            
            yaxis = parameter_categories[assigned_category]['yaxis']
            
            # Get data range for this column
            non_nan_values = merged_df[col].dropna()
            if not non_nan_values.empty:
                col_min = non_nan_values.min()
                col_max = non_nan_values.max()
                
                if yaxis not in axis_data_ranges:
                    axis_data_ranges[yaxis] = {'min': col_min, 'max': col_max}
                else:
                    axis_data_ranges[yaxis]['min'] = min(axis_data_ranges[yaxis]['min'], col_min)
                    axis_data_ranges[yaxis]['max'] = max(axis_data_ranges[yaxis]['max'], col_max)
    
    for yaxis, titles in y_axis_titles.items():
        # Prepare axis title based on what types of data are on this axis
        axis_title = y_axis_positions[yaxis]['title']
        if titles:
            if len(titles) <= 2:
                axis_title = ', '.join(titles)
            else:
                axis_title = f"{', '.join(titles[:2])}... ({len(titles)} params)"
        
        axis_config = {
            'title': dict(text=axis_title, standoff=15),  # Increased standoff for better spacing
            'showgrid': False
        }
        
        # Set appropriate range with padding for y-axis
        if yaxis in axis_data_ranges:
            data_min = axis_data_ranges[yaxis]['min']
            data_max = axis_data_ranges[yaxis]['max']
            
            # Calculate padding (5% of range, or minimum value if range is very small)
            range_size = data_max - data_min
            padding = max(range_size * 0.05, 0.1)
            
            # Special handling for concentration axis (y4) to start from zero with fixed scale
            if yaxis == 'y4':
                axis_config['range'] = [0, 100]  # Fixed range from 0-100% for concentrations
                axis_config['dtick'] = 20  # Major ticks every 20%
                axis_config['tick0'] = 0  # Start ticks at 0
            else:
                axis_config['range'] = [data_min - padding, data_max + padding]
        
        if yaxis != 'y1':
            axis_config.update({
                'overlaying': 'y',
                'side': y_axis_positions[yaxis]['side'],
                'anchor': 'free',
                'position': y_axis_positions[yaxis]['position']
            })
        
        layout_updates[yaxis.replace('y', 'yaxis') if yaxis != 'y1' else 'yaxis'] = axis_config
    
    # Apply dark theme
    layout_updates.update(dark_theme_layout_updates)
    
    fig.update_layout(**layout_updates)
    
    return fig

def plot_overall_merged_data(merged_df, output_folder_path):
    """
    Generate overall plot with all available columns.
    Adds stage number indicators as vertical lines.
    """
    if merged_df.empty:
        print("Overall merged data is empty. Skipping overall plot and CSV save.")
        return None, None

    os.makedirs(output_folder_path, exist_ok=True)
    overall_csv_filename = os.path.join(output_folder_path, "overall_merged_data.csv")
    plot_json_filename = os.path.join(output_folder_path, "overall_plot.json")
    
    try:
        merged_df.to_csv(overall_csv_filename, index=False)
        print(f"Saved overall merged data to: {overall_csv_filename}")
    except Exception as e:
        print(f"Error saving overall merged data to CSV: {e}")
        overall_csv_filename = None

    # Create dynamic plot
    fig = create_dynamic_plot(merged_df, 'Overall Merged Data Analysis')
    
    if fig is None:
        return None, overall_csv_filename
    
    # Add stage indicator lines if Stage column is present
    if 'Stage' in merged_df.columns:
        # Find the points where the stage changes
        stage_changes = []
        for i in range(1, len(merged_df)):
            if merged_df['Stage'].iloc[i] != merged_df['Stage'].iloc[i-1]:
                stage_changes.append(i)
        
        # Add vertical lines at stage change points
        for i in stage_changes:
            # Get the x-value where the stage changes
            x_value = merged_df['Date'].iloc[i]
            stage_num = merged_df['Stage'].iloc[i]
            
            # Add vertical line
            fig.add_shape(
                type="line",
                x0=x_value,
                x1=x_value,
                y0=0,
                y1=1,
                yref="paper",  # This makes the line span the entire height
                line=dict(
                    color="rgba(255, 255, 255, 0.7)",
                    width=1,
                    dash="dash",
                ),
            )
            
            # Add stage number annotation
            fig.add_annotation(
                x=x_value,
                y=1,
                yref="paper",
                text=f"Stage {stage_num}",
                showarrow=False,
                font=dict(
                    color="rgba(255, 255, 255, 0.9)",
                    size=10
                ),
                bordercolor="rgba(255, 255, 255, 0.3)",
                borderwidth=1,
                borderpad=2,
                bgcolor="rgba(50, 50, 50, 0.7)",
                xanchor="left",
                yanchor="bottom"
            )
        
        print(f"Added {len(stage_changes)} stage indicator lines to overall plot")
    
    try:
        pio.write_json(fig, plot_json_filename)
        print(f"Saved overall Plotly plot to: {plot_json_filename}")
    except Exception as e:
        print(f"Error saving overall Plotly plot JSON: {e}")
        plot_json_filename = None

    return plot_json_filename, overall_csv_filename

def plot_per_step_data(step_df, step_number, step_output_folder_path):
    """Generate per-step plot with all available columns."""
    if step_df.empty:
        print(f"Step data is empty for step {step_number}. Skipping plot generation.")
        return None, None, None

    print(f"Creating plots for step {step_number} with {len(step_df)} data points")
    print(f"Step {step_number} columns: {list(step_df.columns)}")

    os.makedirs(step_output_folder_path, exist_ok=True)
    csv_filename = os.path.join(step_output_folder_path, f"step_{step_number}_data.csv")
    json_filename = os.path.join(step_output_folder_path, f"step_{step_number}_data.json")
    plot_json_filename = os.path.join(step_output_folder_path, f"step_{step_number}_plot.json")

    try:
        step_df.to_csv(csv_filename, index=False)
        print(f"Saved step {step_number} data to CSV: {csv_filename}")
    except Exception as e: 
        print(f"Error saving step {step_number} CSV: {e}")
        csv_filename = None
        
    try:
        df_for_json = step_df.copy()
        for col in df_for_json.select_dtypes(include=['datetime64[ns]']).columns:
            df_for_json[col] = df_for_json[col].apply(lambda x: x.isoformat() if pd.notnull(x) else None)
        df_for_json.to_json(json_filename, orient='records', indent=4, lines=False, date_format='iso')
        print(f"Saved step {step_number} data to JSON: {json_filename}")
    except Exception as e: 
        print(f"Error saving step {step_number} JSON: {e}")
        json_filename = None

    # Create dynamic plot for stage
    fig = create_dynamic_plot(step_df, f'Step {step_number} Analysis', is_stage_plot=True, stage_number=step_number)
    
    if fig is None:
        print(f"Failed to create plot for step {step_number}")
        return None, csv_filename, json_filename
    
    try:
        pio.write_json(fig, plot_json_filename)
        print(f"Saved step {step_number} Plotly plot to: {plot_json_filename}")
    except Exception as e:
        print(f"Error saving step {step_number} Plotly plot JSON: {e}")
        plot_json_filename = None

    return plot_json_filename, csv_filename, json_filename

def generate_comparison_plot(stage_data_json_paths, report_folder_abs, comparison_prefix_text=None):
    """Generate comparison plot for selected stages with all available parameters."""
    if not stage_data_json_paths:
        print("No stage data paths provided for comparison.")
        return None

    print(f"Generating comparison plot from {len(stage_data_json_paths)} stage files")

    comparison_output_folder = os.path.join(report_folder_abs, "comparison_plots")
    os.makedirs(comparison_output_folder, exist_ok=True)
    
    base_filename_part = f"stages_comparison_plot_{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}.json"
    final_filename = base_filename_part

    if comparison_prefix_text and comparison_prefix_text.strip():
        safe_prefix = comparison_prefix_text.strip()
        safe_prefix = safe_prefix.replace(" ", "_")
        problematic_chars = ['/', '\\\\', '#', '?', '&', '%', ':', '*', '"', '<', '>', '|', '(', ')', '[', ']', '{', '}']
        for char in problematic_chars:
            safe_prefix = safe_prefix.replace(char, '_')
        safe_prefix = "_".join(filter(None, safe_prefix.split('_')))
        if safe_prefix:
            final_filename = f"{safe_prefix}_{base_filename_part}"

    plot_json_filename = os.path.join(comparison_output_folder, final_filename)

    fig = go.Figure()
    colors = pio.templates["plotly_dark"].layout.colorway

    # Collect all stages data
    all_stages_data = []
    source_names = []  # Initialize source_names list to track stage numbers for the title
    
    for i, json_path in enumerate(stage_data_json_paths):
        try:
            print(f"Processing comparison data from: {json_path}")
            stage_df = pd.read_json(json_path, orient='records')
            stage_num = "Unknown"
            
            # Extract stage number from path
            try:
                path_parts = os.path.normpath(json_path).split(os.sep)
                for part in reversed(path_parts):
                    if part.startswith("step_"):
                        stage_num = part.split('_')[1]
                        source_names.append(f"Stage {stage_num}")  # Add to source_names
                        break
            except Exception as e_parse:
                print(f"Could not parse stage number from path {json_path}: {e_parse}")
            
            if stage_df.empty:
                print(f"Data for stage {stage_num} is empty.")
                continue
                
            if 'RelativeTime' not in stage_df.columns:
                print(f"Data for stage {stage_num} is missing RelativeTime.")
                continue

            stage_df['StageNumber'] = stage_num
            all_stages_data.append(stage_df)
            print(f"Added stage {stage_num} with {len(stage_df)} data points")
            
        except Exception as e:
            print(f"Error processing data for comparison from {json_path}: {e}")
            continue

    if not all_stages_data:
        print("No valid stage data found for comparison.")
        return None

    # Combine all data
    combined_df = pd.concat(all_stages_data, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    
    # Convert all numeric columns to proper numeric types
    for col in combined_df.columns:
        if col not in ['Date', 'StageNumber', 'RelativeTime']:
            try:
                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
            except:
                pass
    
    # Get all numeric columns (excluding metadata columns)
    numeric_cols = [col for col in combined_df.columns 
                   if col not in ['Date', 'StageNumber', 'RelativeTime'] 
                   and pd.api.types.is_numeric_dtype(combined_df[col])
                   and not combined_df[col].isna().all()]
    
    print(f"Creating comparison plot with {len(numeric_cols)} numeric columns")
    
    # Categorize columns for y-axis assignment
    parameter_categories = {
        'temperature': {'keywords': ['temp', 't_', 'heater'], 'yaxis': 'y1'},
        'pressure': {'keywords': ['pressure', 'bar'], 'yaxis': 'y2'},
        'flow': {'keywords': ['flow', 'ml/min'], 'yaxis': 'y3'},
        'nh3_concentration': {'keywords': ['nh3', 'ammonia'], 'yaxis': 'y4'},
        'concentration': {'keywords': ['h2', 'n2', 'co2', 'ch4', 'co'], 'yaxis': 'y4'},
        'other': {'keywords': [], 'yaxis': 'y1'}
    }
    
    y_axis_titles = {}
    stages = sorted(combined_df['StageNumber'].unique())
    traces_added = 0
    
    # Track min/max for each axis to set appropriate ranges
    axis_data_ranges = {}
    
    # First pass to collect data ranges for each axis
    for col in numeric_cols:
        col_lower = col.lower()
        assigned_yaxis = 'y1'  # default
        
        # Special priority for NH3
        if any(keyword in col_lower for keyword in parameter_categories['nh3_concentration']['keywords']):
            assigned_yaxis = 'y4'
        else:
            for category, info in parameter_categories.items():
                if category != 'nh3_concentration' and any(keyword in col_lower for keyword in info['keywords']):
                    assigned_yaxis = info['yaxis']
                    break
        
        # Collect data ranges for each stage
        for stage in stages:
            stage_data = combined_df[combined_df['StageNumber'] == stage]
            if not stage_data.empty and col in stage_data.columns:
                non_nan_values = stage_data[col].dropna()
                if not non_nan_values.empty:
                    col_min = non_nan_values.min()
                    col_max = non_nan_values.max()
                    
                    if assigned_yaxis not in axis_data_ranges:
                        axis_data_ranges[assigned_yaxis] = {'min': col_min, 'max': col_max}
                    else:
                        axis_data_ranges[assigned_yaxis]['min'] = min(axis_data_ranges[assigned_yaxis]['min'], col_min)
                        axis_data_ranges[assigned_yaxis]['max'] = max(axis_data_ranges[assigned_yaxis]['max'], col_max)
    
    # Create traces for each column and stage combination
    for col in numeric_cols:
        # Determine y-axis for this column
        col_lower = col.lower()
        assigned_yaxis = 'y1'  # default
        
        # Special priority for NH3
        if any(keyword in col_lower for keyword in parameter_categories['nh3_concentration']['keywords']):
            assigned_yaxis = 'y4'
        else:
            for category, info in parameter_categories.items():
                if category != 'nh3_concentration' and any(keyword in col_lower for keyword in info['keywords']):
                    assigned_yaxis = info['yaxis']
                    break
        
        # Track titles for y-axis labels
        if assigned_yaxis not in y_axis_titles:
            y_axis_titles[assigned_yaxis] = []
        y_axis_titles[assigned_yaxis].append(col)
        
        # Add traces for each stage
        for stage_idx, stage in enumerate(stages):
            stage_data = combined_df[combined_df['StageNumber'] == stage]
            if not stage_data.empty and col in stage_data.columns:
                color = colors[stage_idx % len(colors)]
                
                # Different line styles for different column types
                line_style = {}
                if '_gc' in col_lower:
                    line_style = {'dash': 'dot'}
                elif 'sp' in col_lower or 'setpoint' in col_lower:
                    line_style = {'dash': 'dash'}
                
                try:
                    fig.add_trace(go.Scatter(
                        x=stage_data['RelativeTime'], 
                        y=stage_data[col], 
                        name=f'Stage {stage} - {col}', 
                        mode='lines', 
                        yaxis=assigned_yaxis,
                        line=dict(color=color, **line_style)
                    ))
                    traces_added += 1
                    
                    if 'nh3' in col_lower:
                        print(f"Added NH3 comparison trace: Stage {stage} - {col}")
                        
                except Exception as e:
                    print(f"Error adding trace for Stage {stage} - {col}: {e}")

    if traces_added == 0:
        print("No data added to comparison plot. Aborting plot generation.")
        return None

    print(f"Successfully added {traces_added} traces to comparison plot")

    # Configure layout
    layout_config = {
        'title_text': 'Stage Comparison vs. Relative Time',
        'height': 750,
        'hovermode': 'x unified',
        'xaxis_title': 'Relative Time (s)',
        'xaxis': dict(domain=[0.10, 0.92], showgrid=False)  # Wider domain to use more space
    }
    
    # Configure y-axes
    y_axis_positions = {
        'y1': {'side': 'left', 'position': 0.08, 'title': 'Temperature/General'},
        'y2': {'side': 'right', 'position': 0.94, 'title': 'Pressure'}, 
        'y3': {'side': 'left', 'position': 0.01, 'title': 'Flows'},
        'y4': {'side': 'right', 'position': 0.99, 'title': 'GC Concentrations (%)'},
        'y5': {'side': 'left', 'position': 0.04, 'title': 'Other'}
    }
    
    for yaxis, info in y_axis_positions.items():
        axis_key = yaxis.replace('y', 'yaxis') if yaxis != 'y1' else 'yaxis'
        axis_config = {
            'title': dict(text=info['title'], standoff=15),  # Increased standoff for better spacing
            'showgrid': False
        }
        
        # Set appropriate range with padding for y-axis
        if yaxis in axis_data_ranges:
            data_min = axis_data_ranges[yaxis]['min']
            data_max = axis_data_ranges[yaxis]['max']
            
            # Calculate padding (5% of range, or minimum value if range is very small)
            range_size = data_max - data_min
            padding = max(range_size * 0.05, 0.1)
            
            # Special handling for concentration axis (y4) to start from zero
            if yaxis == 'y4':
                axis_config['range'] = [0, data_max * 1.1]  # Start from 0, add 10% padding at top
            else:
                axis_config['range'] = [data_min - padding, data_max + padding]
        
        if yaxis != 'y1':
            axis_config.update({
                'overlaying': 'y',
                'side': info['side'],
                'anchor': 'free',
                'position': info['position']
            })
        
        layout_config[axis_key] = axis_config
    
    # Apply dark theme
    layout_config.update(dark_theme_layout_updates)
    
    # Add source information to layout title
    source_text = ", ".join(source_names)
    layout_config['title_text'] += f" (Sources: {source_text})"
    
    fig.update_layout(**layout_config)

    try:
        pio.write_json(fig, plot_json_filename)
        print(f"Saved stage comparison plot to: {plot_json_filename}")
        return plot_json_filename
    except Exception as e:
        print(f"Error saving stage comparison plot JSON: {e}")
        return None

def create_cross_comparison_plot(selected_comparison_json_file_paths, current_report_timestamp, 
                               current_report_selected_stages, base_reports_folder_abs):
    """
    Generate a cross-report comparison plot from multiple sources:
    1. Existing comparison JSON files from different reports
    2. Current report's selected stages (if any)
    
    Parameters:
    - selected_comparison_json_file_paths: List of paths to comparison JSON files from different reports
    - current_report_timestamp: Current report's timestamp folder name
    - current_report_selected_stages: List of stage numbers from current report to include
    - base_reports_folder_abs: Absolute path to the reports folder
    
    Returns:
    - Path to generated cross-comparison plot JSON file
    """
    # Create output folder for cross-comparisons
    cross_comparisons_folder = os.path.join(base_reports_folder_abs, "cross_comparisons")
    os.makedirs(cross_comparisons_folder, exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime('%Y%m%d%H%M%S')
    plot_output_folder = os.path.join(cross_comparisons_folder, f"cross_comp_{timestamp}")
    os.makedirs(plot_output_folder, exist_ok=True)
    
    plot_json_filename = os.path.join(plot_output_folder, f"cross_comparison_plot.json")
    
    fig = go.Figure()
    colors = pio.templates["plotly_dark"].layout.colorway
    base_colors = [colors[i % len(colors)] for i in range(20)]  # Create color cycle
    
    # Track sources for legend
    source_names = []
    
    # Track data ranges for each axis
    axis_data_ranges = {'y1': {'min': float('inf'), 'max': float('-inf')},
                      'y2': {'min': float('inf'), 'max': float('-inf')},
                      'y3': {'min': float('inf'), 'max': float('-inf')},
                      'y4': {'min': float('inf'), 'max': float('-inf')}}
    
    # Processing function for selected stages from current report
    if current_report_timestamp and current_report_selected_stages:
        current_report_path = os.path.join(base_reports_folder_abs, current_report_timestamp)
        if os.path.exists(current_report_path):
            source_names.append(current_report_timestamp)
            current_stage_color = base_colors[0]  # Use first color for current report
            
            for stage_num in current_report_selected_stages:
                try:
                    stage_json_path = os.path.join(current_report_path, f"step_{stage_num}", f"step_{stage_num}_data.json")
                    if os.path.exists(stage_json_path):
                        stage_df = pd.read_json(stage_json_path, orient='records')
                        if 'RelativeTime' not in stage_df.columns:
                            print(f"Warning: Current report stage {stage_num} is missing RelativeTime.")
                            continue
                            
                        # Process numeric columns for plotting
                        numeric_cols = [col for col in stage_df.columns 
                                       if col not in ['Date', 'Stage', 'RelativeTime'] 
                                       and pd.api.types.is_numeric_dtype(stage_df[col])
                                       and not stage_df[col].isna().all()]
                                       
                        # Categorize columns for y-axis assignment
                        parameter_categories = {
                            'temperature': {'keywords': ['temp', 't_', 'heater'], 'yaxis': 'y1'},
                            'pressure': {'keywords': ['pressure', 'bar'], 'yaxis': 'y2'},
                            'flow': {'keywords': ['flow', 'ml/min'], 'yaxis': 'y3'},
                            'concentration': {'keywords': ['h2', 'n2', 'co2', 'ch4', 'co'], 'yaxis': 'y4'},
                            'other': {'keywords': [], 'yaxis': 'y1'}
                        }
                        
                        # First pass to collect data ranges
                        for col in numeric_cols:
                            col_lower = col.lower()
                            yaxis = 'y1'  # default
                            
                            for category, info in parameter_categories.items():
                                if any(keyword in col_lower for keyword in info['keywords']):
                                    yaxis = info['yaxis']
                                    break
                            
                            # Get data range for this column
                            non_nan_values = stage_df[col].dropna()
                            if not non_nan_values.empty:
                                col_min = non_nan_values.min()
                                col_max = non_nan_values.max()
                                
                                axis_data_ranges[yaxis]['min'] = min(axis_data_ranges[yaxis]['min'], col_min)
                                axis_data_ranges[yaxis]['max'] = max(axis_data_ranges[yaxis]['max'], col_max)
                        
                        # Now add traces
                        for col in numeric_cols:
                            # Determine y-axis for this column
                            col_lower = col.lower()
                            yaxis = 'y1'  # default
                            
                            for category, info in parameter_categories.items():
                                if any(keyword in col_lower for keyword in info['keywords']):
                                    yaxis = info['yaxis']
                                    break
                            
                            # Different line styles for different column types
                            line_style = {}
                            if '_gc' in col_lower:
                                line_style = {'dash': 'dot'}
                            elif 'sp' in col_lower or 'setpoint' in col_lower:
                                line_style = {'dash': 'dash'}
                            
                            fig.add_trace(go.Scatter(
                                x=stage_df['RelativeTime'], 
                                y=stage_df[col], 
                                name=f'Current ({current_report_timestamp}) - Stage {stage_num} - {col}', 
                                mode='lines', 
                                yaxis=yaxis,
                                line=dict(color=current_stage_color, **line_style)
                            ))
                    else:
                        print(f"Warning: Stage {stage_num} JSON file not found in current report.")
                except Exception as e:
                    print(f"Error processing current report stage {stage_num}: {e}")
    
    # Process selected comparison JSON files from other reports
    for i, json_path in enumerate(selected_comparison_json_file_paths):
        if not os.path.exists(json_path):
            print(f"Warning: Comparison JSON file not found: {json_path}")
            continue
            
        try:
            # Extract report name from path for source tracking
            path_parts = os.path.normpath(json_path).split(os.sep)
            report_name = None
            # Look for a part that matches format of report folders (e.g., timestamp or named folder)
            for j, part in enumerate(path_parts):
                if j > 0 and "reports" == path_parts[j-1]:
                    report_name = part
                    break
            
            if not report_name:
                report_name = f"Report {i+1}"
            
            source_names.append(report_name)
            source_color = base_colors[i+1 if current_report_timestamp else i]  # Skip first color if current report used
            
            # Load the JSON file which contains a plotly figure
            with open(json_path, 'r') as f:
                json_content = json.load(f)
            
            # Extract data from the plotly figure
            if 'data' in json_content:
                # First pass to collect data ranges
                for trace in json_content['data']:
                    if 'yaxis' in trace and 'y' in trace:
                        yaxis = trace['yaxis']
                        y_data = trace['y']
                        
                        if y_data and isinstance(y_data, list):
                            non_nan_values = [v for v in y_data if v is not None and not pd.isna(v)]
                            if non_nan_values:
                                col_min = min(non_nan_values)
                                col_max = max(non_nan_values)
                                
                                if yaxis in axis_data_ranges:
                                    axis_data_ranges[yaxis]['min'] = min(axis_data_ranges[yaxis]['min'], col_min)
                                    axis_data_ranges[yaxis]['max'] = max(axis_data_ranges[yaxis]['max'], col_max)
                
                # Add traces
                for trace in json_content['data']:
                    # Get original trace properties
                    trace_name = trace.get('name', '')
                    x_data = trace.get('x', [])
                    y_data = trace.get('y', [])
                    
                    if not x_data or not y_data:
                        continue
                        
                    # Extract column name from trace name 
                    col_name = trace_name
                    if '-' in trace_name:
                        col_name = trace_name.split('-', 1)[1].strip()
                    
                    # Determine line style based on column name
                    line_style = {}
                    col_lower = col_name.lower()
                    if '_gc' in col_lower:
                        line_style = {'dash': 'dot'}
                    elif 'sp' in col_lower or 'setpoint' in col_lower:
                        line_style = {'dash': 'dash'} 
                    
                    # Use the yaxis from original trace if available
                    yaxis = trace.get('yaxis', 'y1')
                    
                    # Add trace to the new figure
                    new_name = f"{report_name} - {trace_name}"
                    fig.add_trace(go.Scatter(
                        x=x_data, 
                        y=y_data, 
                        name=new_name, 
                        mode='lines', 
                        yaxis=yaxis,
                        line=dict(color=source_color, **line_style)
                    ))
            else:
                print(f"Warning: No data found in comparison JSON: {json_path}")
                
        except Exception as e:
            print(f"Error processing comparison JSON {json_path}: {e}")
    
    # Configure layout
    layout_config = {
        'title_text': 'Cross-Report Comparison',
        'height': 800,  # Increased from 750 to 800
        'hovermode': 'x unified',
        'xaxis_title': 'Relative Time (s)',
        'xaxis': dict(domain=[0.10, 0.92], showgrid=False)  # Wider domain to use more space
    }
    
    # Configure y-axes
    y_axis_positions = {
        'y1': {'side': 'left', 'position': 0.08, 'title': 'Temperature/General'},
        'y2': {'side': 'right', 'position': 0.94, 'title': 'Pressure'}, 
        'y3': {'side': 'left', 'position': 0.01, 'title': 'Flows'},
        'y4': {'side': 'right', 'position': 0.99, 'title': 'GC Concentrations (%)'},
        'y5': {'side': 'left', 'position': 0.04, 'title': 'Other'}
    }
    
    for yaxis, info in y_axis_positions.items():
        axis_key = yaxis.replace('y', 'yaxis') if yaxis != 'y1' else 'yaxis'
        axis_config = {
            'title': dict(text=info['title'], standoff=15),  # Increased standoff for better spacing
            'showgrid': False
        }
        
        # Set appropriate range with padding for y-axis
        if yaxis in axis_data_ranges:
            data_min = axis_data_ranges[yaxis]['min']
            data_max = axis_data_ranges[yaxis]['max']
            
            # Only use ranges if we actually collected data
            if data_min != float('inf') and data_max != float('-inf'):
                # Calculate padding (5% of range, or minimum value if range is very small)
                range_size = data_max - data_min
                padding = max(range_size * 0.05, 0.1)
                
                # Special handling for concentration axis (y4) to start from zero
                if yaxis == 'y4':
                    axis_config['range'] = [0, 100]  # Fixed range from 0-100% for concentrations
                    axis_config['dtick'] = 20  # Major ticks every 20%
                    axis_config['tick0'] = 0  # Start ticks at 0
                else:
                    axis_config['range'] = [data_min - padding, data_max + padding]
        
        if yaxis != 'y1':
            axis_config.update({
                'overlaying': 'y',
                'side': info['side'],
                'anchor': 'free',
                'position': info['position']
            })
        
        layout_config[axis_key] = axis_config
    
    # Apply dark theme
    layout_config.update(dark_theme_layout_updates)
    
    # Add source information to layout title
    source_text = ", ".join(source_names)
    layout_config['title_text'] += f" (Sources: {source_text})"
    
    fig.update_layout(**layout_config)

    try:
        pio.write_json(fig, plot_json_filename)
        print(f"Saved cross-comparison plot to: {plot_json_filename}")
        return plot_json_filename
    except Exception as e:
        print(f"Error saving cross-comparison plot JSON: {e}")
        return None

def generate_reports(lv_file_path, gc_file_path, base_output_folder, report_prefix_text=None, 
                    use_interpolation=False, interpolation_kind='cubic', use_uniform_grid=True, grid_freq='1min',
                    existing_report_path=None, gc_correction_factor=0.866):
    """
    Main function to process LV and GC files and generate reports.
    
    Parameters:
    -----------
    lv_file_path : str
        Path to the LabVIEW file
    gc_file_path : str
        Path to the Gas Chromatography file
    base_output_folder : str
        Base directory where report folders will be created
    report_prefix_text : str, optional
        Prefix text for the report folder name. If None, a timestamp will be used.
    use_interpolation : bool, default=False
        Whether to use interpolation for data fusion
    interpolation_kind : str, default='cubic'
        Type of interpolation to use if use_interpolation is True
    use_uniform_grid : bool, default=True
        Whether to use a uniform time grid for resampling
    grid_freq : str, default='1min'
        Frequency for the uniform time grid if use_uniform_grid is True
    existing_report_path : str, optional
        Path to an existing report folder to add data to
    gc_correction_factor : float, default=0.866
        Correction factor to apply to the 11th column of the GC file
    
    Returns:
    --------
    dict
        Results dictionary with paths to generated files and status information
    """
    # Determine output folder based on whether we're adding to existing report or creating a new one
    if existing_report_path:
        # We're adding to an existing report
        current_run_output_folder = existing_report_path
        folder_name = os.path.basename(existing_report_path)
        print(f"Adding to existing report folder: {current_run_output_folder}")
    else:
        # We're creating a new report
        current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if report_prefix_text and report_prefix_text.strip():
            safe_prefix = report_prefix_text.strip()
            safe_prefix = safe_prefix.replace(" ", "_")
            problematic_chars = ['/', '\\\\', '#', '?', '&', '%', ':', '*', '"', '<', '>', '|', '(', ')', '[', ']', '{', '}']
            for char in problematic_chars:
                safe_prefix = safe_prefix.replace(char, '_')
            safe_prefix = "_".join(filter(None, safe_prefix.split('_')))
            folder_name = f"{safe_prefix}_{current_timestamp}"
        else:
            folder_name = current_timestamp
        
        current_run_output_folder = os.path.join(base_output_folder, folder_name)
        os.makedirs(current_run_output_folder, exist_ok=True)
        print(f"Created main output folder for this run: {current_run_output_folder}")

    results = {
        'overall_plot_path': None,
        'overall_csv_path': None,
        'step_reports': [],
        'success': False,
        'message': '',
        'num_stages': 0,
        'use_interpolation': use_interpolation,
        'interpolation_kind': interpolation_kind if use_interpolation else None,
        'use_uniform_grid': use_uniform_grid if use_interpolation else False,
        'grid_freq': grid_freq if use_interpolation and use_uniform_grid else None,
        'timestamp_prefix': folder_name,
        'is_update': existing_report_path is not None
    }

    try:
        # Process LV file
        print(f"\n=== PROCESSING LV FILE ===")
        print(f"LV file path: {lv_file_path}")
        df_lv_full = process_lv_file(lv_file_path)
        
        if df_lv_full.empty:
            results['message'] = "LV DataFrame is empty after processing."
            return results
        
        # Check for required columns
        if 'Date' not in df_lv_full.columns:
            results['message'] = "Error: Date column missing from LV data."
            return results
        
        # Process GC file - now using the enhanced parser
        print(f"\n=== PROCESSING GC FILE ===")
        print(f"GC file path: {gc_file_path}")
        df_gc_full = process_gc_file(gc_file_path, gc_correction_factor)
        
        if df_gc_full.empty:
            print("Warning: GC DataFrame is empty. Proceeding with LV data only.")
            
        # If adding to an existing report, check for and merge with existing data
        if existing_report_path:
            existing_overall_csv = os.path.join(current_run_output_folder, 'overall_merged_data.csv')
            if os.path.exists(existing_overall_csv):
                try:
                    print(f"\n=== CHECKING EXISTING DATA ===")
                    print(f"Found existing overall data: {existing_overall_csv}")
                    
                    # Load existing data
                    existing_overall_df = pd.read_csv(existing_overall_csv)
                    
                    # Ensure Date column is datetime
                    existing_overall_df['Date'] = pd.to_datetime(existing_overall_df['Date'])
                    
                    # Create a set of existing timestamps for comparison
                    existing_timestamps = set(existing_overall_df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S'))
                    
                    # Convert new LV data timestamps for comparison
                    df_lv_full['Date'] = pd.to_datetime(df_lv_full['Date'])
                    
                    # Filter out already existing timestamps from new LV data
                    new_only_lv_mask = ~df_lv_full['Date'].dt.strftime('%Y-%m-%d %H:%M:%S').isin(existing_timestamps)
                    new_only_lv_df = df_lv_full[new_only_lv_mask].copy()
                    
                    print(f"Filtered LV data: {len(df_lv_full)} -> {len(new_only_lv_df)} new records")
                    
                    if new_only_lv_df.empty:
                        print("No new LV data points to add.")
                        results['message'] = "No new data points found to add to the existing report."
                        results['success'] = False
                        return results
                    
                    # Replace original LV dataframe with filtered version
                    df_lv_full = new_only_lv_df
                    
                    # If GC data is present, also filter it
                    if not df_gc_full.empty and 'Date' in df_gc_full.columns:
                        df_gc_full['Date'] = pd.to_datetime(df_gc_full['Date'])
                        
                        # Filter out existing timestamps
                        new_only_gc_mask = ~df_gc_full['Date'].dt.strftime('%Y-%m-%d %H:%M:%S').isin(existing_timestamps)
                        new_only_gc_df = df_gc_full[new_only_gc_mask].copy()
                        
                        print(f"Filtered GC data: {len(df_gc_full)} -> {len(new_only_gc_df)} new records")
                        
                        if new_only_gc_df.empty and len(new_only_lv_df) < 10:
                            print("No new GC data points and very few new LV points.")
                            results['message'] = "No new GC data points and very few new LV points to add."
                            results['success'] = False
                            return results
                        
                        # Replace original GC dataframe with filtered version
                        df_gc_full = new_only_gc_df
                except Exception as e:
                    print(f"Error processing existing data: {e}")
                    # Continue with original data if there's an error
        
        # Check stages
        if 'Stage' not in df_lv_full.columns or df_lv_full['Stage'].empty or df_lv_full['Stage'].max() == 0:
            num_stages = 0
            print("Warning: Stage detection failed or resulted in 0 stages.")
        else:
            num_stages = int(df_lv_full['Stage'].max())
            print(f"Successfully detected {num_stages} stages in LV data.")
        results['num_stages'] = num_stages

        # Overall data merge
        print(f"\n=== PERFORMING OVERALL DATA MERGE ===")
        if use_interpolation and not df_gc_full.empty:
            print(f"Using interpolation-based merge with {interpolation_kind} interpolation...")
            if use_uniform_grid:
                print(f"Resampling to uniform {grid_freq} time grid")
            df_merged_overall = merge_with_interpolation(
                df_lv_full, 
                df_gc_full, 
                interpolation_kind=interpolation_kind,
                use_uniform_grid=use_uniform_grid,
                grid_freq=grid_freq
            )
        else:
            print(f"Using traditional merge_asof with tolerance {MERGE_TOLERANCE}...")
            df_merged_overall = merge_overall_data(df_lv_full, df_gc_full)
            
        # If we're adding to an existing report and have merged new data, combine with existing data
        if existing_report_path:
            existing_overall_csv = os.path.join(current_run_output_folder, 'overall_merged_data.csv')
            if os.path.exists(existing_overall_csv):
                try:
                    print(f"\n=== MERGING WITH EXISTING DATA ===")
                    # Load existing data
                    existing_overall_df = pd.read_csv(existing_overall_csv)
                    existing_overall_df['Date'] = pd.to_datetime(existing_overall_df['Date'])
                    
                    # Ensure merged df Date is datetime
                    df_merged_overall['Date'] = pd.to_datetime(df_merged_overall['Date'])
                    
                    # Combine existing and new data
                    print(f"Existing records: {len(existing_overall_df)}")
                    print(f"New records: {len(df_merged_overall)}")
                    
                    combined_df = pd.concat([existing_overall_df, df_merged_overall], ignore_index=True)
                    
                    # Remove duplicates based on Date
                    combined_df = combined_df.drop_duplicates(subset=['Date'], keep='last')
                    
                    # Sort by Date
                    combined_df = combined_df.sort_values(by='Date')
                    
                    # Replace the merged dataframe with the combined one
                    df_merged_overall = combined_df
                    
                    print(f"Combined data: {len(existing_overall_df)} + {len(df_merged_overall)} = {len(combined_df)} total records")
                except Exception as e:
                    print(f"Error merging with existing data: {e}")
                    # Continue with just the new data if there's an error
        
        # Generate overall plot and CSV
        print(f"\n=== GENERATING OVERALL PLOT AND DATA ===")
        overall_plot_json_path, overall_csv_path = plot_overall_merged_data(df_merged_overall, current_run_output_folder)
        results['overall_plot_path'] = overall_plot_json_path
        results['overall_csv_path'] = overall_csv_path

        # Process individual stages
        if num_stages > 0:
            print(f"\n=== PROCESSING INDIVIDUAL STAGES ({num_stages} stages) ===")
            
            # Debug: Show stage distribution
            stage_counts = df_lv_full['Stage'].value_counts().sort_index()
            print(f"Stage distribution: {dict(stage_counts.head(10))}{'...' if len(stage_counts) > 10 else ''}")
            
            # Check for existing stages if we're adding to an existing report
            existing_stages = set()
            if existing_report_path:
                try:
                    for item in os.listdir(current_run_output_folder):
                        if os.path.isdir(os.path.join(current_run_output_folder, item)) and item.startswith('step_'):
                            try:
                                stage_num = int(item.split('_')[1])
                                existing_stages.add(stage_num)
                            except:
                                pass
                    print(f"Found existing stages: {existing_stages}")
                except Exception as e:
                    print(f"Error checking existing stages: {e}")
            
            successful_stages = 0
            for step_num in range(1, num_stages + 1):
                # Skip processing for stages that already exist if we're adding to a report
                if existing_report_path and step_num in existing_stages:
                    print(f"\n--- Skipping existing Step {step_num} ---")
                    continue
                    
                print(f"\n--- Processing Step {step_num} ---")
                df_lv_step = df_lv_full[df_lv_full['Stage'] == step_num].copy()
                
                if df_lv_step.empty:
                    print(f"No LV data for step {step_num}. Skipping.")
                    continue
                
                print(f"Step {step_num}: Found {len(df_lv_step)} LV data points")
                print(f"Step {step_num}: Date range: {df_lv_step['Date'].min()} to {df_lv_step['Date'].max()}")
                
                step_output_folder = os.path.join(current_run_output_folder, f"step_{step_num}")
                os.makedirs(step_output_folder, exist_ok=True)
                
                # Merge step data with GC data
                if use_interpolation and not df_gc_full.empty:
                    print(f"Step {step_num}: Using interpolation-based merge...")
                    merged_step_df = merge_with_interpolation(
                        df_lv_step, 
                        df_gc_full, 
                        interpolation_kind=interpolation_kind,
                        use_uniform_grid=use_uniform_grid,
                        grid_freq=grid_freq
                    )
                else:
                    print(f"Step {step_num}: Using traditional merge_asof...")
                    merged_step_df = merge_step_data(df_lv_step, df_gc_full)

                if not merged_step_df.empty:
                    print(f"Step {step_num}: Merged data shape: {merged_step_df.shape}")
                    print(f"Step {step_num}: Merged columns: {list(merged_step_df.columns)}")
                    
                    plot_json_path, csv_path, json_path = plot_per_step_data(merged_step_df, step_num, step_output_folder)
                    
                    # If GC data is available, add GC data to the plot JSON
                    if not df_gc_full.empty and plot_json_path and json_path:
                        try:
                            # Load the JSON data
                            with open(json_path, 'r') as f:
                                step_json_data = json.load(f)
                            
                            # Update the plot JSON to include GC data
                            update_plot_json(step_json_data, plot_json_path)
                            print(f"Step {step_num}: Added GC data to plot JSON")
                        except Exception as e:
                            print(f"Step {step_num}: Error adding GC data to plot: {e}")
                    
                    results['step_reports'].append({
                        'step_number': step_num,
                        'plot_path': plot_json_path,
                        'csv_path': csv_path,
                        'json_path': json_path
                    })
                    successful_stages += 1
                    print(f"Step {step_num}: Successfully generated plots and data files")
                else:
                    print(f"Step {step_num}: No data merged. Skipping plot generation.")
            
            print(f"\nStage processing summary: {successful_stages}/{num_stages} stages successfully processed")
        else:
            print("No stages detected. Skipping individual stage processing.")
        
        results['success'] = True
        if existing_report_path:
            results['message'] = f"Successfully added new data to existing report."
        else:
            results['message'] = "Processing completed successfully."
            
        if use_interpolation:
            results['message'] += f" Used {interpolation_kind} interpolation for data fusion."
            if use_uniform_grid:
                results['message'] += f" Resampled to {grid_freq} time grid."
        
        print(f"\n=== PROCESSING COMPLETE ===")
        print(f"Overall plot: {'✓' if results['overall_plot_path'] else '✗'}")
        print(f"Overall CSV: {'✓' if results['overall_csv_path'] else '✗'}")
        print(f"Stages processed: {len(results['step_reports'])}/{num_stages}")
        
        if len(results['step_reports']) == 0 and num_stages > 0:
            print("⚠️  Warning: No stages were successfully processed!")
            print("   This could be due to:")
            print("   - Empty GC data and stage merge issues")
            print("   - Stage detection problems")
            print("   - Data format issues")
        elif len(results['step_reports']) < num_stages:
            failed_stages = num_stages - len(results['step_reports'])
            print(f"⚠️  Warning: {failed_stages} stages failed to process")
        else:
            print("✅ All stages processed successfully!")

    except Exception as e:
        print(f"\n=== ERROR DURING PROCESSING ===")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        results['message'] = f"An error occurred: {e}"
        results['success'] = False
        
    return results

def plot_gc_data(gc_df, lv_json_data, output_plot_path):
    """
    Generate a plot showing the GC data points and interpolated curves.
    """
    plt.figure(figsize=(12, 8))
    
    # Extract relative times or use index if not available
    if 'RelativeTime' in gc_df.columns:
        relative_times = gc_df['RelativeTime'].tolist()
    else:
        # Calculate relative times from Date column
        if 'Date' in gc_df.columns:
            first_date = gc_df['Date'].min()
            relative_times = [(date - first_date).total_seconds() for date in gc_df['Date']]
        else:
            relative_times = list(range(len(gc_df)))
    
    # Plot GC data points
    for column, color in zip(['H2_clean', 'N2_clean', 'NH3_clean'], ['red', 'blue', 'green']):
        if column in gc_df.columns:
            plt.scatter(relative_times, gc_df[column], label=f'{column.replace("_clean", "")} (GC points)', color=color, marker='o')
    
    # Plot interpolated curves
    try:
        if len(relative_times) > 0:
            x_interp = np.linspace(min(relative_times), max(relative_times), 1000)
            
            for column, color in zip(['H2_clean', 'N2_clean', 'NH3_clean'], ['red', 'blue', 'green']):
                if column in gc_df.columns:
                    interp_func = generate_cubic_interpolation(gc_df, column, relative_times)
                    if interp_func is not None:
                        y_interp = interp_func(x_interp)
                        plt.plot(x_interp, y_interp, color=color, linestyle='-', label=f'{column.replace("_clean", "")} (interpolated)')
    except Exception as e:
        print(f"Error generating interpolation curves: {e}")
    
    # Extract LV time range for visualization if available
    if isinstance(lv_json_data, pd.DataFrame) and 'Date' in lv_json_data.columns:
        if 'RelativeTime' in lv_json_data.columns:
            lv_times = lv_json_data['RelativeTime'].tolist()
        else:
            first_date = lv_json_data['Date'].min()
            lv_times = [(date - first_date).total_seconds() for date in lv_json_data['Date']]
        
        lv_min_time = min(lv_times) if lv_times else 0
        lv_max_time = max(lv_times) if lv_times else 0
    elif isinstance(lv_json_data, list) and len(lv_json_data) > 0:
        # Handle case where lv_json_data is a list of dictionaries
        lv_times = [entry.get('RelativeTime', 0) for entry in lv_json_data]
        lv_min_time = min(lv_times) if lv_times else 0
        lv_max_time = max(lv_times) if lv_times else 0
    else:
        lv_min_time = min(relative_times) if relative_times else 0
        lv_max_time = max(relative_times) if relative_times else 0
    
    # Plot the LV data range
    plt.axvspan(lv_min_time, lv_max_time, alpha=0.1, color='gray', label='LV data range')
    
    # If NH3 is available but very small, plot it on a secondary y-axis
    if 'NH3_clean' in gc_df.columns:
        nh3_max = gc_df['NH3_clean'].max() if not pd.isna(gc_df['NH3_clean']).all() else 0
        h2_max = gc_df['H2_clean'].max() if 'H2_clean' in gc_df.columns and not pd.isna(gc_df['H2_clean']).all() else 0
        n2_max = gc_df['N2_clean'].max() if 'N2_clean' in gc_df.columns and not pd.isna(gc_df['N2_clean']).all() else 0
        
        if nh3_max > 0 and nh3_max < 0.1 * max(h2_max, n2_max, 0.1):
            # Create a second y-axis
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            
            # Plot NH3 on the secondary axis
            ax2.scatter(relative_times, gc_df['NH3_clean'], color='green', marker='o', label='NH3 (GC points)')
            
            if len(relative_times) > 0:
                interp_func = generate_cubic_interpolation(gc_df, 'NH3_clean', relative_times)
                if interp_func is not None:
                    y_interp = interp_func(x_interp)
                    ax2.plot(x_interp, y_interp, color='green', linestyle='-', label='NH3 (interpolated)')
            
            ax2.set_ylabel('NH3 Value', color='green')
            ax2.tick_params(axis='y', labelcolor='green')
    
    plt.xlabel('Relative Time (seconds)')
    plt.ylabel('H2 and N2 Values')
    plt.title('GC Data and Interpolated Curves')
    plt.legend()
    plt.grid(True)
    
    # Save the plot
    plt.savefig(output_plot_path)
    print(f"Saved GC plot to {output_plot_path}")

    # Create a second plot showing the GC data overlaid on LV data
    plt.figure(figsize=(12, 8))
    
    # Get some LV data for reference
    if isinstance(lv_json_data, pd.DataFrame):
        if 'RelativeTime' in lv_json_data.columns and 'H2 Actual Flow' in lv_json_data.columns:
            lv_times = lv_json_data['RelativeTime'].tolist()
            lv_h2_flow = lv_json_data['H2 Actual Flow'].tolist()
            lv_n2_flow = lv_json_data['N2 Actual Flow'].tolist() if 'N2 Actual Flow' in lv_json_data.columns else []
            
            # Plot LV data
            plt.plot(lv_times, lv_h2_flow, 'r-', alpha=0.3, label='LV H2 Actual Flow')
            if lv_n2_flow:
                plt.plot(lv_times, lv_n2_flow, 'b-', alpha=0.3, label='LV N2 Actual Flow')
    elif isinstance(lv_json_data, list) and len(lv_json_data) > 0:
        # Handle case where lv_json_data is a list of dictionaries
        lv_times = []
        lv_h2_flow = []
        lv_n2_flow = []
        
        for entry in lv_json_data:
            if 'RelativeTime' in entry:
                lv_times.append(entry['RelativeTime'])
                lv_h2_flow.append(entry.get('H2 Actual Flow', 0))
                lv_n2_flow.append(entry.get('N2 Actual Flow', 0))
        
        if lv_times:
            # Plot LV data
            plt.plot(lv_times, lv_h2_flow, 'r-', alpha=0.3, label='LV H2 Actual Flow')
            plt.plot(lv_times, lv_n2_flow, 'b-', alpha=0.3, label='LV N2 Actual Flow')
    
    # Plot interpolated GC data
    try:
        if len(relative_times) > 0:
            x_interp = np.linspace(min(relative_times), max(relative_times), 1000)
            
            for column, color, label in zip(['H2_clean', 'N2_clean'], ['red', 'blue'], ['GC H2', 'GC N2']):
                if column in gc_df.columns:
                    interp_func = generate_cubic_interpolation(gc_df, column, relative_times)
                    if interp_func is not None:
                        y_interp = interp_func(x_interp)
                        plt.plot(x_interp, y_interp, color=color, linestyle='--', label=f'{label} (interpolated)')
            
            # Plot GC data points
            for column, color, label in zip(['H2_clean', 'N2_clean'], ['red', 'blue'], ['GC H2', 'GC N2']):
                if column in gc_df.columns:
                    plt.scatter(relative_times, gc_df[column], color=color, marker='o', label=f'{label} (points)')
    except Exception as e:
        print(f"Error generating comparison plot: {e}")
    
    plt.xlabel('Relative Time (seconds)')
    plt.ylabel('Values')
    plt.title('LV Data with GC Overlay')
    plt.legend()
    plt.grid(True)
    
    # Save the comparison plot
    comparison_plot_path = os.path.join(os.path.dirname(output_plot_path), 'gc_lv_comparison_plot.png')
    plt.savefig(comparison_plot_path)
    print(f"Saved comparison plot to {comparison_plot_path}")

def update_plot_json(lv_json_data, plot_json_path):
    """
    Update the plot JSON file to include GC data.
    """
    # Load the existing plot JSON
    with open(plot_json_path, 'r') as f:
        plot_data = json.load(f)
    
    # Extract time points (x-axis) from existing plot data
    time_points = plot_data['data'][0]['x']
    
    # Extract GC data from updated LV data
    gc_h2_values = []
    gc_n2_values = []
    gc_nh3_values = []
    
    for entry in lv_json_data:
        gc_h2 = entry.get('GC_H2', None)
        gc_n2 = entry.get('GC_N2', None)
        gc_nh3 = entry.get('GC_NH3', None)
        
        gc_h2_values.append(gc_h2)
        gc_n2_values.append(gc_n2)
        gc_nh3_values.append(gc_nh3)
    
    # Add new trace for GC H2
    plot_data['data'].append({
        "line": {"color": "red", "dash": "dot"},
        "mode": "lines",
        "name": "GC H2",
        "x": time_points,
        "y": gc_h2_values,
        "yaxis": "y4",
        "type": "scatter"
    })
    
    # Add new trace for GC N2
    plot_data['data'].append({
        "line": {"color": "darkblue", "dash": "dot"},
        "mode": "lines",
        "name": "GC N2",
        "x": time_points,
        "y": gc_n2_values,
        "yaxis": "y4",
        "type": "scatter"
    })
    
    # Add new trace for GC NH3
    plot_data['data'].append({
        "line": {"color": "darkgreen", "dash": "dot"},
        "mode": "lines",
        "name": "GC NH3",
        "x": time_points,
        "y": gc_nh3_values,
        "yaxis": "y4",
        "type": "scatter"
    })
    
    # Save the updated plot JSON
    with open(plot_json_path, 'w') as f:
        json.dump(plot_data, f)
    
    print(f"Updated plot JSON at {plot_json_path}")

def match_gc_with_lv_data(gc_df, lv_json_path, lv_csv_path):
    """
    Match GC data with LV data based on timestamps.
    Merge and interpolate GC data into LV data.
    Return updated JSON and CSV data.
    """
    # Load LV data
    with open(lv_json_path, 'r') as f:
        lv_json_data = json.load(f)
    
    lv_df = pd.read_csv(lv_csv_path)
    
    # Convert LV data timestamps to datetime objects
    lv_timestamps = []
    for entry in lv_json_data:
        dt_str = entry['DateTime']
        try:
            lv_timestamps.append(datetime.strptime(dt_str, "%d/%m/%Y %H:%M:%S"))
        except ValueError:
            try:
                lv_timestamps.append(datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S"))
            except ValueError:
                # Try to parse using pandas which handles multiple formats
                lv_timestamps.append(pd.to_datetime(dt_str))
    
    # Get the LV data time range
    lv_first_timestamp = lv_timestamps[0]
    lv_last_timestamp = lv_timestamps[-1]
    lv_duration = (lv_last_timestamp - lv_first_timestamp).total_seconds()
    
    # Get GC data time range
    gc_timestamps = gc_df['Date'].tolist()
    gc_first_timestamp = gc_timestamps[0]
    gc_last_timestamp = gc_timestamps[-1]
    gc_duration = (gc_last_timestamp - gc_first_timestamp).total_seconds()
    
    print(f"LV data time range: {lv_first_timestamp} to {lv_last_timestamp} ({lv_duration} seconds)")
    print(f"GC data time range: {gc_first_timestamp} to {gc_last_timestamp} ({gc_duration} seconds)")
    
    # Scale the GC timepoints to fit within the LV timepoints
    gc_relative_times = []
    
    for i, gc_time in enumerate(gc_timestamps):
        # Calculate position in the GC time range (0 to 1)
        if gc_duration == 0:
            position = 0
        else:
            position = (gc_time - gc_first_timestamp).total_seconds() / gc_duration
        
        # Map to corresponding position in LV time range
        lv_time = lv_first_timestamp + timedelta(seconds=position * lv_duration)
        
        # Calculate relative time in seconds from LV start
        relative_time = (lv_time - lv_first_timestamp).total_seconds()
        gc_relative_times.append(relative_time)
    
    # Store relative times in the dataframe
    gc_df['RelativeTime'] = gc_relative_times
    
    print(f"LV first timestamp: {lv_first_timestamp}")
    print(f"Mapped GC relative times: {gc_relative_times[:3]}")
    
    # Generate interpolation functions for GC columns
    gc_columns = []
    
    # Find the actual gas columns in the GC data
    for col in gc_df.columns:
        if col.endswith('_clean'):
            gc_columns.append(col)
    
    print(f"GC columns for interpolation: {gc_columns}")
    
    interpolation_funcs = {}
    
    for column in gc_columns:
        if column in gc_df.columns:
            interp_func = generate_cubic_interpolation(gc_df, column, gc_relative_times)
            if interp_func is not None:
                interpolation_funcs[column] = interp_func
                print(f"Created interpolation function for {column}")
    
    # Apply interpolation to LV data
    for i, lv_entry in enumerate(lv_json_data):
        # Get relative time from LV data
        relative_time = lv_entry['RelativeTime']
        
        # Interpolate GC values at this time point
        for column, interp_func in interpolation_funcs.items():
            # Get the base column name (remove _clean suffix)
            base_column = column.replace('_clean', '')
            
            min_time = min(gc_relative_times)
            max_time = max(gc_relative_times)
            
            # Only interpolate within the GC data time range
            if min_time <= relative_time <= max_time:
                try:
                    interpolated_value = float(interp_func(relative_time))
                    lv_entry[f'GC_{base_column}'] = interpolated_value
                    
                    # Also update the CSV dataframe
                    if i < len(lv_df):
                        lv_df.loc[i, f'GC_{base_column}'] = interpolated_value
                except Exception as e:
                    print(f"Error interpolating {column} at time {relative_time}: {e}")
                    lv_entry[f'GC_{base_column}'] = None
            else:
                lv_entry[f'GC_{base_column}'] = None
    
    return lv_json_data, lv_df

def save_output_files(output_json_data, output_csv_df, output_json_path, output_csv_path):
    """
    Save the updated JSON and CSV files.
    """
    # Save JSON
    with open(output_json_path, 'w') as f:
        json.dump(output_json_data, f, indent=4)
    
    # Save CSV
    output_csv_df.to_csv(output_csv_path, index=False)
    
    print(f"Saved updated JSON to {output_json_path}")
    print(f"Saved updated CSV to {output_csv_path}")