# GC Data Integration

This document describes the process of integrating Gas Chromatography (GC) data with Light Valve (LV) data in the Data Manager system.

## Overview

The GC data integration process extracts data from GC text files, matches it with LV data based on relative timestamps, and generates interpolated functions to merge the datasets for analysis.

## Files

- **GC Input**: `data Light/Exp02_2678_R101_GC.txt` - Original GC data file containing H2, N2, and NH3 readings
- **LV Input**: 
  - `static/reports/[report_id]/step_1/step_1_data.json`
  - `static/reports/[report_id]/step_1/step_1_data.csv`
  - `static/reports/[report_id]/step_1/step_1_plot.json`

- **Generated Outputs**:
  - `static/reports/[report_id]/step_1/step_1_data_with_gc.json` - LV data with integrated GC values
  - `static/reports/[report_id]/step_1/step_1_data_with_gc.csv` - CSV version of the integrated data
  - `static/reports/[report_id]/gc_analysis/gc_interpolation_plot.png` - Visualization of GC data and interpolation curves
  - `static/reports/[report_id]/gc_analysis/gc_lv_comparison_plot.png` - Comparison of GC and LV data

## Using the Web Interface

### Uploading Files

1. Navigate to the home page and click on "Upload Files"
2. Select your LV file in the first file input
3. Select your GC file in the second file input
4. Add an optional report name prefix
5. Choose your preferred data fusion method:
   - Default: Traditional merge with time tolerance
   - Advanced Interpolation: Creates a continuous function between GC data points
6. If using Advanced Interpolation, select:
   - Interpolation method (cubic, linear, quadratic, nearest)
   - Whether to use a uniform time grid (recommended)
   - Grid frequency (1min, 30s, 2min, etc.)
7. Click "Process Files"

### Viewing Results

1. After processing, click "View Results"
2. The visualization page will show:
   - Overall data plot
   - GC Data Analysis section with:
     - GC Data and Interpolation plot
     - GC/LV Data Comparison plot
   - Individual stage plots
3. Use the navigation to explore different stages and plots

## Technical Implementation

### GC Data Parsing

The system identifies and extracts H2, N2, and NH3 columns from the GC file using pattern matching. The system looks for column names containing "H2", "N2", and "NH3" (case-insensitive), and supports variations like "_1", "_2", etc.

### Time Alignment

GC and LV data often have different timestamps and sampling rates. The system aligns them by:

1. Identifying the time range of LV data
2. Mapping GC data timestamps to the LV time range based on relative position
3. Calculating relative time in seconds from the start of the experiment

### Cubic Interpolation

The system uses cubic spline interpolation to generate continuous functions for each gas component:

1. Creates interpolation functions for H2, N2, and NH3
2. Applies these functions to estimate concentration values at each LV timestamp
3. Handles edge cases where interpolation would be extrapolation

### Data Integration

The integrated data is saved in both JSON and CSV formats:

1. For each LV data point, GC values are interpolated
2. The values are added as new columns: `GC_H2`, `GC_N2`, `GC_NH3`
3. These integrated datasets are used for visualization and analysis

## Advanced Usage

### Customizing Interpolation

Different experiments may benefit from different interpolation methods:
- **Cubic**: Best for smooth, continuous processes (default)
- **Linear**: Good for data with sharp transitions
- **Quadratic**: Compromise between linear and cubic
- **Nearest**: Step-function interpolation, good for categorical data

### Working with Sparse GC Data

If GC data is sparse (few data points), consider:
1. Using linear interpolation instead of cubic
2. Increasing the merge tolerance time window
3. Pre-processing GC data to remove outliers

## Troubleshooting

### Common Issues

1. **"No GC data found"**: Check that your GC file contains columns with H2, N2, or NH3 in their names
2. **Misaligned data**: If GC and LV data seem misaligned, try:
   - Adjusting the time alignment manually using offset parameters
   - Checking that the timestamps in both files use consistent formats
3. **Missing interpolation plot**: Ensure the GC file has enough data points for interpolation 