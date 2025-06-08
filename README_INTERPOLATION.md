# Advanced Interpolation Features

This document explains the advanced interpolation features available in the data processing application for merging LV and GC data.

## Overview

The application offers two approaches to merge LabVIEW (LV) and Gas Chromatography (GC) data:

1. **Traditional Merge (merge_asof)**: Simple nearest-neighbor matching with a configurable time tolerance.
2. **Advanced Interpolation**: Creates a continuous function between data points to estimate values at exact timestamps.

## Interpolation Methods

When using advanced interpolation, you can choose from several interpolation methods:

- **Cubic (Default)**: Smooth curves with continuous first derivatives. Best for most scientific data where smooth transitions are expected.
- **Linear**: Simple straight lines between points. Good for data with sharp transitions or when you want to avoid oscillations.
- **Quadratic**: Intermediate complexity between linear and cubic.
- **Nearest**: Step-function interpolation (nearest-neighbor). Similar to traditional merge but with different behavior at boundaries.

## Uniform Time Grid

The application now supports resampling data to a uniform time grid:

- When enabled, all data is resampled to a regular time interval (e.g., 1 minute)
- This ensures consistent spacing between data points
- Improves visualization quality and makes it easier to compare different experiments
- You can select the grid frequency that best matches your data's natural sampling rate

## Usage in the Web Interface

1. Upload your LV and GC files as usual
2. Check the "Use Advanced Interpolation" checkbox
3. Select your preferred interpolation method from the dropdown
4. Choose whether to use a uniform time grid and select the grid frequency
5. Click "Process Files"

## Command-Line Utility

For advanced users, we provide a command-line utility to test interpolation with your actual data:

```bash
python interpolation_cmd.py --lv_file path/to/lv_file.txt --gc_file path/to/gc_file.txt [options]
```

Options:
- `--lv_file FILE`: Path to LV data file (required)
- `--gc_file FILE`: Path to GC data file (required)
- `--method METHOD`: Interpolation method: linear, cubic, quadratic (default: cubic)
- `--grid_freq FREQ`: Time grid frequency: 30s, 1min, 5min, 10min (default: 1min)
- `--no_grid`: Disable uniform time grid (use original LV timestamps)
- `--output FILE`: Output CSV file path (optional)
- `--help`: Show help message

## Best Practices

1. **Choose the Right Interpolation Method**:
   - For most scientific data, **cubic** interpolation provides the best results
   - If your data has sharp transitions or you're concerned about overshooting, use **linear**
   - If cubic seems too smooth or linear too rough, try **quadratic** as a middle ground

2. **Grid Frequency Selection**:
   - Choose a grid frequency that matches your LV data's natural sampling rate
   - A finer grid (e.g., 30s) provides more detail but can introduce artifacts with sparse GC data
   - A coarser grid (e.g., 5min) is more appropriate for slowly changing processes

3. **Validation**:
   - Always validate interpolated results against raw data points
   - Be cautious about interpolation in regions where GC data is sparse

## Technical Implementation

The interpolation process:

1. Converts timestamps to numeric values for interpolation
2. Creates a robust interpolator with proper NaN handling and validation
3. Applies interpolation to either the original LV timestamps or a uniform time grid
4. Uses different interpolation approaches for different column types (e.g., linear for flows, chosen method for temperature)

## Example Use Cases

- **Cubic Interpolation with 1-minute Grid**: Best for standard analysis with good quality data
- **Linear Interpolation with 5-minute Grid**: Good for noisy data or when concerned about artifacts
- **Linear Interpolation without Grid**: When you want to maintain original LV timestamps but still get GC interpolation
- **Cubic Interpolation with 30-second Grid**: For high-resolution analysis with dense data

---

*Note: Interpolation quality depends on the density and quality of your source data. Always validate results.* 