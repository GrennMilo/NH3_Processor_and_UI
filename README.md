# Skid NH3 Synthesis Data Manager

A comprehensive data management and visualization tool for processing and analyzing LabVIEW (LV) and Gas Chromatography (GC) data from NH3 synthesis experiments.

## Recent Updates

- **Enhanced GC Integration**: Improved functionality for processing GC data and merging it with LV data
- **Advanced Interpolation**: Added multiple interpolation methods (cubic, linear, quadratic, nearest) for GC concentration values
- **Uniform Time Grid**: Support for resampling data to consistent time intervals for better visualization
- **Cross-Report Comparison**: Compare data across different experiments with dedicated visualization tools
- **Stage Analysis**: Enhanced statistics and visualization for experimental stages

## Key Features

- **Unified Data Processing**: Process both LV data (temperature, pressure, flow) and GC data (H2, N2, NH3 concentrations) in a single workflow
- **Advanced Data Fusion**: Choose between traditional merging or interpolation methods:
  - Cubic interpolation (smooth curves with continuous first derivatives)
  - Linear interpolation (simple straight lines between points)
  - Quadratic interpolation (intermediate complexity)
  - Nearest interpolation (step-function/nearest-neighbor)
- **Customizable Time Grid**: Create uniform time grids with configurable frequency (30s, 1min, 2min, 5min, 10min)
- **Stage Detection**: Automatically detect and separate experimental stages for detailed analysis
- **Interactive Visualizations**: View comprehensive plots of all process variables with multiple Y-axes
- **Data Export**: Download processed data in CSV and JSON formats for further analysis
- **Cross-Experiment Comparison**: Compare data across different experiments and stages

## Getting Started

1. Clone this repository
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the application:
   ```
   python app.py
   ```
4. Open your browser and navigate to `http://localhost:6007`

## Data Processing Workflow

1. **Upload Data**: 
   - Submit both LV and GC data files through the web interface
   - Add an optional report name prefix for better organization
   - Choose your preferred data fusion method (traditional or advanced interpolation)
   - Configure interpolation settings if applicable

2. **Data Processing**: 
   - The system aligns data based on timestamps
   - Detects experimental stages automatically
   - Applies your chosen interpolation method
   - Creates a unified dataset with all process variables

3. **Results View**: 
   - Explore interactive visualizations of the merged data
   - View dedicated GC data plots showing the interpolation quality
   - Analyze stage-by-stage statistics with grid or chart views

4. **Data Export and Comparison**: 
   - Download individual stage data or combined datasets
   - Compare stages within a single experiment
   - Compare data across different experiments

## GC Data Integration

The system supports seamless integration of GC data with LV data:

- **Automatic Column Mapping**: Intelligently identifies H2, N2, and NH3 concentration columns
- **Multiple Interpolation Methods**: Creates smooth functions to estimate GC values at LV timestamps
- **Time Alignment**: Aligns GC and LV data based on relative timestamps
- **Dedicated Visualization**: Plots showing GC data and its integration with LV data

### GC Interpolation Process

The GC interpolation process involves several steps:
1. Parsing the GC file to identify gas component columns (H2, N2, NH3)
2. Aligning GC timestamps with LV data timeframes
3. Generating interpolation functions for each gas component using your chosen method
4. Applying these functions to estimate gas concentrations at each LV data point
5. Creating dedicated visualizations of the interpolated GC data

## Advanced Interpolation Features

When using advanced interpolation, you can choose from several methods:

- **Cubic (Default)**: Smooth curves with continuous first derivatives. Best for most scientific data where smooth transitions are expected.
- **Linear**: Simple straight lines between points. Good for data with sharp transitions or when you want to avoid oscillations.
- **Quadratic**: Intermediate complexity between linear and cubic.
- **Nearest**: Step-function interpolation (nearest-neighbor). Similar to traditional merge but with different behavior at boundaries.

### Uniform Time Grid

The application supports resampling data to a uniform time grid:

- When enabled, all data is resampled to a regular time interval (e.g., 1 minute)
- This ensures consistent spacing between data points
- Improves visualization quality and makes it easier to compare different experiments
- You can select the grid frequency that best matches your data's natural sampling rate

## Reports and Visualization

The system organizes processed data into comprehensive reports:

- **Overall Analysis**: Combined visualization of all data with proper axis organization
- **GC Data Analysis**: Dedicated section showing GC interpolation quality
- **Stage-by-Stage Analysis**: Detailed statistics for each experimental stage
- **Stage Comparison**: Tools to compare multiple stages within an experiment
- **Cross-Report Comparison**: Advanced tools to compare data across different experiments

### Stage Statistics

For each experimental stage, the system calculates:
- Number of data points
- Duration
- Average temperature
- Average pressure
- Average H₂ and N₂ flow rates
- Average NH₃ concentration
- Average outlet mass flow rate

These statistics can be viewed in either a grid view with visual indicators or a chart view with multiple Y-axes.

## File Structure

- `app.py`: Main application entry point and API routes
- `main_web_processor.py`: Core data processing functionality including GC integration
- `templates/`: Web interface templates
- `static/`: CSS, JavaScript, and generated reports
- `uploads/`: Temporary storage for uploaded data files

## Data Requirements

### LV Data Requirements

The LV data file should contain:
- Date/time information 
- Relative time values (used for stage detection)
- Process variables like temperature, pressure, and flow rates
- Values in a tab-separated format (.txt file)

### GC Data Requirements

The GC data file should contain:
- Date/time information for each measurement
- Concentration data for H2, N2, and NH3 components
- Values in a tab-separated format (.txt file)

## Technologies Used

- **Backend**: Python, Flask, Pandas, NumPy, SciPy
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Plotly, Chart.js
- **Frontend**: HTML, CSS, JavaScript