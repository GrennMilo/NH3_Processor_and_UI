/**
 * Common JavaScript functionality shared across all pages
 */

document.addEventListener('DOMContentLoaded', function() {
    // Theme switcher functionality
    initThemeSwitcher();
    
    // Notification system
    initNotificationSystem();

    // Add CSS styles for the stages detailed statistics table
    const style = document.createElement('style');
    style.textContent = `
        .stages-detailed-view {
            margin-top: 25px;
            background: var(--bg-card);
            border-radius: 8px;
            padding: 15px;
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
        }
        
        .stages-stats-table-container {
            overflow-x: auto;
            margin-top: 15px;
        }
        
        .stages-stats-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }
        
        .stages-stats-table th, 
        .stages-stats-table td {
            padding: 10px;
            text-align: center;
            border-bottom: 1px solid var(--border-color);
        }
        
        .stages-stats-table th {
            background-color: var(--bg-dark-tertiary);
            font-weight: 600;
            position: sticky;
            top: 0;
        }
        
        .stages-stats-table tr:hover {
            background-color: var(--bg-hover);
        }
        
        .stages-stats-table tr:nth-child(even) {
            background-color: var(--bg-dark-secondary);
        }
          
        .with-indicator {
            position: relative;
        }
        
        .value-indicator {
            position: absolute;
            bottom: 0;
            left: 0;
            height: 3px;
            background: linear-gradient(to right, #4CAF50, #8BC34A);
            border-radius: 0 3px 3px 0;
            z-index: 1;
        }
        
        /* Different colors for different metrics */
        td:nth-child(2) .value-indicator { background: linear-gradient(to right, #3498db, #2980b9); } /* Data points */
        td:nth-child(4) .value-indicator { background: linear-gradient(to right, #e74c3c, #c0392b); } /* Temperature */
        td:nth-child(5) .value-indicator { background: linear-gradient(to right, #9b59b6, #8e44ad); } /* Pressure */
        td:nth-child(6) .value-indicator { background: linear-gradient(to right, #2ecc71, #27ae60); } /* H2 Flow */
        td:nth-child(7) .value-indicator { background: linear-gradient(to right, #1abc9c, #16a085); } /* N2 Flow */
        td:nth-child(8) .value-indicator { background: linear-gradient(to right, #f39c12, #d35400); } /* NH3 */
        td:nth-child(9) .value-indicator { background: linear-gradient(to right, #95a5a6, #7f8c8d); } /* Outlet */
    `;
    document.head.appendChild(style);
    
    // Add event listener for stage statistics tables created dynamically
    document.addEventListener('click', function(e) {
        // Check if modal with stages table was created
        if (e.target.classList.contains('report-details-btn')) {
            setTimeout(function() {
                initStageStatisticsTable();
            }, 1000); // Wait for table to load
        }
    });
});

/**
 * Initialize the theme switcher
 */
function initThemeSwitcher() {
    const themeSwitcher = document.getElementById('theme-switcher');
    if (!themeSwitcher) return;
    
    // Check for saved theme preference or use preferred color scheme
    const savedTheme = localStorage.getItem('theme');
    const prefersDarkScheme = window.matchMedia('(prefers-color-scheme: dark)');
    
    if (savedTheme === 'light' || (!savedTheme && !prefersDarkScheme.matches)) {
        document.body.classList.add('light-theme');
        themeSwitcher.innerHTML = '<i class="fas fa-moon"></i>';
    } else {
        document.body.classList.add('dark-theme');
        themeSwitcher.innerHTML = '<i class="fas fa-sun"></i>';
    }
    
    // Toggle theme when clicked
    themeSwitcher.addEventListener('click', function() {
        if (document.body.classList.contains('light-theme')) {
            document.body.classList.replace('light-theme', 'dark-theme');
            localStorage.setItem('theme', 'dark');
            themeSwitcher.innerHTML = '<i class="fas fa-sun"></i>';
        } else {
            document.body.classList.replace('dark-theme', 'light-theme');
            localStorage.setItem('theme', 'light');
            themeSwitcher.innerHTML = '<i class="fas fa-moon"></i>';
        }
        
        // Update all active plots with the new theme colors
        updatePlotsForThemeChange();
    });
}

/**
 * Update all active Plotly plots when theme changes
 */
function updatePlotsForThemeChange() {
    // Find all Plotly containers
    const plotlyContainers = document.querySelectorAll('.plotly-container');
    
    // Apply the enhancePlotlyLayout function to each container
    plotlyContainers.forEach(container => {
        if (container && container._fullLayout) {
            enhancePlotlyLayout(container);
        }
    });
}

/**
 * Initialize notification system
 */
function initNotificationSystem() {
    // Notification element is added in the layout template
    const notification = document.getElementById('notification');
    if (!notification) return;
    
    // Add close button to notification
    notification.addEventListener('click', function() {
        this.style.display = 'none';
    });
}

/**
 * Show a notification message
 * @param {string} message - Message to display
 * @param {string} type - Type of notification (info, success, warning, error)
 * @param {number} duration - Duration in milliseconds (0 for no auto-hide)
 */
function showNotification(message, type = 'info', duration = 5000) {
    const notification = document.getElementById('notification');
    if (!notification) return;
    
    notification.textContent = message;
    notification.className = `notification ${type}`;
    notification.style.display = 'block';
    
    if (duration > 0) {
        setTimeout(() => {
            notification.style.display = 'none';
        }, duration);
    }
}

/**
 * Show or hide the global loading indicator
 * @param {boolean} show - Whether to show or hide the indicator
 */
function showLoading(show) {
    const loadingIndicator = document.getElementById('loading-indicator');
    if (!loadingIndicator) return;
    
    if (show) {
        loadingIndicator.style.display = 'flex';
    } else {
        loadingIndicator.style.display = 'none';
    }
}

/**
 * Format a file size in bytes to human-readable format
 * @param {number} bytes - Size in bytes
 * @returns {string} - Formatted size
 */
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

/**
 * Format a date string to a more readable format
 * @param {string} dateString - ISO date string
 * @returns {string} - Formatted date
 */
function formatDate(dateString) {
    if (!dateString) return '';
    
    const date = new Date(dateString);
    if (isNaN(date.getTime())) return dateString;
    
    return date.toLocaleString();
}

/**
 * Makes an API request with appropriate error handling
 * @param {string} url - API endpoint URL
 * @param {Object} options - Fetch options
 * @returns {Promise} - Resolves with the response data
 */
async function apiRequest(url, options = {}) {
    try {
        showLoading(true);
        
        const response = await fetch(url, options);
        const data = await response.json();
        
        showLoading(false);
        
        if (!data.success) {
            showNotification(data.message || 'API request failed', 'error');
            throw new Error(data.message || 'API request failed');
        }
        
        return data;
    } catch (error) {
        showLoading(false);
        console.error('API request error:', error);
        showNotification('Failed to complete request. Please try again.', 'error');
        throw error;
    }
}

/**
 * Enhances Plotly graphs with improved axis layout and concentration scaling
 * @param {Object} plotlyDiv - The Plotly div element 
 */
function enhancePlotlyLayout(plotlyDiv) {
    if (!plotlyDiv || !plotlyDiv.layout) return;
    
    const layout = plotlyDiv.layout;
    
    // Clear title text to prevent overlap with Plotly title and provide more space
    if (layout.title && typeof layout.title === 'object' && layout.title.text) {
        // Store original title in a data attribute for reference if needed
        plotlyDiv.setAttribute('data-original-title', layout.title.text);
        layout.title.text = '';
    } else if (layout.title && typeof layout.title === 'string') {
        plotlyDiv.setAttribute('data-original-title', layout.title);
        layout.title = '';
    }
    
    // Ensure proper spacing for axes - expand the domain to use more space on the right
    if (layout.xaxis) {
        layout.xaxis.domain = [0.10, 0.92]; // Extend plot area to use more space on the right
    }
    
    // Ensure all axes have proper positioning - adjust position values to match the new xaxis domain
    const axisPositions = {
        yaxis: { position: 0.08, side: 'left', title: { standoff: 15 } },
        yaxis2: { position: 0.94, side: 'right', title: { standoff: 15 } }, // Move right axis further right
        yaxis3: { position: 0.01, side: 'left', title: { standoff: 15 } },
        yaxis4: { position: 0.99, side: 'right', title: { standoff: 15 } }, // Position at far right
        yaxis5: { position: 0.04, side: 'left', title: { standoff: 15 } }
    };
    
    // Apply positions to all axes
    for (const [axisName, position] of Object.entries(axisPositions)) {
        if (layout[axisName]) {
            layout[axisName].position = position.position;
            layout[axisName].side = position.side;
            if (!layout[axisName].title) {
                layout[axisName].title = {};
            }
            layout[axisName].title.standoff = position.title.standoff;
            
            // Only set fixed ranges for concentration axis (yaxis4) if autorange is not already true
            if (axisName === 'yaxis4' && layout[axisName].autorange !== true) {
                layout[axisName].range = [0, 100];
                layout[axisName].dtick = 20;  // Major ticks every 20%
                layout[axisName].tick0 = 0;   // Start ticks at 0
                if (layout[axisName].title && layout[axisName].title.text) {
                    // Add (%) to title if not already present
                    if (!layout[axisName].title.text.includes('(%)')) {
                        layout[axisName].title.text += layout[axisName].title.text.includes('%') ? '' : ' (%)';
                    }
                }
            }
        }
    }
    
    // Update stage division line colors based on theme
    if (layout.shapes && Array.isArray(layout.shapes)) {
        // Determine if we're in light mode
        const isLightTheme = document.body.classList.contains('light-theme');
        const lineColor = isLightTheme ? '#000000' : '#ffffff';
        
        // Update all vertical lines (stage division lines)
        const updates = {};
        layout.shapes.forEach((shape, index) => {
            if (shape.type === 'line' && shape.x0 === shape.x1 && shape.y0 !== shape.y1) {
                updates[`shapes[${index}].line.color`] = lineColor;
            }
        });
        
        // Only apply if we found stage lines
        if (Object.keys(updates).length > 0) {
            Object.assign(layout, updates);
        }
    }
    
    // Increase margins - reduce right margin to make better use of space
    if (layout.margin) {
        layout.margin.l = Math.max(layout.margin.l || 0, 80);
        layout.margin.r = Math.max(layout.margin.r || 0, 60); // Reduced from 80 to 60
        // Increase top margin to prevent overlap with the plot title
        layout.margin.t = Math.max(layout.margin.t || 0, 60);
    } else {
        layout.margin = { l: 80, r: 60, t: 60, b: 40 }; // Reduced right margin
    }
    
    Plotly.relayout(plotlyDiv, layout);
}

/**
 * Initialize a stage statistics table with value indicators
 */
function initStageStatisticsTable() {
    const table = document.querySelector('.stages-stats-table');
    if (!table) return;
    
    // Find all value cells and add indicators
    const rows = table.querySelectorAll('tbody tr');
    
    // Collect max values for each column
    const maxValues = {
        points: 0,
        temp: 0,
        pressure: 0,
        h2Flow: 0,
        n2Flow: 0,
        nh3: 0,
        outlet: 0
    };
    
    // First pass to find maximum values
    rows.forEach(row => {
        const points = parseFloat(row.cells[1].textContent) || 0;
        const temp = parseFloat(row.cells[3].textContent) || 0;
        const pressure = parseFloat(row.cells[4].textContent) || 0;
        const h2Flow = parseFloat(row.cells[5].textContent) || 0;
        const n2Flow = parseFloat(row.cells[6].textContent) || 0;
        const nh3 = parseFloat(row.cells[7].textContent) || 0;
        const outlet = parseFloat(row.cells[8].textContent) || 0;
        
        maxValues.points = Math.max(maxValues.points, points);
        maxValues.temp = Math.max(maxValues.temp, temp);
        maxValues.pressure = Math.max(maxValues.pressure, pressure);
        maxValues.h2Flow = Math.max(maxValues.h2Flow, h2Flow);
        maxValues.n2Flow = Math.max(maxValues.n2Flow, n2Flow);
        maxValues.nh3 = Math.max(maxValues.nh3, nh3);
        maxValues.outlet = Math.max(maxValues.outlet, outlet);
    });
    
    // Second pass to add indicators
    rows.forEach(row => {
        const cells = row.cells;
        
        // Add indicators to each cell with value
        addValueIndicator(cells[1], parseFloat(cells[1].textContent) || 0, maxValues.points); // Points
        addValueIndicator(cells[3], parseFloat(cells[3].textContent) || 0, maxValues.temp); // Temperature
        addValueIndicator(cells[4], parseFloat(cells[4].textContent) || 0, maxValues.pressure); // Pressure
        addValueIndicator(cells[5], parseFloat(cells[5].textContent) || 0, maxValues.h2Flow); // H2 Flow
        addValueIndicator(cells[6], parseFloat(cells[6].textContent) || 0, maxValues.n2Flow); // N2 Flow
        addValueIndicator(cells[7], parseFloat(cells[7].textContent) || 0, maxValues.nh3); // NH3
        addValueIndicator(cells[8], parseFloat(cells[8].textContent) || 0, maxValues.outlet); // Outlet
    });
}

/**
 * Add a value indicator to a cell
 * @param {HTMLElement} cell - The cell to add the indicator to
 * @param {number} value - The value for this cell
 * @param {number} maxValue - The maximum value in this column
 */
function addValueIndicator(cell, value, maxValue) {
    if (cell.textContent.trim() === 'N/A' || value <= 0 || maxValue <= 0) {
        return;
    }
    
    cell.classList.add('with-indicator');
    
    // Calculate width as percentage of max value
    const percentage = (value / maxValue) * 100;
    
    // Create indicator element
    const indicator = document.createElement('div');
    indicator.className = 'value-indicator';
    indicator.style.width = `${percentage}%`;
    
    cell.appendChild(indicator);
} 