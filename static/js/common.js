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