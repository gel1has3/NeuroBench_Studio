/**
 * EEG Foundation Model Dashboard - Core JavaScript
 * Handles theme toggling, auto-refresh, cache management, and UI helpers
 */

(function() {
    'use strict';

    // ==============================================================
    // Theme Management
    // ==============================================================
    
    function updateThemeIcon(theme) {
        const icon = document.getElementById('themeIcon');
        if (icon) {
            if (theme === 'dark') {
                icon.className = 'bi bi-moon-stars-fill';
            } else {
                icon.className = 'bi bi-sun-fill';
            }
        }
    }

    function toggleTheme() {
        const current = document.documentElement.getAttribute('data-bs-theme');
        const next = current === 'dark' ? 'light' : 'dark';
        document.documentElement.setAttribute('data-bs-theme', next);
        localStorage.setItem('eeg-dashboard-theme', next);
        updateThemeIcon(next);
    }

    // ==============================================================
    // Refresh Management
    // ==============================================================

    function refreshData() {
        showLoading(true);
        
        // Call API to clear cache
        fetch('/api/refresh')
            .then(response => response.json())
            .then(data => {
                updateLastRefresh();
                // Reload the page to get fresh data
                window.location.reload();
            })
            .catch(error => {
                console.warn('Refresh API call failed, reloading anyway:', error);
                window.location.reload();
            })
            .finally(() => {
                setTimeout(() => showLoading(false), 1000);
            });
    }

    function clearCache() {
        showLoading(true);
        fetch('/api/refresh')
            .then(response => response.json())
            .then(data => {
                updateLastRefresh();
                showToast('success', 'Cache cleared successfully');
            })
            .catch(error => {
                showToast('danger', 'Failed to clear cache: ' + error.message);
            })
            .finally(() => {
                setTimeout(() => showLoading(false), 1000);
            });
    }

    function exportData() {
        // Collect all visible data and export as JSON
        const data = {
            timestamp: new Date().toISOString(),
            url: window.location.href,
            // Add any data tables visible on the page
            tables: []
        };
        
        document.querySelectorAll('table').forEach((table, idx) => {
            const rows = [];
            table.querySelectorAll('tbody tr').forEach(row => {
                const cells = [];
                row.querySelectorAll('td').forEach(cell => {
                    cells.push(cell.textContent.trim());
                });
                rows.push(cells);
            });
            data.tables.push({
                id: table.id || `table-${idx}`,
                headers: Array.from(table.querySelectorAll('thead th')).map(th => th.textContent.trim()),
                rows: rows
            });
        });
        
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `eeg-dashboard-export-${new Date().toISOString().slice(0, 10)}.json`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        
        showToast('success', 'Data exported successfully');
    }

    // ==============================================================
    // Loading Overlay
    // ==============================================================

    function showLoading(show) {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) {
            if (show) {
                overlay.classList.remove('d-none');
            } else {
                overlay.classList.add('d-none');
            }
        }
    }

    // ==============================================================
    // Toast Notifications
    // ==============================================================

    function showToast(type, message) {
        // Create toast container if it doesn't exist
        let container = document.getElementById('toastContainer');
        if (!container) {
            container = document.createElement('div');
            container.id = 'toastContainer';
            container.className = 'toast-container position-fixed bottom-0 end-0 p-3';
            container.style.zIndex = '9999';
            document.body.appendChild(container);
        }
        
        const toastId = 'toast-' + Date.now();
        const colors = {
            success: 'bg-success text-white',
            danger: 'bg-danger text-white',
            warning: 'bg-warning text-dark',
            info: 'bg-info text-dark',
        };
        
        const bgClass = colors[type] || colors.info;
        
        container.innerHTML += `
            <div id="${toastId}" class="toast ${bgClass}" role="alert">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;
        
        const toastEl = document.getElementById(toastId);
        if (toastEl) {
            const toast = new bootstrap.Toast(toastEl, { delay: 3000 });
            toast.show();
            
            // Clean up after hidden
            toastEl.addEventListener('hidden.bs.toast', function() {
                this.remove();
            });
        }
    }

    // ==============================================================
    // Timestamp Updates
    // ==============================================================

    function updateLastRefresh() {
        const el = document.getElementById('lastRefreshTime');
        if (el) {
            const now = new Date();
            el.textContent = now.toLocaleTimeString();
        }
        const lastUpdated = document.getElementById('lastUpdated');
        if (lastUpdated) {
            const now = new Date();
            lastUpdated.textContent = now.toLocaleString();
        }
    }

    // ==============================================================
    // Checkpoint Details Modal
    // ==============================================================

    function showCheckpointDetails(name, sizeMb, modified, path) {
        document.getElementById('ckptName').textContent = name;
        document.getElementById('ckptSize').textContent = sizeMb + ' MB';
        document.getElementById('ckptModified').textContent = modified.replace('T', ' ');
        document.getElementById('ckptPath').textContent = path;
        
        const modal = new bootstrap.Modal(document.getElementById('checkpointModal'));
        modal.show();
    }

    // ==============================================================
    // Auto-Refresh Timer
    // ==============================================================

    let refreshTimer = null;
    
    function startAutoRefresh(intervalSeconds) {
        if (refreshTimer) {
            clearInterval(refreshTimer);
        }
        if (intervalSeconds > 0) {
            refreshTimer = setInterval(() => {
                updateLastRefresh();
                // For pages that need auto-reload, we can do a soft refresh
                const refreshBtn = document.getElementById('refreshBtn');
                if (refreshBtn) {
                    refreshBtn.classList.add('spinning');
                    setTimeout(() => {
                        refreshBtn.classList.remove('spinning');
                    }, 1000);
                }
            }, intervalSeconds * 1000);
        }
    }

    // ==============================================================
    // API Health Check
    // ==============================================================

    function checkApiHealth() {
        fetch('/api/health')
            .then(response => response.json())
            .then(data => {
                const dot = document.getElementById('apiStatusDot');
                const text = document.getElementById('apiStatusText');
                if (dot && text) {
                    dot.className = 'status-dot status-dot-ok me-2';
                    text.textContent = 'Connected';
                    text.className = 'ms-auto text-success small';
                }
            })
            .catch(error => {
                const dot = document.getElementById('apiStatusDot');
                const text = document.getElementById('apiStatusText');
                if (dot && text) {
                    dot.className = 'status-dot status-dot-err me-2';
                    text.textContent = 'Disconnected';
                    text.className = 'ms-auto text-danger small';
                }
            });
    }

    // ==============================================================
    // Initialization
    // ==============================================================

    document.addEventListener('DOMContentLoaded', function() {
        // Theme toggle
        const themeBtn = document.getElementById('themeToggle');
        if (themeBtn) {
            themeBtn.addEventListener('click', toggleTheme);
        }
        
        // Refresh button
        const refreshBtn = document.getElementById('refreshBtn');
        if (refreshBtn) {
            refreshBtn.addEventListener('click', refreshData);
        }
        
        // Get refresh interval from the status text
        const statusEl = document.getElementById('refreshStatus');
        if (statusEl) {
            const match = statusEl.textContent.match(/(\d+)/);
            if (match) {
                startAutoRefresh(parseInt(match[1]));
            }
        }
        
        // Initial timestamp
        updateLastRefresh();
        
        // Health check
        checkApiHealth();
        
        // Periodic health check
        setInterval(checkApiHealth, 60000);
    });

    // ==============================================================
    // Expose functions globally for inline HTML use
    // ==============================================================
    window.refreshData = refreshData;
    window.clearCache = clearCache;
    window.exportData = exportData;
    window.showLoading = showLoading;
    window.showToast = showToast;
    window.showCheckpointDetails = showCheckpointDetails;
    window.toggleTheme = toggleTheme;

})();