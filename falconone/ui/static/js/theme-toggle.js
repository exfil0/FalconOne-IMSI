// FalconOne Theme Toggle
// Version: 3.0.0

(function() {
    'use strict';
    
    // Initialize theme on page load
    const initTheme = () => {
        // Check for saved theme preference, default to 'light'
        const savedTheme = localStorage.getItem('falconone-theme') || 'light';
        const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
        
        // Use saved theme, or system preference if no saved theme
        const theme = savedTheme !== 'system' ? savedTheme : (prefersDark ? 'dark' : 'light');
        
        setTheme(theme, false);
    };
    
    // Set theme
    const setTheme = (theme, save = true) => {
        document.documentElement.setAttribute('data-theme', theme);
        
        if (save) {
            localStorage.setItem('falconone-theme', theme);
        }
        
        // Update theme toggle button
        updateThemeToggle(theme);
        
        // Emit custom event
        window.dispatchEvent(new CustomEvent('themeChanged', { detail: { theme } }));
        
        // Update Chart.js charts if present
        if (window.Chart) {
            updateChartThemes(theme);
        }
        
        // Announce theme change to screen readers
        announceThemeChange(theme);
    };
    
    // Update theme toggle button
    const updateThemeToggle = (theme) => {
        const toggle = document.querySelector('.theme-toggle');
        if (!toggle) return;
        
        const sunIcon = toggle.querySelector('.fa-sun');
        const moonIcon = toggle.querySelector('.fa-moon');
        
        if (theme === 'dark') {
            if (sunIcon) sunIcon.style.display = 'inline-block';
            if (moonIcon) moonIcon.style.display = 'none';
            toggle.setAttribute('aria-label', 'Switch to light mode');
            toggle.setAttribute('title', 'Switch to light mode');
        } else {
            if (sunIcon) sunIcon.style.display = 'none';
            if (moonIcon) moonIcon.style.display = 'inline-block';
            toggle.setAttribute('aria-label', 'Switch to dark mode');
            toggle.setAttribute('title', 'Switch to dark mode');
        }
    };
    
    // Toggle between light and dark themes
    const toggleTheme = () => {
        const currentTheme = document.documentElement.getAttribute('data-theme') || 'light';
        const newTheme = currentTheme === 'light' ? 'dark' : 'light';
        setTheme(newTheme);
        
        // Add animation class
        const toggle = document.querySelector('.theme-toggle');
        if (toggle) {
            toggle.classList.add('theme-toggle-animate');
            setTimeout(() => toggle.classList.remove('theme-toggle-animate'), 300);
        }
    };
    
    // Update Chart.js chart themes
    const updateChartThemes = (theme) => {
        const isDark = theme === 'dark';
        const textColor = isDark ? '#e9ecef' : '#212529';
        const gridColor = isDark ? '#3d4451' : '#dee2e6';
        
        // Update default Chart.js options
        if (window.Chart && window.Chart.defaults) {
            Chart.defaults.color = textColor;
            Chart.defaults.borderColor = gridColor;
            
            if (Chart.defaults.plugins && Chart.defaults.plugins.legend) {
                Chart.defaults.plugins.legend.labels.color = textColor;
            }
            
            if (Chart.defaults.scales) {
                Chart.defaults.scales.grid = Chart.defaults.scales.grid || {};
                Chart.defaults.scales.grid.color = gridColor;
                Chart.defaults.scales.ticks = Chart.defaults.scales.ticks || {};
                Chart.defaults.scales.ticks.color = textColor;
            }
        }
        
        // Update existing charts
        if (window.Chart && window.Chart.instances) {
            Object.values(Chart.instances).forEach(chart => {
                if (chart && chart.options) {
                    // Update chart colors
                    if (chart.options.plugins && chart.options.plugins.legend) {
                        chart.options.plugins.legend.labels.color = textColor;
                    }
                    
                    if (chart.options.scales) {
                        Object.keys(chart.options.scales).forEach(scaleKey => {
                            const scale = chart.options.scales[scaleKey];
                            if (scale.grid) scale.grid.color = gridColor;
                            if (scale.ticks) scale.ticks.color = textColor;
                        });
                    }
                    
                    chart.update();
                }
            });
        }
    };
    
    // Announce theme change to screen readers
    const announceThemeChange = (theme) => {
        const announcement = document.createElement('div');
        announcement.setAttribute('role', 'status');
        announcement.setAttribute('aria-live', 'polite');
        announcement.className = 'sr-only';
        announcement.textContent = `Theme changed to ${theme} mode`;
        document.body.appendChild(announcement);
        
        // Remove announcement after 1 second
        setTimeout(() => {
            document.body.removeChild(announcement);
        }, 1000);
    };
    
    // Listen for system theme changes
    const watchSystemTheme = () => {
        if (window.matchMedia) {
            const darkModeQuery = window.matchMedia('(prefers-color-scheme: dark)');
            
            darkModeQuery.addEventListener('change', (e) => {
                const savedTheme = localStorage.getItem('falconone-theme');
                
                // Only auto-update if user hasn't manually set a theme
                if (savedTheme === 'system' || !savedTheme) {
                    setTheme(e.matches ? 'dark' : 'light', false);
                }
            });
        }
    };
    
    // Create theme toggle button if it doesn't exist
    const createThemeToggle = () => {
        if (document.querySelector('.theme-toggle')) return;
        
        const toggle = document.createElement('button');
        toggle.className = 'theme-toggle';
        toggle.setAttribute('aria-label', 'Toggle dark mode');
        toggle.setAttribute('title', 'Toggle dark mode');
        toggle.innerHTML = `
            <i class="fas fa-moon" style="display: none;"></i>
            <i class="fas fa-sun" style="display: none;"></i>
        `;
        
        toggle.addEventListener('click', toggleTheme);
        
        document.body.appendChild(toggle);
    };
    
    // Keyboard shortcuts
    const setupKeyboardShortcuts = () => {
        document.addEventListener('keydown', (e) => {
            // Ctrl+Shift+T to toggle theme
            if (e.ctrlKey && e.shiftKey && e.key === 'T') {
                e.preventDefault();
                toggleTheme();
            }
        });
    };
    
    // Export API
    window.ThemeManager = {
        init: () => {
            initTheme();
            createThemeToggle();
            watchSystemTheme();
            setupKeyboardShortcuts();
        },
        setTheme: setTheme,
        toggleTheme: toggleTheme,
        getTheme: () => document.documentElement.getAttribute('data-theme') || 'light'
    };
    
    // Auto-initialize on DOMContentLoaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', window.ThemeManager.init);
    } else {
        window.ThemeManager.init();
    }
})();

// Add CSS animation for theme toggle button
const style = document.createElement('style');
style.textContent = `
    .theme-toggle-animate {
        animation: themeTogglePulse 0.3s ease-in-out;
    }
    
    @keyframes themeTogglePulse {
        0%, 100% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.2) rotate(15deg);
        }
    }
`;
document.head.appendChild(style);
