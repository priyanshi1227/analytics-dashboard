// Sidebar Logout functionality
const sidebarLogoutButton = document.getElementById('sidebarLogoutButton');
sidebarLogoutButton.addEventListener('click', () => {
    // Clear user data
    localStorage.removeItem('token');
    localStorage.removeItem('user');

    // Show logout confirmation
    toast.show('You have been logged out successfully', 'success');

    // Redirect to login page after a short delay
    setTimeout(() => {
        window.location.href = '/login';
    }, 1500);
}
class StorageManager {
    static set(key, value) {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch (error) {
            console.error('Storage error:', error);
            return false;
        }
    }

    static get(key, defaultValue = null) {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch (error) {
            console.error('Storage error:', error);
            return defaultValue;
        }
    }

    static remove(key) {
        try {
            localStorage.removeItem(key);
            return true;
        } catch (error) {
            console.error('Storage error:', error);
            return false;
        }
    }

    static clear() {
        try {
            localStorage.clear();
            return true;
        } catch (error) {
            console.error('Storage error:', error);
            return false;
        }
    }
}

class ValidationUtils {
    static isEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    static isStrongPassword(password) {
        // At least 8 characters, 1 uppercase, 1 lowercase, 1 number
        const passwordRegex = /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/;
        return passwordRegex.test(password);
    }

    static isURL(string) {
        try {
            new URL(string);
            return true;
        } catch (_) {
            return false;
        }
    }

    static isEmpty(value) {
        if (value === null || value === undefined) return true;
        if (typeof value === 'string') return value.trim() === '';
        if (Array.isArray(value)) return value.length === 0;
        if (typeof value === 'object') return Object.keys(value).length === 0;
        return false;
    }
}

class ChartUtils {
    static createGradient(ctx, colorStops) {
        const gradient = ctx.createLinearGradient(0, 0, 0, 400);
        colorStops.forEach(stop => {
            gradient.addColorStop(stop.offset, stop.color);
        });
        return gradient;
    }

    static getRandomColor() {
        const colors = [
            '#3b82f6', '#ef4444', '#10b981', '#f59e0b', '#8b5cf6',
            '#06b6d4', '#84cc16', '#f97316', '#ec4899', '#6366f1'
        ];
        return colors[Math.floor(Math.random() * colors.length)];
    }

    static formatChartValue(value, type = 'number') {
        switch (type) {
            case 'percent':
                return `${value}%`;
            case 'currency':
                return DashboardUtils.formatCurrency(value);
            case 'short':
                return DashboardUtils.formatNumber(value);
            default:
                return value;
        }
    }
}
// notion-sync.js - Frontend integration for Notion live sync

class NotionSyncManager {
    constructor(apiBaseUrl = '/api') {
        this.apiBaseUrl = apiBaseUrl;
        this.syncTasks = new Map();
    }

    // Initialize with ntn token
    async initialize(ntnToken) {
        this.ntnToken = ntnToken;

        try {
            const response = await fetch(`${this.apiBaseUrl}/notion/verify-token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'Authorization': `Bearer ${await this.getAuthToken()}`
                },
                body: JSON.stringify({ token: ntnToken })
            });

            if (!response.ok) {
                throw new Error('Invalid Notion token');
            }

            return await response.json();
        } catch (error) {
            console.error('Notion initialization failed:', error);
            throw error;
        }
    }

    // Get all databases
    async getDatabases() {
        const response = await fetch(`${this.apiBaseUrl}/notion/databases`, {
            headers: {
                'Authorization': `Bearer ${await this.getAuthToken()}`,
                'X-Notion-Token': this.ntnToken
            }
        });
        return await response.json();
    }

    // Start live sync
    async startSync(databaseId, config = {}) {
        const syncConfig = {
            database_id: databaseId,
            sync_interval: config.syncInterval || 60,
            webhook_url: config.webhookUrl,
            sync_actions: config.syncActions || {},
            filters: config.filters
        };

        const response = await fetch(`${this.apiBaseUrl}/notion/sync/start`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${await this.getAuthToken()}`,
                'X-Notion-Token': this.ntnToken
            },
            body: JSON.stringify(syncConfig)
        });

        const result = await response.json();

        if (result.sync_id) {
            this.syncTasks.set(result.sync_id, {
                databaseId,
                config: syncConfig,
                status: 'running'
            });
        }

        return result;
    }

    // Stop sync
    async stopSync(syncId) {
        const response = await fetch(`${this.apiBaseUrl}/notion/sync/stop/${syncId}`, {
            method: 'POST',
            headers: {
                'Authorization': `Bearer ${await this.getAuthToken()}`,
                'X-Notion-Token': this.ntnToken
            }
        });

        if (response.ok) {
            this.syncTasks.delete(syncId);
        }

        return await response.json();
    }

    // Get sync status
    async getSyncStatus(syncId) {
        const response = await fetch(`${this.apiBaseUrl}/notion/sync/status/${syncId}`, {
            headers: {
                'Authorization': `Bearer ${await this.getAuthToken()}`,
                'X-Notion-Token': this.ntnToken
            }
        });
        return await response.json();
    }

    // Query database
    async queryDatabase(databaseId, query = {}) {
        const response = await fetch(`${this.apiBaseUrl}/notion/databases/${databaseId}/query`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Authorization': `Bearer ${await this.getAuthToken()}`,
                'X-Notion-Token': this.ntnToken
            },
            body: JSON.stringify(query)
        });
        return await response.json();
    }

    // Helper method to get auth token (implement based on your auth system)
    async getAuthToken() {
        // Return your JWT token or other auth token
        return localStorage.getItem('auth_token');
    }
}

}
// Error handling utilities
class ErrorHandler {
    static handleAPIError(error, fallbackMessage = 'An error occurred') {
        console.error('API Error:', error);

        if (error.status === 401) {
            // Unauthorized - redirect to login
            StorageManager.remove('token');
            StorageManager.remove('user');
            window.location.href = '/login';
            return 'Session expired. Please log in again.';
        }

        if (error.status === 403) {
            return 'You do not have permission to perform this action.';
        }

        if (error.status === 404) {
            return 'The requested resource was not found.';
        }

        if (error.status >= 500) {
            return 'Server error. Please try again later.';
        }

        return error.message || fallbackMessage;
    }

    static logError(error, context = '') {
        const timestamp = new Date().toISOString();
        const errorInfo = {
            timestamp,
            context,
            error: error.message,
            stack: error.stack,
            url: window.location.href,
            userAgent: navigator.userAgent
        };

        console.error('Application Error:', errorInfo);

        // In a real app, you might send this to an error tracking service
        // Sentry.captureException(error, { extra: errorInfo });
    }
}

// Export additional utilities
window.StorageManager = StorageManager;
window.ValidationUtils = ValidationUtils;
window.ChartUtils = ChartUtils;
window.ErrorHandler = ErrorHandler;