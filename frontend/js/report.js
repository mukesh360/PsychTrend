/**
 * PsychTrend - Report Page JavaScript
 */

// API Base URL
const API_BASE = '';

// DOM Elements
const loadingState = document.getElementById('loadingState');
const noDataState = document.getElementById('noDataState');
const reportContent = document.getElementById('reportContent');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    loadReport();
    setupEventListeners();
});

/**
 * Theme Management
 */
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 'dark';
    document.documentElement.setAttribute('data-bs-theme', savedTheme);
    updateThemeIcon(savedTheme);
}

function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-bs-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    document.documentElement.setAttribute('data-bs-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
}

function updateThemeIcon(theme) {
    if (themeIcon) {
        themeIcon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
    }
}

/**
 * Event Listeners
 */
function setupEventListeners() {
    themeToggle.addEventListener('click', toggleTheme);
}

/**
 * Get Session ID from URL or localStorage
 */
function getSessionId() {
    // Check URL params first
    const urlParams = new URLSearchParams(window.location.search);
    const sessionFromUrl = urlParams.get('session');

    if (sessionFromUrl) {
        return sessionFromUrl;
    }

    // Fall back to localStorage
    return localStorage.getItem('sessionId');
}

/**
 * Load Report Data
 */
async function loadReport() {
    const sessionId = getSessionId();

    if (!sessionId) {
        showNoData();
        return;
    }

    try {
        const response = await fetch(`${API_BASE}/report/${sessionId}`);

        if (!response.ok) {
            if (response.status === 404) {
                showNoData();
                return;
            }
            throw new Error('Failed to load report');
        }

        const data = await response.json();
        displayReport(data);

    } catch (error) {
        console.error('Error loading report:', error);
        showNoData();
    }
}

/**
 * Show No Data State
 */
function showNoData() {
    loadingState.style.display = 'none';
    noDataState.style.display = 'block';
    reportContent.style.display = 'none';
}

/**
 * Display Report
 */
function displayReport(data) {
    loadingState.style.display = 'none';
    noDataState.style.display = 'none';
    reportContent.style.display = 'block';

    // Header info
    document.getElementById('userName').textContent = data.user_name || 'User';
    document.getElementById('reportDate').textContent = formatDate(data.generated_at);

    // Executive Summary
    document.getElementById('executiveSummary').textContent = data.executive_summary;

    // Disclaimer
    if (data.disclaimer) {
        document.getElementById('disclaimerText').textContent = data.disclaimer;
    }

    // Trend Analysis
    displayTrends(data.trend_analysis);

    // Behavioral Profile
    displayBehavioralProfile(data.behavioral_profile);

    // Predictions
    displayPredictions(data.predictions);

    // Strengths & Growth
    displayStrengths(data.strengths);
    displayGrowthAreas(data.growth_opportunities);
}

/**
 * Display Trend Analysis
 */
function displayTrends(trends) {
    if (!trends) return;

    // Motivation
    if (trends.motivation) {
        const score = Math.round(trends.motivation.score * 100);
        document.getElementById('motivationScore').textContent = `${score}%`;
        document.getElementById('motivationBar').style.width = `${score}%`;
        document.getElementById('motivationDesc').textContent = trends.motivation.description || '-';
    }

    // Consistency
    if (trends.consistency) {
        const score = Math.round(trends.consistency.score * 100);
        document.getElementById('consistencyScore').textContent = `${score}%`;
        document.getElementById('consistencyBar').style.width = `${score}%`;
        document.getElementById('consistencyDesc').textContent = trends.consistency.description || '-';
    }

    // Growth Orientation
    if (trends.growth_orientation) {
        const score = Math.round(trends.growth_orientation.score * 100);
        document.getElementById('growthScore').textContent = `${score}%`;
        document.getElementById('growthBar').style.width = `${score}%`;
        document.getElementById('growthDesc').textContent = trends.growth_orientation.description || '-';
    }

    // Stress Response
    if (trends.stress_response) {
        const score = Math.round(trends.stress_response.score * 100);
        document.getElementById('stressScore').textContent = `${score}%`;
        document.getElementById('stressBar').style.width = `${score}%`;
        document.getElementById('stressDesc').textContent = trends.stress_response.description || '-';
    }
}

/**
 * Display Behavioral Profile
 */
function displayBehavioralProfile(profile) {
    if (!profile) return;

    const archetype = profile.primary_archetype;
    if (archetype) {
        document.getElementById('archetypeName').textContent =
            archetype.cluster_name ? archetype.cluster_name.charAt(0).toUpperCase() + archetype.cluster_name.slice(1) : '-';
        document.getElementById('archetypeAffinity').textContent =
            archetype.affinity ? `${Math.round(archetype.affinity * 100)}% affinity` : '-';
        document.getElementById('archetypeDescription').textContent =
            archetype.description || '-';

        // Display traits
        const traitsContainer = document.getElementById('archetypeTraits');
        traitsContainer.innerHTML = '';

        if (archetype.traits && archetype.traits.length > 0) {
            archetype.traits.forEach(trait => {
                const badge = document.createElement('span');
                badge.className = 'trait-badge';
                badge.textContent = trait;
                traitsContainer.appendChild(badge);
            });
        }
    }
}

/**
 * Display Predictions
 */
function displayPredictions(predictions) {
    const container = document.getElementById('predictionsContainer');
    container.innerHTML = '';

    if (!predictions || predictions.length === 0) {
        container.innerHTML = '<p class="text-muted">No predictions available.</p>';
        return;
    }

    predictions.forEach(pred => {
        const col = document.createElement('div');
        col.className = 'col-md-6 mb-3';

        const probability = pred.probability ? `${Math.round(pred.probability * 100)}%` : 'N/A';
        const confidenceClass = getConfidenceClass(pred.confidence);

        col.innerHTML = `
            <div class="prediction-card">
                <div class="d-flex justify-content-between align-items-start mb-2">
                    <h6 class="mb-0">${pred.type || 'Prediction'}</h6>
                    <span class="prediction-probability">${probability}</span>
                </div>
                <p class="small mb-2">${pred.explanation || ''}</p>
                <div class="d-flex align-items-center">
                    <span class="badge ${confidenceClass} me-2">
                        ${pred.confidence || 'medium'} confidence
                    </span>
                </div>
                ${pred.factors && pred.factors.length > 0 ? `
                    <div class="mt-2">
                        <small class="text-muted">Contributing factors:</small>
                        <ul class="small mb-0">
                            ${pred.factors.map(f => `<li>${f}</li>`).join('')}
                        </ul>
                    </div>
                ` : ''}
            </div>
        `;

        container.appendChild(col);
    });
}

/**
 * Display Strengths
 */
function displayStrengths(strengths) {
    const container = document.getElementById('strengthsList');
    container.innerHTML = '';

    if (!strengths || strengths.length === 0) {
        container.innerHTML = '<li class="text-muted">No strengths identified yet.</li>';
        return;
    }

    strengths.forEach(strength => {
        const item = document.createElement('li');
        item.className = 'strength-item';
        item.innerHTML = `
            <i class="fas fa-check-circle"></i>
            <span>${strength}</span>
        `;
        container.appendChild(item);
    });
}

/**
 * Display Growth Areas
 */
function displayGrowthAreas(areas) {
    const container = document.getElementById('growthList');
    container.innerHTML = '';

    if (!areas || areas.length === 0) {
        container.innerHTML = '<li class="text-muted">No growth areas identified yet.</li>';
        return;
    }

    areas.forEach(area => {
        const item = document.createElement('li');
        item.className = 'growth-item';
        item.innerHTML = `
            <i class="fas fa-arrow-circle-up"></i>
            <span>${area}</span>
        `;
        container.appendChild(item);
    });
}

/**
 * Helper Functions
 */
function formatDate(dateString) {
    if (!dateString) return '-';
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'long',
        day: 'numeric',
        hour: '2-digit',
        minute: '2-digit'
    });
}

function getConfidenceClass(confidence) {
    switch (confidence) {
        case 'high':
            return 'bg-success';
        case 'low':
            return 'bg-warning text-dark';
        default:
            return 'bg-info';
    }
}
