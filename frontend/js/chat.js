/**
 * PsychTrend - Enhanced Chat Interface
 */

// API Base URL
const API_BASE = '';

// State
let sessionId = null;
let isProcessing = false;

// DOM Elements
const welcomeCard = document.getElementById('welcomeCard');
const chatContainer = document.getElementById('chatContainer');
const chatMessages = document.getElementById('chatMessages');
const chatForm = document.getElementById('chatForm');
const messageInput = document.getElementById('messageInput');
const sendBtn = document.getElementById('sendBtn');
const startBtn = document.getElementById('startBtn');
const resetBtn = document.getElementById('resetBtn');
const themeToggle = document.getElementById('themeToggle');
const themeIcon = document.getElementById('themeIcon');
const progressBar = document.getElementById('progressBar');
const categoryBadge = document.getElementById('categoryBadge');
const statusText = document.getElementById('statusText');
const reportSection = document.getElementById('reportSection');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initTheme();
    checkExistingSession();
    setupEventListeners();
    addInputAnimations();
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
 * Session Management
 */
function checkExistingSession() {
    const savedSession = localStorage.getItem('sessionId');
    if (savedSession) {
        fetch(`${API_BASE}/session/${savedSession}`)
            .then(res => {
                if (res.ok) return res.json();
                throw new Error('Session not found');
            })
            .then(data => {
                sessionId = savedSession;
                if (data.is_complete) {
                    showChat();
                    loadConversationHistory(data.conversation_history);
                    showReportSection();
                } else if (data.conversation_history && data.conversation_history.length > 0) {
                    showChat();
                    loadConversationHistory(data.conversation_history);
                    updateProgress(data.current_category);
                }
            })
            .catch(() => {
                localStorage.removeItem('sessionId');
            });
    }
}

function loadConversationHistory(history) {
    chatMessages.innerHTML = '';
    history.forEach((msg, index) => {
        setTimeout(() => {
            addMessage(msg.content, msg.role === 'user' ? 'user' : 'bot', false);
        }, index * 50);
    });
    setTimeout(scrollToBottom, history.length * 50 + 100);
}

/**
 * Event Listeners
 */
function setupEventListeners() {
    startBtn.addEventListener('click', startConversation);
    chatForm.addEventListener('submit', handleSubmit);
    themeToggle.addEventListener('click', toggleTheme);
    resetBtn.addEventListener('click', resetData);

    messageInput.addEventListener('keypress', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    });
}

/**
 * Input Animations
 */
function addInputAnimations() {
    if (messageInput) {
        messageInput.addEventListener('focus', () => {
            messageInput.parentElement.style.boxShadow = '0 0 0 3px rgba(99, 102, 241, 0.2)';
        });

        messageInput.addEventListener('blur', () => {
            messageInput.parentElement.style.boxShadow = 'none';
        });
    }
}

/**
 * Start New Conversation
 */
async function startConversation() {
    try {
        startBtn.disabled = true;
        startBtn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Initializing...';

        const response = await fetch(`${API_BASE}/session`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({})
        });

        if (!response.ok) throw new Error('Failed to create session');

        const data = await response.json();
        sessionId = data.session_id;
        localStorage.setItem('sessionId', sessionId);

        showChat();

        // Add welcome message with typing effect
        showTypingIndicator();
        setTimeout(() => {
            hideTypingIndicator();
            addMessage(data.message, 'bot');
        }, 800);

    } catch (error) {
        console.error('Error starting conversation:', error);
        showNotification('Failed to start conversation. Please try again.', 'error');
        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Your Journey';
    }
}

/**
 * Handle Message Submit
 */
async function handleSubmit(e) {
    e.preventDefault();

    const message = messageInput.value.trim();
    if (!message || isProcessing || !sessionId) return;

    isProcessing = true;
    sendBtn.disabled = true;
    sendBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i>';

    // Add user message with animation
    addMessage(message, 'user');
    messageInput.value = '';

    // Show typing indicator
    showTypingIndicator();

    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: sessionId,
                message: message
            })
        });

        if (!response.ok) throw new Error('Failed to send message');

        const data = await response.json();

        // Simulate thinking time
        await new Promise(resolve => setTimeout(resolve, 500 + Math.random() * 500));

        hideTypingIndicator();
        addMessage(data.message, 'bot');

        updateProgress(data.current_category, data.progress);

        if (data.is_complete) {
            showReportSection();
            messageInput.disabled = true;
            messageInput.placeholder = '✨ Conversation complete!';
            showNotification('Your insight report is ready!', 'success');
        }

    } catch (error) {
        console.error('Error sending message:', error);
        hideTypingIndicator();
        addMessage('Oops! Something went wrong. Please try again.', 'bot');
    } finally {
        isProcessing = false;
        sendBtn.disabled = false;
        sendBtn.innerHTML = '<i class="fas fa-paper-plane"></i>';
        messageInput.focus();
    }
}

/**
 * Add Message to Chat
 */
function addMessage(text, role, animate = true) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}`;

    if (!animate) {
        messageDiv.style.animation = 'none';
    }

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.innerHTML = role === 'user'
        ? '<i class="fas fa-user"></i>'
        : '<i class="fas fa-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;

    const time = document.createElement('div');
    time.className = 'message-time';
    time.textContent = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    content.appendChild(time);

    if (role === 'user') {
        messageDiv.appendChild(content);
        messageDiv.appendChild(avatar);
    } else {
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
    }

    chatMessages.appendChild(messageDiv);
    scrollToBottom();
}

/**
 * Typing Indicator
 */
function showTypingIndicator() {
    const indicator = document.createElement('div');
    indicator.className = 'message bot';
    indicator.id = 'typingIndicator';

    indicator.innerHTML = `
        <div class="message-avatar">
            <i class="fas fa-robot"></i>
        </div>
        <div class="message-content typing-indicator">
            <span></span>
            <span></span>
            <span></span>
        </div>
    `;

    chatMessages.appendChild(indicator);
    scrollToBottom();

    statusText.innerHTML = '<i class="fas fa-circle" style="font-size: 6px; color: #f59e0b;"></i> Typing...';
}

function hideTypingIndicator() {
    const indicator = document.getElementById('typingIndicator');
    if (indicator) {
        indicator.remove();
    }
    statusText.innerHTML = '<i class="fas fa-circle" style="font-size: 6px;"></i> Online';
}

/**
 * Update Progress
 */
function updateProgress(category, progress = 0) {
    if (category) {
        const formatted = category.charAt(0).toUpperCase() + category.slice(1);
        categoryBadge.textContent = formatted;

        // Add pulse animation
        categoryBadge.style.animation = 'none';
        setTimeout(() => {
            categoryBadge.style.animation = 'pulse 0.5s ease';
        }, 10);
    }

    const percentage = Math.round(progress * 100);
    progressBar.style.width = `${percentage}%`;
}

/**
 * UI Helpers
 */
function showChat() {
    welcomeCard.style.display = 'none';
    chatContainer.style.display = 'flex';
    chatContainer.classList.add('active');
    messageInput.focus();
}

function showReportSection() {
    reportSection.style.display = 'block';
    const viewReportBtn = document.getElementById('viewReportBtn');
    if (viewReportBtn) {
        viewReportBtn.href = `report.html?session=${sessionId}`;
    }
}

function scrollToBottom() {
    chatMessages.scrollTo({
        top: chatMessages.scrollHeight,
        behavior: 'smooth'
    });
}

function showNotification(message, type = 'info') {
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        info: '#6366f1'
    };

    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 100px;
        right: 20px;
        background: ${colors[type]};
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        z-index: 9999;
        animation: slideIn 0.3s ease;
        font-weight: 500;
    `;
    notification.textContent = message;

    document.body.appendChild(notification);

    setTimeout(() => {
        notification.style.animation = 'fadeOut 0.3s ease forwards';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

/**
 * Reset Data
 */
async function resetData() {
    if (!confirm('Are you sure you want to delete all your data? This cannot be undone.')) {
        return;
    }

    try {
        await fetch(`${API_BASE}/reset`, { method: 'POST' });

        localStorage.removeItem('sessionId');
        sessionId = null;

        // Reset UI
        chatMessages.innerHTML = '';
        welcomeCard.style.display = 'block';
        chatContainer.style.display = 'none';
        chatContainer.classList.remove('active');
        reportSection.style.display = 'none';
        messageInput.disabled = false;
        messageInput.placeholder = 'Type your response...';
        progressBar.style.width = '0%';
        categoryBadge.textContent = 'Introduction';

        startBtn.disabled = false;
        startBtn.innerHTML = '<i class="fas fa-play me-2"></i>Start Your Journey';

        showNotification('All data deleted successfully!', 'success');

    } catch (error) {
        console.error('Error resetting data:', error);
        showNotification('Failed to reset data. Please try again.', 'error');
    }
}

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from { transform: translateX(100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
    @keyframes fadeOut {
        from { opacity: 1; }
        to { opacity: 0; transform: translateY(-20px); }
    }
`;
document.head.appendChild(style);
