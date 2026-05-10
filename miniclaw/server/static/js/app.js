const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const fileInput = document.getElementById('fileInput');
const emojiBtn = document.getElementById('emojiBtn');
const emojiPanel = document.getElementById('emojiPanel');
const themeToggle = document.getElementById('themeToggle');
const newChatBtn = document.getElementById('newChatBtn');
const sessionList = document.getElementById('sessionList');
const currentSessionTitle = document.getElementById('currentSessionTitle');

let currentSessionId = null;
const STORAGE_KEY = 'miniclaw_sessions';
const HIDDEN_SESSIONS_KEY = 'miniclaw_hidden_sessions';
const SESSION_PATH = 'C:\\Users\\Administrator\\.miniclaw\\sessions';

document.addEventListener('DOMContentLoaded', () => {
    localStorage.removeItem('miniclaw_theme');
    document.body.className = '';
    updateThemeIcon();

    initSessions();

    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    if (emojiBtn) emojiBtn.addEventListener('click', toggleEmojiPanel);
    if (newChatBtn) newChatBtn.addEventListener('click', createNewSession);

    window.addEventListener('storage', handleStorageChange);
});

function handleStorageChange(e) {
    if (e.key === STORAGE_KEY) {
        console.log('Storage changed from another tab, syncing sessions...');
        initSessions();
    }
}

function toggleTheme() {
    const body = document.body;
    if (body.className === 'dark-theme') {
        body.className = '';
        localStorage.setItem('miniclaw_theme', '');
    } else {
        body.className = 'dark-theme';
        localStorage.setItem('miniclaw_theme', 'dark-theme');
    }
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('.theme-icon');
    if (icon) {
        icon.textContent = document.body.className === 'dark-theme' ? '☀️' : '🌙';
    }
}

function toggleEmojiPanel() {
    emojiPanel.classList.toggle('show');
}

function getHiddenSessions() {
    try {
        const data = localStorage.getItem(HIDDEN_SESSIONS_KEY);
        return data ? JSON.parse(data) : [];
    } catch {
        return [];
    }
}

async function deleteSession(sessionId) {
    try {
        const res = await fetch(`/sessions/${sessionId}`, { method: 'DELETE' });
        const data = await res.json();
        
        if (data.success) {
            const sessions = getSessions();
            delete sessions[sessionId];
            localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));

            if (currentSessionId === sessionId) {
                const remainingIds = Object.keys(sessions);
                if (remainingIds.length > 0) {
                    currentSessionId = remainingIds.sort((a, b) =>
                        sessions[b].updatedAt - sessions[a].updatedAt
                    )[0];
                } else {
                    currentSessionId = await createDefaultSession();
                }
                loadSession(currentSessionId);
            }

            await initSessions();
        }
    } catch (err) {
        console.error('Failed to delete session:', err);
    }
}

function renameSession(sessionId, newTitle) {
    const sessions = getSessions();
    if (sessions[sessionId]) {
        sessions[sessionId].title = newTitle || '新对话';
        sessions[sessionId].updatedAt = Date.now();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        renderSessionList(sessions);
        if (currentSessionId === sessionId) {
            currentSessionTitle.textContent = sessions[sessionId].title;
        }
        
        fetch(`/sessions/${sessionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ title: sessions[sessionId].title })
        }).catch(err => console.error('Failed to sync title:', err));
    }
}

function pinSession(sessionId) {
    const sessions = getSessions();
    if (sessions[sessionId]) {
        sessions[sessionId].isPinned = !sessions[sessionId].isPinned;
        sessions[sessionId].updatedAt = Date.now();
        localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        renderSessionList(sessions);
        
        fetch(`/sessions/${sessionId}`, {
            method: 'PUT',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ is_pinned: sessions[sessionId].isPinned })
        }).catch(err => console.error('Failed to sync pin status:', err));
    }
}

function showSessionPath(sessionId) {
    const existingTooltip = document.querySelector('.path-tooltip');
    if (existingTooltip) {
        existingTooltip.remove();
    }

    const tooltip = document.createElement('div');
    tooltip.className = 'path-tooltip';
    tooltip.innerHTML = `
        <div class="path-tooltip-title">会话文件路径</div>
        <div class="path-tooltip-path">${SESSION_PATH}\\${sessionId}.json</div>
    `;

    const sessionItem = document.querySelector(`[data-session-id="${sessionId}"]`);
    if (sessionItem) {
        const rect = sessionItem.getBoundingClientRect();
        tooltip.style.top = `${rect.bottom + 8}px`;
        tooltip.style.left = `${rect.left}px`;
    }

    document.body.appendChild(tooltip);

    tooltip.addEventListener('click', () => tooltip.remove());
    setTimeout(() => {
        const existing = document.querySelector('.path-tooltip');
        if (existing) {
            existing.remove();
        }
    }, 3000);
}

function startRename(sessionId, e) {
    e.stopPropagation();
    const sessions = getSessions();
    const session = sessions[sessionId];
    if (!session) return;

    const sessionItem = document.querySelector(`[data-session-id="${sessionId}"]`);
    const sessionInfo = sessionItem.querySelector('.session-info');

    const input = document.createElement('input');
    input.type = 'text';
    input.className = 'session-title-input';
    input.value = session.title;

    const titleSpan = sessionInfo.querySelector('.session-title');
    titleSpan.style.display = 'none';

    sessionInfo.appendChild(input);
    input.focus();
    input.select();

    const finishRename = () => {
        const newTitle = input.value.trim();
        if (newTitle) {
            renameSession(sessionId, newTitle);
        } else {
            titleSpan.style.display = '';
            input.remove();
        }
    };

    input.addEventListener('blur', finishRename);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            input.blur();
        } else if (e.key === 'Escape') {
            input.value = session.title;
            input.blur();
        }
    });
}

async function initSessions() {
    try {
        const res = await fetch('/sessions');
        const data = await res.json();

        if (data.sessions && data.sessions.length > 0) {
            const sessions = {};
            const hidden = getHiddenSessions();

            for (const s of data.sessions) {
                if (!hidden.includes(s.key)) {
                    const detailRes = await fetch(`/sessions/${s.key}`);
                    const detailData = await detailRes.json();
                    
                    let title = '会话 ' + new Date(s.updated_at).toLocaleDateString('zh-CN');
                    let messages = [];
                    let isPinned = false;

                    if (detailData.success && detailData.session) {
                        messages = detailData.session.messages.map((msg, index) => ({
                            id: index,
                            role: msg.role,
                            content: msg.content,
                            timestamp: msg.timestamp || new Date(s.created_at).getTime() + index
                        }));
                        
                        if (detailData.session.title && detailData.session.title.trim()) {
                            title = detailData.session.title;
                        } else if (messages.length > 0) {
                            const firstUserMsg = messages.find(m => m.role === 'user');
                            if (firstUserMsg) {
                                title = firstUserMsg.content.length > 20 
                                    ? firstUserMsg.content.substring(0, 20) + '...' 
                                    : firstUserMsg.content;
                            }
                        }
                        
                        isPinned = detailData.session.is_pinned !== undefined 
                            ? detailData.session.is_pinned 
                            : false;
                    }
                    
                    sessions[s.key] = {
                        id: s.key,
                        title: title,
                        messages: messages,
                        createdAt: new Date(s.created_at).getTime(),
                        updatedAt: new Date(s.updated_at).getTime(),
                        isPinned: isPinned
                    };
                }
            }

            localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
        } else {
            localStorage.removeItem(STORAGE_KEY);
        }
    } catch (err) {
        console.error('Failed to fetch sessions from backend:', err);
    }

    let sessions = getSessions();
    const hidden = getHiddenSessions();

    Object.keys(sessions).forEach(key => {
        if (hidden.includes(key)) {
            delete sessions[key];
        }
    });

    if (Object.keys(sessions).length === 0) {
        currentSessionId = createDefaultSession();
        sessions = getSessions();
    } else {
        currentSessionId = Object.keys(sessions).sort((a, b) => {
            if (sessions[a].isPinned && !sessions[b].isPinned) return -1;
            if (!sessions[a].isPinned && sessions[b].isPinned) return 1;
            return sessions[b].updatedAt - sessions[a].updatedAt;
        })[0];
    }

    renderSessionList(sessions);
    loadSession(currentSessionId);
}

async function createDefaultSession() {
    const sessionId = 'session_' + Date.now();
    const sessions = getSessions();
    sessions[sessionId] = {
        id: sessionId,
        title: '新对话',
        messages: [{
            id: 'msg_1',
            role: 'bot',
            content: '你好！我是MiniClaw，你的AI助手。有什么我可以帮助你的吗？',
            timestamp: Date.now()
        }],
        createdAt: Date.now(),
        updatedAt: Date.now(),
        isPinned: false
    };
    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    
    fetch(`/sessions/${sessionId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title: '新对话', is_pinned: false })
    }).catch(err => console.error('Failed to create session on backend:', err));
    
    return sessionId;
}

async function createNewSession() {
    currentSessionId = await createDefaultSession();
    renderSessionList(getSessions());
    loadSession(currentSessionId);
    messageInput.focus();
}

function getSessions() {
    try {
        const data = localStorage.getItem(STORAGE_KEY);
        return data ? JSON.parse(data) : {};
    } catch {
        return {};
    }
}

function renderSessionList(sessions) {
    sessionList.innerHTML = '';

    const sortedIds = Object.keys(sessions).sort((a, b) => {
        if (sessions[a].isPinned && !sessions[b].isPinned) return -1;
        if (!sessions[a].isPinned && sessions[b].isPinned) return 1;
        return sessions[b].updatedAt - sessions[a].updatedAt;
    });

    sortedIds.forEach(sessionId => {
        const session = sessions[sessionId];
        const item = document.createElement('div');
        item.className = `session-item-wrapper`;
        item.dataset.sessionId = sessionId;

        const timeStr = formatTime(session.updatedAt);

        item.innerHTML = `
            <div class="session-item-inner">
                <div class="session-item ${sessionId === currentSessionId ? 'active' : ''}" onclick="selectSession('${sessionId}')">
                    <div class="session-icon">${session.isPinned ? '📌' : '💬'}</div>
                    <div class="session-info">
                        <div class="session-title">${escapeHtml(session.title)}</div>
                        <div class="session-time">${timeStr}</div>
                    </div>
                    <button class="dropdown-btn" data-session-id="${sessionId}" title="更多操作" onclick="event.stopPropagation(); toggleSessionActions('${sessionId}')">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M6 9l6 6 6-6"></path>
                        </svg>
                    </button>
                </div>
            </div>
            <div class="session-actions">
                <button class="session-action-btn" onclick="handleAction('${sessionId}', 'rename')" title="重命名">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
                        <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
                    </svg>
                </button>
                <button class="session-action-btn ${session.isPinned ? 'pinned' : ''}" onclick="handleAction('${sessionId}', 'pin')" title="${session.isPinned ? '取消置顶' : '置顶'}">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M12 2L12 22"></path>
                        <path d="M5 12L12 5L19 12"></path>
                        <path d="M5 12L12 19L19 12"></path>
                    </svg>
                </button>
                <button class="session-action-btn" onclick="handleAction('${sessionId}', 'path')" title="路径">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path>
                    </svg>
                </button>
                <button class="session-action-btn delete" onclick="handleAction('${sessionId}', 'delete')" title="删除">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M18 6L6 18"></path>
                        <path d="M6 6l12 12"></path>
                    </svg>
                </button>
            </div>
        `;

        sessionList.appendChild(item);
    });
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatTime(timestamp) {
    if (typeof timestamp === 'string') {
        timestamp = new Date(timestamp).getTime();
    }

    const now = Date.now();
    const diff = now - timestamp;

    if (diff < 60000) {
        return '刚刚';
    } else if (diff < 3600000) {
        return `${Math.floor(diff / 60000)}分钟前`;
    } else if (diff < 86400000) {
        return `${Math.floor(diff / 3600000)}小时前`;
    } else {
        const date = new Date(timestamp);
        return `${date.getMonth() + 1}/${date.getDate()}`;
    }
}

function selectSession(sessionId) {
    currentSessionId = sessionId;
    loadSession(sessionId);
    renderSessionList(getSessions());
    messageInput.focus();
}

function loadSession(sessionId) {
    const sessions = getSessions();
    const session = sessions[sessionId];

    if (!session) return;

    currentSessionTitle.textContent = session.title;
    chatMessages.innerHTML = '';

    if (session.messages && session.messages.length > 0) {
        session.messages.forEach(msg => {
            addMessage(msg.content, msg.role, false, msg.timestamp);
        });
    } else {
        addMessage('你好！我是MiniClaw，你的AI助手。有什么我可以帮助你的吗？', 'bot');
    }

    scrollToBottom();
}

function saveMessage(role, content) {
    const sessions = getSessions();
    const session = sessions[currentSessionId];

    if (!session) return;

    const newMsg = {
        id: 'msg_' + Date.now(),
        role,
        content,
        timestamp: Date.now()
    };

    session.messages.push(newMsg);
    session.updatedAt = Date.now();

    if (role === 'user' && session.messages.length === 2) {
        session.title = content.length > 20 ? content.substring(0, 20) + '...' : content;
    }

    localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
    renderSessionList(sessions);
    currentSessionTitle.textContent = session.title;
}

function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
    }
}

async function sendMessage() {
    const message = messageInput.value.trim();
    if (!message) return;

    addMessage(message, 'user');
    messageInput.value = '';

    showTyping();

    try {
        const res = await fetch('/webhook/msg', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, session_key: currentSessionId })
        });

        const data = await res.json();
        hideTyping();

        if (data.error) {
            addMessage('错误: ' + data.error, 'bot', true);
        } else {
            addMessage(data.result, 'bot');
        }

        await syncSessionFromBackend(currentSessionId);
    } catch (err) {
        hideTyping();
        addMessage('网络错误，请重试', 'bot', true);
    }
}

async function syncSessionFromBackend(sessionId) {
    try {
        const res = await fetch(`/sessions/${sessionId}`);
        const data = await res.json();
        
        if (data.success && data.session) {
            const sessions = getSessions();
            sessions[sessionId] = {
                id: sessionId,
                title: data.session.title && data.session.title.trim() 
                    ? data.session.title 
                    : sessions[sessionId]?.title || '会话 ' + new Date().toLocaleDateString('zh-CN'),
                messages: data.session.messages.map((msg, index) => ({
                    id: index,
                    role: msg.role,
                    content: msg.content,
                    timestamp: msg.timestamp || Date.now() + index
                })),
                createdAt: sessions[sessionId]?.createdAt || Date.now(),
                updatedAt: Date.now(),
                isPinned: data.session.is_pinned !== undefined 
                    ? data.session.is_pinned 
                    : sessions[sessionId]?.isPinned || false
            };
            
            localStorage.setItem(STORAGE_KEY, JSON.stringify(sessions));
            renderSessionList(sessions);
            
            if (currentSessionId === sessionId) {
                currentSessionTitle.textContent = sessions[sessionId].title;
            }
        }
    } catch (err) {
        console.error('Failed to sync session from backend:', err);
    }
}

function parseMarkdown(text) {
    if (!text) return '';

    let html = text;

    html = html.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    html = html.replace(/~~(.*?)~~/g, '<del>$1</del>');
    html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

    html = html.replace(/```(\w+)?\n([\s\S]*?)```/g, (match, lang, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });
    html = html.replace(/```([\s\S]*?)```/g, (match, code) => {
        return `<pre><code>${code.trim()}</code></pre>`;
    });

    html = html.replace(/^\#{3}\s+(.*$)/gim, '<h3>$1</h3>');
    html = html.replace(/^\#{2}\s+(.*$)/gim, '<h2>$1</h2>');
    html = html.replace(/^\#\s+(.*$)/gim, '<h1>$1</h1>');

    html = html.replace(/^\>\s+(.*$)/gim, '<blockquote>$1</blockquote>');

    html = html.replace(/^\-{3,}$/gim, '<hr>');

    html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');

    html = html.replace(/^\s*\d+\.\s+(.*$)/gim, '<li>$1</li>');
    html = html.replace(/^\s*[-*+]\s+(.*$)/gim, '<li>$1</li>');

    html = html.replace(/(<li>.*<\/li>)/gim, (match) => {
        if (!match.includes('</ul>') && !match.includes('</ol>')) {
            return `<ul>${match}</ul>`;
        }
        return match;
    });

    html = html.replace(/<\/ul>\s*<ul>/g, '');
    html = html.replace(/<\/ol>\s*<ol>/g, '');

    html = html.replace(/\n/g, '<br>');

    return html;
}

function addMessage(text, sender, isError = false, timestamp = Date.now()) {
    const message = document.createElement('div');
    message.className = `message ${sender}-message`;
    if (isError) {
        message.classList.add('error-message');
    }

    const avatar = document.createElement('div');
    avatar.className = `avatar ${sender}-avatar`;
    avatar.textContent = sender === 'user' ? '👤' : '🤖';

    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.innerHTML = parseMarkdown(text);

    const time = document.createElement('span');
    time.className = 'message-time';
    time.textContent = new Date(timestamp).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });

    bubble.appendChild(content);
    bubble.appendChild(time);
    message.appendChild(avatar);
    message.appendChild(bubble);

    chatMessages.appendChild(message);
    scrollToBottom();
}

function showTyping() {
    const typing = document.createElement('div');
    typing.className = 'message bot-message';
    typing.innerHTML = `
        <div class="avatar bot-avatar">🤖</div>
        <div class="typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    chatMessages.appendChild(typing);
    scrollToBottom();
}

function hideTyping() {
    const typing = chatMessages.querySelector('.typing-indicator');
    if (typing) typing.parentElement.remove();
}

function scrollToBottom() {
    chatMessages.scrollTo({ top: chatMessages.scrollHeight, behavior: 'smooth' });
}

function insertEmoji(e) {
    if (e.target.nodeName === 'SPAN') {
        messageInput.value += e.target.textContent;
        messageInput.focus();
        emojiPanel.classList.remove('show');
    }
}

emojiPanel.addEventListener('click', insertEmoji);

document.addEventListener('click', (e) => {
    if (!emojiPanel.contains(e.target) && e.target !== emojiBtn) {
        emojiPanel.classList.remove('show');
    }
    const actionBtn = e.target.closest('.session-action-btn');
    const tooltip = document.querySelector('.path-tooltip');
    if (tooltip && !tooltip.contains(e.target) && !actionBtn) {
        tooltip.remove();
    }
});

fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const maxSize = 10 * 1024 * 1024;
    if (file.size > maxSize) {
        alert('文件超过10MB限制');
        fileInput.value = '';
        return;
    }

    const formData = new FormData();
    formData.append('file', file);

    addMessage(`上传文件: ${file.name}`, 'user');
    saveMessage('user', `上传文件: ${file.name}`);

    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();

        if (data.success) {
            addMessage(`文件上传成功: ${data.filename}`, 'bot');
            saveMessage('bot', `文件上传成功: ${data.filename}`);
        } else {
            addMessage('上传失败: ' + data.error, 'bot', true);
            saveMessage('bot', '上传失败: ' + data.error);
        }
    } catch (err) {
        addMessage('上传失败，请重试', 'bot', true);
        saveMessage('bot', '上传失败，请重试');
    }

    fileInput.value = '';
});

function toggleSessionActions(sessionId) {
    const wrapper = document.querySelector(`[data-session-id="${sessionId}"]`);
    const isOpen = wrapper && wrapper.classList.contains('open');
    
    closeAllSessionActions();
    
    if (wrapper && !isOpen) {
        wrapper.classList.add('open');
    }
}

function handleAction(sessionId, action) {
    console.log('handleAction called:', sessionId, action);
    closeAllSessionActions();
    
    switch (action) {
        case 'rename':
            startRename(sessionId, event);
            break;
        case 'pin':
            pinSession(sessionId);
            break;
        case 'path':
            showSessionPath(sessionId);
            break;
        case 'delete':
            deleteSession(sessionId);
            break;
    }
}

function closeAllSessionActions() {
    document.querySelectorAll('.session-item-wrapper').forEach(wrapper => {
        wrapper.classList.remove('open');
    });
}

document.addEventListener('click', (e) => {
    const sessionActions = e.target.closest('.session-actions');
    const dropdownBtn = e.target.closest('.dropdown-btn');
    
    if (!sessionActions && !dropdownBtn) {
        closeAllSessionActions();
    }
});
