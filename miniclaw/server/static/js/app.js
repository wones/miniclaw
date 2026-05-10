const chatMessages = document.getElementById('chatMessages');
const messageInput = document.getElementById('messageInput');
const fileInput = document.getElementById('fileInput');
const emojiBtn = document.getElementById('emojiBtn');
const emojiPanel = document.getElementById('emojiPanel');
const themeToggle = document.getElementById('themeToggle');

let sessionKey = 'user_' + Date.now();

document.addEventListener('DOMContentLoaded', () => {
    const savedTheme = localStorage.getItem('miniclaw_theme');
    if (savedTheme) {
        document.body.className = savedTheme;
        updateThemeIcon();
    }
    
    const savedSession = localStorage.getItem('miniclaw_session');
    if (savedSession) sessionKey = savedSession;
    
    if (themeToggle) themeToggle.addEventListener('click', toggleTheme);
    if (emojiBtn) emojiBtn.addEventListener('click', toggleEmojiPanel);
    if (emojiPanel) emojiPanel.addEventListener('click', insertEmoji);
    if (fileInput) fileInput.addEventListener('change', handleFileUpload);
});

function toggleTheme() {
    const body = document.body;
    if (body.className === 'light-theme') {
        body.className = '';
        localStorage.setItem('miniclaw_theme', '');
    } else {
        body.className = 'light-theme';
        localStorage.setItem('miniclaw_theme', 'light-theme');
    }
    updateThemeIcon();
}

function updateThemeIcon() {
    const icon = document.querySelector('.theme-icon');
    if (icon) {
        icon.textContent = document.body.className === 'light-theme' ? '🌙' : '☀️';
    }
}

function toggleEmojiPanel() {
    console.log('toggleEmojiPanel called');
    console.log('emojiPanel:', emojiPanel);
    emojiPanel.classList.toggle('show');
    console.log('emojiPanel classes:', emojiPanel.classList);
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
            body: JSON.stringify({ message, session_key: sessionKey })
        });
        
        const data = await res.json();
        hideTyping();
        
        if (data.error) {
            addMessage('错误: ' + data.error, 'bot', true);
        } else {
            addMessage(data.result, 'bot');
        }
    } catch (err) {
        hideTyping();
        addMessage('网络错误，请重试', 'bot', true);
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

function addMessage(text, sender, isError = false) {
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
    time.textContent = new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' });
    
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
    console.log('insertEmoji called');
    console.log('target:', e.target);
    console.log('target.nodeName:', e.target.nodeName);
    if (e.target.nodeName === 'SPAN') {
        console.log('inserting emoji:', e.target.textContent);
        messageInput.value += e.target.textContent;
        messageInput.focus();
        emojiPanel.classList.remove('show');
    }
}

document.addEventListener('click', (e) => {
    if (!emojiPanel.contains(e.target) && e.target !== emojiBtn) {
        emojiPanel.classList.remove('show');
    }
});

async function handleFileUpload(e) {
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
    
    try {
        const res = await fetch('/upload', { method: 'POST', body: formData });
        const data = await res.json();
        
        if (data.success) {
            addMessage(`文件上传成功: ${data.filename}`, 'bot');
        } else {
            addMessage('上传失败: ' + data.error, 'bot', true);
        }
    } catch (err) {
        addMessage('上传失败，请重试', 'bot', true);
    }
    
    fileInput.value = '';
}

window.addEventListener('beforeunload', () => {
    localStorage.setItem('miniclaw_session', sessionKey);
});
