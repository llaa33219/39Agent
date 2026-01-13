let ws = null;
let selectedCharacter = null;
let isRecording = false;
let currentSessionId = null;
let reconnectAttempts = 0;
const MAX_RECONNECT_ATTEMPTS = 5;
const RECONNECT_DELAY = 2000;

document.addEventListener('DOMContentLoaded', async () => {
    await loadCharacters();
    await loadISOs();
    await checkExistingSession();
});

async function checkExistingSession() {
    const savedSessionId = localStorage.getItem('39agent_session_id');
    if (!savedSessionId) return;
    
    try {
        const response = await fetch(`/api/session/${savedSessionId}`);
        const data = await response.json();
        
        if (data.exists && data.running) {
            console.log('[*] Found active session, reconnecting...');
            await reconnectToSession(savedSessionId, data.character);
        } else {
            clearSessionStorage();
        }
    } catch (e) {
        console.error('Failed to check existing session:', e);
        clearSessionStorage();
    }
}

function clearSessionStorage() {
    localStorage.removeItem('39agent_session_id');
    localStorage.removeItem('39agent_character');
}

async function reconnectToSession(sessionId, character) {
    ws = new WebSocket(`ws://${window.location.host}/ws/session`);
    
    ws.onopen = () => {
        console.log('[*] WebSocket connected, sending reconnect request...');
        ws.send(JSON.stringify({
            action: 'reconnect',
            session_id: sessionId
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        
        if (data.type === 'error' && data.message === 'Session not found or expired') {
            console.log('[!] Session expired, showing setup screen');
            clearSessionStorage();
            return;
        }
        
        handleMessage(data);
    };
    
    ws.onclose = () => {
        console.log('[*] WebSocket closed, attempting reconnect...');
        attemptReconnect();
    };
    
    ws.onerror = (e) => {
        console.error('WebSocket error:', e);
    };
    
    currentSessionId = sessionId;
    
    if (character) {
        showRunningScreen(character);
    }
}

function attemptReconnect() {
    if (reconnectAttempts >= MAX_RECONNECT_ATTEMPTS) {
        console.log('[!] Max reconnect attempts reached');
        clearSessionStorage();
        document.getElementById('running-screen').classList.remove('active');
        document.getElementById('setup-screen').classList.add('active');
        return;
    }
    
    reconnectAttempts++;
    console.log(`[*] Reconnect attempt ${reconnectAttempts}/${MAX_RECONNECT_ATTEMPTS}`);
    
    setTimeout(async () => {
        const savedSessionId = localStorage.getItem('39agent_session_id');
        if (savedSessionId) {
            try {
                const response = await fetch(`/api/session/${savedSessionId}`);
                const data = await response.json();
                if (data.exists && data.running) {
                    await reconnectToSession(savedSessionId, data.character);
                    reconnectAttempts = 0;
                } else {
                    clearSessionStorage();
                }
            } catch (e) {
                attemptReconnect();
            }
        }
    }, RECONNECT_DELAY);
}

function saveSession(sessionId, character) {
    localStorage.setItem('39agent_session_id', sessionId);
    localStorage.setItem('39agent_character', JSON.stringify(character));
    currentSessionId = sessionId;
}

async function loadCharacters() {
    try {
        const response = await fetch('/api/characters');
        const data = await response.json();
        
        const container = document.getElementById('character-list');
        
        if (data.characters.length === 0) {
            container.innerHTML = `
                <div class="character-card selected" onclick="selectCharacter('default')">
                    <div class="placeholder">?</div>
                    <span>Default</span>
                </div>
            `;
            selectedCharacter = 'default';
            return;
        }
        
        container.innerHTML = data.characters.map(char => `
            <div class="character-card" data-name="${char.name}" onclick="selectCharacter('${char.name}')">
                ${char.has_image 
                    ? `<img src="/api/character/${char.name}/image" alt="${char.display_name}">`
                    : `<div class="placeholder">${char.display_name[0].toUpperCase()}</div>`
                }
                <span>${char.display_name}</span>
            </div>
        `).join('');
        
        if (data.characters.length > 0) {
            selectCharacter(data.characters[0].name);
        }
    } catch (e) {
        console.error('Failed to load characters:', e);
    }
}

async function loadISOs() {
    try {
        const response = await fetch('/api/isos');
        const data = await response.json();
        
        const select = document.getElementById('iso-select');
        
        data.isos.forEach(iso => {
            const option = document.createElement('option');
            option.value = iso;
            option.textContent = iso;
            select.appendChild(option);
        });
    } catch (e) {
        console.error('Failed to load ISOs:', e);
    }
}

function selectCharacter(name) {
    document.querySelectorAll('.character-card').forEach(card => {
        card.classList.remove('selected');
    });
    
    const card = document.querySelector(`.character-card[data-name="${name}"]`);
    if (card) {
        card.classList.add('selected');
    }
    
    selectedCharacter = name;
}

function toggleAdvanced() {
    const settings = document.getElementById('advanced-settings');
    settings.classList.toggle('collapsed');
}

async function startSession() {
    const task = document.getElementById('task-input').value.trim();
    const isoPath = document.getElementById('iso-select').value;
    const diskGb = parseInt(document.getElementById('disk-gb').value);
    const ramMb = parseInt(document.getElementById('ram-mb').value);
    const vramMb = parseInt(document.getElementById('vram-mb').value);
    
    if (!task) {
        alert('Please enter a task for the agent.');
        return;
    }
    
    document.getElementById('start-btn').disabled = true;
    
    ws = new WebSocket(`ws://${window.location.host}/ws/session`);
    
    ws.onopen = () => {
        ws.send(JSON.stringify({
            action: 'start',
            character: selectedCharacter || 'default',
            task: task,
            iso_path: isoPath || null,
            disk_gb: diskGb,
            ram_mb: ramMb,
            vram_mb: vramMb
        }));
    };
    
    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };
    
    ws.onclose = () => {
        console.log('WebSocket closed');
        if (currentSessionId) {
            attemptReconnect();
        } else {
            document.getElementById('start-btn').disabled = false;
        }
    };
    
    ws.onerror = (e) => {
        console.error('WebSocket error:', e);
        alert('Connection error. Please try again.');
        document.getElementById('start-btn').disabled = false;
    };
}

function handleMessage(data) {
    switch (data.type) {
        case 'status':
            updateStatus(data.status);
            if (data.status === 'running' && data.session_id) {
                saveSession(data.session_id, data.character);
                showRunningScreen(data.character);
            } else if (data.status === 'running') {
                showRunningScreen(data.character);
            }
            break;
            
        case 'screen':
            updateScreen(data.image);
            break;
            
        case 'speak':
            updateSpeech(data.text);
            break;
            
        case 'tool':
            addToolEntry(data.name, data.params, true);
            break;
            
        case 'agent_response':
            if (data.tool_result) {
                updateLastToolResult(data.tool_result.success, data.tool_result.message);
            }
            break;
            
        case 'error':
            console.error('Error:', data.message);
            updateStatus('error');
            break;
    }
}

function updateStatus(status) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');
    
    dot.className = 'dot';
    
    switch (status) {
        case 'initializing':
            text.textContent = 'Initializing...';
            break;
        case 'starting_vm':
            text.textContent = 'Starting VM...';
            break;
        case 'running':
            dot.classList.add('running');
            text.textContent = 'Running';
            break;
        case 'stopped':
            text.textContent = 'Stopped';
            break;
        case 'error':
            dot.classList.add('error');
            text.textContent = 'Error';
            break;
        case 'restarted':
            dot.classList.add('running');
            text.textContent = 'Restarted';
            break;
    }
}

function showRunningScreen(character) {
    document.getElementById('setup-screen').classList.remove('active');
    document.getElementById('running-screen').classList.add('active');
    
    const hudImg = document.getElementById('hud-character-image');
    if (hudImg && character && character.has_image) {
        hudImg.src = `/api/character/${character.name}/image`;
    }
}

function updateScreen(base64Image) {
    const img = document.getElementById('screen-image');
    const noScreen = document.getElementById('no-screen');
    
    img.src = `data:image/png;base64,${base64Image}`;
    img.classList.add('visible');
    noScreen.style.display = 'none';
}

function updateSpeech(text) {
    const speechText = document.getElementById('hud-speech-text');
    if (!speechText) return;
    
    speechText.style.opacity = '0';
    speechText.textContent = text;
    
    void speechText.offsetWidth;
    
    speechText.style.transition = 'opacity 0.3s ease';
    speechText.style.opacity = '1';
}

function addToolEntry(name, params, pending = false) {
    const container = document.getElementById('tool-history');
    
    if (container) {
        const entry = document.createElement('div');
        entry.className = 'tool-entry' + (pending ? '' : ' success');
        entry.innerHTML = `
            <div class="tool-name">${name}</div>
            <div class="tool-params">${JSON.stringify(params, null, 2)}</div>
        `;
        
        container.insertBefore(entry, container.firstChild);
        
        if (container.children.length > 20) {
            container.removeChild(container.lastChild);
        }
    }
    
    const hudTool = document.getElementById('hud-tool-status');
    if (hudTool) hudTool.textContent = name;
}

function updateLastToolResult(success, message) {
    const container = document.getElementById('tool-history');
    const firstEntry = container.firstChild;
    
    if (firstEntry) {
        firstEntry.classList.remove('pending');
        firstEntry.classList.add(success ? 'success' : 'error');
    }
}

function toggleRecording() {
    isRecording = !isRecording;
    const btn = document.getElementById('record-btn');
    
    if (isRecording) {
        btn.classList.add('recording');
        btn.innerHTML = '<span class="icon">&#9632;</span> Stop Recording';
    } else {
        btn.classList.remove('recording');
        btn.innerHTML = '<span class="icon">&#9679;</span> Record';
    }
}

function restartAgent() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'restart' }));
    }
}

function stopAgent() {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ action: 'stop' }));
    }
    
    clearSessionStorage();
    currentSessionId = null;
    reconnectAttempts = 0;
    
    document.getElementById('running-screen').classList.remove('active');
    document.getElementById('setup-screen').classList.add('active');
    document.getElementById('start-btn').disabled = false;
    
    document.getElementById('screen-image').classList.remove('visible');
    document.getElementById('no-screen').style.display = 'block';
    
    const hudSpeech = document.getElementById('hud-speech-text');
    const hudTool = document.getElementById('hud-tool-status');
    const toolHistory = document.getElementById('tool-history');
    
    if (hudSpeech) hudSpeech.textContent = '';
    if (hudTool) hudTool.textContent = '';
    if (toolHistory) toolHistory.innerHTML = '';
}
