// Bulletproof Global State for Module Scope
window.STATE = {
    activeGen: 1,
    maxGenSeen: 1,
    currentRunID: 'latest',
    generations: {}
};

window.switchGen = async (gen) => {
    window.STATE.activeGen = gen;
    if (gen > window.STATE.maxGenSeen) window.STATE.maxGenSeen = gen;
    
    // Initialize if not present
    if (!window.STATE.generations[gen]) {
        window.STATE.generations[gen] = { gen: gen, title: `GEN ${gen}: DISCOVERY`, tasks: {}, logs: [], topology: [] };
    }
    
    let genData = window.STATE.generations[gen];

    // Dynamic Fetch
    try {
        const path = window.STATE.currentRunID === 'latest' 
            ? `./data/gen_${gen}.json?v=${Date.now()}` 
            : `./data/runs/${window.STATE.currentRunID}/gen_${gen}.json?v=${Date.now()}`;

        const resp = await fetch(path);
        if (resp.ok) {
            const dynamicData = await resp.json();
            genData = { ...genData, ...dynamicData };
            window.STATE.generations[gen] = genData;
        }
    } catch (e) { 
        console.warn(`Gen ${gen} data load unavailable.`); 
    }

    renderGeneration(genData);
    updateNav();
}

function updateNav() {
    const nav = document.getElementById('generation-nav');
    const hb = window.lastHeartbeat;
    if (!nav) return;

    // Use heartbeat to determine the scale of the lineage
    const hbMax = hb ? parseInt(hb.active_gen) : 1;
    const localMax = Object.keys(window.STATE.generations).length ? Math.max(...Object.keys(window.STATE.generations).map(Number)) : 1;
    const maxGen = Math.max(hbMax, localMax, window.STATE.maxGenSeen);

    let html = '';
    // Reverse chronological for better UX in large lineages
    for (let genNum = maxGen; genNum >= 1; genNum--) {
        let statusDot = '';
        const isCurrentlyActiveInHeartbeat = hb && parseInt(hb.active_gen) === genNum;
        
        if (isCurrentlyActiveInHeartbeat) {
            statusDot = hb.status === "SPAWNING" 
                ? '<span class="status-dot spawning"></span>' 
                : '<span class="status-dot live"></span>';
        } else if (genNum < (hb ? parseInt(hb.active_gen) : window.STATE.activeGen)) {
            statusDot = '<span class="status-dot completed"></span>';
        }

        html += `
            <div class="nav-item ${window.STATE.activeGen === genNum ? 'active' : ''}" onclick="switchGen(${genNum})">
                ${statusDot}Gen ${genNum}
            </div>
        `;
    }
    nav.innerHTML = html;
}

function renderGeneration(genData) {
    document.getElementById('current-gen-title').textContent = genData.title || `GEN ${genData.gen}`;
    document.getElementById('global-score').textContent = (genData.accuracy || genData.score || 0).toFixed(2);
    document.getElementById('accuracy-val').textContent = `${Math.round((genData.accuracy || genData.score || 0) * 100)}%`;

    renderLineageTrail(genData.gen, window.lastHeartbeat);

    const ring = document.querySelector('.progress-ring__circle');
    const radius = ring.r.baseVal.value;
    const circumference = 2 * Math.PI * radius;
    const scoreValue = (genData.accuracy || genData.score || 0);
    const offset = circumference - scoreValue * circumference;
    ring.style.strokeDasharray = `${circumference} ${circumference}`;
    ring.style.strokeDashoffset = isNaN(offset) ? circumference : offset;

    renderTopology(genData.topology);
    renderLogs(genData);
    renderTasks(genData.tasks);
    renderTrends();

    const costBar = document.getElementById('cost-bar');
    const timeBar = document.getElementById('time-bar');
    if (costBar) costBar.style.width = `${genData.cost_eff || 0}%`;
    if (timeBar) timeBar.style.width = `${genData.time_eff || 0}%`;

    const stats = document.querySelectorAll('.bench-stats .val');
    if (stats.length >= 2) {
        stats[0].textContent = genData.swe_stats || `0/30`;
        stats[1].textContent = genData.lcb_stats || "0/15";
    }
}

function renderLineageTrail(currentGen, heartbeat = null) {
    const container = document.getElementById('lineage-trail');
    let trailHtml = '';
    const totalGens = Math.max(Object.keys(window.STATE.generations).length, heartbeat ? heartbeat.active_gen : 0);

    for (let i = 1; i <= totalGens; i++) {
        let classes = ['step'];
        if (i === currentGen) classes.push('active');
        if (heartbeat && heartbeat.active_gen === i) {
            classes.push('live');
        }
        trailHtml += `<div class="${classes.join(' ')}" onclick="switchGen(${i})">GEN ${i}</div>`;
        if (i < totalGens) trailHtml += '<span class="arrow">➔</span>';
    }
    container.innerHTML = trailHtml;
}

function renderTopology(nodes) {
    const container = document.getElementById('topology-map');
    if (!nodes || nodes.length === 0) {
        container.innerHTML = '<div style="text-align:center; padding: 50px; color: #64748b">Topology not yet discovered.</div>';
        return;
    }
    const width = 600, height = 400;
    const centerX = width / 2, centerY = height / 2, radius = 130;

    const nodesHtml = nodes.map((node, i) => {
        const angle = (i / nodes.length) * 2 * Math.PI;
        const x = centerX + radius * Math.cos(angle);
        const y = centerY + radius * Math.sin(angle);
        const lx = centerX + (radius + 65) * Math.cos(angle);
        const ly = centerY + (radius + 65) * Math.sin(angle);

        return `
            <g class="node-group" data-node-id="${node.id}">
                <circle cx="${x}" cy="${y}" r="45" fill="var(--bg)" stroke="${node.color}" stroke-width="2"/>
                <text x="${lx}" y="${ly}" text-anchor="middle" fill="${node.color}" style="font-size: 0.7rem; font-weight: 700;">${node.label}</text>
            </g>
        `;
    }).join('');

    container.innerHTML = `
        <svg width="100%" height="220" viewBox="0 0 ${width} ${height}" style="overflow: visible">
            ${nodesHtml}
        </svg>
    `;
}

function renderLogs(genData) {
    const logFeed = document.getElementById('log-feed');
    logFeed.innerHTML = (genData.logs || []).map(log => `
        <div class="log-entry ${log.type}" onclick="showForensics(${window.STATE.activeGen}, '${log.taskId || 'demo'}')">
            ${log.text}
        </div>
    `).join('');
}

function renderTasks(tasks) {
    const grid = document.getElementById('task-grid');
    if (!tasks || Object.keys(tasks).length === 0) {
        grid.innerHTML = '<tr><td colspan="6" style="text-align:center">No task data. Waiting for 2nd Run to commence.</td></tr>';
        return;
    }

    grid.innerHTML = Object.entries(tasks).map(([key, t]) => `
        <tr>
            <td class="code-font">${t.id}</td>
            <td><span class="status-pass">PASSED</span></td>
            <td class="progress-cell">
                <div class="task-progress-track">
                    <div class="task-progress-fill" style="width: ${t.pass_rate || '100%'}"></div>
                </div>
            </td>
            <td>${t.time}s</td>
            <td>$0.02</td>
            <td><button class="btn-audit" onclick="showForensics(${window.STATE.activeGen}, '${key}')">AUDIT</button></td>
        </tr>
    `).join('');
}

window.showForensics = (gen, taskId) => {
    const panel = document.getElementById('forensic-panel');
    const genData = window.STATE.generations[gen];
    if (!genData || !genData.tasks) return;
    const task = genData.tasks[taskId];
    if (!task) return;

    panel.classList.remove('hidden');
    document.getElementById('panel-title').textContent = task.title || "Audit Log";
    document.getElementById('panel-problem-id').textContent = task.id;
    document.getElementById('panel-platform').textContent = task.platform || "Lineage";
    document.getElementById('panel-code').textContent = task.code || "";
    
    // NEW: Genetic Diff Logic
    renderGeneticDiff(gen, taskId);

    if (window.hljs) hljs.highlightElement(document.getElementById('panel-code'));
};

function renderGeneticDiff(genNum, taskId) {
    const container = document.getElementById('panel-genetic-diff');
    if (genNum <= 1) {
        container.innerHTML = '<div class="info">GEN 1: No parental mutation trail.</div>';
        return;
    }
    const current = window.STATE.generations[genNum]?.tasks?.[taskId]?.code || "";
    const parent = window.STATE.generations[genNum-1]?.tasks?.[taskId]?.code || "";
    
    if (!parent) {
        container.innerHTML = '<div class="info">Parental DNA not found for comparison.</div>';
        return;
    }

    const cLines = current.split('\n');
    const pLines = parent.split('\n');
    let html = '';
    
    // Simple line-based diff for "WOW" effect
    const max = Math.max(cLines.length, pLines.length);
    for (let i = 0; i < max; i++) {
        if (cLines[i] === pLines[i]) continue; // Hide unchanged for brevity
        if (pLines[i]) html += `<div class="diff-rem">- ${pLines[i]}</div>`;
        if (cLines[i]) html += `<div class="diff-add">+ ${cLines[i]}</div>`;
    }
    container.innerHTML = html || '<div class="info">No logic mutation detected in this task.</div>';
}

function renderTrends() {
    const container = document.getElementById('historical-trends');
    const gens = Object.keys(window.STATE.generations).map(Number).sort((a,b) => a-b);
    if (gens.length < 1) return;

    const accuracyData = gens.map(g => (window.STATE.generations[g].accuracy || window.STATE.generations[g].score || 0) * 100);
    const utilityData = gens.map(g => ((window.STATE.generations[g].cost_eff || 50) + (window.STATE.generations[g].time_eff || 50)) / 2);

    container.innerHTML = `
        <div class="trend-item">
            <span class="label">Accuracy</span>
            ${generateSparkline(accuracyData, 'var(--cyan)')}
        </div>
        <div class="trend-item">
            <span class="label">Efficiency</span>
            ${generateSparkline(utilityData, 'var(--magenta)')}
        </div>
    `;
}

function generateSparkline(data, color) {
    if (data.length < 2) return `<span style="color:${color}; font-family:var(--font-mono); font-size:0.7rem;">STATIC</span>`;
    const width = 60, height = 20;
    const max = Math.max(...data, 100);
    const min = Math.min(...data, 0);
    const range = max - min || 1;
    
    const points = data.map((v, i) => {
        const x = (i / (data.length - 1)) * width;
        const y = height - ((v - min) / range) * height;
        return `${x},${y}`;
    }).join(' ');

    return `
        <svg class="trend-svg">
            <polyline class="trend-path" points="${points}" stroke="${color}" />
            <circle cx="${width}" cy="${height - ((data[data.length-1] - min) / range) * height}" r="2" fill="${color}" />
        </svg>
    `;
}

window.closePanel = () => document.getElementById('forensic-panel').classList.add('hidden');

// Cloud Sync Sim
async function pullHeartbeat() {
    try {
        const path = `./data/heartbeat.json?t=${Date.now()}`;
        const resp = await fetch(path);
        if (!resp.ok) return;
        const hb = await resp.json();
        window.lastHeartbeat = hb;

        const liveBadge = document.getElementById('live-status-indicator');
        const lastUpdated = new Date(hb.last_updated).getTime();
        const isLive = Math.abs(Date.now() - lastUpdated) < 60000;

        // 1. Badge & Health Dots
        if (liveBadge) {
            liveBadge.classList.toggle('active', isLive);
            liveBadge.innerHTML = isLive ? `ONLINE: GEN ${hb.active_gen}` : 'RUNNER OFFLINE';
        }

        const dots = {
            'RUNNING': 'health-dot-live',
            'STALLED': 'health-dot-stalled',
            'ERROR': 'health-dot-error',
            'SPAWNING': 'health-dot-spawning'
        };

        Object.values(dots).forEach(id => {
            const el = document.getElementById(id);
            if (el) el.classList.remove('live', 'stalled', 'error', 'spawning');
        });

        if (isLive && hb.status && dots[hb.status]) {
            const activeDot = document.getElementById(dots[hb.status]);
            if (activeDot) activeDot.classList.add(hb.status.toLowerCase());
        }

        // 2. Active Probe
        const probeVal = document.getElementById('active-task-id');
        if (probeVal) {
            probeVal.textContent = isLive ? (hb.active_task_id || 'IDLE') : 'OFFLINE';
        }

        // 3. Progress Bar
        const pBar = document.getElementById('benchmark-progress-bar');
        const pVal = document.getElementById('benchmark-progress-val');
        if (hb.progress && pBar) {
            const parts = hb.progress.split('/');
            if (parts.length === 2) {
                const pct = Math.round((parseInt(parts[0]) / parseInt(parts[1])) * 100);
                pBar.style.width = `${pct}%`;
                pVal.textContent = `${pct}%`;
            }
        }

        updateNav();
    } catch (e) {
        console.error("Heartbeat sync failed:", e);
    }
}

window.triggerNewRun = async () => {
    const prompt = document.getElementById('initial-prompt').value;
    alert("Triggering 2nd Run with prompt: " + prompt);
    // In a real scenario, this would POST to a trigger endpoint
}

setInterval(pullHeartbeat, 5000);
window.switchGen(1);
pullHeartbeat();
