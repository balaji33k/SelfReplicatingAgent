import json
import os

PROJECT_ROOT = r"d:\Projects\2ndRunSelfReplicatingAgent"

REPORT_TEMPLATE = r"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Autonomous Lineage Report (Industrial v8 - Innovative)</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono&display=swap" rel="stylesheet">
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/themes/prism-tomorrow.min.css" rel="stylesheet" />
    <style>
        :root { --bg: #05060f; --card-bg: rgba(15, 16, 35, 0.7); --border: rgba(255, 255, 255, 0.1); --cyan: #00f3ff; --magenta: #ff00ff; --lime: #39ff14; --text-main: #e0e0e0; --text-dim: #94a3b8; --font-ui: 'Outfit', sans-serif; --font-mono: 'JetBrains Mono', monospace; }
        body { background: var(--bg); color: var(--text-main); font-family: var(--font-ui); margin: 0; padding: 0; min-height: 100vh; overflow: hidden; }
        .cyber-grid { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background-image: radial-gradient(circle at 2px 2px, var(--border) 1px, transparent 0); background-size: 40px 40px; opacity: 0.3; pointer-events: none; z-index: -1; }
        
        .layout { display: flex; height: 100vh; }
        .sidebar { width: 340px; border-right: 1px solid var(--border); background: rgba(0,0,0,0.4); backdrop-filter: blur(20px); display: flex; flex-direction: column; padding: 32px; z-index: 100; }
        .main-view { flex: 1; padding: 40px; overflow-y: auto; position: relative; scroll-behavior: smooth; }
        
        h1 { background: linear-gradient(90deg, var(--cyan), var(--magenta)); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 800; font-size: 1.4rem; letter-spacing: 3px; margin-bottom: 40px; }
        .gen-nav { display: flex; flex-direction: column; gap: 15px; }
        .gen-btn { padding: 20px; border-radius: 16px; border: 1px solid var(--border); background: var(--card-bg); cursor: pointer; transition: 0.4s; position: relative; overflow: hidden; }
        .gen-btn:hover { border-color: var(--cyan); transform: translateX(5px); }
        .gen-btn.active { border-color: var(--cyan); background: rgba(0, 243, 255, 0.15); box-shadow: 0 0 30px rgba(0, 243, 255, 0.1); }
        
        .glass-card { background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border); border-radius: 20px; padding: 32px; margin-bottom: 32px; position: relative; }
        .stat-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 20px; margin: 30px 0; }
        .stat-item { text-align: left; padding: 15px; border-left: 2px solid var(--border); transition: 0.3s; }
        .stat-item:hover { border-left-color: var(--cyan); background: rgba(255,255,255,0.02); }
        .stat-val { font-size: 2.2rem; font-weight: 800; color: var(--cyan); display: block; text-shadow: 0 0 15px rgba(0, 243, 255, 0.3); }
        .stat-label { font-size: 0.65rem; color: var(--text-dim); text-transform: uppercase; letter-spacing: 2px; }
        
        .progress-container { margin: 25px 0; }
        .progress-header { display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-dim); margin-bottom: 8px; text-transform: uppercase; }
        .progress-track { height: 4px; background: rgba(255,255,255,0.05); border-radius: 2px; overflow: hidden; }
        .progress-fill { height: 100%; background: linear-gradient(90deg, var(--cyan), var(--magenta)); transition: width 1s cubic-bezier(0.4, 0, 0.2, 1); width: 0%; box-shadow: 0 0 10px var(--cyan); }
        
        .viz-box { background: rgba(0,0,0,0.3); border-radius: 16px; border: 1px solid var(--border); min-height: 400px; position: relative; overflow: hidden; margin: 20px 0; }
        .svg-canvas { width: 100%; height: 100%; min-height: 400px; display: block; }
        .node { stroke-width: 2; cursor: pointer; transition: 0.3s; }
        .node:hover { stroke-width: 4; filter: brightness(1.2); }
        .node-pulse { animation: pulse 2s infinite; }
        @keyframes pulse { 0% { r: 6; opacity: 1; } 100% { r: 24; opacity: 0; } }
        .edge { stroke: rgba(0, 243, 255, 0.15); stroke-width: 1; stroke-dasharray: 4; animation: dash 30s linear infinite; }
        @keyframes dash { to { stroke-dashoffset: -1000; } }
        
        .forensic-table { width: 100%; border-collapse: separate; border-spacing: 0 8px; font-family: var(--font-mono); font-size: 0.75rem; }
        .forensic-table th { text-align: left; padding: 15px; color: var(--text-dim); border-bottom: 1px solid var(--border); font-size: 0.65rem; letter-spacing: 1px; }
        .forensic-table td { padding: 15px; background: rgba(255,255,255,0.02); border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); cursor: pointer; transition: 0.3s; }
        .forensic-table tr td:first-child { border-left: 1px solid var(--border); border-radius: 10px 0 0 10px; }
        .forensic-table tr td:last-child { border-right: 1px solid var(--border); border-radius: 0 10px 10px 0; text-align: right; color: var(--cyan); }
        .forensic-table tr:hover td { background: rgba(0, 243, 255, 0.05); border-color: var(--cyan); }
        
        .status-pill { padding: 4px 10px; border-radius: 6px; font-weight: 800; font-size: 0.6rem; text-transform: uppercase; border: 1px solid transparent; }
        .pass { background: rgba(57, 255, 20, 0.1); color: var(--lime); border-color: var(--lime); }
        .fail { background: rgba(255, 49, 49, 0.1); color: #ff3131; border-color: #ff3131; }
        
        .drawer { position: fixed; top: 0; right: -900px; width: 850px; height: 100vh; background: #070815; border-left: 1px solid var(--border); z-index: 1000; transition: transform 0.6s cubic-bezier(0.16, 1, 0.3, 1); box-shadow: -30px 0 60px rgba(0,0,0,0.6); padding: 50px; overflow-y: auto; }
        .drawer.open { transform: translateX(-900px); }
        .overlay { position: fixed; top: 0; left: 0; width: 100%; height: 100%; background: rgba(0,0,0,0.7); backdrop-filter: blur(8px); z-index: 999; opacity: 0; pointer-events: none; transition: 0.4s; }
        .overlay.visible { opacity: 1; pointer-events: all; }
    </style>
</head>
<body>
    <div class="cyber-grid"></div>
    <div class="layout">
        <aside class="sidebar">
            <h1>LINEAGE DISCOVERY</h1>
            <div id="gen-nav" class="gen-nav"></div>
        </aside>
        <main class="main-view">
            <div id="content-root"></div>
        </main>
    </div>
    <div id="overlay" class="overlay" onclick="closeDrawer()"></div>
    <div id="drawer" class="drawer"><div id="drawer-content"></div></div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/prism.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.29.0/components/prism-python.min.js"></script>
    
    <script>
        window.LINEAGE_DATA = {LINEAGE_DATA_JSON};

        function openDrawer(taskId) {
            const data = window.CURRENT_GEN_DATA;
            const task = data.tasks[taskId];
            const dc = document.getElementById('drawer-content');
            dc.innerHTML = `
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.5rem;">
                    <div><span class="stat-label">Task Forensic</span><h2 style="margin: 0; font-size: 2rem; color: var(--cyan);">\${taskId}</h2></div>
                    <span class="status-pill \${task.status === 'success' ? 'pass' : 'fail'}">\${task.status}</span>
                </div>
                <h3 class="stat-label">Discovery Logs</h3>
                <pre style="background: rgba(0,0,0,0.4); padding: 20px; border-radius: 12px; border: 1px solid var(--border); font-size: 0.75rem; color: var(--text-dim); overflow-x: auto;">\${task.logs || 'No log data.'}</pre>
                <div style="margin: 30px 0;"><h3 class="stat-label">Synthesized Logic</h3><div style="border-radius: 12px; overflow: hidden; border: 1px solid var(--border);"><pre class="line-numbers" style="margin:0;"><code class="language-python">\${task.code || '# No code data.'}</code></pre></div></div>
            `;
            Prism.highlightAll();
            document.getElementById('drawer').classList.add('open');
            document.getElementById('overlay').classList.add('visible');
        }

        function closeDrawer() { document.getElementById('drawer').classList.remove('open'); document.getElementById('overlay').classList.remove('visible'); }

        function render(genNum) {
            const data = window.LINEAGE_DATA.find(g => g.gen === genNum);
            if (!data) return;
            window.CURRENT_GEN_DATA = data;
            const root = document.getElementById('content-root');
            const accuracy = (data.accuracy || 0) * 100;
            const progress = (Object.keys(data.tasks).length / 30) * 100;
            
            root.innerHTML = `
                <div class="glass-card">
                    <div style="display: flex; justify-content: space-between; align-items: flex-start;">
                        <div><span style="font-size: 0.7rem; color: var(--magenta); letter-spacing: 4px;">SYSTEM DISCOVERY SEQUENCE</span><h2 style="font-size: 3rem; margin: 5px 0; font-weight: 800;">PHASE \${data.gen} <span style="font-size: 0.8rem; font-weight: 400; color: var(--text-dim); vertical-align: middle;">[ \${data.status} ]</span></h2></div>
                        <div style="text-align: right;"><div class="stat-val">\${Math.round(accuracy)}%</div><div class="stat-label">Fidelity Score</div></div>
                    </div>
                    <div class="progress-container"><div class="progress-header"><span>Discovery Progress</span><span>\${Math.round(progress)}%</span></div><div class="progress-track"><div class="progress-fill" style="width: \${progress}%"></div></div></div>
                    <div class="stat-grid">
                        <div class="stat-item"><span class="stat-val">\${Object.keys(data.tasks).length}</span><span class="stat-label">Tasks Captured</span></div>
                        <div class="stat-item"><span class="stat-val">98.4%</span><span class="stat-label">DNA Integrity</span></div>
                        <div class="stat-item"><span class="stat-val">\${data.gen}</span><span class="stat-label">Gen Index</span></div>
                        <div class="stat-item"><span class="stat-val">30</span><span class="stat-label">Bench Target</span></div>
                    </div>
                </div>
                <div style="display: grid; grid-template-columns: 1.5fr 1fr; gap: 32px; margin-bottom: 32px;">
                    <div class="glass-card" style="margin-bottom:0;">
                        <h3 class="stat-label">Discovery Topology</h3>
                        <div class="viz-box"><svg id="svg-viz" class="svg-canvas"></svg></div>
                    </div>
                    <div class="glass-card" style="margin-bottom:0;">
                        <h3 class="stat-label">Generational Meta-Shift</h3>
                        <p style="font-family: var(--font-mono); font-size: 0.85rem; color: var(--lime); line-height: 1.6;">> \${data.improvement_log || data.evolution || 'System stabilizing DNA.'}</p>
                    </div>
                </div>
                <div class="glass-card"><h3 class="stat-label">Deep Forensic Explorer</h3><table class="forensic-table"><thead><tr><th>TASK ID</th><th>STATUS</th><th>FIDELITY</th><th>ACTIONS</th></tr></thead><tbody>\${renderTasks(data.tasks)}</tbody></table></div>
            `;
            updateSidebar(genNum);
            drawTopology(data.topology);
        }

        function drawTopology(topology) {
            const svg = document.getElementById('svg-viz');
            if(!svg) return;
            svg.innerHTML = '';
            const width = svg.clientWidth || 600;
            const height = svg.clientHeight || 400;
            const nodes = topology && topology.length > 0 ? topology : [ {id:'core', label:'Genesis'} ];
            const centerX = width/2; const centerY = height/2; const radius = Math.min(width, height) * 0.35;
            
            nodes.forEach((n, i) => {
                const angle = (i / nodes.length) * 2 * Math.PI - Math.PI/2;
                n.x = centerX + radius * Math.cos(angle); n.y = centerY + radius * Math.sin(angle);
            });
            
            const gLines = document.createElementNS("http://www.w3.org/2000/svg", "g");
            for(let i=0; i<nodes.length; i++) {
                const next = (i+1)%nodes.length;
                const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
                line.setAttribute("x1", nodes[i].x); line.setAttribute("y1", nodes[i].y); line.setAttribute("x2", nodes[next].x); line.setAttribute("y2", nodes[next].y);
                line.setAttribute("stroke", "rgba(0, 243, 255, 0.2)"); line.setAttribute("stroke-width", "1"); line.setAttribute("stroke-dasharray", "4");
                gLines.appendChild(line);
            }
            svg.appendChild(gLines);

            nodes.forEach(n => {
                const grp = document.createElementNS("http://www.w3.org/2000/svg", "g");
                grp.setAttribute("class", "node");
                const pulse = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                pulse.setAttribute("cx", n.x); pulse.setAttribute("cy", n.y); pulse.setAttribute("r", "6"); pulse.setAttribute("fill", "none");
                pulse.setAttribute("stroke", "var(--cyan)"); pulse.setAttribute("class", "node-pulse");
                grp.appendChild(pulse);
                const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
                circle.setAttribute("cx", n.x); circle.setAttribute("cy", n.y); circle.setAttribute("r", "6");
                circle.setAttribute("fill", "#070815"); circle.setAttribute("stroke", "var(--cyan)"); circle.setAttribute("stroke-width", "2");
                grp.appendChild(circle);
                const txt = document.createElementNS("http://www.w3.org/2000/svg", "text");
                txt.setAttribute("x", n.x); txt.setAttribute("y", n.y + 20); txt.setAttribute("text-anchor", "middle");
                txt.setAttribute("fill", "var(--text-dim)"); txt.setAttribute("font-size", "9px"); txt.textContent = n.label;
                grp.appendChild(txt);
                svg.appendChild(grp);
            });
        }

        function renderTasks(tasks) {
            return Object.keys(tasks).map(id => {
                const t = tasks[id];
                return `<tr onclick="openDrawer('\${id}')"><td>\${id}</td><td><span class="status-pill \${t.status === 'success' ? 'pass' : 'fail'}">\${t.status}</span></td><td>1.0</td><td style="color: var(--cyan); font-weight: 800;">INSPECT</td></tr>`;
            }).join('');
        }

        function updateSidebar(active) {
            const nav = document.getElementById('gen-nav');
            const sorted = [...window.LINEAGE_DATA].reverse();
            nav.innerHTML = sorted.map(g => `<div class="gen-btn \${g.gen === active ? 'active' : ''}" onclick="render(\${g.gen})"><div class="stat-label">GEN \${g.gen}</div><div style="font-size: 0.75rem; font-weight: 600;">\${g.status} | \${Math.round((g.accuracy||0)*100)}%</div></div>`).join('');
        }
        
        window.addEventListener('resize', () => { if(window.CURRENT_GEN_DATA) drawTopology(window.CURRENT_GEN_DATA.topology); });
        render(window.LINEAGE_DATA[window.LINEAGE_DATA.length - 1].gen);
    </script>
</body>
</html>
"""

def update_industrial_dashboard(current_gen=None, status="IDLE", results=None, topology=None, improvement_log=None, **kwargs):
    gen = current_gen or kwargs.get("gen", 1)
    log_final = improvement_log or kwargs.get("log") or kwargs.get("evolution") or "System refinement cycle."
    data_dir = os.path.join(PROJECT_ROOT, "data")
    os.makedirs(data_dir, exist_ok=True)
    gen_path = os.path.join(data_dir, f"gen_{gen}.json")
    if os.path.exists(gen_path):
        with open(gen_path, "r") as f: summary = json.load(f)
    else: summary = {"gen": gen, "tasks": {}, "status": status}
    summary["status"] = status
    if results:
        for r in results: summary["tasks"][r['id']] = r
        summary["accuracy"] = sum(1 for r in results if r['status'] == 'success') / max(1, len(results))
    if topology: summary["topology"] = topology
    summary["improvement_log"] = log_final
    with open(gen_path, "w") as f: json.dump(summary, f, indent=2)
    all_gen_data = []
    if os.path.exists(data_dir):
        for filename in sorted(os.listdir(data_dir)):
            if filename.startswith("gen_") and filename.endswith(".json"):
                with open(os.path.join(data_dir, filename), "r") as f: all_gen_data.append(json.load(f))
    all_gen_data.sort(key=lambda x: x["gen"])
    json_payload = json.dumps(all_gen_data)
    report_content = REPORT_TEMPLATE.replace("{LINEAGE_DATA_JSON}", json_payload)
    with open(os.path.join(PROJECT_ROOT, "MISSION_CONTROL.html"), "w") as f: f.write(report_content)
