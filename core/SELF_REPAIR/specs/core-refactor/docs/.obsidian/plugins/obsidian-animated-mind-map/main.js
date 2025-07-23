// Animated Mind Map+ Obsidian Plugin
const { Plugin, ItemView, PluginSettingTab, Setting, Modal } = require('obsidian');

const ANIMATED_MM_VIEW_TYPE = 'animated-mindmap';
const KANBAN_GRAPH_VIEW_TYPE = 'semantic-kanban-graph';

// Node type definitions
const NODE_TYPES = [
  'default',
  'logical-and',
  'logical-or',
  'logical-not',
  'conditional-if',
  'conditional-else',
  'multiplex',
  'demultiplex'
];

class AnimatedMindMapSettings {
  constructor() {
    this.animationSpeed = 500;
    this.layoutType = 'wfc'; // 'wfc' | 'force-directed'
    this.enableSpeculativeGrouping = true;
    this.enableSpeculativeLinking = true;
  }
}

// Add CSS for controls and fallback UI
const style = document.createElement('style');
style.textContent = `
.animated-mm-controls, .animated-mm-suggestions, .animated-mm-groups, .animated-mm-links {
  background: var(--background-secondary, #f8f8f8);
  padding: 8px;
  margin-bottom: 8px;
  border-radius: 6px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.04);
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}
.animated-mm-controls label { font-weight: bold; margin-right: 4px; }
.animated-mm-controls select, .animated-mm-controls input, .animated-mm-controls button {
  margin-right: 8px;
  min-width: 80px;
}
.animated-mm-fallback {
  color: #b00;
  background: #fff0f0;
  padding: 8px;
  border-radius: 6px;
  margin-bottom: 8px;
  font-weight: bold;
}
`;
document.head.appendChild(style);

class AnimatedMindMapSettingsTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    try {
      new Setting(containerEl)
        .setName('Animation Speed (ms)')
        .setDesc('Speed of animated transitions')
        .addText(text => text
          .setValue(this.plugin.settings.animationSpeed.toString())
          .onChange(value => {
            this.plugin.settings.animationSpeed = parseInt(value);
            this.plugin.saveData(this.plugin.settings);
          }));
      new Setting(containerEl)
        .setName('Layout Type')
        .setDesc('Choose layout algorithm')
        .addDropdown(drop => drop
          .addOption('wfc', 'Wave Function Collapse')
          .addOption('force-directed', 'Force-Directed')
          .setValue(this.plugin.settings.layoutType)
          .onChange(value => {
            this.plugin.settings.layoutType = value;
            this.plugin.saveData(this.plugin.settings);
          }));
      new Setting(containerEl)
        .setName('Speculative Grouping')
        .setDesc('Enable speculative node grouping')
        .addToggle(toggle => toggle
          .setValue(this.plugin.settings.enableSpeculativeGrouping)
          .onChange(value => {
            this.plugin.settings.enableSpeculativeGrouping = value;
            this.plugin.saveData(this.plugin.settings);
          }));
      new Setting(containerEl)
        .setName('Speculative Linking')
        .setDesc('Enable speculative node linking')
        .addToggle(toggle => toggle
          .setValue(this.plugin.settings.enableSpeculativeLinking)
          .onChange(value => {
            this.plugin.settings.enableSpeculativeLinking = value;
            this.plugin.saveData(this.plugin.settings);
          }));
    } catch (e) {
      containerEl.createDiv('animated-mm-fallback', { text: 'Failed to render settings: ' + e.message });
    }
  }
}

// Utility: Speculate on nodes to add based on current structure
function speculateNodes(existingNodes) {
  const suggestions = [];
  // Example: If no logical nodes, suggest adding AND/OR/NOT
  if (!existingNodes.some(n => n.type.startsWith('logical-'))) {
    suggestions.push({ type: 'logical-and', label: 'AND' });
    suggestions.push({ type: 'logical-or', label: 'OR' });
    suggestions.push({ type: 'logical-not', label: 'NOT' });
  }
  // If no conditional, suggest IF/ELSE
  if (!existingNodes.some(n => n.type.startsWith('conditional-'))) {
    suggestions.push({ type: 'conditional-if', label: 'IF' });
    suggestions.push({ type: 'conditional-else', label: 'ELSE' });
  }
  // If no mux/demux, suggest them
  if (!existingNodes.some(n => n.type === 'multiplex')) {
    suggestions.push({ type: 'multiplex', label: 'MUX' });
  }
  if (!existingNodes.some(n => n.type === 'demultiplex')) {
    suggestions.push({ type: 'demultiplex', label: 'DEMUX' });
  }
  // If orphans exist, suggest linking or grouping
  if (existingNodes.some(n => n.orphan)) {
    suggestions.push({ type: 'default', label: 'Link Orphans', action: 'link-orphans' });
  }
  // Suggest common workflow nodes
  if (!existingNodes.some(n => n.label === 'Start')) {
    suggestions.push({ type: 'default', label: 'Start' });
  }
  if (!existingNodes.some(n => n.label === 'End')) {
    suggestions.push({ type: 'default', label: 'End' });
  }
  return suggestions;
}

// Utility: A* pathfinding for grid routing
function astarPath(start, goal, occupied, gridW, gridH) {
  // start, goal: {x, y}; occupied: Set('x,y'); gridW, gridH: grid size
  const key = (x, y) => `${x},${y}`;
  const open = [start];
  const cameFrom = {};
  const gScore = { [key(start.x, start.y)]: 0 };
  const fScore = { [key(start.x, start.y)]: Math.abs(goal.x - start.x) + Math.abs(goal.y - start.y) };
  while (open.length > 0) {
    open.sort((a, b) => fScore[key(a.x, a.y)] - fScore[key(b.x, b.y)]);
    const current = open.shift();
    if (current.x === goal.x && current.y === goal.y) {
      // Reconstruct path
      const path = [current];
      let k = key(current.x, current.y);
      while (cameFrom[k]) {
        k = cameFrom[k];
        const [x, y] = k.split(',').map(Number);
        path.unshift({ x, y });
      }
      return path;
    }
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = current.x + dx, ny = current.y + dy;
      if (nx < 0 || ny < 0 || nx >= gridW || ny >= gridH) continue;
      if (occupied.has(key(nx, ny)) && (nx !== goal.x || ny !== goal.y)) continue;
      const neighbor = { x: nx, y: ny };
      const nKey = key(nx, ny);
      const tentativeG = gScore[key(current.x, current.y)] + 1;
      if (tentativeG < (gScore[nKey] ?? Infinity)) {
        cameFrom[nKey] = key(current.x, current.y);
        gScore[nKey] = tentativeG;
        fScore[nKey] = tentativeG + Math.abs(goal.x - nx) + Math.abs(goal.y - ny);
        if (!open.some(n => n.x === nx && n.y === ny)) open.push(neighbor);
      }
    }
  }
  return [start, goal]; // fallback: straight line
}

function intelligentCircuitLayout(nodes, links, gridSize = 8) {
  // Place nodes on a grid, try to minimize link length and crossings
  // Returns: { positions: { [idx]: {x, y} }, routes: { [linkIdx]: [ {x, y}, ... ] } }
  const positions = {};
  const occupied = new Set();
  let x = 0, y = 0, gridW = 8, gridH = 8;
  nodes.forEach((node, idx) => {
    positions[idx] = { x, y };
    occupied.add(`${x},${y}`);
    x += gridSize;
    if (x > gridSize * 4) { x = 0; y += gridSize; }
  });
  gridW = Math.max(gridW, x / gridSize + 1);
  gridH = Math.max(gridH, y / gridSize + 1);
  // A* routing for links
  const routes = {};
  links.forEach((link, i) => {
    const from = positions[link.from], to = positions[link.to];
    routes[i] = astarPath(from, to, occupied, gridW, gridH);
  });
  return { positions, routes };
}

// Helper for smooth animation
function animatePosition(from, to, duration, onUpdate, onComplete) {
  console.log('[ANIMATE POSITION] from', from, 'to', to, 'over', duration, 'ms');
  const start = performance.now();
  function frame(now) {
    const t = Math.min(1, (now - start) / duration);
    const x = from.x + (to.x - from.x) * t;
    const y = from.y + (to.y - from.y) * t;
    onUpdate({ x, y });
    if (t < 1) {
      requestAnimationFrame(frame);
    } else {
      if (onComplete) onComplete();
      console.log('[ANIMATE POSITION DONE]');
    }
  }
  requestAnimationFrame(frame);
}

// Minimal d3-interpolate for transforms and paths
function interpolateTransformSvg(a, b) {
  // Parse translate(x, y)
  const parse = str => {
    const match = /translate\(([-\d.]+),([-.\d]+)\)/.exec(str);
    return match ? [parseFloat(match[1]), parseFloat(match[2])] : [0, 0];
  };
  const [ax, ay] = parse(a);
  const [bx, by] = parse(b);
  return t => `translate(${ax + (bx - ax) * t},${ay + (by - ay) * t})`;
}
function interpolatePath(a, b) {
  // Simple linear interpolation for matching point count
  // For real use, use d3-interpolate-path or flubber
  const pa = a.match(/[-\d.]+/g).map(Number);
  const pb = b.match(/[-\d.]+/g).map(Number);
  if (pa.length !== pb.length) return () => b;
  return t => {
    const pts = pa.map((v, i) => v + (pb[i] - v) * t);
    let out = b;
    let j = 0;
    out = out.replace(/[-\d.]+/g, () => pts[j++]);
    return out;
  };
}
function animateSVGAttrD3(element, attr, from, to, duration, type) {
  console.log(`[ANIMATE] ${type} ${attr} from`, from, 'to', to, 'over', duration, 'ms');
  const start = performance.now();
  let interp;
  if (type === 'transform') interp = interpolateTransformSvg(from, to);
  else if (type === 'path') interp = interpolatePath(from, to);
  else return;
  function frame(now) {
    const t = cubicInOut(Math.min(1, (now - start) / duration));
    element.setAttribute(attr, interp(t));
    if (t < 1) requestAnimationFrame(frame);
    else console.log(`[ANIMATE DONE] ${type} ${attr}`);
  }
  requestAnimationFrame(frame);
}

// Cubic easing for animation
function cubicInOut(t) {
  return ((t *= 2) <= 1 ? t * t * t : (t -= 2) * t * t + 2) / 2;
}

// Modal for settings
class MindMapSettingsModal extends Modal {
  constructor(app, plugin, onUpdate) {
    super(app);
    this.plugin = plugin;
    this.onUpdate = onUpdate;
  }
  onOpen() {
    const { contentEl } = this;
    contentEl.empty();
    contentEl.createEl('h2', { text: 'Animated Mind Map+ Settings' });
    new Setting(contentEl)
      .setName('Animation Speed (ms)')
      .setDesc('Speed of animated transitions')
      .addText(text => text
        .setValue(this.plugin.settings.animationSpeed.toString())
        .onChange(value => {
          this.plugin.settings.animationSpeed = parseInt(value);
          this.plugin.saveData(this.plugin.settings);
          this.onUpdate();
        }));
    new Setting(contentEl)
      .setName('Layout Type')
      .setDesc('Choose layout algorithm')
      .addDropdown(drop => drop
        .addOption('wfc', 'Wave Function Collapse')
        .addOption('force-directed', 'Force-Directed')
        .setValue(this.plugin.settings.layoutType)
        .onChange(value => {
          this.plugin.settings.layoutType = value;
          this.plugin.saveData(this.plugin.settings);
          this.onUpdate();
        }));
    new Setting(contentEl)
      .setName('Speculative Grouping')
      .setDesc('Enable speculative node grouping')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.enableSpeculativeGrouping)
        .onChange(value => {
          this.plugin.settings.enableSpeculativeGrouping = value;
          this.plugin.saveData(this.plugin.settings);
          this.onUpdate();
        }));
    new Setting(contentEl)
      .setName('Speculative Linking')
      .setDesc('Enable speculative node linking')
      .addToggle(toggle => toggle
        .setValue(this.plugin.settings.enableSpeculativeLinking)
        .onChange(value => {
          this.plugin.settings.enableSpeculativeLinking = value;
          this.plugin.saveData(this.plugin.settings);
          this.onUpdate();
        }));
  }
  onClose() {
    this.contentEl.empty();
  }
}

// Defensive wrapper for createDiv and createEl
function safeCreateDiv(parent, arg, opts) {
  if (typeof arg === 'function' || typeof arg === 'undefined') {
    console.error('createDiv called with invalid argument:', arg, opts);
    throw new Error('createDiv: first argument must be a string (class) or options object, not a function or undefined');
  }
  return parent.createDiv(arg, opts);
}
function safeCreateEl(parent, tag, opts) {
  if (typeof tag === 'function' || typeof tag === 'undefined') {
    console.error('createEl called with invalid argument:', tag, opts);
    throw new Error('createEl: first argument must be a string (tag) or options object, not a function or undefined');
  }
  return parent.createEl(tag, opts);
}

class AnimatedMindMapView extends ItemView {
  constructor(settings, leaf, plugin) {
    super(leaf);
    this.settings = settings;
    this.plugin = plugin;
    this.containerEl.addClass('animated-mindmap-view');
    // Load persisted data
    const data = plugin.getMindMapData();
    this.nodes = data.nodes || [];
    this.links = data.links || [];
    this.groups = data.groups || [
      { name: 'Group 1', color: '#A3D9C9' },
      { name: 'Group 2', color: '#B7B5E4' },
      { name: 'Orphans', color: '#F6E6E6' }
    ];
    this.draggingNode = null;
    this.dragOverNode = null;
    this.draggingGroup = null;
    this._resizeListener = null;
    this._cssListener = null;
    this._lastPositions = {};
  }
  getViewType() { return ANIMATED_MM_VIEW_TYPE; }
  getDisplayText() { return 'Animated Mind Map (WFC/Quantum Orbit)'; }
  getIcon() { return 'dot-network'; }
  async onOpen() {
    this.renderSkeleton();
    // Responsive redraw on resize/theme change
    this._resizeListener = () => this.renderSkeleton();
    this._cssListener = () => this.renderSkeleton();
    this.plugin._listeners.push(
      this.app.workspace.on('resize', this._resizeListener),
      this.app.workspace.on('css-change', this._cssListener)
    );
  }
  async onClose() {
    // Remove listeners
    if (this._resizeListener) this.app.workspace.off('resize', this._resizeListener);
    if (this._cssListener) this.app.workspace.off('css-change', this._cssListener);
  }
  renderSkeleton() {
    this.containerEl.empty();
    // Settings (gear) icon in top-right
    const gear = safeCreateEl(this.containerEl, 'div', { cls: 'animated-mm-gear' });
    gear.style.position = 'absolute';
    gear.style.top = '12px';
    gear.style.right = '16px';
    gear.style.cursor = 'pointer';
    gear.innerHTML = '<svg width="24" height="24" viewBox="0 0 24 24"><path fill="#888" d="M12 15.5A3.5 3.5 0 1 0 12 8.5a3.5 3.5 0 0 0 0 7zm7.43-2.9l1.77-1.02a.5.5 0 0 0 .18-.68l-1.7-2.94a.5.5 0 0 0-.61-.23l-2.08.83a7.03 7.03 0 0 0-1.5-.87l-.32-2.23A.5.5 0 0 0 14.5 5h-3a.5.5 0 0 0-.5.42l-.32 2.23c-.53.22-1.03.5-1.5.87l-2.08-.83a.5.5 0 0 0-.61.23l-1.7 2.94a.5.5 0 0 0 .18.68l1.77 1.02c-.04.32-.07.65-.07.98s.03.66.07.98l-1.77 1.02a.5.5 0 0 0-.18.68l1.7 2.94c.13.23.4.32.61.23l2.08-.83c.47.37.97.65 1.5.87l.32 2.23a.5.5 0 0 0 .5.42h3a.5.5 0 0 0 .5-.42l.32-2.23c.53-.22 1.03-.5 1.5-.87l2.08.83c.21.09.48 0 .61-.23l1.7-2.94a.5.5 0 0 0-.18-.68l-1.77-1.02c.04-.32.07-.65.07-.98s-.03-.66-.07-.98z"/></svg>';
    gear.onclick = () => {
      new MindMapSettingsModal(this.app, this.plugin, () => this.renderSkeleton()).open();
    };
    // Mind map SVG/canvas
    const canvas = safeCreateDiv(this.containerEl, 'animated-mm-canvas');
    canvas.style.position = 'relative';
    // SVG for nodes and links
    let svg = canvas.querySelector('svg');
    if (!svg) {
      svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('class', 'animated-mm-svg');
      svg.style.position = 'absolute';
      svg.style.top = '0';
      svg.style.left = '0';
      svg.style.width = '100%';
      svg.style.height = '100%';
      canvas.appendChild(svg);
    }
    // Intelligent layout
    const { positions, routes } = intelligentCircuitLayout(this.nodes, this.links);
    // --- Render links as SVG <path> ---
    const linkMap = {};
    this.links.forEach((link, i) => {
      linkMap[`${link.from}-${link.to}`] = i;
    });
    Array.from(svg.querySelectorAll('path.animated-mm-link')).forEach(path => {
      if (!linkMap[path.dataset.key]) svg.removeChild(path);
    });
    this.links.forEach((link, i) => {
      const route = routes[i];
      if (!route) return;
      const d = `M${route[0].x * 10 + 60},${route[0].y * 10 + 60} ` + route.slice(1).map(pt => `L${pt.x * 10 + 60},${pt.y * 10 + 60}`).join(' ');
      let path = svg.querySelector(`path[data-key='${link.from}-${link.to}']`);
      if (!path) {
        path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        path.setAttribute('class', 'animated-mm-link');
        path.setAttribute('stroke', '#888');
        path.setAttribute('stroke-width', '2');
        path.setAttribute('fill', 'none');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        path.dataset.key = `${link.from}-${link.to}`;
        path.onclick = () => {
          if (confirm('Delete this link?')) {
            this.links.splice(i, 1);
            this.saveAndRender();
          }
        };
        svg.appendChild(path);
        // Always animate from empty to d
        animateSVGAttrD3(path, 'd', '', d, this.settings.animationSpeed, 'path');
      } else {
        const prevD = path.getAttribute('d') || '';
        if (prevD !== d) animateSVGAttrD3(path, 'd', prevD, d, this.settings.animationSpeed, 'path');
      }
    });
    // --- Render nodes as SVG <g> ---
    const nodeKeys = this.nodes.map((_, i) => `node-${i}`);
    Array.from(svg.querySelectorAll('g.animated-mm-node')).forEach(g => {
      if (!nodeKeys.includes(g.dataset.key)) svg.removeChild(g);
    });
    this.nodes.forEach((node, idx) => {
      const pos = positions[idx];
      let g = svg.querySelector(`g[data-key='node-${idx}']`);
      const next = `translate(${pos.x * 10 + 60},${pos.y * 10 + 60})`;
      if (!g) {
        g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        g.setAttribute('class', 'animated-mm-node');
        g.dataset.key = `node-${idx}`;
        g.style.cursor = 'pointer';
        const icon = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        icon.setAttribute('x', 0);
        icon.setAttribute('y', 0);
        icon.setAttribute('font-size', '18');
        icon.setAttribute('font-weight', 'bold');
        icon.setAttribute('text-anchor', 'middle');
        icon.setAttribute('alignment-baseline', 'middle');
        if (node.type === 'logical-and') icon.textContent = '∧';
        else if (node.type === 'logical-or') icon.textContent = '∨';
        else if (node.type === 'logical-not') icon.textContent = '¬';
        else if (node.type === 'conditional-if') icon.textContent = '?';
        else if (node.type === 'conditional-else') icon.textContent = ':';
        else if (node.type === 'multiplex') icon.textContent = 'MUX';
        else if (node.type === 'demultiplex') icon.textContent = 'DEMUX';
        else icon.textContent = '●';
        g.appendChild(icon);
        const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        label.setAttribute('x', 0);
        label.setAttribute('y', 22);
        label.setAttribute('font-size', '12');
        label.setAttribute('text-anchor', 'middle');
        label.setAttribute('alignment-baseline', 'hanging');
        label.textContent = node.label;
        g.appendChild(label);
        g.onclick = () => {
          this.selectedNodeIdx = idx;
          const newLabel = prompt('Edit node label:', node.label);
          if (newLabel !== null) {
            node.label = newLabel;
            this.saveAndRender();
          }
        };
        g.oncontextmenu = (e) => {
          e.preventDefault();
          if (confirm('Delete node?')) {
            this.nodes.splice(idx, 1);
            this.saveAndRender();
          }
        };
        svg.appendChild(g);
        // Always animate from empty to next
        animateSVGAttrD3(g, 'transform', '', next, this.settings.animationSpeed, 'transform');
      } else {
        const prev = g.getAttribute('transform') || '';
        if (prev !== next) animateSVGAttrD3(g, 'transform', prev, next, this.settings.animationSpeed, 'transform');
      }
    });
    // Arrowhead marker (unchanged)
    if (!svg.querySelector('marker#arrowhead')) {
      const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
      const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
      marker.setAttribute('id', 'arrowhead');
      marker.setAttribute('markerWidth', '10');
      marker.setAttribute('markerHeight', '7');
      marker.setAttribute('refX', '10');
      marker.setAttribute('refY', '3.5');
      marker.setAttribute('orient', 'auto');
      const arrowPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
      arrowPath.setAttribute('d', 'M0,0 L10,3.5 L0,7 Z');
      arrowPath.setAttribute('fill', '#888');
      marker.appendChild(arrowPath);
      defs.appendChild(marker);
      svg.appendChild(defs);
    }
    // Animate node transitions
    Object.entries(positions).forEach(([idx, pos]) => {
      const prev = this._lastPositions[idx];
      if (prev && (prev.x !== pos.x || prev.y !== pos.y)) {
        animatePosition(prev, pos, this.settings.animationSpeed, (cur) => {
          this._lastPositions[idx] = { ...cur };
          // Update node DOM position if exists
          const nodeEl = canvas.querySelector(`[data-node-idx='${idx}']`);
          if (nodeEl) {
            nodeEl.style.left = `${cur.x * 10 + 40}px`;
            nodeEl.style.top = `${cur.y * 10 + 40}px`;
            nodeEl.classList.add('animated-mm-pulse');
            setTimeout(() => nodeEl.classList.remove('animated-mm-pulse'), 300);
          }
        });
      } else {
        this._lastPositions[idx] = { ...pos };
      }
    });
    // Group boundaries
    this.groups.forEach(group => {
      const groupDiv = safeCreateDiv(canvas, 'animated-mm-group');
      groupDiv.style.background = group.color + '22';
      groupDiv.style.border = '2px solid ' + group.color;
      groupDiv.style.margin = '8px';
      groupDiv.style.padding = '8px';
      groupDiv.style.display = 'inline-block';
      groupDiv.style.verticalAlign = 'top';
      groupDiv.style.minWidth = '120px';
      groupDiv.style.minHeight = '120px';
      safeCreateEl(groupDiv, 'h5', { text: group.name });
      // Drag node into group
      groupDiv.ondragover = e => { e.preventDefault(); this.draggingGroup = group.name; };
      groupDiv.ondrop = e => {
        if (this.draggingNode != null) {
          this.nodes[this.draggingNode].group = group.name;
          this.draggingNode = null;
          this.renderSkeleton();
        }
      };
      // Render nodes in group
      this.nodes.forEach((node, idx) => {
        if ((node.group || 'Orphans') === group.name) {
          const nodeDiv = safeCreateDiv(groupDiv, 'animated-mm-node');
          nodeDiv.setText(node.label);
          nodeDiv.setAttr('data-type', node.type);
          nodeDiv.setAttr('draggable', 'true');
          nodeDiv.style.transition = 'all 0.3s cubic-bezier(.4,2,.6,1)';
          // Position node using intelligent layout
          const pos = positions[idx];
          if (pos) {
            nodeDiv.style.position = 'absolute';
            nodeDiv.style.left = `${pos.x * 10 + 40}px`;
            nodeDiv.style.top = `${pos.y * 10 + 40}px`;
          }
          // Render icons/visuals for node types
          const icon = document.createElement('span');
          icon.className = 'animated-mm-node-icon';
          if (node.type === 'logical-and') icon.textContent = '∧';
          else if (node.type === 'logical-or') icon.textContent = '∨';
          else if (node.type === 'logical-not') icon.textContent = '¬';
          else if (node.type === 'conditional-if') icon.textContent = '?';
          else if (node.type === 'conditional-else') icon.textContent = ':';
          else if (node.type === 'multiplex') icon.textContent = 'MUX';
          else if (node.type === 'demultiplex') icon.textContent = 'DEMUX';
          else icon.textContent = '●';
          nodeDiv.prepend(icon);
          // Node editing
          nodeDiv.onclick = () => {
            this.selectedNodeIdx = idx;
            const newLabel = prompt('Edit node label:', node.label);
            if (newLabel !== null) {
              node.label = newLabel;
              this.renderSkeleton();
            }
          };
          // Node deletion
          nodeDiv.oncontextmenu = (e) => {
            e.preventDefault();
            if (confirm('Delete node?')) {
              this.nodes.splice(idx, 1);
              this.renderSkeleton();
            }
          };
          // Drag to connect nodes
          nodeDiv.ondragstart = e => { this.draggingNode = idx; };
          nodeDiv.ondragend = e => { this.draggingNode = null; };
          nodeDiv.ondrop = e => {
            if (this.draggingNode != null && this.draggingNode !== idx) {
              this.addLink(this.draggingNode, idx);
              this.draggingNode = null;
              this.renderSkeleton();
            }
          };
          nodeDiv.ondragover = e => { e.preventDefault(); this.dragOverNode = idx; };
        }
      });
    });
    // Link configuration
    safeCreateEl(this.containerEl, 'h4', { text: 'Links' });
    safeCreateDiv(this.containerEl, 'animated-mm-links');
    // Group in/out configuration
    safeCreateEl(this.containerEl, 'h4', { text: 'Groups' });
    safeCreateDiv(this.containerEl, 'animated-mm-groups', { text: 'Drag nodes into group areas to assign them. Group boundaries are shown as colored boxes.' });
    // Comments: further extension: more advanced PCB-like optimization, node/link metadata, etc.
  }
  addNode(node) {
    if (!node.label || node.label.trim() === '') {
      new Notice('Node label cannot be empty.');
      return;
    }
    this.nodes.push({ ...node });
    this.saveAndRender();
  }
  addLink(fromIdx, toIdx) {
    if (fromIdx === toIdx) return;
    if (!this.links.some(l => l.from === fromIdx && l.to === toIdx)) {
      this.links.push({ from: fromIdx, to: toIdx });
      this.saveAndRender();
    } else {
      new Notice('Link already exists.');
    }
  }
  saveAndRender() {
    this.plugin.saveMindMapData({ nodes: this.nodes, links: this.links, groups: this.groups });
    this.renderSkeleton();
  }
}

class KanbanGraphSettings {
  constructor() {
    this.lens = 'semantic'; // 'semantic' | 'link' | 'workflow' | 'trait'
    this.animationSpeed = 500;
    this.colorTheme = 'zen';
  }
}

class KanbanGraphView extends ItemView {
  constructor(settings, leaf) {
    super(leaf);
    this.settings = settings;
    this.containerEl.addClass('kanban-graph-view');
    // Placeholder: lanes and force-directed node layout
    this.renderSkeleton();
  }
  getViewType() { return KANBAN_GRAPH_VIEW_TYPE; }
  getDisplayText() { return 'Semantic Kanban Graph'; }
  getIcon() { return 'layout-kanban'; }
  renderSkeleton() {
    this.containerEl.empty();
    const lanes = ['Group A', 'Group B', 'Orphans'];
    const lanesDiv = safeCreateDiv(this.containerEl, 'kanban-graph-lanes');
    for (const lane of lanes) {
      const laneDiv = safeCreateDiv(lanesDiv, 'kanban-graph-lane');
      safeCreateEl(laneDiv, 'h3', { text: lane });
      // Placeholder: nodes in lane
      safeCreateDiv(laneDiv, 'kanban-graph-nodes', { text: '(nodes here)' });
    }
    // Placeholder: drag-and-drop, force-directed layout, animated transitions
  }
}

class KanbanGraphSettingsTab extends PluginSettingTab {
  constructor(app, plugin) {
    super(app, plugin);
    this.plugin = plugin;
  }
  display() {
    const { containerEl } = this;
    containerEl.empty();
    new Setting(containerEl)
      .setName('Lens')
      .setDesc('Choose the lens for grouping and layout')
      .addDropdown(drop => drop
        .addOption('semantic', 'Semantic')
        .addOption('link', 'Link Structure')
        .addOption('workflow', 'Workflow')
        .addOption('trait', 'Trait')
        .setValue(this.plugin.settings.kanbanGraph.lens)
        .onChange(async (value) => {
          this.plugin.settings.kanbanGraph.lens = value;
          await this.plugin.saveData(this.plugin.settings);
        }));
    new Setting(containerEl)
      .setName('Animation Speed')
      .setDesc('Set the animation speed (ms)')
      .addText(text => text
        .setValue(this.plugin.settings.kanbanGraph.animationSpeed.toString())
        .onChange(async (value) => {
          this.plugin.settings.kanbanGraph.animationSpeed = parseInt(value);
          await this.plugin.saveData(this.plugin.settings);
        }));
    new Setting(containerEl)
      .setName('Color Theme')
      .setDesc('Choose the color theme')
      .addDropdown(drop => drop
        .addOption('zen', 'Zen')
        .addOption('classic', 'Classic')
        .setValue(this.plugin.settings.kanbanGraph.colorTheme)
        .onChange(async (value) => {
          this.plugin.settings.kanbanGraph.colorTheme = value;
          await this.plugin.saveData(this.plugin.settings);
        }));
  }
}

// Add CSS for pulse/highlight animation
const animStyle = document.createElement('style');
animStyle.textContent = `
.animated-mm-pulse {
  animation: mm-pulse 0.3s;
}
@keyframes mm-pulse {
  0% { box-shadow: 0 0 0 0 #4caf50; }
  70% { box-shadow: 0 0 8px 8px #4caf5044; }
  100% { box-shadow: 0 0 0 0 #4caf5000; }
}
`;
document.head.appendChild(animStyle);

module.exports = class AnimatedMindMapPlus extends Plugin {
  async onload() {
    this.settings = Object.assign(new AnimatedMindMapSettings(), await this.loadData() || {});
    // Load mind map data
    const savedData = await this.loadData();
    this.mindMapData = savedData && savedData.mindMapData ? savedData.mindMapData : { nodes: [], links: [], groups: undefined };
    // Add left sidebar ribbon icon
    this.addRibbonIcon('dot-network', 'Open Animated Mind Map+ (Sidebar)', () => {
      this.activateView();
    });
    this.registerView(
      ANIMATED_MM_VIEW_TYPE,
      leaf => new AnimatedMindMapView(this.settings, leaf, this)
    );
    this.addCommand({
      id: 'open-animated-mindmap',
      name: 'Open Animated Mind Map+',
      callback: () => this.activateView()
    });
    this.addSettingTab(new AnimatedMindMapSettingsTab(this.app, this));
    if (!this.settings.kanbanGraph) this.settings.kanbanGraph = new KanbanGraphSettings();
    this.registerView(KANBAN_GRAPH_VIEW_TYPE, leaf => new KanbanGraphView(this.settings.kanbanGraph, leaf));
    this.addCommand({
      id: 'open-kanban-graph-view',
      name: 'Open Semantic Kanban Graph',
      callback: () => {
        this.app.workspace.getRightLeaf(false).setViewState({ type: KANBAN_GRAPH_VIEW_TYPE });
      }
    });
    this.addSettingTab(new KanbanGraphSettingsTab(this.app, this));
    this.addCommand({
      id: 'open-animated-mindmap-view',
      name: 'Open Animated Mind Map (WFC/Quantum Orbit)',
      callback: () => {
        this.app.workspace.getRightLeaf(false).setViewState({ type: ANIMATED_MM_VIEW_TYPE });
      }
    });
    this._listeners = [];
  }
  async activateView() {
    let leaf = this.app.workspace.getLeavesOfType(ANIMATED_MM_VIEW_TYPE)[0];
    if (!leaf) {
      leaf = this.app.workspace.getRightLeaf(false);
      await leaf.setViewState({ type: ANIMATED_MM_VIEW_TYPE });
    }
    this.app.workspace.revealLeaf(leaf);
  }
  onunload() {
    // Clean up listeners, intervals, etc.
    if (this._listeners) {
      this._listeners.forEach(unreg => unreg && unreg());
      this._listeners = [];
    }
  }
  async saveMindMapData(data) {
    this.mindMapData = data;
    await this.saveData({ ...this.settings, mindMapData: data });
  }
  getMindMapData() {
    return this.mindMapData;
  }
}; 