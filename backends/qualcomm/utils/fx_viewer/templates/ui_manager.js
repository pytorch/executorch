// Manages all non-canvas DOM elements: taskbar, search, layers dropdown, legend, and info panel.
class UIManager {
    constructor(container, viewer, options = {}) {
        this.container = container;
        this.viewer = viewer;
        this.controller = viewer.controller;
        this.options = options;
        this._teardownFns = [];
        this.controls = {
            toolbar: true,
            search: true,
            layers: true,
            colorBy: true,
            theme: true,
            legend: true,
            zoomButtons: true,
            fullscreenButton: false,
            highlightButton: true,
            ...(options.controls || {}),
        };
        this.mounts = options.mounts || {};
        this.layerCheckboxes = new Map();
        this.colorByRadios = new Map();
        this.buildUI();
    }

    _createTaskbarButton({ html, title, onClick, className = 'fx-button' }) {
        const btn = document.createElement('button');
        btn.className = className;
        btn.innerHTML = html;
        if (title) btn.title = title;
        if (typeof onClick === 'function') btn.onclick = onClick;
        return btn;
    }

    buildUI() {
        this.taskbar = document.createElement('div');
        this.taskbar.className = 'fx-taskbar';

        this.searchContainer = null;
        this.layersContainer = null;
        this.layersMenu = null;
        this.themeSelect = null;
        this.btnHighlight = null;
        this.btnZoomFit = null;
        this.btnFullscreen = null;
        this.btnClear = null;

        if (this.controls.search) {
            this.searchContainer = document.createElement('div');
            this.searchContainer.className = 'fx-search-container';

            this.searchInput = document.createElement('input');
            this.searchInput.className = 'fx-search-input';
            this.searchInput.placeholder = 'Search nodes (fuzzy)...';

            this.searchMenu = document.createElement('div');
            this.searchMenu.className = 'fx-search-menu';

            this.searchContainer.appendChild(this.searchInput);
            this.searchContainer.appendChild(this.searchMenu);
            this.taskbar.appendChild(this.searchContainer);
        } else {
            this.searchInput = null;
            this.searchMenu = null;
        }

        if (this.controls.layers || this.controls.colorBy) {
            this.layersContainer = document.createElement('div');
            this.layersContainer.style.position = 'relative';
            this.layersContainer.style.marginLeft = '10px';

            this.btnLayers = this._createTaskbarButton({
                html: '&#x1F4DA; Layers',
            });

            this.layersMenu = document.createElement('div');
            this.layersMenu.className = 'fx-layers-menu';
            this.rebuildLayersMenu();

            this.btnLayers.onclick = () => {
                this.layersMenu.style.display = this.layersMenu.style.display === 'block' ? 'none' : 'block';
            };

            this.layersContainer.appendChild(this.btnLayers);
            this.layersContainer.appendChild(this.layersMenu);
            this.taskbar.appendChild(this.layersContainer);
        }

        if (this.controls.highlightButton) {
            this.btnHighlight = this._createTaskbarButton({
                html: '&#x1F517;',
                title: 'Toggle Highlight Ancestors/Descendants',
                className: 'fx-button active',
                onClick: () => {
                    this.controller.state.highlightAncestors = !this.controller.state.highlightAncestors;
                    this.btnHighlight.classList.toggle('active', this.controller.state.highlightAncestors);
                    this.controller.setState({});
                },
            });
            this.taskbar.appendChild(this.btnHighlight);
        }

        if (this.controls.zoomButtons) {
            this.btnZoomFit = this._createTaskbarButton({
                html: '&#x2922;',
                title: 'Zoom to Fit',
                onClick: () => this.controller.zoomToFit(),
            });
            this.taskbar.appendChild(this.btnZoomFit);
        }

        if (this.controls.fullscreenButton) {
            this.btnFullscreen = this._createTaskbarButton({
                html: '&#x26F6;',
                title: 'Enter Fullscreen',
                onClick: async () => {
                    if (document.fullscreenElement) {
                        await this.viewer.exitFullscreen();
                    } else {
                        await this.viewer.enterFullscreen();
                    }
                    this.syncFullscreenButton();
                },
            });
            this.taskbar.appendChild(this.btnFullscreen);
            this._onFullscreenChange = () => this.syncFullscreenButton();
            fxOn(this._teardownFns, document, 'fullscreenchange', this._onFullscreenChange);
        }

        if (this.controls.theme) {
            this.themeSelect = document.createElement('select');
            this.themeSelect.className = 'fx-select';
            this.themeSelect.innerHTML = `<option value="light">&#x2600; Light</option><option value="dark">&#x1F319; Dark</option>`;
            this.themeSelect.onchange = (e) => {
                this.controller.setState({ themeName: e.target.value });
            };
            this.taskbar.appendChild(this.themeSelect);
        }

        this.infoPanel = document.createElement('div');
        this.infoPanel.className = 'fx-info-panel';
        this.infoPanel.innerHTML = '<div style="color: #888; text-align: center; margin-top: 20px;">No node selected<br><br>Hover or click a node</div>';

        this.legendOverlay = document.createElement('div');
        this.legendOverlay.className = 'fx-legend-overlay';

        const toolbarContainer = this.mounts.toolbarContainer || this.container;
        const legendContainer = this.mounts.legendContainer || this.container;
        const infoContainer = this.mounts.infoContainer || this.viewer.sidebar || this.container;
        if (this.controls.toolbar) {
            toolbarContainer.appendChild(this.taskbar);
        }
        if (this.controls.legend) {
            legendContainer.appendChild(this.legendOverlay);
        } else {
            this.legendOverlay.style.display = 'none';
        }
        infoContainer.appendChild(this.infoPanel);

        this.applyThemeToDOM();
        this.renderLegend();

        if (this.searchInput) {
            fxOn(this._teardownFns, this.searchInput, 'input', (e) => {
                this.controller.handleSearch(e.target.value);
                if (e.target.value) {
                    this._showSearchMenu();
                } else {
                    this.searchMenu.style.display = 'none';
                }
            });

            fxOn(this._teardownFns, this.searchInput, 'keydown', (e) => {
                if (e.key === 'ArrowDown') {
                    e.preventDefault();
                    this.controller.handleSearchNavigate(1);
                } else if (e.key === 'ArrowUp') {
                    e.preventDefault();
                    this.controller.handleSearchNavigate(-1);
                } else if (e.key === 'Enter') {
                    e.preventDefault();
                    this.controller.handleSearchSelect();
                }
            });
        }

        this._onDocumentClick = (e) => {
            if (this.searchContainer && !this.searchContainer.contains(e.target)) {
                this.closeSearchMenu();
                if (this.controller.state.searchCandidates.length > 0) {
                    this.controller.setState({ searchCandidates: [], searchSelectedIndex: -1, previewNodeId: null });
                }
            }
            if (this.layersContainer && !this.layersContainer.contains(e.target)) {
                this.layersMenu.style.display = 'none';
            }
        };
        fxOn(this._teardownFns, document, 'click', this._onDocumentClick);

        this.syncControlsFromState();
        this.syncFullscreenButton();
    }

    rebuildLayersMenu() {
        if (!this.layersMenu) return;
        this.layerCheckboxes.clear();
        this.colorByRadios.clear();
        const radioName = `fx-color-by-${this.viewer.containerId || 'viewer'}`;

        let layersHtml = '';
        if (this.controls.layers) {
            layersHtml += `<div style="padding: 5px; font-weight: bold; border-bottom: 1px solid #ccc;">Extensions</div>`;
            for (const [extId, extData] of Object.entries(this.viewer.store.extensions)) {
                layersHtml += `
                    <label style="display: block; padding: 5px; cursor: pointer;">
                        <input type="checkbox" class="fx-layer-checkbox" value="${fxEsc(extId)}">
                        ${fxEsc(extData.name)}
                    </label>
                `;
            }
        }

        if (this.controls.colorBy) {
            layersHtml += `<div style="padding: 5px; font-weight: bold; border-bottom: 1px solid #ccc; margin-top: 10px;">Color By</div>`;
            layersHtml += `
                <label style="display: block; padding: 5px; cursor: pointer;">
                    <input type="radio" name="${radioName}" value="base">
                    Base Graph
                </label>
            `;
            for (const [extId, extData] of Object.entries(this.viewer.store.extensions)) {
                if (extData.legend && extData.legend.length > 0) {
                    layersHtml += `
                        <label style="display: block; padding: 5px; cursor: pointer;">
                            <input type="radio" name="${radioName}" value="${fxEsc(extId)}">
                            ${fxEsc(extData.name)}
                        </label>
                    `;
                }
            }
        }

        this.layersMenu.innerHTML = layersHtml;
        this.layersMenu.querySelectorAll('.fx-layer-checkbox').forEach(cb => {
            this.layerCheckboxes.set(cb.value, cb);
            cb.onchange = (e) => {
                const active = new Set(this.controller.state.activeExtensions);
                if (e.target.checked) active.add(e.target.value);
                else active.delete(e.target.value);
                this.controller.setState({ activeExtensions: active });
            };
        });
        this.layersMenu.querySelectorAll('input[type="radio"]').forEach(radio => {
            this.colorByRadios.set(radio.value, radio);
            radio.onchange = (e) => {
                this.controller.setState({ colorBy: e.target.value });
            };
        });
    }

    syncControlsFromState() {
        const state = this.controller.state;
        if (this.themeSelect) {
            this.themeSelect.value = state.themeName;
        }
        if (this.btnHighlight) {
            this.btnHighlight.classList.toggle('active', !!state.highlightAncestors);
        }
        if (this.layerCheckboxes.size > 0) {
            this.layerCheckboxes.forEach((checkbox, extId) => {
                checkbox.checked = state.activeExtensions.has(extId);
            });
        }
        if (this.colorByRadios.size > 0) {
            this.colorByRadios.forEach((radio, extId) => {
                radio.checked = extId === state.colorBy;
            });
        }
    }

    setControlVisibility(flags = {}) {
        if ('toolbar' in flags && this.taskbar) {
            this.taskbar.style.display = flags.toolbar ? '' : 'none';
        }
        if ('search' in flags && this.searchContainer) {
            this.searchContainer.style.display = flags.search ? '' : 'none';
        }
        if ('layers' in flags && this.layersContainer) {
            this.layersContainer.style.display = flags.layers ? '' : 'none';
        }
        if ('theme' in flags && this.themeSelect) {
            this.themeSelect.style.display = flags.theme ? '' : 'none';
        }
        if ('legend' in flags && this.legendOverlay) {
            this.legendOverlay.style.display = flags.legend ? '' : 'none';
        }
        if ('fullscreenButton' in flags && this.btnFullscreen) {
            this.btnFullscreen.style.display = flags.fullscreenButton ? '' : 'none';
        }
        if ('highlightButton' in flags && this.btnHighlight) {
            this.btnHighlight.style.display = flags.highlightButton ? '' : 'none';
        }
    }

    syncFullscreenButton() {
        if (!this.btnFullscreen) return;
        const active = !!document.fullscreenElement;
        this.btnFullscreen.title = active ? 'Exit Fullscreen' : 'Enter Fullscreen';
        this.btnFullscreen.innerHTML = active ? '&#x2715;' : '&#x26F6;';
    }

    renderLegend() {
        if (!this.controls.legend || !this.legendOverlay) return;
        const colorBy = this.controller.state.colorBy;
        let legendData = [];
        let title = "Legend";
        const theme = THEMES[this.controller.state.themeName];

        if (colorBy === 'base') {
            legendData = this.viewer.store.baseData.legend || [];
            title = "Base Graph";
        } else if (this.viewer.store.extensions[colorBy]) {
            legendData = this.viewer.store.extensions[colorBy].legend || [];
            title = this.viewer.store.extensions[colorBy].name;
        }

        if (legendData.length === 0) {
            this.legendOverlay.style.display = 'none';
            return;
        }

        this.legendOverlay.style.display = 'block';
        const shadeColor = (color, percent) => {
            if (!color || !color.startsWith('#')) return color;
            let R = parseInt(color.substring(1,3), 16);
            let G = parseInt(color.substring(3,5), 16);
            let B = parseInt(color.substring(5,7), 16);
            R = parseInt(R * (100 + percent) / 100);
            G = parseInt(G * (100 + percent) / 100);
            B = parseInt(B * (100 + percent) / 100);
            R = (R<255)?R:255;  G = (G<255)?G:255;  B = (B<255)?B:255;
            R = (R>0)?R:0;      G = (G>0)?G:0;      B = (B>0)?B:0;
            const RR = ((R.toString(16).length==1)?"0"+R.toString(16):R.toString(16));
            const GG = ((G.toString(16).length==1)?"0"+G.toString(16):G.toString(16));
            const BB = ((B.toString(16).length==1)?"0"+B.toString(16):B.toString(16));
            return "#"+RR+GG+BB;
        };

        let html = `<strong>${title}</strong><div style="margin-top:5px;">`;
        legendData.forEach(item => {
            const swatchColor = theme && theme === THEMES.dark ? shadeColor(item.color, -20) : item.color;
            html += `<div style="display:flex; align-items:center; margin-bottom:2px;">
                <div style="width:12px; height:12px; background-color:${swatchColor}; border:1px solid #ccc; margin-right:5px;"></div>
                <span style="font-size:11px;">${fxEsc(item.label)}</span>
            </div>`;
        });
        html += `</div>`;
        this.legendOverlay.innerHTML = html;
    }

    applyThemeToDOM() {
        const theme = THEMES[this.controller.state.themeName];
        this.viewer.wrapper.style.setProperty('--fx-ui-hover', theme.uiHover);
        this.viewer.wrapper.style.backgroundColor = theme.bg;
        this.viewer.wrapper.style.color = theme.text;
        if (this.viewer.sidebar) {
            this.viewer.sidebar.style.backgroundColor = theme.bg;
            this.viewer.sidebar.style.borderLeftColor = theme.uiBorder;
        }

        if (this.taskbar) {
            this.taskbar.style.backgroundColor = theme.uiBg;
            this.taskbar.style.borderColor = theme.uiBorder;
        }
        if (this.searchMenu) {
            this.searchMenu.style.backgroundColor = theme.uiBg;
            this.searchMenu.style.borderColor = theme.uiBorder;
        }

        if (this.legendOverlay) {
            this.legendOverlay.style.backgroundColor = theme.legendBg;
            this.legendOverlay.style.borderColor = theme.uiBorder;
        }

        this.infoPanel.style.backgroundColor = theme.bg;
        
        const controls = this.viewer.wrapper.querySelectorAll('.fx-button, .fx-search-input, .fx-select, .fx-layers-menu');
        controls.forEach(ctrl => {
            ctrl.style.borderColor = theme.uiBorder;
            ctrl.style.color = theme.text;
            ctrl.style.backgroundColor = theme.uiBg;
        });
        
        document.querySelectorAll('.fx-search-item').forEach(item => {
            item.style.borderBottomColor = theme.uiBorder;
        });
        
        this.viewer.resizer.style.backgroundColor = theme.uiBorder;
        if (this.viewer.resizerH) {
            this.viewer.resizerH.style.backgroundColor = theme.uiBorder;
        }
        
        if (this.viewer.minimapRenderer && this.viewer.minimapRenderer.container) {
            this.viewer.minimapRenderer.container.style.borderTopColor = theme.uiBorder;
            this.viewer.minimapRenderer.generateThumbnail();
            this.viewer.minimapRenderer.render();
        }
    }

    closeSearchMenu() {
        if (this.searchMenu) this.searchMenu.style.display = 'none';
    }

    _showSearchMenu() {
        if (!this.searchMenu) return;
        const canvas = this.viewer.canvasRenderer?.canvas;
        if (canvas) {
            const rect = canvas.getBoundingClientRect();
            this.searchMenu.style.maxHeight = Math.floor(rect.height * 0.5) + 'px';
            this.searchMenu.style.maxWidth = Math.floor(rect.width * 0.5) + 'px';
        }
        this.searchMenu.style.display = 'block';
    }

    updateSearchResults(candidates, selectedIndex) {
        if (!this.searchMenu) return;
        this.searchMenu.innerHTML = '';
        if (candidates.length === 0) return;
        this._showSearchMenu();
        
        this.visibleCandidatesCount = 50;
        this.renderSearchCandidatesChunk(candidates, selectedIndex, 0, this.visibleCandidatesCount);

        this.searchMenu.onscroll = () => {
            if (this.searchMenu.scrollTop + this.searchMenu.clientHeight >= this.searchMenu.scrollHeight - 10) {
                if (this.visibleCandidatesCount < candidates.length) {
                    const start = this.visibleCandidatesCount;
                    this.visibleCandidatesCount += 20;
                    this.renderSearchCandidatesChunk(candidates, this.controller.state.searchSelectedIndex, start, this.visibleCandidatesCount);
                }
            }
        };
    }

    updateSearchActiveItem(selectedIndex) {
        if (!this.searchMenu) return;
        Array.from(this.searchMenu.children).forEach((item, idx) => {
            if (idx === selectedIndex) {
                item.classList.add('active');
            } else {
                item.classList.remove('active');
            }
        });
        if (selectedIndex >= 0 && selectedIndex < this.searchMenu.children.length) {
            const childNode = this.searchMenu.children[selectedIndex];
            if (childNode) childNode.scrollIntoView({ block: 'nearest' });
        }
    }

    renderSearchCandidatesChunk(candidates, selectedIndex, start=0, end=50) {
        if (!this.searchMenu) return;
        const theme = THEMES[this.controller.state.themeName];
        for (let idx = start; idx < Math.min(end, candidates.length); idx++) {
            const cand = candidates[idx];
            const item = document.createElement('div');
            item.className = 'fx-search-item' + (idx === selectedIndex ? ' active' : '');
            item.style.borderBottomColor = theme.uiBorder;
            
            let matchText = '';
            if (cand.matchField === 'id') {
                matchText = ''; 
            } else if (cand.matchField) {
                matchText = `<i>${cand.matchField}:</i> ${cand.matchString}`;
            }
            
            // highlightedId and matchString contain intentional HTML (highlight spans from SearchEngine)
            item.innerHTML = `<div><strong>${cand.highlightedId}</strong></div><div style="font-size: 10px; color: ${theme.textMuted};">${matchText}</div>`;
            
            item.onmouseenter = () => this.controller.handleSearchHover(idx);
            item.onmousedown = (e) => {
                e.preventDefault(); 
                this.controller.handleSearchSelect(idx);
            };
            
            this.searchMenu.appendChild(item);
        }
    }

    updateInfoPanel(nodeId) {
        const node = this.viewer.store.activeNodeMap.get(nodeId);
        if (!node) return;
        
        this.infoPanel.style.display = 'block';
        const theme = THEMES[this.controller.state.themeName];
        
        let html = `<h3>Node: ${fxEsc(node.id)}</h3>`;
        html += `<table class="fx-info-table" style="border-color: ${theme.uiBorder}">`;
        
        const renderRow = (key, val) => {
            html += `<tr><th style="border-color: ${theme.uiBorder}">${fxEsc(key)}</th><td style="border-color: ${theme.uiBorder}">${val}</td></tr>`;
        };

        // 1. Core PyTorch Properties
        const coreKeys = ['op', 'name', 'target', 'args', 'kwargs', 'shape', 'dtype', 'tensor_shape'];
        if (node.info) {
            coreKeys.forEach(k => {
                if (k in node.info) {
                    let val = node.info[k];
                    if (k === 'args' || k === 'kwargs') {
                        if (val !== '()' && val !== '{}') {
                            renderRow(k.charAt(0).toUpperCase() + k.slice(1), `<pre style="margin:0; font-size:10px; white-space:pre-wrap; max-width: 250px;">${val}</pre>`);
                        }
                    } else if (k === 'shape' || k === 'tensor_shape') {
                        renderRow("Shape", JSON.stringify(val).replace(/"/g, ''));
                    } else if (k === 'dtype') {
                        renderRow("Dtype", val.replace('torch.', ''));
                    } else {
                        renderRow(k.charAt(0).toUpperCase() + k.slice(1), val);
                    }
                }
            });
        }

        const inEdges = this.viewer.store.revAdjList.get(nodeId) || [];
        if (inEdges.length > 0) {
            let links = inEdges.map(e => `<div class="fx-link" data-node="${fxEsc(e.v)}">${fxEsc(e.v)}</div>`).join('<br>');
            renderRow("Inputs", links);
        }

        const outEdges = this.viewer.store.adjList.get(nodeId) || [];
        if (outEdges.length > 0) {
            let links = outEdges.map(e => `<div class="fx-link" data-node="${fxEsc(e.w)}">${fxEsc(e.w)}</div>`).join('<br>');
            renderRow("Outputs", links);
        }
        
        // Render base custom meta that isn't from an extension
        if (node.info) {
            for (const [key, value] of Object.entries(node.info)) {
                if (coreKeys.includes(key) || key.includes('.')) continue; // skip core and extensions
                let valStr = typeof value === 'object' ? JSON.stringify(value, null, 2) : String(value);
                renderRow(`Meta: ${key}`, `<pre style="margin:0; font-size:10px; white-space:pre-wrap; max-width: 250px; overflow-x: auto;">${valStr}</pre>`);
            }
        }
        html += `</table>`;

        // 2. Extension Groups (Split by Prefix)
        if (node.info) {
            let extensionGroups = {};
            for (const [key, value] of Object.entries(node.info)) {
                if (key.includes('.')) {
                    const parts = key.split('.');
                    const extName = parts[0];
                    const subKey = parts.slice(1).join('.');
                    if (!extensionGroups[extName]) extensionGroups[extName] = {};
                    extensionGroups[extName][subKey] = value;
                }
            }

            for (const [extName, extDict] of Object.entries(extensionGroups)) {
                html += `<div class="fx-ext-header" style="border-bottom: 1px solid ${theme.uiBorder};">--- ${fxEsc(extName)} ---</div>`;
                html += `<table class="fx-info-table" style="border-color: ${theme.uiBorder}; margin-top: 5px;">`;
                for (const [k, v] of Object.entries(extDict)) {
                    let valStr = typeof v === 'object' ? JSON.stringify(v, null, 2) : String(v);
                    html += `<tr><th style="border-color: ${theme.uiBorder}">${fxEsc(k)}</th><td style="border-color: ${theme.uiBorder}"><pre style="margin:0; font-size:10px; white-space:pre-wrap; max-width: 250px; overflow-x: auto;">${fxEsc(valStr)}</pre></td></tr>`;
                }
                html += `</table>`;
            }
        }

        this.infoPanel.innerHTML = html;
        
        const links = this.infoPanel.querySelectorAll('.fx-link');
        links.forEach(link => {
            link.onclick = (e) => {
                const targetNode = e.target.getAttribute('data-node');
                this.controller.selectNode(targetNode);
                this.controller.animateToNode(targetNode);
            };
        });
    }

    updateEdgeInfoPanel(edge) {
        const srcNode = this.viewer.store.activeNodeMap.get(edge.v);
        const dstNode = this.viewer.store.activeNodeMap.get(edge.w);
        if (!srcNode || !dstNode) return;
        
        const theme = THEMES[this.controller.state.themeName];
        let html = `<h3>Edge: ${fxEsc(srcNode.id)} &rarr;<br>${fxEsc(dstNode.id)}</h3>`;
        html += `<table class="fx-info-table" style="border-color: ${theme.uiBorder}">`;

        let shapeStr = '', dtypeStr = '';
        if (srcNode.info && srcNode.info.shape) shapeStr = JSON.stringify(srcNode.info.shape).replace(/"/g, '');
        else if (srcNode.info && srcNode.info.tensor_shape) shapeStr = JSON.stringify(srcNode.info.tensor_shape).replace(/"/g, '');

        if (srcNode.info && srcNode.info.dtype && typeof srcNode.info.dtype === "string") dtypeStr = srcNode.info.dtype.replace('torch.', '');

        if (shapeStr) html += `<tr><th style="border-color: ${theme.uiBorder}">Shape</th><td style="border-color: ${theme.uiBorder}">${fxEsc(shapeStr)}</td></tr>`;
        if (dtypeStr) html += `<tr><th style="border-color: ${theme.uiBorder}">Dtype</th><td style="border-color: ${theme.uiBorder}">${fxEsc(dtypeStr)}</td></tr>`;

        html += `<tr><th style="border-color: ${theme.uiBorder}">Src Node</th><td style="border-color: ${theme.uiBorder}"><div class="fx-link" data-node="${fxEsc(srcNode.id)}">${fxEsc(srcNode.id)}</div></td></tr>`;
        html += `<tr><th style="border-color: ${theme.uiBorder}">Dst Node</th><td style="border-color: ${theme.uiBorder}"><div class="fx-link" data-node="${fxEsc(dstNode.id)}">${fxEsc(dstNode.id)}</div></td></tr>`;
        
        html += `</table>`;
        this.infoPanel.innerHTML = html;
        
        const links = this.infoPanel.querySelectorAll('.fx-link');
        links.forEach(link => {
            link.onclick = (e) => {
                const targetNode = e.target.getAttribute('data-node');
                this.controller.selectNode(targetNode);
                this.controller.animateToNode(targetNode);
            };
        });
    }

    hideInfoPanel() {
        this.infoPanel.innerHTML = '<div style="color: #888; text-align: center; margin-top: 20px;">No node selected<br><br>Hover or click a node</div>';
    }

    destroy() {
        fxOffAll(this._teardownFns);
        if (this.taskbar && this.taskbar.parentNode) this.taskbar.parentNode.removeChild(this.taskbar);
        if (this.legendOverlay && this.legendOverlay.parentNode) this.legendOverlay.parentNode.removeChild(this.legendOverlay);
        if (this.infoPanel && this.infoPanel.parentNode) this.infoPanel.parentNode.removeChild(this.infoPanel);
    }
}
