class FXGraphViewer {
    static create(config) {
        return new FXGraphViewer(config);
    }

    static registerTheme(name, themeTokens) {
        if (!name || !themeTokens || typeof themeTokens !== 'object') {
            throw new Error('registerTheme(name, themeTokens) requires valid inputs');
        }
        THEMES[name] = themeTokens;
    }

    constructor(arg1, arg2) {
        this._listeners = new Map();
        this._layoutState = {};
        this._teardownFns = [];

        this.config = this._normalizeConfig(arg1, arg2);
        this.containerId = this.config._resolved.root.id || 'fx-viewer-root';
        this.rootContainer = this.config._resolved.root;
        this._injectStyles();
        this._buildShell();
        this.config._resolved.slots = this._resolveSlots((this.config.mount && this.config.mount.slots) || {});

        this.store = new GraphDataStore(this.config.payload);
        this.searchEngine = new SearchEngine(this.store);
        this.controller = new ViewerController(this, this.config.state || {});

        const slots = this.config._resolved.slots;
        const canvasMount = slots.canvas || this.mainArea;
        this.canvasRenderer = new CanvasRenderer(canvasMount, this);
        this.canvasContainer = this.canvasRenderer.canvasContainer;

        if (this._shouldUseInternalSidebar()) {
            this.resizerH = document.createElement('div');
            this.resizerH.className = 'fx-resizer-h';
            this.resizerH.title = 'Drag to resize minimap height.';
        } else {
            this.resizerH = null;
        }

        this.ui = new UIManager(this.mainArea, this, {
            controls: this.config.ui.controls,
            mounts: {
                toolbarContainer: slots.toolbar || this.mainArea,
                legendContainer: slots.legend || this.mainArea,
                infoContainer: slots.info || (this._shouldUseInternalSidebar() ? this.sidebar : this.mainArea),
            },
        });

        if (this._isMinimapVisible()) {
            const minimapMount = slots.minimap || (this._shouldUseInternalSidebar() ? this.sidebar : this.mainArea);
            this.minimapRenderer = new MinimapRenderer(minimapMount, this);
            const minimapHeight = this.config.layout?.panels?.minimap?.height;
            if (typeof minimapHeight === 'number' && minimapHeight > 0) {
                this.minimapRenderer.container.style.height = `${minimapHeight}px`;
            }
            if (this.resizerH && minimapMount === this.sidebar) {
                this.sidebar.insertBefore(this.resizerH, this.minimapRenderer.container);
            }
        } else {
            this.minimapRenderer = null;
        }

        this.setupResizer();
        this.applyLayout(this.config.layout);
    }

    _normalizeConfig(arg1, arg2) {
        const isNewConfig = arg1 && typeof arg1 === 'object' && 'payload' in arg1;
        const config = isNewConfig
            ? { ...arg1 }
            : {
                payload: arg2,
                mount: { root: typeof arg1 === 'string' ? `#${arg1}` : arg1 },
            };

        if (!config.payload) {
            throw new Error('FXGraphViewer requires a payload');
        }

        const root = this._resolveElement(config.mount && config.mount.root);
        if (!root) {
            throw new Error(`FXGraphViewer root mount not found: ${String(config.mount && config.mount.root)}`);
        }

        const preset = ((config.layout && config.layout.preset) || 'split').toLowerCase();
        const presetDefaults = this._presetDefaults(preset);
        const mergedLayout = this._deepMerge(presetDefaults.layout, config.layout || {});

        const mergedUI = this._deepMerge(presetDefaults.ui, config.ui || {});
        if (
            mergedLayout &&
            mergedLayout.fullscreen &&
            Object.prototype.hasOwnProperty.call(mergedLayout.fullscreen, 'button')
        ) {
            mergedUI.controls.fullscreenButton = !!mergedLayout.fullscreen.button;
        }
        const mergedState = this._deepMerge(presetDefaults.state, config.state || {});

        const slots = (config.mount && config.mount.slots) || {};
        const resolvedSlots = this._resolveSlots(slots);

        if (Array.isArray(mergedState.activeExtensions)) {
            mergedState.activeExtensions = mergedState.activeExtensions.slice();
        }

        return {
            payload: config.payload,
            mount: config.mount || { root },
            layout: mergedLayout,
            ui: mergedUI,
            state: mergedState,
            _resolved: {
                root,
                slots: resolvedSlots,
                preset,
            },
        };
    }

    _resolveElement(ref) {
        if (!ref) return null;
        if (typeof ref === 'string') {
            if (ref.startsWith('#') || ref.startsWith('.')) {
                return document.querySelector(ref);
            }
            return document.getElementById(ref) || document.querySelector(ref);
        }
        if (ref instanceof HTMLElement) return ref;
        return null;
    }

    _resolveSlots(slots) {
        return {
            canvas: this._resolveElement(slots.canvas),
            toolbar: this._resolveElement(slots.toolbar),
            info: this._resolveElement(slots.info),
            minimap: this._resolveElement(slots.minimap),
            legend: this._resolveElement(slots.legend),
        };
    }

    _presetDefaults(preset) {
        const split = {
            layout: {
                preset,
                panels: {
                    sidebar: { visible: true, width: 500, resizable: true, collapsible: true },
                    info: { visible: true, dock: 'sidebar' },
                    minimap: { visible: true, dock: 'sidebar', height: 500, resizable: true },
                    legend: { visible: true, dock: 'canvas' },
                },
                fullscreen: { enabled: true, button: false },
            },
            ui: {
                controls: {
                    toolbar: true,
                    search: true,
                    layers: true,
                    colorBy: true,
                    theme: true,
                    legend: true,
                    zoomButtons: true,
                    fullscreenButton: false,
                    clearButton: true,
                    highlightButton: true,
                },
            },
            state: {
                theme: 'light',
                colorBy: 'base',
                highlightAncestors: true,
            },
        };

        if (preset === 'compact') {
            split.layout.panels.sidebar.visible = false;
            split.layout.panels.minimap.visible = false;
            split.layout.panels.info.visible = false;
        }
        if (preset === 'headless') {
            split.layout.panels.sidebar.visible = false;
            split.layout.panels.minimap.visible = false;
            split.layout.panels.info.visible = false;
            split.ui.controls.toolbar = false;
            split.ui.controls.search = false;
            split.ui.controls.layers = false;
            split.ui.controls.colorBy = false;
            split.ui.controls.theme = false;
            split.ui.controls.legend = false;
        }
        if (preset === 'custom') {
            split.layout.panels.sidebar.visible = false;
            split.layout.panels.minimap.visible = false;
            split.layout.panels.info.visible = false;
        }
        return split;
    }

    _deepMerge(base, patch) {
        if (!patch || typeof patch !== 'object') {
            return Array.isArray(base) ? base.slice() : { ...base };
        }
        const out = Array.isArray(base) ? base.slice() : { ...base };
        Object.keys(patch).forEach((key) => {
            const patchVal = patch[key];
            const baseVal = out[key];
            if (
                patchVal &&
                typeof patchVal === 'object' &&
                !Array.isArray(patchVal) &&
                baseVal &&
                typeof baseVal === 'object' &&
                !Array.isArray(baseVal)
            ) {
                out[key] = this._deepMerge(baseVal, patchVal);
            } else {
                out[key] = patchVal;
            }
        });
        return out;
    }

    _addListener(target, eventName, handler, options) {
        if (!target || !target.addEventListener || !target.removeEventListener) return;
        target.addEventListener(eventName, handler, options);
        this._teardownFns.push(() => target.removeEventListener(eventName, handler, options));
    }

    _injectStyles() {
        if (document.getElementById('fx-viewer-styles')) return;
        const style = document.createElement('style');
        style.id = 'fx-viewer-styles';
        style.innerHTML = `
            .fx-viewer-wrapper { display: flex; flex-direction: row; width: 100%; height: 100%; overflow: hidden; font-family: sans-serif; }
            .fx-main-area { flex: 1; position: relative; overflow: hidden; }
            .fx-resizer { width: 6px; background: #ccc; cursor: col-resize; z-index: 20; transition: background 0.2s; }
            .fx-resizer:hover, .fx-resizer.dragging { background: #999; }
            .fx-sidebar { width: 500px; display: flex; flex-direction: column; background: #fff; border-left: 1px solid #ccc; z-index: 10; }
            .fx-sidebar.collapsed { display: none; }
            .fx-canvas { display: block; width: 100%; height: 100%; }
            .fx-taskbar { position: absolute; top: 10px; left: 10px; right: 10px; min-height: 40px; border-radius: 4px; display: flex; align-items: center; padding: 0 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 10; border: 1px solid transparent; overflow: visible; flex-wrap: wrap; gap: 6px; }
            .fx-search-container { position: relative; flex: 1; max-width: 400px; }
            .fx-search-input { width: 100%; padding: 6px; box-sizing: border-box; }
            .fx-search-menu { position: absolute; top: 100%; left: 0; right: 0; max-height: 300px; overflow-y: auto; display: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid transparent; border-top: none; z-index: 100; }
            .fx-layers-menu { position: absolute; top: 100%; right: 0; min-width: 260px; max-width: 420px; max-height: 60vh; overflow-y: auto; display: none; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border: 1px solid transparent; z-index: 200; }
            .fx-search-item { padding: 8px; cursor: pointer; border-bottom: 1px solid transparent; }
            .fx-search-item:hover, .fx-search-item.active { background: var(--fx-ui-hover, #f0f8ff); }
            .fx-minimap-container { width: 100%; height: 500px; border-top: 1px solid transparent; flex-shrink: 0; }
            .fx-minimap { width: 100%; height: 100%; display: block; cursor: crosshair; }
            .fx-info-panel { flex: 1; overflow-y: auto; padding: 15px; font-size: 13px; display: block; }
            .fx-info-panel h3 { margin-top: 0; margin-bottom: 15px; font-size: 15px; word-break: break-all; }
            .fx-info-table { width: 100%; border-collapse: collapse; border: 1px solid transparent; }
            .fx-info-table th, .fx-info-table td { border: 1px solid transparent; padding: 6px; text-align: left; vertical-align: top; }
            .fx-info-table th { width: 60px; font-weight: bold; }
            .fx-ext-header { margin-top: 15px; padding: 4px 6px; font-weight: bold; font-size: 12px; letter-spacing: 0.5px; background: rgba(0,0,0,0.03); }
            .fx-legend-overlay { position: absolute; right: 10px; bottom: 10px; padding: 8px 10px; border: 1px solid transparent; border-radius: 4px; font-size: 12px; max-width: 260px; max-height: 40vh; overflow-y: auto; box-shadow: 0 2px 6px rgba(0,0,0,0.1); z-index: 15; }
            .fx-link { color: #0366d6; cursor: pointer; text-decoration: none; font-family: monospace; display: inline-block; margin-bottom: 4px; word-break: break-all; }
            .fx-link:hover { text-decoration: underline; }
            .fx-button { margin-left: 10px; padding: 6px 12px; cursor: pointer; background: transparent; border: 1px solid transparent; border-radius: 4px; font-size: 16px; display: flex; align-items: center; justify-content: center; transition: background 0.2s; }
            .fx-select { margin-left: 10px; padding: 4px; border-radius: 4px; font-size: 14px; }
            .fx-resizer-h { height: 6px; background: #ccc; cursor: row-resize; z-index: 20; transition: background 0.2s; flex-shrink: 0; }
            .fx-resizer-h:hover, .fx-resizer-h.dragging { background: #999; }
            .fx-hidden { display: none !important; }
        `;
        document.head.appendChild(style);
    }

    _buildShell() {
        const root = this.rootContainer;
        const oldWrappers = root.querySelectorAll(':scope > .fx-viewer-wrapper[data-fx-viewer-owned="true"]');
        oldWrappers.forEach((node) => node.remove());

        this.wrapper = document.createElement('div');
        this.wrapper.className = 'fx-viewer-wrapper';
        this.wrapper.dataset.fxViewerOwned = 'true';
        root.appendChild(this.wrapper);

        this.mainArea = document.createElement('div');
        this.mainArea.className = 'fx-main-area';
        this.wrapper.appendChild(this.mainArea);

        this.resizer = document.createElement('div');
        this.resizer.className = 'fx-resizer';
        this.resizer.title = 'Drag to resize sidebar. Double click to toggle.';
        this.wrapper.appendChild(this.resizer);

        this.sidebar = document.createElement('div');
        this.sidebar.className = 'fx-sidebar';
        this.wrapper.appendChild(this.sidebar);
    }

    _shouldUseInternalSidebar() {
        const slots = this.config._resolved.slots;
        const panels = this.config.layout?.panels || {};
        const infoInternal = panels.info?.visible !== false && !slots.info;
        const minimapInternal = panels.minimap?.visible !== false && !slots.minimap;
        return !!(infoInternal || minimapInternal);
    }

    _isSidebarVisible() {
        const panels = this.config.layout?.panels || {};
        if (panels.sidebar && panels.sidebar.visible === false) {
            return false;
        }
        return this._shouldUseInternalSidebar();
    }

    _isMinimapVisible() {
        const panels = this.config.layout?.panels || {};
        return !(panels.minimap && panels.minimap.visible === false);
    }

    setupResizer() {
        let isResizing = false;
        let isResizingH = false;

        if (!this.resizer) return;

        const onResizerMouseDown = (e) => {
            if (!this._isSidebarVisible() || this.config.layout?.panels?.sidebar?.resizable === false) return;
            isResizing = true;
            this.resizer.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        };
        this._addListener(this.resizer, 'mousedown', onResizerMouseDown);

        if (this.resizerH) {
            const onResizerHMouseDown = (e) => {
                if (this.config.layout?.panels?.minimap?.resizable === false) return;
                isResizingH = true;
                this.resizerH.classList.add('dragging');
                document.body.style.cursor = 'row-resize';
                e.preventDefault();
            };
            this._addListener(this.resizerH, 'mousedown', onResizerHMouseDown);
        }

        const onWindowMouseMove = (e) => {
            if (isResizing) {
                const containerRect = this.wrapper.getBoundingClientRect();
                let newWidth = containerRect.right - e.clientX;
                newWidth = Math.max(150, Math.min(newWidth, containerRect.width - 200));
                this.sidebar.style.width = `${newWidth}px`;
                this._layoutState.sidebarWidth = newWidth;

                this.canvasRenderer.resize();
                if (this.minimapRenderer) {
                    this.minimapRenderer.resize();
                    this.minimapRenderer.generateThumbnail();
                }
                this.renderAll();
            } else if (isResizingH && this.minimapRenderer) {
                const containerRect = this.wrapper.getBoundingClientRect();
                let newHeight = containerRect.bottom - e.clientY;
                newHeight = Math.max(100, Math.min(newHeight, containerRect.height - 100));
                this.minimapRenderer.container.style.height = `${newHeight}px`;
                this._layoutState.minimapHeight = newHeight;

                this.minimapRenderer.resize();
                this.minimapRenderer.generateThumbnail();
                this.renderAll();
            }
        };
        this._addListener(window, 'mousemove', onWindowMouseMove);

        const onWindowMouseUp = () => {
            if (isResizing) {
                isResizing = false;
                this.resizer.classList.remove('dragging');
                document.body.style.cursor = '';
            }
            if (isResizingH && this.resizerH) {
                isResizingH = false;
                this.resizerH.classList.remove('dragging');
                document.body.style.cursor = '';
            }
        };
        this._addListener(window, 'mouseup', onWindowMouseUp);

        const onResizerDblClick = () => {
            if (this.config.layout?.panels?.sidebar?.collapsible === false) return;
            this.sidebar.classList.toggle('collapsed');
            requestAnimationFrame(() => {
                this.canvasRenderer.resize();
                this.renderAll();
            });
        };
        this._addListener(this.resizer, 'dblclick', onResizerDblClick);
    }

    applyLayout(layoutPatch) {
        if (layoutPatch) {
            this.config.layout = this._deepMerge(this.config.layout, layoutPatch);
        }
        const panels = this.config.layout?.panels || {};

        const sidebarVisible = this._isSidebarVisible();
        this.sidebar.style.display = sidebarVisible ? '' : 'none';
        this.resizer.style.display = sidebarVisible && panels.sidebar?.resizable !== false ? '' : 'none';

        if (panels.sidebar && typeof panels.sidebar.width === 'number') {
            this.sidebar.style.width = `${panels.sidebar.width}px`;
            this._layoutState.sidebarWidth = panels.sidebar.width;
        }

        if (this.ui && this.ui.infoPanel) {
            this.ui.infoPanel.style.display = panels.info?.visible === false ? 'none' : '';
        }

        if (this.minimapRenderer) {
            this.minimapRenderer.container.style.display = panels.minimap?.visible === false ? 'none' : '';
            if (typeof panels.minimap?.height === 'number') {
                this.minimapRenderer.container.style.height = `${panels.minimap.height}px`;
                this._layoutState.minimapHeight = panels.minimap.height;
            }
        }

        if (this.resizerH) {
            const showResizerH = panels.minimap?.visible !== false && panels.minimap?.resizable !== false;
            this.resizerH.style.display = showResizerH ? '' : 'none';
        }

        if (this.ui) {
            this.ui.setControlVisibility({
                toolbar: this.config.ui.controls.toolbar,
                search: this.config.ui.controls.search,
                layers: this.config.ui.controls.layers || this.config.ui.controls.colorBy,
                theme: this.config.ui.controls.theme,
                legend: this.config.ui.controls.legend && panels.legend?.visible !== false,
                fullscreenButton: !!this.config.ui.controls.fullscreenButton,
            });
        }

        this.canvasRenderer.resize();
        if (this.minimapRenderer) {
            this.minimapRenderer.resize();
            this.minimapRenderer.generateThumbnail();
        }
        this.renderAll();
    }

    init() {
        if (this.minimapRenderer) {
            this.minimapRenderer.generateThumbnail();
        }

        if (this.store.baseData.nodes.length > 10) {
            const firstNode = this.store.baseData.nodes[0];
            const k = 0.5;
            const rect = this.canvasContainer.getBoundingClientRect();
            this.controller.transform.k = k;
            this.controller.transform.x = rect.width / 2 - firstNode.x * k;
            this.controller.transform.y = rect.height / 2 - firstNode.y * k;
            this.renderAll();
        } else {
            this.controller.zoomToFit();
        }
    }

    renderAll() {
        if (this.canvasRenderer) this.canvasRenderer.render();
        if (this.minimapRenderer) this.minimapRenderer.render();
    }

    on(eventName, listener) {
        if (!this._listeners.has(eventName)) {
            this._listeners.set(eventName, new Set());
        }
        this._listeners.get(eventName).add(listener);
        return () => this.off(eventName, listener);
    }

    off(eventName, listener) {
        if (!this._listeners.has(eventName)) return;
        this._listeners.get(eventName).delete(listener);
    }

    _emit(eventName, payload) {
        const listeners = this._listeners.get(eventName);
        if (!listeners) return;
        const event = {
            type: eventName,
            timestamp: Date.now(),
            ...payload,
        };
        listeners.forEach((cb) => {
            try {
                cb(event);
            } catch (err) {
                console.error(`FXGraphViewer listener error on '${eventName}':`, err);
            }
        });
    }

    getState() {
        const state = this.controller.snapshotState();
        return {
            ...state,
            layoutState: {
                ...this._layoutState,
                sidebarCollapsed: this.sidebar && this.sidebar.classList.contains('collapsed'),
            },
            uiVisibility: {
                ...(state.uiVisibility || {}),
                toolbar: this.ui && this.ui.taskbar ? this.ui.taskbar.style.display !== 'none' : false,
                search: this.ui && this.ui.searchContainer ? this.ui.searchContainer.style.display !== 'none' : false,
                layers: this.ui && this.ui.layersContainer ? this.ui.layersContainer.style.display !== 'none' : false,
                theme: this.ui && this.ui.themeSelect ? this.ui.themeSelect.style.display !== 'none' : false,
                legend: this.ui && this.ui.legendOverlay ? this.ui.legendOverlay.style.display !== 'none' : false,
                fullscreenButton: this.ui && this.ui.btnFullscreen ? this.ui.btnFullscreen.style.display !== 'none' : false,
            },
        };
    }

    setState(patch, opts = {}) {
        const source = opts.source || 'api';
        const nextPatch = { ...(patch || {}) };
        if (nextPatch.camera && typeof nextPatch.camera === 'object') {
            if (Number.isFinite(nextPatch.camera.x)) this.controller.transform.x = nextPatch.camera.x;
            if (Number.isFinite(nextPatch.camera.y)) this.controller.transform.y = nextPatch.camera.y;
            if (Number.isFinite(nextPatch.camera.k)) this.controller.transform.k = nextPatch.camera.k;
            delete nextPatch.camera;
        }
        const hasSearchQuery = typeof nextPatch.searchQuery === 'string';
        const searchQuery = hasSearchQuery ? nextPatch.searchQuery : null;
        if (hasSearchQuery) delete nextPatch.searchQuery;

        this.controller.setState(nextPatch, { source });
        if (hasSearchQuery) {
            if (this.ui && this.ui.searchInput) {
                this.ui.searchInput.value = searchQuery;
            }
            this.controller.handleSearch(searchQuery);
        }
        this.renderAll();
    }

    replaceState(nextState, opts = {}) {
        const source = opts.source || 'api';
        const ns = nextState || {};
        const replacePatch = {
            hoveredNodeId: null,
            hoveredEdge: null,
            selectedNodeId: ns.selectedNodeId || null,
            selectedEdge: ns.selectedEdge || null,
            previewNodeId: null,
            ancestors: new Set(),
            descendants: new Set(),
            searchCandidates: [],
            searchSelectedIndex: -1,
            highlightAncestors: ns.highlightAncestors !== false,
            themeName: ns.themeName || ns.theme || 'light',
            activeExtensions: new Set(ns.activeExtensions || []),
            colorBy: ns.colorBy || 'base',
            uiVisibility: { ...(ns.uiVisibility || {}) },
        };
        this.controller.setState(replacePatch, { source });

        if (ns.camera && typeof ns.camera === 'object') {
            if (Number.isFinite(ns.camera.x)) this.controller.transform.x = ns.camera.x;
            if (Number.isFinite(ns.camera.y)) this.controller.transform.y = ns.camera.y;
            if (Number.isFinite(ns.camera.k)) this.controller.transform.k = ns.camera.k;
            this.renderAll();
        }
        if (typeof ns.searchQuery === 'string') {
            if (this.ui && this.ui.searchInput) {
                this.ui.searchInput.value = ns.searchQuery;
            }
            this.controller.handleSearch(ns.searchQuery);
        }
    }

    batch(fn) {
        if (typeof fn === 'function') {
            fn();
        }
    }

    setTheme(themeName) {
        if (!(themeName in THEMES)) {
            const err = new Error(`Unknown theme '${themeName}'`);
            this._emit('error', { error: err, source: 'api' });
            throw err;
        }
        this.setState({ themeName }, { source: 'api' });
    }

    setLayers(layerIds) {
        this.setState({ activeExtensions: new Set(layerIds || []) }, { source: 'api' });
    }

    setColorBy(layerId) {
        this.setState({ colorBy: layerId || 'base' }, { source: 'api' });
    }

    selectNode(nodeId, opts = {}) {
        this.controller.selectNode(nodeId);
        if (opts.animate) {
            this.controller.animateToNode(nodeId, opts.k || null);
        } else if (opts.center !== false) {
            this.controller.panToNode(nodeId);
        }
    }

    clearSelection() {
        this.controller.clearSelection();
    }

    search(query) {
        if (this.ui && this.ui.searchInput) {
            this.ui.searchInput.value = query;
        }
        this.controller.handleSearch(query);
    }

    zoomToFit() {
        this.controller.zoomToFit();
    }

    panToNode(nodeId) {
        this.controller.panToNode(nodeId);
    }

    animateToNode(nodeId, options = {}) {
        const targetK = Object.prototype.hasOwnProperty.call(options, 'k') ? options.k : null;
        this.controller.animateToNode(nodeId, targetK);
    }

    setUIVisibility(flags = {}) {
        if (!this.ui) return;
        const prev = this.getState();
        this.ui.setControlVisibility(flags);
        this.controller.state.uiVisibility = {
            ...(this.controller.state.uiVisibility || {}),
            ...flags,
        };
        this._emit('statechange', {
            prevState: prev,
            nextState: this.getState(),
            source: 'api',
        });
    }

    setLayout(layoutPatch = {}) {
        const prev = this.getState();
        this.applyLayout(layoutPatch);
        this._emit('layoutchange', {
            prevState: prev,
            nextState: this.getState(),
            source: 'api',
        });
    }

    _refreshAfterLayerMutation({ rebuildMenu = false } = {}) {
        this._refreshLayerControls({ rebuildMenu });
        this.controller.setState({}, { source: 'api' });
    }

    _refreshLayerControls({ rebuildMenu = false } = {}) {
        if (!this.ui) return;
        if (rebuildMenu) this.ui.rebuildLayersMenu();
        this.ui.syncControlsFromState();
        this.ui.renderLegend();
    }

    upsertLayer(layerId, layerPayload) {
        this.store.upsertExtension(layerId, layerPayload);
        this._refreshAfterLayerMutation({ rebuildMenu: true });
    }

    removeLayer(layerId) {
        this.store.removeExtension(layerId);
        const active = new Set(this.controller.state.activeExtensions);
        active.delete(layerId);
        const nextColorBy = this.controller.state.colorBy === layerId ? 'base' : this.controller.state.colorBy;
        this.controller.setState({ activeExtensions: active, colorBy: nextColorBy }, { source: 'api' });
        this._refreshLayerControls({ rebuildMenu: true });
    }

    patchLayerNodes(layerId, patchByNodeId) {
        this.store.patchExtensionNodes(layerId, patchByNodeId);
        this._refreshAfterLayerMutation();
    }

    setLayerLabel(layerId, label) {
        this.store.setExtensionLabel(layerId, label);
        this._refreshAfterLayerMutation({ rebuildMenu: true });
    }

    setColorRule(layerId, colorRule) {
        const ext = this.store.extensions[layerId];
        if (!ext || !ext.nodes) return;

        if (typeof colorRule === 'function') {
            Object.entries(ext.nodes).forEach(([nodeId, nodeData]) => {
                const nextColor = colorRule(nodeData, nodeId);
                if (typeof nextColor === 'string') {
                    nodeData.fill_color = nextColor;
                }
            });
            this._refreshAfterLayerMutation();
            return;
        }

        if (colorRule && colorRule.type === 'threshold' && colorRule.field && Array.isArray(colorRule.thresholds)) {
            const thresholds = colorRule.thresholds
                .filter((x) => x && typeof x.value === 'number' && typeof x.color === 'string')
                .sort((a, b) => a.value - b.value);
            Object.values(ext.nodes).forEach((nodeData) => {
                const v = Number(nodeData.info && nodeData.info[colorRule.field]);
                if (!Number.isFinite(v)) return;
                let chosen = thresholds.length > 0 ? thresholds[0].color : null;
                thresholds.forEach((t) => {
                    if (v >= t.value) chosen = t.color;
                });
                if (chosen) nodeData.fill_color = chosen;
            });
            this._refreshAfterLayerMutation();
        }
    }

    enterFullscreen() {
        const target = this.rootContainer;
        if (target.requestFullscreen) {
            return target.requestFullscreen();
        }
        return Promise.resolve();
    }

    exitFullscreen() {
        if (document.fullscreenElement && document.exitFullscreen) {
            return document.exitFullscreen();
        }
        return Promise.resolve();
    }

    destroy() {
        this._listeners.clear();
        if (this.canvasRenderer && this.canvasRenderer.destroy) {
            this.canvasRenderer.destroy();
        }
        if (this.minimapRenderer && this.minimapRenderer.destroy) {
            this.minimapRenderer.destroy();
        }
        if (this.ui && this.ui.destroy) {
            this.ui.destroy();
        }
        while (this._teardownFns.length > 0) {
            const off = this._teardownFns.pop();
            try {
                off();
            } catch (_) {}
        }
        if (this.wrapper && this.wrapper.parentNode) {
            this.wrapper.parentNode.removeChild(this.wrapper);
        }
    }
}

class FXGraphCompare {
    static create(config) {
        return new FXGraphCompare(config);
    }

    constructor(config = {}) {
        this.viewers = Array.isArray(config.viewers) ? config.viewers : [];
        this.sync = {
            selection: true,
            camera: false,
            theme: false,
            layers: false,
            ...(config.sync || {}),
        };
        this.layout = {
            columns: (config.layout && config.layout.columns) || 2,
            compact: !!(config.layout && config.layout.compact),
        };
        this.container = null;
        if (config.layout && config.layout.container) {
            if (typeof config.layout.container === 'string') {
                this.container = document.querySelector(config.layout.container);
            } else if (config.layout.container instanceof HTMLElement) {
                this.container = config.layout.container;
            }
        }
        this._guards = new WeakSet();
        this._offs = [];
        this._compactSnapshots = new WeakMap();

        this._wireSelectionSync();
        this._wireStateSync();
        this._applyColumns();
        this._applyCompact(this.layout.compact);
    }

    _wireSelectionSync() {
        this.viewers.forEach((viewer) => {
            const off = viewer.on('selectionchange', (evt) => {
                if (!this.sync.selection) return;
                if (this._guards.has(viewer)) return;
                if (!evt.nextSelection) {
                    this.viewers.forEach((other) => {
                        if (other === viewer) return;
                        this._applyGuarded(other, () => other.clearSelection());
                    });
                    return;
                }

                this.viewers.forEach((other) => {
                    if (other === viewer) return;
                    if (!other.store.activeNodeMap.has(evt.nextSelection)) return;
                    this._applyGuarded(other, () => {
                        other.selectNode(evt.nextSelection, { animate: true, center: true });
                    });
                });
            });
            this._offs.push(off);
        });
    }

    _wireStateSync() {
        this.viewers.forEach((viewer) => {
            const off = viewer.on('statechange', (evt) => {
                if (this._guards.has(viewer)) return;
                const prev = evt.prevState || {};
                const next = evt.nextState || {};

                this.viewers.forEach((other) => {
                    if (other === viewer) return;
                    this._applyGuarded(other, () => {
                        if (this.sync.theme && prev.theme !== next.theme && typeof next.theme === 'string') {
                            other.setTheme(next.theme);
                        }

                        if (this.sync.layers) {
                            const prevLayers = JSON.stringify(prev.activeExtensions || []);
                            const nextLayers = JSON.stringify(next.activeExtensions || []);
                            if (prevLayers !== nextLayers && Array.isArray(next.activeExtensions)) {
                                other.setLayers(next.activeExtensions);
                            }
                            if (prev.colorBy !== next.colorBy && typeof next.colorBy === 'string') {
                                other.setColorBy(next.colorBy);
                            }
                        }

                        if (this.sync.camera) {
                            const pc = prev.camera || {};
                            const nc = next.camera || {};
                            const changed = pc.x !== nc.x || pc.y !== nc.y || pc.k !== nc.k;
                            if (changed && Number.isFinite(nc.x) && Number.isFinite(nc.y) && Number.isFinite(nc.k)) {
                                other.setState({ camera: { x: nc.x, y: nc.y, k: nc.k } });
                            }
                        }
                    });
                });
            });
            this._offs.push(off);
        });
    }

    setColumns(columns) {
        this.layout.columns = Math.max(1, Number(columns) || 1);
        this._applyColumns();
    }

    setCompact(compact) {
        this.layout.compact = !!compact;
        this._applyCompact(this.layout.compact);
    }

    setSync(syncPatch = {}) {
        this.sync = { ...this.sync, ...syncPatch };
    }

    _applyGuarded(viewer, fn) {
        this._guards.add(viewer);
        try {
            fn();
        } finally {
            setTimeout(() => this._guards.delete(viewer), 0);
        }
    }

    _applyColumns() {
        if (!this.container) return;
        this.container.style.display = 'grid';
        this.container.style.gridTemplateColumns = `repeat(${this.layout.columns}, minmax(320px, 1fr))`;
        this.container.style.gap = this.container.style.gap || '10px';
    }

    _applyCompact(compact) {
        this.viewers.forEach((viewer) => {
            if (compact) {
                if (!this._compactSnapshots.has(viewer)) {
                    this._compactSnapshots.set(viewer, JSON.parse(JSON.stringify(viewer.config.layout || {})));
                }
                viewer.setLayout({ panels: { sidebar: { visible: false }, minimap: { visible: false }, info: { visible: false } } });
                return;
            }
            if (this._compactSnapshots.has(viewer)) {
                viewer.setLayout(this._compactSnapshots.get(viewer));
                this._compactSnapshots.delete(viewer);
            }
        });
    }

    destroy() {
        this._offs.forEach((off) => {
            try {
                off();
            } catch (_) {}
        });
        this._offs = [];
    }
}
