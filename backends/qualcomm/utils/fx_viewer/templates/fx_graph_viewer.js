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

        if (slots.info) {
            slots.info.style.overflow = 'hidden';
            slots.info.style.minHeight = '0';
            if (this.ui && this.ui.infoPanel) {
                this.ui.infoPanel.style.height = '100%';
                this.ui.infoPanel.style.overflowY = 'auto';
                this.ui.infoPanel.style.boxSizing = 'border-box';
            }
        }
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
                    minimap: { visible: true, dock: 'sidebar', height: 240, resizable: true },
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

    _injectStyles() {
        if (document.getElementById('fx-viewer-styles')) return;
        const style = document.createElement('style');
        style.id = 'fx-viewer-styles';
        style.innerHTML = `
            .fx-viewer-wrapper { display: flex; flex-direction: row; width: 100%; height: 100%; overflow: hidden; font-family: sans-serif; }
            .fx-main-area { flex: 1; position: relative; overflow: hidden; min-width: 60%; }
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
            .fx-compare-root { display: flex; flex-direction: column; width: 100%; height: 100%; overflow: hidden; }
            .fx-compare-taskbar { flex: 0 0 auto; display: flex; align-items: center; padding: 6px 10px; gap: 8px; border-bottom: 1px solid #ccc; background: rgba(255,255,255,0.95); flex-wrap: wrap; }
            .fx-compare-grid { flex: 1; min-height: 0; display: grid; gap: 8px; padding: 8px; }
            .fx-compare-col { display: flex; flex-direction: column; min-height: 0; overflow: hidden; border: 1px solid #e5e7eb; border-radius: 6px; }
            .fx-compare-col-header { flex: 0 0 auto; padding: 4px 8px; font-size: 12px; font-weight: 600; background: rgba(0,0,0,0.03); border-bottom: 1px solid #e5e7eb; }
            .fx-compare-minimap-row { flex: 0 0 auto; overflow: hidden; border-bottom: 1px solid #e5e7eb; }
            .fx-compare-minimap-row .fx-minimap-container { width: 100%; height: 100%; border-top: none; }
            .fx-compare-canvas-row { flex: 1; min-height: 0; position: relative; overflow: hidden; }
            .fx-compare-canvas-row .fx-main-area { min-width: 0; }
            .fx-compare-info-bar { flex: 0 0 auto; overflow-y: auto; border-top: 1px solid #ccc; font-size: 13px; max-height: 220px; background: #fff; }
            .fx-compare-info-grid { display: grid; padding: 4px 8px; gap: 0 8px; font-size: 12px; }
            .fx-compare-info-hdr { font-weight: 600; padding: 3px 6px; border-bottom: 2px solid #ccc; background: rgba(0,0,0,0.04); position: sticky; top: 0; }
            .fx-compare-info-prop { font-weight: 600; padding: 3px 6px; border-bottom: 1px solid #eee; background: rgba(0,0,0,0.02); }
            .fx-compare-info-val { padding: 3px 6px; border-bottom: 1px solid #eee; }
            .fx-compare-info-diff { background: rgba(255,200,0,0.2); }
            .fx-compare-info-placeholder { color: #888; text-align: center; padding: 12px; }
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
        fxOn(this._teardownFns, this.resizer, 'mousedown', onResizerMouseDown);

        if (this.resizerH) {
            const onResizerHMouseDown = (e) => {
                if (this.config.layout?.panels?.minimap?.resizable === false) return;
                isResizingH = true;
                this.resizerH.classList.add('dragging');
                document.body.style.cursor = 'row-resize';
                e.preventDefault();
            };
            fxOn(this._teardownFns, this.resizerH, 'mousedown', onResizerHMouseDown);
        }

        const onWindowMouseMove = (e) => {
            if (isResizing) {
                const containerRect = this.wrapper.getBoundingClientRect();
                let newWidth = containerRect.right - e.clientX;
                newWidth = Math.max(150, Math.min(newWidth, containerRect.width * 0.4));
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
        fxOn(this._teardownFns, window, 'mousemove', onWindowMouseMove);

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
        fxOn(this._teardownFns, window, 'mouseup', onWindowMouseUp);

        const onResizerDblClick = () => {
            if (this.config.layout?.panels?.sidebar?.collapsible === false) return;
            this.sidebar.classList.toggle('collapsed');
            requestAnimationFrame(() => {
                this.canvasRenderer.resize();
                this.renderAll();
            });
        };
        fxOn(this._teardownFns, this.resizer, 'dblclick', onResizerDblClick);
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
        fxOffAll(this._teardownFns);
        if (this.wrapper && this.wrapper.parentNode) {
            this.wrapper.parentNode.removeChild(this.wrapper);
        }
    }
}

class FXCompareTaskbar {
    constructor(root, compareInstance, controls = {}) {
        this._root = root;
        this.compare = compareInstance;
        this._teardownFns = [];

        this.el = document.createElement('div');
        this.el.className = 'fx-compare-taskbar';
        root.insertBefore(this.el, root.firstChild);

        this._build(controls);
    }

    _build(controls) {
        const el = this.el;

        if (controls.theme !== false) {
            const sel = document.createElement('select');
            sel.className = 'fx-select';
            sel.innerHTML = `<option value="light">&#x2600; Light</option><option value="dark">&#x1F319; Dark</option>`;
            sel.onchange = (e) => this.compare.viewers.forEach((v) => v.setTheme(e.target.value));
            el.appendChild(sel);
            this._themeSelect = sel;
        }

        if (controls.layers !== false) {
            this._layersBtn = document.createElement('button');
            this._layersBtn.className = 'fx-button';
            this._layersBtn.title = 'Layers / Color By';
            this._layersBtn.textContent = 'Layers';
            this._layersMenu = document.createElement('div');
            this._layersMenu.className = 'fx-layers-menu';
            this._layersMenu.style.display = 'none';
            this._layersBtn.onclick = () => {
                this._rebuildLayersMenu();
                this._layersMenu.style.display = this._layersMenu.style.display === 'none' ? 'block' : 'none';
            };
            const wrap = document.createElement('div');
            wrap.style.position = 'relative';
            wrap.appendChild(this._layersBtn);
            wrap.appendChild(this._layersMenu);
            el.appendChild(wrap);
            fxOn(this._teardownFns, document, 'click', (e) => {
                if (!wrap.contains(e.target)) this._layersMenu.style.display = 'none';
            });
        }

        if (controls.zoomFit !== false) {
            const btn = document.createElement('button');
            btn.className = 'fx-button';
            btn.innerHTML = '&#x2922;';
            btn.title = 'Zoom to Fit All';
            btn.onclick = () => this.compare.viewers.forEach((v) => v.controller.zoomToFit());
            el.appendChild(btn);
        }

        if (controls.syncMode !== false) {
            const sel = document.createElement('select');
            sel.className = 'fx-select';
            this._syncSelect = sel;
            this._rebuildSyncSelect();
            sel.onchange = (e) => {
                const val = e.target.value;
                if (val === 'none') {
                    this.compare.setSync({ mode: 'none' });
                } else if (val === 'id') {
                    this.compare.setSync({ mode: 'id' });
                } else {
                    const [layer, field] = val.split('::');
                    this.compare.setSync({ mode: 'layer', layer, field });
                }
            };
            el.appendChild(sel);
        }

        if (controls.fullscreen !== false) {
            const btn = document.createElement('button');
            btn.className = 'fx-button';
            btn.innerHTML = '&#x26F6;';
            btn.title = 'Fullscreen';
            btn.onclick = () => {
                if (document.fullscreenElement) {
                    document.exitFullscreen();
                } else {
                    this._root.requestFullscreen && this._root.requestFullscreen();
                }
            };
            el.appendChild(btn);
        }
    }

    _rebuildLayersMenu() {
        if (!this._layersMenu) return;
        const allLayers = new Map();
        this.compare.viewers.forEach((v) => {
            Object.entries(v.store.extensions || {}).forEach(([id, ext]) => {
                if (!allLayers.has(id)) allLayers.set(id, ext.name || id);
            });
        });
        let html = '<div style="padding:5px;font-weight:bold;border-bottom:1px solid #ccc;">Extensions</div>';
        allLayers.forEach((name, id) => {
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="checkbox" value="${id}"> ${name}</label>`;
        });
        html += '<div style="padding:5px;font-weight:bold;border-top:1px solid #ccc;margin-top:4px;">Color By</div>';
        html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="base"> Base</label>`;
        allLayers.forEach((name, id) => {
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="${id}"> ${name}</label>`;
        });
        this._layersMenu.innerHTML = html;
        this._layersMenu.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
            cb.onchange = (e) => {
                this.compare.viewers.forEach((v) => {
                    const active = new Set(v.controller.state.activeExtensions);
                    if (e.target.checked) active.add(e.target.value);
                    else active.delete(e.target.value);
                    v.setLayers([...active]);
                });
            };
        });
        // Sync current colorBy state from first viewer
        const currentColorBy = this.compare.viewers[0]?.controller?.state?.colorBy || 'base';
        const matchingRadio = this._layersMenu.querySelector(`input[type="radio"][name="cmp_colorby"][value="${currentColorBy}"]`);
        if (matchingRadio) matchingRadio.checked = true;
        this._layersMenu.querySelectorAll('input[type="radio"][name="cmp_colorby"]').forEach((rb) => {
            rb.onchange = (e) => {
                if (e.target.checked) {
                    this.compare.viewers.forEach((v) => v.setColorBy(e.target.value));
                }
            };
        });
    }

    _rebuildSyncSelect() {
        if (!this._syncSelect) return;
        let html = '<option value="none">Don\'t sync</option><option value="id">Sync by ID</option>';
        const allFields = new Map();
        this.compare.viewers.forEach((v) => {
            Object.entries(v.store.extensions || {}).forEach(([extId, ext]) => {
                const firstNode = ext.nodes && Object.values(ext.nodes)[0];
                if (firstNode && firstNode.info) {
                    Object.keys(firstNode.info).forEach((field) => {
                        const key = `${extId}::${field}`;
                        if (!allFields.has(key)) allFields.set(key, `Sync by ${extId}.${field}`);
                    });
                }
            });
        });
        allFields.forEach((label, key) => {
            html += `<option value="${key}">${label}</option>`;
        });
        this._syncSelect.innerHTML = html;
        const sync = this.compare.sync;
        if (sync.mode === 'none') this._syncSelect.value = 'none';
        else if (sync.mode === 'layer' && sync.layer && sync.field) this._syncSelect.value = `${sync.layer}::${sync.field}`;
        else this._syncSelect.value = 'id';
    }

    destroy() {
        fxOffAll(this._teardownFns);
        if (this.el && this.el.parentNode) this.el.parentNode.removeChild(this.el);
    }
}

class FXGraphCompare {
    static create(config) {
        return new FXGraphCompare(config);
    }

    constructor(config = {}) {
        this.viewers = Array.isArray(config.viewers) ? config.viewers : [];
        this.sync = {
            mode: 'id',
            layer: '',
            field: '',
            ...(config.sync || {}),
        };
        this.layout = {
            columns: (config.layout && config.layout.columns) || 2,
        };
        this.sharedTaskbar = {
            enabled: !!(config.sharedTaskbar && config.sharedTaskbar.enabled),
            controls: (config.sharedTaskbar && config.sharedTaskbar.controls) || {},
        };
        this._minimapHeight = (config.layout && config.layout.minimapHeight) || 180;
        this._infoHeight = (config.layout && config.layout.infoHeight) || 200;
        this._canvasHeightRatio = (config.layout && config.layout.canvasHeightRatio != null)
            ? config.layout.canvasHeightRatio : 0.7;

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
        this._taskbar = null;
        this._root = null;
        this._grid = null;
        this._infoBar = null;
        this._colResizeObservers = [];
        this._domSnapshots = new WeakMap();

        if (this.container) {
            this._buildCompareDOM();
        }

        this._wireSelectionSync();
        this._wireStateSync();

        // Always suppress per-viewer UI in compare mode
        this.viewers.forEach((v) => {
            const keepToolbar = this.sharedTaskbar.enabled;
            v.setUIVisibility({
                toolbar: keepToolbar,
                search: keepToolbar,
                layers: false,
                theme: false,
                zoomButtons: false,
                fullscreenButton: false,
                highlightButton: false,
            });
            if (keepToolbar && v.ui && v.ui.searchContainer) {
                v.ui.searchContainer.style.flex = '1';
                v.ui.searchContainer.style.maxWidth = 'none';
            }
        });
        if (this.sharedTaskbar.enabled && this._root) {
            this._taskbar = new FXCompareTaskbar(this._root, this, this.sharedTaskbar.controls);
        }
    }

    _buildCompareDOM() {
        // Create root flex-column container inside user's container
        this._root = document.createElement('div');
        this._root.className = 'fx-compare-root';
        this.container.appendChild(this._root);

        // Ensure root has a defined height so the flex chain doesn't collapse.
        // If the container has an explicit height, use 100%; otherwise fall back to 90vh.
        const _syncRootHeight = () => {
            if (this.container.offsetHeight >= 100) {
                this._root.style.height = '100%';
            } else {
                this._root.style.height = Math.round(window.innerHeight * 0.9) + 'px';
            }
        };
        _syncRootHeight();
        if (typeof ResizeObserver !== 'undefined') {
            const ro = new ResizeObserver(_syncRootHeight);
            ro.observe(this.container);
            this._colResizeObservers.push(ro);
        }

        // Create the grid (columns of viewers)
        this._grid = document.createElement('div');
        this._grid.className = 'fx-compare-grid';
        this._grid.style.gridTemplateColumns = `repeat(${this.layout.columns}, 1fr)`;
        this._root.appendChild(this._grid);

        // For each viewer, create a column and move its mainArea + minimap into it
        this.viewers.forEach((viewer, i) => {
            // Snapshot original DOM positions for teardown
            this._domSnapshots.set(viewer, {
                mainAreaParent: viewer.mainArea.parentNode,
                mainAreaNextSibling: viewer.mainArea.nextSibling,
                minimapParent: viewer.minimapRenderer ? viewer.minimapRenderer.container.parentNode : null,
                minimapNextSibling: viewer.minimapRenderer ? viewer.minimapRenderer.container.nextSibling : null,
                wrapperDisplay: viewer.wrapper.style.display,
            });

            const col = document.createElement('div');
            col.className = 'fx-compare-col';

            // Optional column header (viewer title)
            const title = (viewer.config && viewer.config.title) || `Graph ${i + 1}`;
            const header = document.createElement('div');
            header.className = 'fx-compare-col-header';
            header.textContent = title;
            col.appendChild(header);

            // Minimap row
            if (viewer.minimapRenderer) {
                const minimapRow = document.createElement('div');
                minimapRow.className = 'fx-compare-minimap-row';
                minimapRow.style.height = `${this._minimapHeight}px`;
                minimapRow.appendChild(viewer.minimapRenderer.container);
                col.appendChild(minimapRow);
            }

            // Canvas row — move mainArea here
            const canvasRow = document.createElement('div');
            canvasRow.className = 'fx-compare-canvas-row';
            canvasRow.appendChild(viewer.mainArea);
            col.appendChild(canvasRow);

            // Hide the viewer's own wrapper shell (sidebar, resizer, etc.)
            viewer.wrapper.style.display = 'none';

            this._grid.appendChild(col);

            // ResizeObserver on canvas row to keep canvas sized correctly
            if (typeof ResizeObserver !== 'undefined') {
                const ro = new ResizeObserver(() => {
                    viewer.canvasRenderer.resize();
                    viewer.renderAll();
                });
                ro.observe(canvasRow);
                this._colResizeObservers.push(ro);
            }
        });

        // Shared info bar at the bottom
        this._infoBar = document.createElement('div');
        this._infoBar.className = 'fx-compare-info-bar';
        this._infoBar.innerHTML = '<div class="fx-compare-info-placeholder">No node selected — click a node to compare</div>';
        this._root.appendChild(this._infoBar);

        // Resize all viewers after DOM is settled
        requestAnimationFrame(() => {
            this.viewers.forEach((v) => {
                v.canvasRenderer.resize();
                if (v.minimapRenderer) {
                    v.minimapRenderer.resize();
                    v.minimapRenderer.generateThumbnail();
                }
                v.renderAll();
            });
        });
    }

    _teardownCompareDOM() {
        // Disconnect resize observers
        this._colResizeObservers.forEach((ro) => ro.disconnect());
        this._colResizeObservers = [];

        // Restore each viewer's DOM
        this.viewers.forEach((viewer) => {
            const snap = this._domSnapshots.get(viewer);
            if (!snap) return;

            // Restore mainArea
            if (snap.mainAreaParent) {
                if (snap.mainAreaNextSibling) {
                    snap.mainAreaParent.insertBefore(viewer.mainArea, snap.mainAreaNextSibling);
                } else {
                    snap.mainAreaParent.appendChild(viewer.mainArea);
                }
            }

            // Restore minimap
            if (viewer.minimapRenderer && snap.minimapParent) {
                if (snap.minimapNextSibling) {
                    snap.minimapParent.insertBefore(viewer.minimapRenderer.container, snap.minimapNextSibling);
                } else {
                    snap.minimapParent.appendChild(viewer.minimapRenderer.container);
                }
            }

            // Restore wrapper visibility
            viewer.wrapper.style.display = snap.wrapperDisplay;
            this._domSnapshots.delete(viewer);
        });

        // Remove compare root
        if (this._root && this._root.parentNode) {
            this._root.parentNode.removeChild(this._root);
        }
        this._root = null;
        this._grid = null;
        this._infoBar = null;
    }

    _wireSelectionSync() {
        this.viewers.forEach((viewer) => {
            const off = viewer.on('selectionchange', (evt) => {
                if (this._guards.has(viewer)) return;
                if (!evt.nextSelection) {
                    this.viewers.forEach((other) => {
                        if (other === viewer) return;
                        this._applyGuarded(other, () => other.clearSelection());
                    });
                    this._updateMergedInfo(null);
                    return;
                }

                const nodeIdMap = new Map([[viewer, evt.nextSelection]]);

                this.viewers.forEach((other) => {
                    if (other === viewer) return;
                    const targetId = this._findSyncTarget(viewer, evt.nextSelection, other);
                    if (targetId) {
                        nodeIdMap.set(other, targetId);
                        this._applyGuarded(other, () => {
                            other.selectNode(targetId, { animate: true, center: true });
                        });
                    } else {
                        this._applyGuarded(other, () => other.clearSelection());
                    }
                });

                this._updateMergedInfo(nodeIdMap);
            });
            this._offs.push(off);
        });
    }

    _findSyncTarget(sourceViewer, nodeId, targetViewer) {
        const mode = this.sync.mode;
        if (mode === 'none') return null;
        if (mode === 'id') {
            return targetViewer.store.activeNodeMap.has(nodeId) ? nodeId : null;
        }
        if (mode === 'layer') {
            const { layer, field } = this.sync;
            const srcVal = sourceViewer.store.extensions[layer]?.nodes[nodeId]?.info[field];
            if (srcVal === undefined) return null;
            const candidates = targetViewer.store.activeNodes.filter(
                (n) => targetViewer.store.extensions[layer]?.nodes[n.id]?.info[field] === srcVal
            );
            if (candidates.length === 0) return null;
            if (candidates.length === 1) return candidates[0].id;
            const withTopo = candidates.filter((n) => targetViewer.store.extensions[layer]?.nodes[n.id]?.info?.topo_index !== undefined);
            if (withTopo.length > 0) {
                return withTopo.reduce((best, n) => {
                    const bv = Number(targetViewer.store.extensions[layer]?.nodes[best.id]?.info?.topo_index);
                    const nv = Number(targetViewer.store.extensions[layer]?.nodes[n.id]?.info?.topo_index);
                    return nv > bv ? n : best;
                }).id;
            }
            return candidates[candidates.length - 1].id;
        }
        return null;
    }

    _updateMergedInfo(nodeIdMap) {
        if (!this._infoBar) return;
        if (!nodeIdMap) {
            this._infoBar.innerHTML = '<div class="fx-compare-info-placeholder">No node selected — click a node to compare</div>';
            return;
        }
        const N = this.viewers.length;
        const viewerNames = this.viewers.map((v, i) => (v.config && v.config.title) || `Graph ${i + 1}`);
        const allProps = new Set();
        const rowData = new Map();
        nodeIdMap.forEach((nid, v) => {
            const node = v.store.activeNodeMap.get(nid);
            if (!node) return;
            const props = { id: nid, op: node.op || '', ...(node.info || {}) };
            Object.keys(props).forEach((k) => allProps.add(k));
            rowData.set(v, props);
        });
        let html = `<div class="fx-compare-info-grid" style="grid-template-columns: auto repeat(${N}, 1fr)">`;
        // Header row
        html += `<div class="fx-compare-info-hdr">Property</div>`;
        viewerNames.forEach((name) => { html += `<div class="fx-compare-info-hdr">${name}</div>`; });
        // Data rows
        allProps.forEach((prop) => {
            const vals = this.viewers.map((v) => {
                const d = rowData.get(v);
                return d ? (d[prop] !== undefined ? String(d[prop]) : '—') : '—';
            });
            const allSame = vals.every((v) => v === vals[0]);
            html += `<div class="fx-compare-info-prop">${prop}</div>`;
            vals.forEach((val) => { html += `<div class="fx-compare-info-val${allSame ? '' : ' fx-compare-info-diff'}">${val}</div>`; });
        });
        html += `</div>`;
        this._infoBar.innerHTML = html;
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
                        if (prev.theme !== next.theme && typeof next.theme === 'string') {
                            other.setTheme(next.theme);
                        }
                    });
                });
            });
            this._offs.push(off);
        });
    }

    setColumns(columns) {
        this.layout.columns = Math.max(1, Number(columns) || 1);
        if (this._grid) {
            this._grid.style.gridTemplateColumns = `repeat(${this.layout.columns}, 1fr)`;
        }
        requestAnimationFrame(() => {
            this.viewers.forEach((v) => {
                v.canvasRenderer.resize();
                if (v.minimapRenderer) {
                    v.minimapRenderer.resize();
                    v.minimapRenderer.generateThumbnail();
                }
                v.renderAll();
            });
        });
    }

    setSync(syncPatch = {}) {
        this.sync = { ...this.sync, ...syncPatch };
        if (this._taskbar) this._taskbar._rebuildSyncSelect();
    }

    /** @deprecated No-op — tiled layout is always used in compare mode */
    setTiled() {}

    /** @deprecated No-op — use sharedTaskbar config instead */
    setCompact() {}

    _applyGuarded(viewer, fn) {
        this._guards.add(viewer);
        try {
            fn();
        } finally {
            setTimeout(() => this._guards.delete(viewer), 0);
        }
    }

    destroy() {
        if (this._taskbar) {
            this._taskbar.destroy();
            this._taskbar = null;
        }
        this._teardownCompareDOM();
        this._offs.forEach((off) => {
            try { off(); } catch (_) {}
        });
        this._offs = [];
    }
}

if (typeof globalThis !== 'undefined') {
    globalThis.FXGraphViewer = FXGraphViewer;
    globalThis.FXGraphCompare = FXGraphCompare;
}
