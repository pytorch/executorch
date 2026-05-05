// FXGraphCompare: orchestrates multi-view compare DOM, selection sync, and lifecycle.
// Layout: 3-row × (N+1)-col CSS grid.
//   Col 0 (sidebar): shared controls spanning rows 0-1.
//   Cols 1..N: per-graph minimap (row 0) and canvas (row 1).
//   Row 2: info row using CSS subgrid for aligned property columns.
//
// Selection sync modes (config.sync.mode):
//   'auto'  (default) — tries from_node_root first, then debug_handle set-intersection, falls back to node-ID match.
//   'id'              — matches by node id only.
//   'layer'           — matches by extensions[layer].nodes[nodeId].info[field] value equality;
//                       picks last in topological order on multiple matches.
//   'none'            — no cross-viewer selection propagation.
//
// debug_handle normalization (_normalizeHandle):
//   int dh      → Set{dh}  (if non-zero)
//   int[] dh    → Set(dh.filter(x => x !== 0))
//   null/0/[]   → empty Set (no match)
// Two nodes match if their normalized sets have a non-empty intersection.
//
// Sidebar sync selector options (rebuilt by _rebuildSyncPanel):
//   "Auto (from_node→handle→id)"  → mode: 'auto'
//   "ID only"                     → mode: 'id'
//   "Don't sync"                  → mode: 'none'
//   "Ext: <extId>.<field>"        → mode: 'layer' (one per registered sync_key)

class FXGraphCompare {
    static create(config) {
        return new FXGraphCompare(config);
    }

    constructor(config = {}) {
        // Accept Map<name, viewer> or Array<viewer> (backward compat)
        if (config.viewers instanceof Map) {
            this._viewerMap = config.viewers;
        } else if (Array.isArray(config.viewers)) {
            this._viewerMap = new Map(config.viewers.map((v, i) =>
                [(v.config && v.config.title) || `Graph ${i + 1}`, v]
            ));
        } else {
            this._viewerMap = new Map();
        }
        this.viewers = [...this._viewerMap.values()];
        this._viewerNames = [...this._viewerMap.keys()];
        this._visibleViewers = new Set(this._viewerNames);

        this.sync = {
            mode: 'auto',
            layer: '',
            field: '',
            ...(config.sync || {}),
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
        this._root = null;
        this._grid = null;
        this._infoRow = null;
        this._minimapCells = [];
        this._nameCells = [];
        this._canvasCells = [];
        this._colResizeObservers = [];
        this._domSnapshots = new WeakMap();
        this._followSelection = true;
        this._openPortalMenus = [];
        this._currentTheme = this.viewers[0]?.controller?.state?.themeName || 'light';
        this._layoutRefreshQueued = false;
        this._layoutRefreshAgain = false;
        this._pendingRefreshAgainResetView = false;
        this._needsResetOnNextVisibleLayout = false;

        if (this.container) {
            this._buildCompareDOM();
        }

        this._wireSelectionSync();
        this._wireStateSync();

        // Suppress per-viewer UI in compare mode (keep taskbar + search only)
        this.viewers.forEach((v) => {
            v.setUIVisibility({
                toolbar: true,
                search: true,
                layers: false,
                theme: false,
                zoomButtons: false,
                fullscreenButton: false,
                highlightButton: false,
            });
        });
    }

    _buildCompareDOM() {
        this._root = document.createElement('div');
        this._root.className = 'fx-compare-root';
        // Fallback height if container has no defined height
        if (this.container.offsetHeight < 100) {
            this._root.style.height = Math.round(window.innerHeight * 0.85) + 'px';
        }
        this.container.appendChild(this._root);

        const N = this.viewers.length;

        // Main grid: col 0 = 160px sidebar, cols 1..N = 1fr each
        this._grid = document.createElement('div');
        this._grid.className = 'fx-compare-grid';
        this._grid.style.gridTemplateColumns = `160px repeat(${N}, 1fr)`;
        this._root.appendChild(this._grid);

        // Sidebar cell (col 1, rows 1-3)
        const sidebar = document.createElement('div');
        sidebar.className = 'fx-compare-sidebar-cell';
        this._grid.appendChild(sidebar);
        this._buildSidebar(sidebar);
        this._sidebarEl = sidebar;

        // Per-viewer cells
        this._minimapCells = [];
        this._nameCells = [];
        this._canvasCells = [];

        this.viewers.forEach((viewer, i) => {
            // Snapshot original DOM positions for teardown
            this._domSnapshots.set(viewer, {
                mainAreaParent: viewer.mainArea.parentNode,
                mainAreaNextSibling: viewer.mainArea.nextSibling,
                minimapParent: viewer.minimapRenderer ? viewer.minimapRenderer.container.parentNode : null,
                minimapNextSibling: viewer.minimapRenderer ? viewer.minimapRenderer.container.nextSibling : null,
                wrapperDisplay: viewer.wrapper.style.display,
            });

            // Minimap cell: col i+2, row 1
            const minimapCell = document.createElement('div');
            minimapCell.className = 'fx-compare-minimap-cell';
            minimapCell.style.gridColumn = String(i + 2);
            minimapCell.style.gridRow = '1';
            if (viewer.minimapRenderer) {
                minimapCell.appendChild(viewer.minimapRenderer.container);
            }
            this._grid.appendChild(minimapCell);
            this._minimapCells.push(minimapCell);

            // Name cell: col i+2, row 2
            const nameCell = document.createElement('div');
            nameCell.className = 'fx-compare-name-cell';
            nameCell.style.gridColumn = String(i + 2);
            nameCell.textContent = this._viewerNames[i];
            this._grid.appendChild(nameCell);
            this._nameCells.push(nameCell);

            // Canvas cell: col i+2, row 3
            const canvasCell = document.createElement('div');
            canvasCell.className = 'fx-compare-canvas-cell';
            canvasCell.style.gridColumn = String(i + 2);
            canvasCell.style.gridRow = '3';
            canvasCell.appendChild(viewer.mainArea);
            this._grid.appendChild(canvasCell);
            this._canvasCells.push(canvasCell);

            // Hide viewer's own wrapper shell
            viewer.wrapper.style.display = 'none';

            // ResizeObserver on canvas cell
            if (typeof ResizeObserver !== 'undefined') {
                const ro = new ResizeObserver(() => this._scheduleLayoutRefresh());
                ro.observe(canvasCell);
                if (viewer.minimapRenderer) {
                    ro.observe(minimapCell);
                }
                this._colResizeObservers.push(ro);
            }
        });

        // Info row (col 1..-1, row 4) — uses CSS subgrid
        this._infoRow = document.createElement('div');
        this._infoRow.className = 'fx-compare-info-row';
        this._grid.appendChild(this._infoRow);
        this._updateMergedInfo(null);

        this._applyCompareTheme(this.viewers[0]?.controller?.state?.themeName || 'light');

        this._scheduleLayoutRefresh({ resetView: true });
    }

    _scheduleLayoutRefresh(options = {}) {
        if (!this._root) return;
        if (this._layoutRefreshQueued) {
            this._layoutRefreshAgain = true;
            this._pendingRefreshAgainResetView = this._pendingRefreshAgainResetView || !!options.resetView;
            return;
        }
        this._pendingRefreshResetView = this._pendingRefreshResetView || !!options.resetView;
        this._layoutRefreshQueued = true;
        requestAnimationFrame(() => {
            requestAnimationFrame(() => {
                this._layoutRefreshQueued = false;
                if (!this._root) return;
                const resetView = !!this._pendingRefreshResetView || this._needsResetOnNextVisibleLayout;
                this._pendingRefreshResetView = false;
                this._refreshViewerLayout({ resetView });
                if (this._layoutRefreshAgain) {
                    const resetAgain = this._pendingRefreshAgainResetView;
                    this._layoutRefreshAgain = false;
                    this._pendingRefreshAgainResetView = false;
                    this._scheduleLayoutRefresh({ resetView: resetAgain });
                }
            });
        });
    }

    _refreshViewerLayout(options = {}) {
        const resetView = !!options.resetView;
        let sawInvalidLayout = false;
        this.viewers.forEach((viewer, i) => {
            if (!this._visibleViewers.has(this._viewerNames[i])) return;
            if (viewer.canvasRenderer && typeof viewer.canvasRenderer.resetInteractionState === 'function') {
                viewer.canvasRenderer.resetInteractionState();
            }
            if (viewer.minimapRenderer && typeof viewer.minimapRenderer.resetInteractionState === 'function') {
                viewer.minimapRenderer.resetInteractionState();
            }

            const canvasRect = viewer.canvasContainer && viewer.canvasContainer.getBoundingClientRect();
            const minimapRect = viewer.minimapRenderer && viewer.minimapRenderer.container.getBoundingClientRect();
            const hasCanvasLayout = canvasRect && canvasRect.width > 0 && canvasRect.height > 0;
            const hasMinimapLayout = !viewer.minimapRenderer || (minimapRect && minimapRect.width > 0 && minimapRect.height > 0);
            if (!hasCanvasLayout || !hasMinimapLayout) {
                sawInvalidLayout = true;
                return;
            }

            if (viewer.canvasRenderer) viewer.canvasRenderer.resize();
            if (viewer.minimapRenderer) {
                viewer.minimapRenderer.resize();
                viewer.minimapRenderer.generateThumbnail();
            }
            if (resetView) viewer.init();
            else viewer.renderAll();
        });
        this._needsResetOnNextVisibleLayout = sawInvalidLayout;
    }

    _buildSidebar(sidebar) {
        // Layers button + menu (portal pattern — menu appended to document.body when open)
        const layersWrap = document.createElement('div');
        layersWrap.style.position = 'relative';
        const layersBtn = document.createElement('button');
        layersBtn.className = 'fx-button';
        layersBtn.title = 'Layers / Color By';
        layersBtn.textContent = 'Layers';
        layersBtn.style.marginLeft = '0';
        layersBtn.style.boxSizing = 'border-box';
        layersBtn.style.width = '100%';
        const layersMenu = document.createElement('div');
        layersMenu.className = 'fx-compare-portal-menu';
        layersBtn.onclick = () => {
            if (layersMenu.parentNode) {
                this._closePortalMenu(layersMenu);
            } else {
                this._rebuildLayersMenu(layersMenu);
                this._openPortalMenu(layersBtn, layersMenu);
            }
        };
        fxOn(this._offs, document, 'click', (e) => {
            if (!layersWrap.contains(e.target) && !layersMenu.contains(e.target)) {
                this._closePortalMenu(layersMenu);
            }
        });
        layersWrap.appendChild(layersBtn);
        sidebar.appendChild(layersWrap);

        // Theme selector
        const themeSel = document.createElement('select');
        themeSel.className = 'fx-select';
        themeSel.style.marginLeft = '0';
        themeSel.style.boxSizing = 'border-box';
        themeSel.style.width = '100%';
        themeSel.innerHTML = `<option value="light">&#x2600; Light</option><option value="dark">&#x1F319; Dark</option>`;
        themeSel.onchange = (e) => this.viewers.forEach((v) => v.setTheme(e.target.value));
        sidebar.appendChild(themeSel);
        this._themeSelect = themeSel;

        // Zoom-fit button
        const zoomBtn = document.createElement('button');
        zoomBtn.className = 'fx-button';
        zoomBtn.style.marginLeft = '0';
        zoomBtn.innerHTML = '&#x2922; Fit';
        zoomBtn.title = 'Zoom to Fit All';
        zoomBtn.style.boxSizing = 'border-box';
        zoomBtn.style.width = '100%';
        zoomBtn.onclick = () => this.viewers.forEach((v) => v.controller.zoomToFit());
        sidebar.appendChild(zoomBtn);

        // Fullscreen button
        const fsBtn = document.createElement('button');
        fsBtn.className = 'fx-button';
        fsBtn.style.marginLeft = '0';
        fsBtn.innerHTML = '&#x26F6; Full';
        fsBtn.title = 'Fullscreen';
        fsBtn.style.boxSizing = 'border-box';
        fsBtn.style.width = '100%';
        fsBtn.onclick = () => {
            if (document.fullscreenElement) {
                document.exitFullscreen();
            } else {
                this._root.requestFullscreen && this._root.requestFullscreen();
            }
        };
        fxOn(this._offs, document, 'fullscreenchange', () => {
            fsBtn.innerHTML = document.fullscreenElement ? '&#x2715; Exit' : '&#x26F6; Full';
            fsBtn.title = document.fullscreenElement ? 'Exit Fullscreen' : 'Fullscreen';
        });
        sidebar.appendChild(fsBtn);

        // Sync mode selector (only registered sync_keys)
        const syncSel = document.createElement('select');
        syncSel.className = 'fx-select';
        syncSel.style.marginLeft = '0';
        syncSel.style.boxSizing = 'border-box';
        syncSel.style.width = '100%';
        this._syncSelect = syncSel;
        this._rebuildSyncPanel();
        syncSel.onchange = (e) => {
            const val = e.target.value;
            if (val === 'none') { this.setSync({ mode: 'none' }); }
            else if (val === 'id') { this.setSync({ mode: 'id' }); }
            else if (val === 'auto') { this.setSync({ mode: 'auto' }); }
            else { const [layer, field] = val.split('::'); this.setSync({ mode: 'layer', layer, field }); }
        };
        sidebar.appendChild(syncSel);

        // Visible graphs toggle (portal pattern)
        const visWrap = document.createElement('div');
        visWrap.style.position = 'relative';
        const visBtn = document.createElement('button');
        visBtn.className = 'fx-button';
        visBtn.style.marginLeft = '0';
        visBtn.style.boxSizing = 'border-box';
        visBtn.style.width = '100%';
        visBtn.title = 'Toggle visible graphs';
        visBtn.textContent = 'Graphs';
        const visMenu = document.createElement('div');
        visMenu.className = 'fx-compare-portal-menu';
        visBtn.onclick = () => {
            if (visMenu.parentNode) {
                this._closePortalMenu(visMenu);
            } else {
                this._rebuildVisMenu(visMenu);
                this._openPortalMenu(visBtn, visMenu);
            }
        };
        fxOn(this._offs, document, 'click', (e) => {
            if (!visWrap.contains(e.target) && !visMenu.contains(e.target)) {
                this._closePortalMenu(visMenu);
            }
        });
        visWrap.appendChild(visBtn);
        sidebar.appendChild(visWrap);

        // Follow selection toggle
        const followBtn = document.createElement('button');
        followBtn.className = 'fx-button';
        followBtn.style.marginLeft = '0';
        followBtn.style.boxSizing = 'border-box';
        followBtn.style.width = '100%';
        followBtn.title = 'Toggle: zoom-to-fit on selection sync';
        const updateFollowBtn = () => {
            followBtn.textContent = this._followSelection ? '\u2299 Zoom-Fit' : '\u25cb Zoom-Fit';
            followBtn.style.opacity = this._followSelection ? '1' : '0.6';
        };
        updateFollowBtn();
        followBtn.onclick = () => {
            this._followSelection = !this._followSelection;
            updateFollowBtn();
        };
        sidebar.appendChild(followBtn);

        // Highlight ancestors toggle
        const hlBtn = document.createElement('button');
        hlBtn.className = 'fx-button';
        hlBtn.style.marginLeft = '0';
        hlBtn.style.boxSizing = 'border-box';
        hlBtn.style.width = '100%';
        hlBtn.innerHTML = '&#x1F517;';
        hlBtn.title = 'Toggle Highlight Ancestors/Descendants';
        const updateHlBtn = () => {
            const on = this.viewers[0]?.controller?.state?.highlightAncestors !== false;
            hlBtn.style.opacity = on ? '1' : '0.6';
        };
        updateHlBtn();
        hlBtn.onclick = () => {
            const on = this.viewers[0]?.controller?.state?.highlightAncestors !== false;
            this.viewers.forEach((v) => {
                v.controller.state.highlightAncestors = !on;
                v.controller.setState({});
            });
            updateHlBtn();
        };
        sidebar.appendChild(hlBtn);
        this._hlBtn = hlBtn;
    }

    _rebuildLayersMenu(menu) {
        const allLayers = new Map();
        this.viewers.forEach((v) => {
            Object.entries(v.store.extensions || {}).forEach(([id, ext]) => {
                if (!allLayers.has(id)) allLayers.set(id, ext.name || id);
            });
        });
        let html = '<div style="padding:5px;font-weight:bold;border-bottom:1px solid #ccc;">Extensions</div>';
        allLayers.forEach((name, id) => {
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="checkbox" value="${fxEsc(id)}"> ${fxEsc(name)}</label>`;
        });
        html += '<div style="padding:5px;font-weight:bold;border-top:1px solid #ccc;margin-top:4px;">Color By</div>';
        html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="base"> Base</label>`;
        allLayers.forEach((name, id) => {
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="${fxEsc(id)}"> ${fxEsc(name)}</label>`;
        });
        menu.innerHTML = html;
        menu.querySelectorAll('input[type="checkbox"]').forEach((cb) => {
            cb.onchange = (e) => {
                this.viewers.forEach((v) => {
                    const active = new Set(v.controller.state.activeExtensions);
                    if (e.target.checked) active.add(e.target.value);
                    else active.delete(e.target.value);
                    v.setLayers([...active]);
                });
            };
        });
        const currentColorBy = this.viewers[0]?.controller?.state?.colorBy || 'base';
        const matchingRadio = menu.querySelector(`input[type="radio"][name="cmp_colorby"][value="${currentColorBy}"]`);
        if (matchingRadio) matchingRadio.checked = true;
        menu.querySelectorAll('input[type="radio"][name="cmp_colorby"]').forEach((rb) => {
            rb.onchange = (e) => {
                if (e.target.checked) this.viewers.forEach((v) => v.setColorBy(e.target.value));
            };
        });
    }

    _rebuildSyncPanel() {
        if (!this._syncSelect) return;
        let html = '<option value="auto">Auto (from_node&#x2192;handle&#x2192;id)</option>'
                 + '<option value="id">ID only</option>'
                 + '<option value="none">Don\'t sync</option>';
        const seen = new Set(['auto', 'id', 'none']);
        this.viewers.forEach((v) => {
            Object.entries(v.store.extensions || {}).forEach(([extId, ext]) => {
                (ext.sync_keys || []).forEach((field) => {
                    const key = `${extId}::${field}`;
                    if (!seen.has(key)) {
                        seen.add(key);
                        html += `<option value="${fxEsc(key)}">Ext: ${fxEsc(extId)}.${fxEsc(field)}</option>`;
                    }
                });
            });
        });
        this._syncSelect.innerHTML = html;
        const sync = this.sync;
        if (sync.mode === 'none') this._syncSelect.value = 'none';
        else if (sync.mode === 'id') this._syncSelect.value = 'id';
        else if (sync.mode === 'layer' && sync.layer && sync.field) this._syncSelect.value = `${sync.layer}::${sync.field}`;
        else this._syncSelect.value = 'auto';
    }

    _rebuildVisMenu(menu) {
        menu.innerHTML = '';
        this._viewerNames.forEach((name) => {
            const label = document.createElement('label');
            label.style.cssText = 'display:block;padding:5px;cursor:pointer;';
            const cb = document.createElement('input');
            cb.type = 'checkbox';
            cb.checked = this._visibleViewers.has(name);
            cb.onchange = (e) => this._setViewerVisible(name, e.target.checked);
            label.appendChild(cb);
            label.appendChild(document.createTextNode(' ' + name));
            menu.appendChild(label);
        });
    }

    _openPortalMenu(btn, menu) {
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        const btnRect = btn.getBoundingClientRect();
        const rootRect = this._root.getBoundingClientRect();
        const scrollTop = this._root.scrollTop;
        const scrollLeft = this._root.scrollLeft;
        const top = btnRect.top - rootRect.top + scrollTop;
        const left = btnRect.right - rootRect.left + scrollLeft + 4;
        menu.style.top = top + 'px';
        menu.style.left = left + 'px';
        menu.style.maxHeight = Math.min(window.innerHeight - btnRect.top - 8, window.innerHeight * 0.6) + 'px';
        this._root.appendChild(menu);
        this._openPortalMenus.push(menu);
        const theme = (typeof THEMES !== 'undefined' && THEMES[this._currentTheme]) || THEMES?.light;
        if (theme) {
            menu.style.backgroundColor = theme.uiBg;
            menu.style.color = theme.text;
            menu.style.borderColor = theme.uiBorder;
            menu.querySelectorAll('label, .fx-button, .fx-select').forEach((el) => {
                el.style.color = theme.text;
            });
        }
    }

    _closePortalMenu(menu) {
        if (menu.parentNode) menu.parentNode.removeChild(menu);
        const idx = this._openPortalMenus.indexOf(menu);
        if (idx !== -1) this._openPortalMenus.splice(idx, 1);
    }

    _setViewerVisible(name, visible) {
        if (visible) this._visibleViewers.add(name);
        else this._visibleViewers.delete(name);

        const visCount = [...this._viewerNames].filter((n) => this._visibleViewers.has(n)).length;
        this._grid.style.gridTemplateColumns = `160px repeat(${Math.max(1, visCount)}, 1fr)`;

        let colIdx = 2;
        this.viewers.forEach((v, i) => {
            const isVis = this._visibleViewers.has(this._viewerNames[i]);
            if (this._minimapCells[i]) {
                this._minimapCells[i].style.display = isVis ? '' : 'none';
                if (isVis) this._minimapCells[i].style.gridColumn = String(colIdx);
            }
            if (this._nameCells[i]) {
                this._nameCells[i].style.display = isVis ? '' : 'none';
                if (isVis) this._nameCells[i].style.gridColumn = String(colIdx);
            }
            if (this._canvasCells[i]) {
                this._canvasCells[i].style.display = isVis ? '' : 'none';
                if (isVis) this._canvasCells[i].style.gridColumn = String(colIdx);
            }
            if (isVis) {
                colIdx++;
            }
        });

        this._scheduleLayoutRefresh();

        if (visible) {
            const newViewer = this.viewers[this._viewerNames.indexOf(name)];
            requestAnimationFrame(() => {
                const canvasRect = newViewer.canvasContainer && newViewer.canvasContainer.getBoundingClientRect();
                if (!canvasRect || canvasRect.width <= 0 || canvasRect.height <= 0) {
                    this._needsResetOnNextVisibleLayout = true;
                    return;
                }
                if (newViewer.canvasRenderer) newViewer.canvasRenderer.resize();
                if (newViewer.minimapRenderer) {
                    newViewer.minimapRenderer.resize();
                    newViewer.minimapRenderer.generateThumbnail();
                }
                let srcViewer = null, srcNodeId = null;
                this.viewers.forEach((v, i) => {
                    if (v === newViewer || !this._visibleViewers.has(this._viewerNames[i])) return;
                    const sel = v.controller?.state?.selectedNodeId;
                    if (sel && !srcNodeId) { srcViewer = v; srcNodeId = sel; }
                });
                if (srcViewer && srcNodeId) {
                    const targetId = this._findSyncTarget(srcViewer, srcNodeId, newViewer);
                    if (targetId) {
                        newViewer.selectNode(targetId, { center: false });
                        newViewer.controller.zoomToFit();
                    } else {
                        newViewer.controller.zoomToFit();
                    }
                } else {
                    newViewer.controller.zoomToFit();
                }
            });
        }

        this._updateMergedInfo(null);
    }

    _teardownCompareDOM() {
        // Close any open portal menus
        this._openPortalMenus.slice().forEach((m) => this._closePortalMenu(m));

        this._colResizeObservers.forEach((ro) => ro.disconnect());
        this._colResizeObservers = [];

        this.viewers.forEach((viewer) => {
            const snap = this._domSnapshots.get(viewer);
            if (!snap) return;

            if (snap.mainAreaParent) {
                if (snap.mainAreaNextSibling) {
                    snap.mainAreaParent.insertBefore(viewer.mainArea, snap.mainAreaNextSibling);
                } else {
                    snap.mainAreaParent.appendChild(viewer.mainArea);
                }
            }

            if (viewer.minimapRenderer && snap.minimapParent) {
                if (snap.minimapNextSibling) {
                    snap.minimapParent.insertBefore(viewer.minimapRenderer.container, snap.minimapNextSibling);
                } else {
                    snap.minimapParent.appendChild(viewer.minimapRenderer.container);
                }
            }

            viewer.wrapper.style.display = snap.wrapperDisplay;
            this._domSnapshots.delete(viewer);
        });

        if (this._root && this._root.parentNode) {
            this._root.parentNode.removeChild(this._root);
        }
        this._root = null;
        this._grid = null;
        this._infoRow = null;
        this._minimapCells = [];
        this._nameCells = [];
        this._canvasCells = [];
    }

    _wireSelectionSync() {
        this.viewers.forEach((viewer) => {
            const off = viewer.on('selectionchange', (evt) => {
                if (evt.nextSelection){
                    if (this._followSelection) viewer.controller.zoomToFit();
                } 

                if (this._guards.has(viewer)) return;
                if (!evt.nextSelection) {
                    this.viewers.forEach((other) => {
                        if (other === viewer) return;
                        this._applyGuarded(other, () => other.clearSelection());
                    });
                    this._updateMergedInfo(null);
                    this.viewers.forEach((v) => v.removeHighlightGroup('_sync_candidates'));
                    return;
                }

                const nodeIdMap = this._buildSyncedNodeMap(viewer, evt.nextSelection);

                this.viewers.forEach((other) => {
                    if (other === viewer) return;
                    const targetId = nodeIdMap.get(other);
                    if (targetId) {
                        this._applyGuarded(other, () => {
                            other.selectNode(targetId, { center: false });
                            if (this._followSelection) other.controller.panToNode(targetId, {});
                        });
                    } else {
                        this._applyGuarded(other, () => other.clearSelection());
                    }
                });

                this._updateMergedInfo(nodeIdMap);
                this._applyAutoCandidateHighlights(viewer, evt.nextSelection);
            });
            this._offs.push(off);
        });
    }

    _buildSyncedNodeMap(sourceViewer, nodeId) {
        const map = new Map();
        if (!nodeId) return map;
        map.set(sourceViewer, nodeId);
        this.viewers.forEach((viewer) => {
            if (viewer === sourceViewer) return;
            const targetId = this._findSyncTarget(sourceViewer, nodeId, viewer);
            if (targetId) {
                map.set(viewer, targetId);
            }
        });
        return map;
    }

    _collectCurrentSelectionMap() {
        const map = new Map();
        this.viewers.forEach((viewer) => {
            const selectedNodeId = viewer.controller?.state?.selectedNodeId;
            if (selectedNodeId) {
                map.set(viewer, selectedNodeId);
            }
        });
        return map.size > 0 ? map : null;
    }

    _buildAutoCandidates(sourceViewer, nodeId, targetViewer) {
        const rootCandidates = this._getAllFromNodeRootCandidates(sourceViewer, nodeId, targetViewer);
        const handleCandidates = this._getAllDebugHandleCandidates(sourceViewer, nodeId, targetViewer);
        const targetId = this._findSyncTarget(sourceViewer, nodeId, targetViewer);
        const candidates = [...rootCandidates, ...handleCandidates];
        if (targetId) candidates.push(targetId);
        if (targetViewer === sourceViewer) candidates.push(nodeId);
        return [...new Set(candidates)];
    }

    _applyAutoCandidateHighlights(sourceViewer, nodeId) {
        if (this.sync.mode !== 'auto' || !nodeId) {
            this.viewers.forEach((v) => v.removeHighlightGroup('_sync_candidates'));
            return;
        }

        this.viewers.forEach((viewer) => {
            const allCandidates = this._buildAutoCandidates(sourceViewer, nodeId, viewer);
            if (allCandidates.length > 0) {
                viewer.addHighlightGroup('_sync_candidates', allCandidates, '#ffaa00');
            } else {
                viewer.removeHighlightGroup('_sync_candidates');
            }
        });
    }

    _syncPreviewAcrossViewers(sourceViewer, previewNodeId) {
        if (!previewNodeId) {
            this.viewers.forEach((other) => {
                if (other === sourceViewer) return;
                this._applyGuarded(other, () => {
                    const selectedNodeId = other.controller.state.selectedNodeId;
                    const selectedEdge = other.controller.state.selectedEdge;
                    let ancestors = new Set();
                    let descendants = new Set();
                    if (selectedNodeId) {
                        ancestors = other.store.getAncestors(selectedNodeId);
                        descendants = other.store.getDescendants(selectedNodeId);
                    } else if (selectedEdge) {
                        ancestors = other.store.getAncestors(selectedEdge.v);
                        descendants = other.store.getDescendants(selectedEdge.w);
                    }
                    other.controller.setState({
                        previewNodeId: null,
                        ancestors,
                        descendants,
                    }, { source: 'compare-preview-sync' });
                });
            });
            this._updateMergedInfo(this._collectCurrentSelectionMap());
            this._applyAutoCandidateHighlights(sourceViewer, sourceViewer.controller?.state?.selectedNodeId || null);
            return;
        }

        const nodeIdMap = this._buildSyncedNodeMap(sourceViewer, previewNodeId);
        this.viewers.forEach((other) => {
            if (other === sourceViewer) return;
            const targetPreviewId = nodeIdMap.get(other) || null;
            this._applyGuarded(other, () => {
                if (!targetPreviewId) {
                    const selectedNodeId = other.controller.state.selectedNodeId;
                    const selectedEdge = other.controller.state.selectedEdge;
                    let ancestors = new Set();
                    let descendants = new Set();
                    if (selectedNodeId) {
                        ancestors = other.store.getAncestors(selectedNodeId);
                        descendants = other.store.getDescendants(selectedNodeId);
                    } else if (selectedEdge) {
                        ancestors = other.store.getAncestors(selectedEdge.v);
                        descendants = other.store.getDescendants(selectedEdge.w);
                    }
                    other.controller.setState({
                        previewNodeId: null,
                        ancestors,
                        descendants,
                    }, { source: 'compare-preview-sync' });
                    return;
                }

                other.controller.setState({
                    previewNodeId: targetPreviewId,
                    ancestors: other.store.getAncestors(targetPreviewId),
                    descendants: other.store.getDescendants(targetPreviewId),
                }, { source: 'compare-preview-sync' });
                if (this._followSelection) {
                    other.controller.panToNode(targetPreviewId);
                }
            });
        });
        this._updateMergedInfo(nodeIdMap);
        this._applyAutoCandidateHighlights(sourceViewer, previewNodeId);
    }

    _findSyncTarget(sourceViewer, nodeId, targetViewer) {
        const mode = this.sync.mode;
        if (mode === 'none') return null;
        if (mode === 'auto') {
            const byRoot = this._syncByFromNodeRoot(sourceViewer, nodeId, targetViewer);
            if (byRoot) return byRoot;
            const byHandle = this._syncByDebugHandle(sourceViewer, nodeId, targetViewer);
            if (byHandle) return byHandle;
            return targetViewer.store.activeNodeMap.has(nodeId) ? nodeId : null;
        }
        if (mode === 'id') {
            return targetViewer.store.activeNodeMap.has(nodeId) ? nodeId : null;
        }
        if (mode === 'layer') {
            const { layer, field } = this.sync;
            const srcVal = sourceViewer.store.extensions[layer]?.nodes[nodeId]?.info[field];
            if (srcVal === undefined) return null;
            const sourceNode = sourceViewer.store.activeNodeMap.get(nodeId);
            const candidates = targetViewer.store.activeNodes.filter(
                (n) => targetViewer.store.extensions[layer]?.nodes[n.id]?.info[field] === srcVal
            );
            const picked = this._pickCandidateByTargetMode(sourceNode, candidates);
            return picked ? picked.id : null;
        }
        return null;
    }

    _getTargetMode(node) {
        const target = String(node?.info?.target || '').toLowerCase();
        if (target.includes('dequantize')) return 'dequantize';
        if (target.includes('quantize') || target.includes('activation_post_process')) return 'quantize';
        return 'none';
    }

    _pickCandidateByTargetMode(sourceNode, candidates) {
        if (!Array.isArray(candidates) || candidates.length === 0) return null;
        const sourceMode = this._getTargetMode(sourceNode);
        for (let i = candidates.length - 1; i >= 0; i--) {
            if (this._getTargetMode(candidates[i]) === sourceMode) return candidates[i];
        }
        return candidates[candidates.length - 1];
    }

    // Normalize debug_handle (int | int[] | null) → Set<int>
    _normalizeHandle(dh) {
        if (!dh && dh !== 0) return new Set();
        if (typeof dh === 'number') return dh !== 0 ? new Set([dh]) : new Set();
        if (Array.isArray(dh)) return new Set(dh.filter((x) => typeof x === 'number' && x !== 0));
        return new Set();
    }

    _syncByFromNodeRoot(sourceViewer, nodeId, targetViewer) {
        const srcNode = sourceViewer.store.activeNodeMap.get(nodeId);
        const srcRoot = srcNode?.info?.from_node_root;
        if (!srcRoot) return null;
        const candidates = targetViewer.store.activeNodes.filter((n) => {
            const tgtNode = targetViewer.store.activeNodeMap.get(n.id);
            return tgtNode?.info?.from_node_root === srcRoot;
        });
        const picked = this._pickCandidateByTargetMode(srcNode, candidates);
        return picked ? picked.id : null;
    }

    _getAllFromNodeRootCandidates(sourceViewer, nodeId, targetViewer) {
        const srcNode = sourceViewer.store.activeNodeMap.get(nodeId);
        const srcRoot = srcNode?.info?.from_node_root;
        if (!srcRoot) return [];
        return targetViewer.store.activeNodes
            .filter((n) => {
                const tgtNode = targetViewer.store.activeNodeMap.get(n.id);
                return tgtNode?.info?.from_node_root === srcRoot;
            })
            .map((n) => n.id);
    }

    _syncByDebugHandle(sourceViewer, nodeId, targetViewer) {
        const srcNode = sourceViewer.store.activeNodeMap.get(nodeId);
        const srcSet = this._normalizeHandle(srcNode?.info?.debug_handle);
        if (srcSet.size === 0) return null;
        const candidates = targetViewer.store.activeNodes.filter((n) => {
            const tgtSet = this._normalizeHandle(
                targetViewer.store.activeNodeMap.get(n.id)?.info?.debug_handle
            );
            for (const v of srcSet) { if (tgtSet.has(v)) return true; }
            return false;
        });
        const picked = this._pickCandidateByTargetMode(srcNode, candidates);
        return picked ? picked.id : null;
    }

    _getAllDebugHandleCandidates(sourceViewer, nodeId, targetViewer) {
        const srcNode = sourceViewer.store.activeNodeMap.get(nodeId);
        const srcSet = this._normalizeHandle(srcNode?.info?.debug_handle);
        if (srcSet.size === 0) return [];
        return targetViewer.store.activeNodes
            .filter((n) => {
                const tgtSet = this._normalizeHandle(
                    targetViewer.store.activeNodeMap.get(n.id)?.info?.debug_handle
                );
                for (const v of srcSet) { if (tgtSet.has(v)) return true; }
                return false;
            })
            .map((n) => n.id);
    }

    _updateMergedInfo(nodeIdMap) {
        if (!this._infoRow) return;
        while (this._infoRow.firstChild) this._infoRow.removeChild(this._infoRow.firstChild);

        if (!nodeIdMap) {
            const ph = document.createElement('div');
            ph.className = 'fx-compare-info-placeholder';
            ph.textContent = 'No node selected — click a node to compare';
            this._infoRow.appendChild(ph);
            return;
        }

        const makeCell = (cls, text) => {
            const el = document.createElement('div');
            el.className = cls;
            el.textContent = text;
            return el;
        };

        const allProps = [];
        const rowData = new Map();
        nodeIdMap.forEach((nid, v) => {
            const node = v.store.activeNodeMap.get(nid);
            if (!node) return;
            const props = { id: nid, op: node.op || '', ...(node.info || {}) };
            Object.keys(props).forEach((k) => { if (!allProps.includes(k)) allProps.push(k); });
            rowData.set(v, props);
        });

        // Header row
        this._infoRow.appendChild(makeCell('fx-compare-info-hdr fx-compare-info-prop', 'Property'));
        this._viewerNames.forEach((name) => {
            if (!this._visibleViewers.has(name)) return;
            this._infoRow.appendChild(makeCell('fx-compare-info-hdr', name));
        });

        // Data rows
        allProps.forEach((prop, idx) => {
            const rowCls = idx % 2 === 1 ? ' fx-compare-info-row-alt' : '';
            const visViewerPairs = this.viewers
                .map((v, i) => ({ v, name: this._viewerNames[i] }))
                .filter(({ name }) => this._visibleViewers.has(name));
            const vals = visViewerPairs.map(({ v }) => {
                const d = rowData.get(v);
                if (!d || d[prop] === undefined) return ' -- ';
                const raw = d[prop];
                return (raw !== null && typeof raw === 'object') ? JSON.stringify(raw, null, 2) : String(raw);
            });
            const allSame = vals.every((v) => v === vals[0]);
            this._infoRow.appendChild(makeCell('fx-compare-info-prop' + rowCls, prop));
            vals.forEach((val) => {
                this._infoRow.appendChild(makeCell('fx-compare-info-val' + rowCls + (allSame ? '' : ' fx-compare-info-diff'), val));
            });
        });
    }

    _wireStateSync() {
        this.viewers.forEach((viewer) => {
            const off = viewer.on('statechange', (evt) => {
                if (this._guards.has(viewer)) return;
                const prev = evt.prevState || {};
                const next = evt.nextState || {};
                const themeChanged = prev.theme !== next.theme && typeof next.theme === 'string';
                if (themeChanged) {
                    this.viewers.forEach((other) => {
                        if (other === viewer) return;
                        this._applyGuarded(other, () => {
                            other.setTheme(next.theme);
                        });
                    });
                }

                const themeNameChanged = prev.themeName !== next.themeName && typeof next.themeName === 'string';
                if (themeNameChanged && this._themeSelect) {
                    this._themeSelect.value = next.themeName;
                }
                if (themeNameChanged) {
                    this._applyCompareTheme(next.themeName);
                }

                if (prev.previewNodeId !== next.previewNodeId) {
                    this._syncPreviewAcrossViewers(viewer, next.previewNodeId || null);
                }
            });
            this._offs.push(off);
        });
    }

    setSync(syncPatch = {}) {
        this.sync = { ...this.sync, ...syncPatch };
        if (this.sync.mode !== 'auto') {
            this.viewers.forEach((v) => v.removeHighlightGroup('_sync_candidates'));
        }
        this._rebuildSyncPanel();
    }

    /** @deprecated No-op — tiled layout is always used in compare mode */
    setTiled() {}

    /** @deprecated No-op — sidebar replaces sharedTaskbar */
    setCompact() {}

    _applyCompareTheme(themeName) {
        this._currentTheme = themeName;
        const isDark = themeName === 'dark';
        const r = this._root.style;
        if (isDark) {
            r.setProperty('--cmp-bg', '#1e1e1e');
            r.setProperty('--cmp-text', '#ffffff');
            r.setProperty('--cmp-border', '#444444');
            r.setProperty('--cmp-border-strong', '#555555');
            r.setProperty('--cmp-sidebar-bg', 'rgba(255,255,255,0.05)');
            r.setProperty('--cmp-info-bg', '#1e1e1e');
            r.setProperty('--cmp-prop-bg', 'rgba(255,255,255,0.05)');
            r.setProperty('--cmp-hdr-bg', 'rgba(255,255,255,0.08)');
            r.setProperty('--cmp-diff-bg', 'rgba(255,160,40,0.10)');
            r.setProperty('--cmp-diff-accent', '#c87830');
            r.setProperty('--cmp-name-bg', 'rgba(255,255,255,0.1)');
            r.setProperty('--cmp-ui-bg', 'rgba(30,30,30,0.95)');
            r.setProperty('--cmp-ui-hover', '#333333');
        } else {
            r.setProperty('--cmp-bg', '#ffffff');
            r.setProperty('--cmp-text', '#000000');
            r.setProperty('--cmp-border', '#e5e7eb');
            r.setProperty('--cmp-border-strong', '#cccccc');
            r.setProperty('--cmp-sidebar-bg', 'rgba(0,0,0,0.02)');
            r.setProperty('--cmp-info-bg', '#ffffff');
            r.setProperty('--cmp-prop-bg', 'rgba(0,0,0,0.02)');
            r.setProperty('--cmp-hdr-bg', 'rgba(0,0,0,0.04)');
            r.setProperty('--cmp-diff-bg', 'rgba(255,140,0,0.06)');
            r.setProperty('--cmp-diff-accent', '#e08a3c');
            r.setProperty('--cmp-name-bg', 'rgba(0,0,0,0.06)');
            r.setProperty('--cmp-ui-bg', 'rgba(255,255,255,0.95)');
            r.setProperty('--cmp-ui-hover', '#f0f8ff');
        }
        const theme = (typeof THEMES !== 'undefined' && THEMES[themeName]) || THEMES?.light;
        if (theme && this._sidebarEl) {
            this._sidebarEl.querySelectorAll('.fx-button, .fx-select').forEach((el) => {
                el.style.backgroundColor = theme.uiBg;
                el.style.color = theme.text;
                el.style.borderColor = theme.uiBorder;
            });
        }
        if (theme) {
            for (const menu of this._openPortalMenus) {
                menu.style.backgroundColor = theme.uiBg;
                menu.style.color = theme.text;
                menu.style.borderColor = theme.uiBorder;
                menu.querySelectorAll('label, .fx-button, .fx-select').forEach((el) => {
                    el.style.color = theme.text;
                });
            }
        }
    }

    _applyGuarded(viewer, fn) {
        this._guards.add(viewer);
        try {
            fn();
        } finally {
            setTimeout(() => this._guards.delete(viewer), 0);
        }
    }

    setViewerVisible(name, visible) {
        this._setViewerVisible(name, visible);
    }

    refreshLayout(options = {}) {
        this._scheduleLayoutRefresh(options);
    }

    destroy() {
        this._teardownCompareDOM();
        this._offs.forEach((off) => {
            try { off(); } catch (_) {}
        });
        this._offs = [];
    }
}

if (typeof globalThis !== 'undefined') {
    globalThis.FXGraphCompare = FXGraphCompare;
}
