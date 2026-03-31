// FXCompareTaskbar: shared taskbar for multi-viewer compare mode.
// FXGraphCompare: orchestrates multi-view compare DOM, selection sync, and lifecycle.

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
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="checkbox" value="${fxEsc(id)}"> ${fxEsc(name)}</label>`;
        });
        html += '<div style="padding:5px;font-weight:bold;border-top:1px solid #ccc;margin-top:4px;">Color By</div>';
        html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="base"> Base</label>`;
        allLayers.forEach((name, id) => {
            html += `<label style="display:block;padding:5px;cursor:pointer;"><input type="radio" name="cmp_colorby" value="${fxEsc(id)}"> ${fxEsc(name)}</label>`;
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
            html += `<option value="${fxEsc(key)}">${fxEsc(label)}</option>`;
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
        viewerNames.forEach((name) => { html += `<div class="fx-compare-info-hdr">${fxEsc(name)}</div>`; });
        // Data rows
        allProps.forEach((prop) => {
            const vals = this.viewers.map((v) => {
                const d = rowData.get(v);
                return d ? (d[prop] !== undefined ? String(d[prop]) : '—') : '—';
            });
            const allSame = vals.every((v) => v === vals[0]);
            html += `<div class="fx-compare-info-prop">${fxEsc(prop)}</div>`;
            vals.forEach((val) => { html += `<div class="fx-compare-info-val${allSame ? '' : ' fx-compare-info-diff'}">${fxEsc(val)}</div>`; });
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
    globalThis.FXGraphCompare = FXGraphCompare;
}
