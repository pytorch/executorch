// Centralized state machine managing interactions, camera transforms, selections, and extension visibility.
class ViewerController {
    constructor(viewer, initialState = {}) {
        this.viewer = viewer;
        this.store = viewer.store;
        this.transform = { x: 0, y: 0, k: 1 };

        const initialTheme = initialState.themeName || initialState.theme || 'light';
        const initialExtensions = initialState.activeExtensions
            ? new Set(initialState.activeExtensions)
            : new Set(Object.keys(this.store.extensions));
        const initialColorBy = initialState.colorBy || 'base';
        
        this.state = {
            hoveredNodeId: null,
            hoveredEdge: null,
            selectedNodeId: null,
            selectedEdge: null,
            previewNodeId: null,
            ancestors: new Set(),
            descendants: new Set(),
            searchCandidates: [],
            searchSelectedIndex: -1,
            highlightAncestors: initialState.highlightAncestors !== false,
            themeName: initialTheme,
            uiVisibility: { ...(initialState.uiVisibility || {}) },
            
            // V3 Extensibility State
            activeExtensions: initialExtensions,
            colorBy: initialColorBy
        };
        
        // Initial computation of the virtual graph
        this.store.computeActiveGraph(this.state.activeExtensions, this.state.colorBy);
    }
    
    snapshotState() {
        return {
            hoveredNodeId: this.state.hoveredNodeId,
            hoveredEdge: this.state.hoveredEdge,
            selectedNodeId: this.state.selectedNodeId,
            selectedEdge: this.state.selectedEdge,
            previewNodeId: this.state.previewNodeId,
            searchCandidates: this.state.searchCandidates.slice(),
            searchSelectedIndex: this.state.searchSelectedIndex,
            highlightAncestors: this.state.highlightAncestors,
            themeName: this.state.themeName,
            theme: this.state.themeName,
            activeExtensions: Array.from(this.state.activeExtensions),
            colorBy: this.state.colorBy,
            searchQuery: this.viewer.ui && this.viewer.ui.searchInput ? this.viewer.ui.searchInput.value : "",
            camera: { ...this.transform },
            uiVisibility: { ...(this.state.uiVisibility || {}) },
        };
    }

    setState(newState, options = {}) {
        const prev = this.snapshotState();

        const patch = { ...newState };
        if ('theme' in patch && !('themeName' in patch)) {
            patch.themeName = patch.theme;
        }
        if ('activeExtensions' in patch && !(patch.activeExtensions instanceof Set)) {
            patch.activeExtensions = new Set(patch.activeExtensions || []);
        }

        Object.assign(this.state, patch);
        
        // If graph structure or color changed, we must recompute and update UI
        if ('activeExtensions' in patch || 'colorBy' in patch) {
            this.store.computeActiveGraph(this.state.activeExtensions, this.state.colorBy);
            
            if (this.viewer.minimapRenderer) {
                this.viewer.minimapRenderer.generateThumbnail();
            }
            if (this.viewer.ui) {
                this.viewer.ui.renderLegend();
                if (this.state.selectedNodeId) {
                    this.viewer.ui.updateInfoPanel(this.state.selectedNodeId);
                }
            }
        }

        if ('themeName' in patch || 'theme' in patch) {
            if (this.viewer.ui) {
                this.viewer.ui.applyThemeToDOM();
            }
            if (this.viewer.minimapRenderer) {
                this.viewer.minimapRenderer.generateThumbnail();
            }
        }

        if (this.viewer.ui) {
            this.viewer.ui.syncControlsFromState();
        }
        
        this.viewer.renderAll();

        const next = this.snapshotState();
        this.viewer._emit('statechange', { prevState: prev, nextState: next, source: options.source || 'api' });
        if (prev.selectedNodeId !== next.selectedNodeId) {
            this.viewer._emit('selectionchange', {
                prevSelection: prev.selectedNodeId,
                nextSelection: next.selectedNodeId,
                source: options.source || 'api',
            });
        }
        if (prev.theme !== next.theme) {
            this.viewer._emit('themechange', { prevTheme: prev.theme, nextTheme: next.theme, source: options.source || 'api' });
        }
    }

    animateToTransform(targetX, targetY, targetK, duration = 300) {
        const startX = this.transform.x;
        const startY = this.transform.y;
        const startK = this.transform.k;
        const startTime = performance.now();
        
        const animate = (currentTime) => {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const ease = 1 - Math.pow(1 - progress, 3); // easeOutCubic
            
            this.transform.x = startX + (targetX - startX) * ease;
            this.transform.y = startY + (targetY - startY) * ease;
            this.transform.k = startK + (targetK - startK) * ease;
            
            this.viewer.renderAll();
            
            if (progress < 1) {
                requestAnimationFrame(animate);
            }
        };
        requestAnimationFrame(animate);
    }

    zoomToFit() {
        const padding = 50;
        const rect = this.viewer.canvasContainer.getBoundingClientRect();
        const availableW = rect.width - padding * 2;
        const availableH = rect.height - padding * 2;

        let bounds = this.store.graphBounds;

        if (this.state.selectedNodeId) {
            bounds = this.store.computeBoundsForNodes(
                this._collect2HopNeighbors(this.state.selectedNodeId)
            ) || bounds;
        } else if (this.state.selectedEdge) {
            bounds = this.store.computeBoundsForNodes(
                this._collectEdgeNeighbors(this.state.selectedEdge)
            ) || bounds;
        }

        if (bounds.width === 0 || bounds.height === 0) return;

        const scaleW = availableW / bounds.width;
        const scaleH = availableH / bounds.height;
        let targetK = Math.min(scaleW, scaleH);
        if (this.state.selectedNodeId || this.state.selectedEdge) {
            targetK = Math.min(targetK, 1.2);
        }

        const centerX = bounds.minX + bounds.width / 2;
        const centerY = bounds.minY + bounds.height / 2;

        const targetX = (rect.width / 2) - centerX * targetK;
        const targetY = (rect.height / 2) - centerY * targetK;

        this.animateToTransform(targetX, targetY, targetK);
    }

    _collect2HopNeighbors(nodeId) {
        const nodes = new Set([nodeId]);
        for (const e of this.store.revAdjList.get(nodeId) || []) {
            nodes.add(e.v);
            for (const e2 of this.store.revAdjList.get(e.v) || []) nodes.add(e2.v);
        }
        for (const e of this.store.adjList.get(nodeId) || []) {
            nodes.add(e.w);
            for (const e2 of this.store.adjList.get(e.w) || []) nodes.add(e2.w);
        }
        return nodes;
    }

    _collectEdgeNeighbors(edge) {
        const nodes = new Set([edge.v, edge.w]);
        for (const e of this.store.revAdjList.get(edge.v) || []) nodes.add(e.v);
        for (const e of this.store.adjList.get(edge.v) || []) nodes.add(e.w);
        for (const e of this.store.revAdjList.get(edge.w) || []) nodes.add(e.v);
        for (const e of this.store.adjList.get(edge.w) || []) nodes.add(e.w);
        return nodes;
    }

    panToNode(nodeId) {
        const node = this.store.activeNodeMap.get(nodeId);
        if (!node) return;
        const rect = this.viewer.canvasContainer.getBoundingClientRect();
        this.transform.x = rect.width / 2 - node.x * this.transform.k;
        this.transform.y = rect.height / 2 - node.y * this.transform.k;
        this.viewer.renderAll();
    }

    animateToNode(nodeId, targetK = null) {
        const node = this.store.activeNodeMap.get(nodeId);
        if (!node) return;
        const rect = this.viewer.canvasContainer.getBoundingClientRect();
        const k = targetK !== null ? targetK : this.transform.k;
        const targetX = rect.width / 2 - node.x * k;
        const targetY = rect.height / 2 - node.y * k;
        this.animateToTransform(targetX, targetY, k);
    }
    
    handleHover(nodeId, edge) {
        if (this.state.hoveredNodeId !== nodeId || this.state.hoveredEdge !== edge) {
            this.setState({ hoveredNodeId: nodeId, hoveredEdge: edge });
        }
    }
    
    handleClick(nodeId, edge) {
        if (nodeId) {
            this.selectNode(nodeId);
        } else if (edge) {
            this.selectEdge(edge);
        } else {
            this.clearSelection();
        }
    }
    
    selectNode(nodeId) {
        const ancestors = this.store.getAncestors(nodeId);
        const descendants = this.store.getDescendants(nodeId);
        this.setState({ 
            selectedNodeId: nodeId, 
            selectedEdge: null,
            ancestors, 
            descendants,
            previewNodeId: null
        });
        this.viewer.ui.updateInfoPanel(nodeId);
    }

    selectEdge(edge) {
        const ancestors = this.store.getAncestors(edge.v);
        const descendants = this.store.getDescendants(edge.w);
        this.setState({ 
            selectedNodeId: null, 
            selectedEdge: edge,
            ancestors, 
            descendants,
            previewNodeId: null
        });
        this.viewer.ui.updateEdgeInfoPanel(edge);
    }
    
    clearSelection() {
        this.setState({
            selectedNodeId: null,
            selectedEdge: null,
            ancestors: new Set(),
            descendants: new Set(),
            previewNodeId: null
        });
        this.viewer.ui.hideInfoPanel();
    }

    handleSearch(query) {
        if (!query) {
            this.setState({ searchCandidates: [], searchSelectedIndex: -1, previewNodeId: null });
            this.viewer.ui.updateSearchResults([], -1);
            if (this.state.selectedNodeId) {
                this.viewer.ui.updateInfoPanel(this.state.selectedNodeId);
                this.panToNode(this.state.selectedNodeId);
            } else {
                this.viewer.ui.hideInfoPanel();
            }
            return;
        }
        const candidates = this.viewer.searchEngine.search(query);
        this.setState({ searchCandidates: candidates, searchSelectedIndex: -1, previewNodeId: null });
        this.viewer.ui.updateSearchResults(candidates, -1);
    }

    handleSearchNavigate(direction) {
        const { searchCandidates, searchSelectedIndex } = this.state;
        if (searchCandidates.length === 0) return;
        let newIndex = searchSelectedIndex + direction;
        if (newIndex < 0) newIndex = searchCandidates.length - 1;
        if (newIndex >= searchCandidates.length) newIndex = 0;
        
        const previewNode = searchCandidates[newIndex].node.id;
        this.setState({ searchSelectedIndex: newIndex, previewNodeId: previewNode });
        this.viewer.ui.updateSearchActiveItem(newIndex);
        this.viewer.ui.updateInfoPanel(previewNode);
        this.panToNode(previewNode);
    }

    handleSearchSelect(index) {
        const { searchCandidates } = this.state;
        const idx = index !== undefined ? index : this.state.searchSelectedIndex;
        if (idx >= 0 && idx < searchCandidates.length) {
            const nodeId = searchCandidates[idx].node.id;
            this.selectNode(nodeId);
            this.panToNode(nodeId);
            this.viewer.ui.closeSearchMenu();
            this.setState({ searchCandidates: [], searchSelectedIndex: -1, previewNodeId: null });
            if (this.viewer.ui.searchInput) this.viewer.ui.searchInput.value = '';
        }
    }

    handleSearchHover(index) {
        const { searchCandidates } = this.state;
        if (index >= 0 && index < searchCandidates.length) {
            const previewNode = searchCandidates[index].node.id;
            this.setState({ searchSelectedIndex: index, previewNodeId: previewNode });
            this.viewer.ui.updateSearchActiveItem(index);
            this.viewer.ui.updateInfoPanel(previewNode);
            this.panToNode(previewNode);
        }
    }
}
