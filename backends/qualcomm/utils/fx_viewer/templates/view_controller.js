/**
 * RFC v1 notes:
 * - Controller owns interaction state and camera transform.
 * - `setState` triggers canonical recompute/repaint and emits viewer events.
 * - Theme/layer/colorBy mutations propagate through store -> minimap -> legend -> canvas.
 *
 * UX impact:
 * - Every interaction path (UI or API) converges on one state pipeline.
 * - Search, selection, and camera animation remain consistent across embeds.
 */
/**
 * ============================================================================
 * CLASS: ViewerController
 * ============================================================================
 * Centralized state machine managing interactions, camera transforms, selections,
 * and Extension visibility.
 * 
 * RELATED VARIABLES & STATE:
 * - `viewer`: Reference to the `FXGraphViewer` facade (used to trigger `renderAll()`).
 * - `store`: Reference to `GraphDataStore` (used to query node adjacency and bounds).
 * - `transform`: Object {x, y, k} defining the current camera pan (x,y) and zoom (k).
 * - `state.hoveredNodeId` / `state.hoveredEdge`: Elements currently under the cursor.
 * - `state.selectedNodeId` / `state.selectedEdge`: Elements actively clicked/selected.
 * - `state.previewNodeId`: Node currently highlighted via keyboard search navigation.
 * - `state.ancestors` / `state.descendants`: Sets of Node IDs connected to the selection.
 * - `state.searchCandidates`: Array of matched items from the SearchEngine.
 * - `state.highlightAncestors`: Boolean toggling whether to dim unrelated branches.
 * - `state.themeName`: String ('light' or 'dark') tracking the current color palette.
 * - `activeExtensions`: Set[str]. The user-selected layers (from the UIManager 
 *   Layers dropdown) currently injected into the visual graph.
 * - `colorBy`: String. The explicit source of `fill_color` for the nodes 
 *   ('base' or an extension ID).
 * 
 * USE CASES & METHOD CALLS & EXPECTED BEHAVIORS:
 *
 * - Interaction Routing: Every UI click, search input, or canvas drag routes 
 *   through this class. It intercepts the event, updates its internal `state`, 
 *   and issues commands to other modules.
 * 
 * - `setState(newState)`: Merges state and dynamically issues re-render commands.
 *   If `activeExtensions` or `colorBy` change, it tells the `GraphDataStore` to 
 *   recompute the virtual graph before painting.
 * - `animateToTransform(targetX, targetY, targetK)`:
 *   - Trigger: Called by `zoomToFit` and `animateToNode` for smooth camera transitions.
 *   - Effect: Uses `requestAnimationFrame` with easeOutCubic to interpolate `transform.x, y, k` 
 *     over 300ms, calling `renderAll()` on each frame.
 * 
 * - `zoomToFit()`:
 *   - Trigger: User clicks the "⛶" (Zoom to Fit) button in the taskbar.
 *   - Logic: If a node/edge is selected, it traces 2-hops of ancestors and descendants 
 *     via `this.store`, calculates their combined bounding box, and fits the screen to 
 *     that specific local neighborhood. If nothing is selected, it fits the entire graph.
 *   - Effect: Animates the camera to perfectly frame the calculated bounds.
 * 
 * - `panToNode(nodeId)`:
 *   - Trigger: User clicks a search candidate or arrows through the search dropdown.
 *   - Effect: Instantly teleports the camera `transform.x` and `y` to center the target node.
 * 
 * - `animateToNode(nodeId)`:
 *   - Trigger: User clicks an Input/Output link inside the Info Panel.
 *   - Effect: Calculates the target coordinate to center the node and invokes 
 *     `animateToTransform` for a smooth gliding animation.
 * 
 * - `handleHover(nodeId, edge)`:
 *   - Trigger: `mousemove` event inside `CanvasRenderer` detects a collision.
 *   - Effect: Updates `state.hoveredNodeId` or `state.hoveredEdge` and re-renders. 
 *     This causes the hovered element to receive a dashed red border in the canvas.
 * 
 * - `handleClick(nodeId, edge)`:
 *   - Trigger: `click` event inside `CanvasRenderer` on an element or empty space.
 *   - Effect: Routes to `selectNode`, `selectEdge`, or `clearSelection`.
 * 
 * - `selectNode(nodeId)` / `selectEdge(edge)`:
 *   - Trigger: Clicking an element in canvas, clicking an Info Panel link, or hitting 
 *     Enter on a search candidate.
 *   - Logic: Queries `this.store.getAncestors` and `getDescendants`.
 *   - Effect: Sets `state.selectedNodeId` or `state.selectedEdge`, populates the `ancestors` 
 *     and `descendants` sets for canvas highlighting, and opens the Info Panel overlay.
 * 
 * - `clearSelection()`:
 *   - Trigger: User clicks the "✖" button or clicks empty white space in the canvas.
 *   - Effect: Nullifies active selection states, clears ancestor/descendant sets, and 
 *     hides the Info Panel.
 * 
 * - `handleSearch(query)`:
 *   - Trigger: User types into the taskbar search input.
 *   - Effect: Passes query to `SearchEngine`, stores results in `state.searchCandidates`, 
 *     and passes them to the DOM via `UIManager.updateSearchResults`.
 * 
 * - `handleSearchNavigate(direction)`:
 *   - Trigger: User presses ArrowUp (-1) or ArrowDown (+1) inside the search input.
 *   - Effect: Shifts `state.searchSelectedIndex`, teleports the camera to preview the 
 *     highlighted node, and updates the Info Panel.
 * 
 * - `handleSearchSelect(index)`:
 *   - Trigger: User presses Enter while navigating search, or physically clicks a dropdown item.
 *   - Effect: Fully selects the highlighted node, closes the dropdown menu, and clears 
 *     the search input to reset the minimap red highlights.
 * 
 * - `handleSearchHover(index)`:
 *   - Trigger: User hovers the mouse over a specific search candidate in the dropdown menu.
 *   - Effect: Updates the `searchSelectedIndex` to match the hovered item, instantly panning 
 *     the camera to preview the node in the canvas.
 * 
 * ALGORITHM & INFO FLOW:
 * - Extension Toggling (`setState`): 
 *   When the user checks a box in the Layers menu, `UIManager` calls `setState()`. 
 *   The controller detects this mutation, calls `store.computeActiveGraph()`, 
 *   tells the `MinimapRenderer` to regenerate its static thumbnail using the new 
 *   colors/labels, tells `UIManager` to swap the Legend overlay, and finally 
 *   calls `viewer.renderAll()` to repaint the Canvas.
 * - Zoom & Pan (`zoomToFit`, `animateToNode`): 
 *   Uses `requestAnimationFrame` and an easeOutCubic interpolation to glide 
 *   the camera smoothly to a target X/Y coordinate.
 * - Selection Engine: 
 *   When a node is selected, queries the `store` for BFS ancestors/descendants. 
 *   Passes these arrays to the Canvas to highlight the execution dependency chain.
 * 
 * USER EXPERIENCE (UX):
 * - Centralizing state prevents React-like prop-drilling in vanilla JS. When an 
 *   extension is toggled, all UI elements (Canvas, Minimap, Info Panel, Legend) 
 *   update synchronously without race conditions.
 * - The smooth camera animations give the tool a premium, desktop-client feel 
 *   rather than a static webpage.
 * ============================================================================
 */
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
            let localNodes = new Set();
            localNodes.add(this.state.selectedNodeId);

            const p1 = this.store.revAdjList.get(this.state.selectedNodeId) || [];
            p1.forEach(e => {
                localNodes.add(e.v);
                const p2 = this.store.revAdjList.get(e.v) || [];
                p2.forEach(e2 => localNodes.add(e2.v));
            });

            const c1 = this.store.adjList.get(this.state.selectedNodeId) || [];
            c1.forEach(e => {
                localNodes.add(e.w);
                const c2 = this.store.adjList.get(e.w) || [];
                c2.forEach(e2 => localNodes.add(e2.w));
            });

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            localNodes.forEach(nid => {
                const node = this.store.activeNodeMap.get(nid);
                if (node) {
                    minX = Math.min(minX, node.x - node.width/2);
                    maxX = Math.max(maxX, node.x + node.width/2);
                    minY = Math.min(minY, node.y - node.height/2);
                    maxY = Math.max(maxY, node.y + node.height/2);
                }
            });

            if (minX !== Infinity) {
                bounds = { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
            }
        } else if (this.state.selectedEdge) {
            let localNodes = new Set();
            const v = this.state.selectedEdge.v;
            const w = this.state.selectedEdge.w;
            localNodes.add(v);
            localNodes.add(w);

            (this.store.revAdjList.get(v) || []).forEach(e => localNodes.add(e.v));
            (this.store.adjList.get(v) || []).forEach(e => localNodes.add(e.w));

            (this.store.revAdjList.get(w) || []).forEach(e => localNodes.add(e.v));
            (this.store.adjList.get(w) || []).forEach(e => localNodes.add(e.w));

            let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
            localNodes.forEach(nid => {
                const node = this.store.activeNodeMap.get(nid);
                if (node) {
                    minX = Math.min(minX, node.x - node.width/2);
                    maxX = Math.max(maxX, node.x + node.width/2);
                    minY = Math.min(minY, node.y - node.height/2);
                    maxY = Math.max(maxY, node.y + node.height/2);
                }
            });

            if (minX !== Infinity) {
                bounds = { minX, maxX, minY, maxY, width: maxX - minX, height: maxY - minY };
            }
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
