// Minimap overview rendering with viewport tracking and click/drag navigation.
class MinimapRenderer {
    constructor(container, viewer) {
        this.viewer = viewer;
        this._teardownFns = [];
        const mountPoint = container || this.viewer.sidebar || this.viewer.mainArea;
        this.container = document.createElement('div');
        this.container.className = 'fx-minimap-container';
        mountPoint.appendChild(this.container);
        
        this.canvas = document.createElement('canvas');
        this.canvas.className = 'fx-minimap';
        this.container.appendChild(this.canvas);
        this.ctx = this.canvas.getContext('2d');
        
        this.thumbnailCanvas = document.createElement('canvas');
        this.thumbnailCtx = this.thumbnailCanvas.getContext('2d');
        
        this.minimapScale = 1;
        this.thumbnailOffset = { x: 0, y: 0 };
        this.isDragging = false;
        
        this.resize();
        this._onWindowResize = () => {
            this.resize();
            this.generateThumbnail();
            this.render();
        };
        fxOn(this._teardownFns, window, 'resize', this._onWindowResize);
        
        this.setupEvents();

        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(() => {
                const rect = this.container.getBoundingClientRect();
                if (rect.width > 0 && rect.height > 0) {
                    this.resize();
                    this.generateThumbnail();
                    this.render();
                }
            });
            this._resizeObserver.observe(this.container);
            this._teardownFns.push(() => this._resizeObserver.disconnect());
        }
    }
    
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.container.getBoundingClientRect();
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
    }
    
    generateThumbnail() {
        const mw = this.canvas.width;
        const mh = this.canvas.height;
        if (mw === 0 || mh === 0) return;
        
        this.thumbnailCanvas.width = mw;
        this.thumbnailCanvas.height = mh;
        
        const bounds = this.viewer.store.graphBounds;
        if (bounds.width === 0) return;
        
        const scaleX = mw / bounds.width;
        const scaleY = mh / bounds.height;
        this.minimapScale = Math.min(scaleX, scaleY) * 0.9;
        
        this.thumbnailOffset = {
            x: (mw - bounds.width * this.minimapScale) / 2,
            y: (mh - bounds.height * this.minimapScale) / 2
        };
        
        const ctx = this.thumbnailCtx;
        const theme = THEMES[this.viewer.controller.state.themeName];
        ctx.fillStyle = theme.bg;
        ctx.fillRect(0, 0, mw, mh);
        ctx.save();
        ctx.translate(this.thumbnailOffset.x, this.thumbnailOffset.y);
        ctx.scale(this.minimapScale, this.minimapScale);
        
        ctx.strokeStyle = theme.edgeNormal;
        const dpr = window.devicePixelRatio || 1;
        ctx.lineWidth = Math.max(1 / this.minimapScale, dpr / this.minimapScale);
        ctx.beginPath();
        this.viewer.store.baseData.edges.forEach(edge => {
            const v = this.viewer.store.activeNodeMap.get(edge.v);
            const w = this.viewer.store.activeNodeMap.get(edge.w);
            if (v && w) {
                if (edge.points && edge.points.length > 0) {
                    ctx.moveTo(edge.points[0].x, edge.points[0].y);
                    for (let i = 1; i < edge.points.length; i++) {
                        ctx.lineTo(edge.points[i].x, edge.points[i].y);
                    }
                } else {
                    ctx.moveTo(v.x, v.y);
                    ctx.lineTo(w.x, w.y);
                }
            }
        });
        ctx.stroke();
        
        this.viewer.store.activeNodes.forEach(node => {
            ctx.fillStyle = node.fill_color ? node.fill_color : theme.nodeFill;
            const minSize = 2 / this.minimapScale;
            const w = Math.max(node.width, minSize);
            const h = Math.max(node.height, minSize);
            ctx.fillRect(node.x - w/2, node.y - h/2, w, h);
        });
        ctx.restore();
    }
    
    setupEvents() {
        const onMouseDown = (e) => {
            this.isDragging = true;
            this.handleDrag(e);
        };
        fxOn(this._teardownFns, this.canvas, 'mousedown', onMouseDown);

        const onMouseMove = (e) => {
            if (this.isDragging) this.handleDrag(e);
        };
        fxOn(this._teardownFns, window, 'mousemove', onMouseMove);

        const onMouseUp = () => {
            this.isDragging = false;
        };
        fxOn(this._teardownFns, window, 'mouseup', onMouseUp);
        
        const onWheel = (e) => {
            e.preventDefault();
            const zoomIntensity = 0.1;
            const wheel = e.deltaY < 0 ? 1 : -1;
            const zoomFactor = Math.exp(wheel * zoomIntensity);
            
            const mainCanvasRect = this.viewer.canvasRenderer.canvas.getBoundingClientRect();
            const mouseX = mainCanvasRect.width / 2;
            const mouseY = mainCanvasRect.height / 2;

            const transform = this.viewer.controller.transform;
            const graphX = (mouseX - transform.x) / transform.k;
            const graphY = (mouseY - transform.y) / transform.k;

            transform.k *= zoomFactor;
            transform.x = mouseX - graphX * transform.k;
            transform.y = mouseY - graphY * transform.k;
            
            this.viewer.renderAll();
        };
        fxOn(this._teardownFns, this.canvas, 'wheel', onWheel, { passive: false });
    }
    
    handleDrag(e) {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvas.getBoundingClientRect();
        const mx = (e.clientX - rect.left) * dpr;
        const my = (e.clientY - rect.top) * dpr;

        const graphX = (mx - this.thumbnailOffset.x) / this.minimapScale;
        const graphY = (my - this.thumbnailOffset.y) / this.minimapScale;

        const canvasRect = this.viewer.canvasContainer.getBoundingClientRect();
        const transform = this.viewer.controller.transform;

        transform.x = (canvasRect.width / 2) - (graphX * transform.k);
        transform.y = (canvasRect.height / 2) - (graphY * transform.k);

        this.viewer.renderAll();
    }

    resetInteractionState() {
        this.isDragging = false;
    }

    render() {
        const dpr = window.devicePixelRatio || 1;
        if (this.canvas.width === 0 || this.canvas.height === 0) return;
        
        const state = this.viewer.controller.state;
        const theme = THEMES[state.themeName];
        const minEdgeWidth = dpr / Math.max(this.minimapScale, 1e-6);
        
        this.ctx.setTransform(1, 0, 0, 1, 0, 0);
        this.ctx.fillStyle = theme.bg;
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        const isSelectionMode = !!state.selectedNodeId || !!state.previewNodeId || !!state.selectedEdge;
        
        if (isSelectionMode && state.highlightAncestors) {
            this.ctx.globalAlpha = 0.2;
        }
        this.ctx.drawImage(this.thumbnailCanvas, 0, 0);
        this.ctx.globalAlpha = 1.0;
        
        this.ctx.save();
        this.ctx.translate(this.thumbnailOffset.x, this.thumbnailOffset.y);
        this.ctx.scale(this.minimapScale, this.minimapScale);
        
        const drawNodes = (nodes, padding = 0, fillColor = null) => {
            nodes.forEach(nid => {
                const node = this.viewer.store.activeNodeMap.get(nid);
                if (node) {
                    this.ctx.fillStyle = fillColor || (node.fill_color ? node.fill_color : theme.nodeFill);
                    const minSize = 3 / this.minimapScale;
                    const w = Math.max(node.width, minSize) + padding;
                    const h = Math.max(node.height, minSize) + padding;
                    this.ctx.fillRect(node.x - w/2, node.y - h/2, w, h);
                }
            });
        };

        const drawEdgePath = (edge, v, w) => {
            if (edge.points && edge.points.length > 0) {
                this.ctx.moveTo(edge.points[0].x, edge.points[0].y);
                for (let i = 1; i < edge.points.length; i++) {
                    this.ctx.lineTo(edge.points[i].x, edge.points[i].y);
                }
            } else {
                this.ctx.moveTo(v.x, v.y);
                this.ctx.lineTo(w.x, w.y);
            }
        };

        const drawEdges = (edges, color, width) => {
            const edgeList = Array.from(edges || []);
            if (edgeList.length === 0) return;
            this.ctx.strokeStyle = color;
            this.ctx.lineWidth = Math.max(width, minEdgeWidth);
            this.ctx.beginPath();
            edgeList.forEach((edge) => {
                const v = this.viewer.store.activeNodeMap.get(edge.v);
                const w = this.viewer.store.activeNodeMap.get(edge.w);
                if (!v || !w) return;
                drawEdgePath(edge, v, w);
            });
            this.ctx.stroke();
        };

        const selectionEdges = new Set();
        const target = state.previewNodeId || state.selectedNodeId;

        if (target) {
            (this.viewer.store.revAdjList.get(target) || []).forEach((edge) => selectionEdges.add(edge));
            (this.viewer.store.adjList.get(target) || []).forEach((edge) => selectionEdges.add(edge));
        }
        if (state.selectedEdge) selectionEdges.add(state.selectedEdge);
        drawEdges(selectionEdges, theme.edgeInput, 2 / this.minimapScale);
        
        if (target) {
            if (state.highlightAncestors) {
                drawNodes(Array.from(state.ancestors));
                drawNodes(Array.from(state.descendants));
            }
        }

        if (state.highlightGroups && state.highlightGroups.size > 0) {
            state.highlightGroups.forEach(({ nodeIds, color }) => {
                const nodeSet = new Set(nodeIds || []);
                const edges = this.viewer.store.baseData.edges.filter(
                    (edge) => nodeSet.has(edge.v) && nodeSet.has(edge.w)
                );
                drawEdges(edges, color, 2 / this.minimapScale);
                drawNodes(Array.from(nodeSet), 2/this.minimapScale, color);
            });
        }

        if (state.selectedEdge) {
            drawNodes([state.selectedEdge.v, state.selectedEdge.w], 2/this.minimapScale, theme.nodeSelected);
        }

        if (state.searchCandidates.length > 0) {
            drawNodes(state.searchCandidates.map(c => c.node.id), 2/this.minimapScale, theme.nodeSelected);
        }

        if (target) {
            drawNodes([target], 3/ this.minimapScale, theme.nodeSelected);
        }
       
        this.ctx.restore();
        
        const transform = this.viewer.controller.transform;
        const canvasRect = this.viewer.canvasContainer.getBoundingClientRect();
        
        const vx = -transform.x / transform.k;
        const vy = -transform.y / transform.k;
        const vw = canvasRect.width / transform.k;
        const vh = canvasRect.height / transform.k;
        
        const mx = vx * this.minimapScale + this.thumbnailOffset.x;
        const my = vy * this.minimapScale + this.thumbnailOffset.y;
        const mw = vw * this.minimapScale;
        const mh = vh * this.minimapScale;
        
        this.ctx.strokeStyle = theme.minimapBorder;
        this.ctx.lineWidth = 2 * dpr;
        this.ctx.strokeRect(mx, my, mw, mh);
        this.ctx.fillStyle = theme.minimapBox;
        this.ctx.fillRect(mx, my, mw, mh);
    }

    destroy() {
        fxOffAll(this._teardownFns);
    }
}
