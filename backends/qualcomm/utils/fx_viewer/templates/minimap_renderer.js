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
        ctx.lineWidth = 1 / this.minimapScale;
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

    render() {
        const dpr = window.devicePixelRatio || 1;
        if (this.canvas.width === 0 || this.canvas.height === 0) return;
        
        const state = this.viewer.controller.state;
        const theme = THEMES[state.themeName];
        
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
        
        const drawNodes = (nodes, padding=0) => {
            nodes.forEach(nid => {
                const node = this.viewer.store.activeNodeMap.get(nid);
                if (node) {
                    this.ctx.fillStyle = node.fill_color ? node.fill_color : theme.nodeFill;
                    const minSize = 3 / this.minimapScale;
                    const w = Math.max(node.width, minSize) + padding;
                    const h = Math.max(node.height, minSize) + padding;
                    this.ctx.fillRect(node.x - w/2, node.y - h/2, w, h);
                }
            });
        };
        
        if (state.searchCandidates.length > 0) {
            drawNodes(state.searchCandidates.map(c => c.node.id), 7 / this.minimapScale);
        }

        if (state.selectedNodeId || state.previewNodeId) {
            if (state.highlightAncestors) {
                drawNodes(Array.from(state.ancestors));
                drawNodes(Array.from(state.descendants));
            }
            const target = state.previewNodeId || state.selectedNodeId;
            drawNodes([target], 7 / this.minimapScale);
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
