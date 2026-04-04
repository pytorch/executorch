// High-performance 2D canvas rendering of the main graph with pan/zoom and hover interactions.
//
// render() pass order:
//   1. Clear canvas, fill theme background.
//   2. Apply DPR scale + camera transform (translate + scale).
//   3. Draw edges (with midpoint tensor-shape labels).
//   4. Draw node rectangles with multi-line labels and interaction-state coloring.
//   5. Highlight group overlay pass — iterates state.highlightGroups; draws a thick solid border
//      outside each node rect with an explicit outer gap from the single-selection border.
//      Multiple groups coexist; last group in Map iteration order wins on overlapping nodes.
//   6. Draw smart tooltip for hovered node or edge.
class CanvasRenderer {
    constructor(container, viewer) {
        this.container = container;
        this.viewer = viewer;
        this._teardownFns = [];
        
        this.canvasContainer = document.createElement('div');
        this.canvasContainer.style.width = '100%';
        this.canvasContainer.style.height = '100%';
        this.container.appendChild(this.canvasContainer);

        this.canvas = document.createElement('canvas');
        this.canvas.className = 'fx-canvas';
        this.canvasContainer.appendChild(this.canvas);
        
        this.ctx = this.canvas.getContext('2d');
        
        this.isDragging = false;
        this.lastMousePos = { x: 0, y: 0 };
        
        this.resize();
        this._onWindowResize = () => this.resize();
        fxOn(this._teardownFns, window, 'resize', this._onWindowResize);
        this._resizeObserver = null;
        if (typeof ResizeObserver !== 'undefined') {
            this._resizeObserver = new ResizeObserver(() => this.resize());
            this._resizeObserver.observe(this.canvasContainer);
        }
        this.setupEvents();
    }
    
    resize() {
        const dpr = window.devicePixelRatio || 1;
        const rect = this.canvasContainer.getBoundingClientRect();
        if (rect.width <= 0 || rect.height <= 0) return;
        this.canvas.width = rect.width * dpr;
        this.canvas.height = rect.height * dpr;
        this.viewer.renderAll();
    }

    destroy() {
        if (this._resizeObserver) {
            this._resizeObserver.disconnect();
            this._resizeObserver = null;
        }
        fxOffAll(this._teardownFns);
    }
    
    setupEvents() {
        const onMouseDown = (e) => {
            this.isDragging = true;
            this.dragMoved = false;
            this.lastMousePos = { x: e.clientX, y: e.clientY };
        };
        fxOn(this._teardownFns, this.canvas, 'mousedown', onMouseDown);
        
        const onMouseMove = (e) => {
            if (this.isDragging) {
                const dx = e.clientX - this.lastMousePos.x;
                const dy = e.clientY - this.lastMousePos.y;
                if (Math.abs(dx) > 2 || Math.abs(dy) > 2) {
                    this.dragMoved = true;
                }
                this.viewer.controller.transform.x += dx;
                this.viewer.controller.transform.y += dy;
                this.lastMousePos = { x: e.clientX, y: e.clientY };
                this.viewer.renderAll();
            } else {
                const rect = this.canvas.getBoundingClientRect();
                const mouseX = e.clientX - rect.left;
                const mouseY = e.clientY - rect.top;
                
                const transform = this.viewer.controller.transform;
                const graphX = (mouseX - transform.x) / transform.k;
                const graphY = (mouseY - transform.y) / transform.k;
                
                this.detectHover(graphX, graphY);
            }
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
            
            const rect = this.canvas.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;

            const transform = this.viewer.controller.transform;
            const graphX = (mouseX - transform.x) / transform.k;
            const graphY = (mouseY - transform.y) / transform.k;

            transform.k *= zoomFactor;
            transform.x = mouseX - graphX * transform.k;
            transform.y = mouseY - graphY * transform.k;
            
            this.viewer.renderAll();
        };
        fxOn(this._teardownFns, this.canvas, 'wheel', onWheel, { passive: false });
        
        const onClick = (e) => {
            if (this.dragMoved) return;
            const state = this.viewer.controller.state;
            this.viewer.controller.handleClick(state.hoveredNodeId, state.hoveredEdge);
        };
        fxOn(this._teardownFns, this.canvas, 'click', onClick);
    }

    detectHover(graphX, graphY) {
        let nearestNode = null;
        let nearestEdge = null;
        
        for (let i = 0; i < this.viewer.store.baseData.nodes.length; i++) {
            const node = this.viewer.store.baseData.nodes[i];
            const w = node.width;
            const h = node.height;
            if (graphX >= node.x - w/2 && graphX <= node.x + w/2 &&
                graphY >= node.y - h/2 && graphY <= node.y + h/2) {
                nearestNode = node.id;
                break; 
            }
        }
        
        if (!nearestNode) {
            const transform = this.viewer.controller.transform;
            const hoverDist = 5 / transform.k;
            for (let i = 0; i < this.viewer.store.baseData.edges.length; i++) {
                const edge = this.viewer.store.baseData.edges[i];
                if (!edge.bounds) continue;
                if (graphX < edge.bounds.minX - hoverDist || graphX > edge.bounds.maxX + hoverDist ||
                    graphY < edge.bounds.minY - hoverDist || graphY > edge.bounds.maxY + hoverDist) continue;

                const v = this.viewer.store.activeNodeMap.get(edge.v);
                const w = this.viewer.store.activeNodeMap.get(edge.w);
                let min_d = Infinity;
                if (edge.points && edge.points.length > 0) {
                    for (let j = 0; j < edge.points.length - 1; j++) {
                        const d = this.distToSegment({x: graphX, y: graphY}, edge.points[j], edge.points[j+1]);
                        min_d = Math.min(min_d, d);
                    }
                } else if (v && w) {
                    min_d = this.distToSegment({x: graphX, y: graphY}, v, w);
                }
                if (min_d <= hoverDist) {
                    nearestEdge = edge;
                    break;
                }
            }
        }

        this.viewer.controller.handleHover(nearestNode, nearestEdge);
    }
    
    distToSegment(p, v, w) {
        const l2 = (v.x - w.x)**2 + (v.y - w.y)**2;
        if (l2 === 0) return Math.hypot(p.x - v.x, p.y - v.y);
        let t = ((p.x - v.x) * (w.x - v.x) + (p.y - v.y) * (w.y - v.y)) / l2;
        t = Math.max(0, Math.min(1, t));
        return Math.hypot(p.x - (v.x + t * (w.x - v.x)), p.y - (v.y + t * (w.y - v.y)));
    }

    render() {
        const dpr = window.devicePixelRatio || 1;
        const ctx = this.ctx;
        const transform = this.viewer.controller.transform;
        const state = this.viewer.controller.state;
        const theme = THEMES[state.themeName];
        const edgeLineWidths = {
            normal: 2,
            input: 4,
            output: 4,
            hover: 5,
        };
        const nodeBorderWidths = {
            selected: 5,
            preview: 5,
            hover: 5,
        };
        const hoverDashPattern = [8, 6];
        const groupBorderWidth = 5;
        const groupBorderGap = 5;
        
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.fillStyle = theme.bg;
        ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);
        
        ctx.save();
        ctx.scale(dpr, dpr);
        ctx.translate(transform.x, transform.y);
        ctx.scale(transform.k, transform.k);
        
        const inNodes = new Set();
        const outNodes = new Set();

        // nodes under selection or hovering
        const activeNodes = [state.previewNodeId, state.selectedNodeId, state.hoveredNodeId]
        activeNodes.forEach(
            (activeNode) => {
                (this.viewer.store.revAdjList.get(activeNode) || []).forEach(e => inNodes.add(e.v));
                (this.viewer.store.adjList.get(activeNode) || []).forEach(e => outNodes.add(e.w));
            }
        )

        const isSelectionMode = !!state.selectedNodeId || !!state.previewNodeId || !!state.selectedEdge;
        
        this.viewer.store.baseData.edges.forEach(edge => {
            const v = this.viewer.store.activeNodeMap.get(edge.v);
            const w = this.viewer.store.activeNodeMap.get(edge.w);
            if (!v || !w) return;

            let opacity = 1.0;
            if (isSelectionMode) {
                const targetNode = state.previewNodeId || state.selectedNodeId;
                const isSelectedNodeEdge = (targetNode && (edge.v === targetNode || edge.w === targetNode)) || state.selectedEdge === edge;
                if (state.highlightAncestors) {
                    const inAncestors = state.ancestors.has(edge.v) && state.ancestors.has(edge.w);
                    const inDescendants = state.descendants.has(edge.v) && state.descendants.has(edge.w);
                    if (!inAncestors && !inDescendants && !isSelectedNodeEdge) {
                        opacity = 0.15;
                    }
                }
                // If highlightAncestors is false, opacity remains 1.0 for all edges
            }

            const isHovered = state.hoveredEdge === edge || state.selectedEdge === edge;
            const isInputEdge = activeNodes.includes(edge.w);
            const isOutputEdge = activeNodes.includes(edge.v);
            
            if (isHovered) {
                ctx.strokeStyle = theme.edgeHover;
                ctx.globalAlpha = opacity;
                ctx.lineWidth = edgeLineWidths.hover;
            } else if (isInputEdge) {
                ctx.strokeStyle = theme.edgeInput;
                ctx.globalAlpha = opacity;
                ctx.lineWidth = edgeLineWidths.input;
            } else if (isOutputEdge) {
                ctx.strokeStyle = theme.edgeOutput;
                ctx.globalAlpha = opacity;
                ctx.lineWidth = edgeLineWidths.output;
            } else {
                ctx.strokeStyle = theme.edgeNormal;
                ctx.globalAlpha = opacity;
                ctx.lineWidth = edgeLineWidths.normal;
            }
            
            ctx.beginPath();
            let midX = 0, midY = 0;
            if (edge.points && edge.points.length > 0) {
                ctx.moveTo(edge.points[0].x, edge.points[0].y);
                for (let i = 1; i < edge.points.length; i++) {
                    ctx.lineTo(edge.points[i].x, edge.points[i].y);
                }
                const midIdx = Math.floor(edge.points.length / 2);
                midX = edge.points[midIdx].x;
                midY = edge.points[midIdx].y;
                if (edge.points.length % 2 === 0 && midIdx > 0) {
                    midX = (edge.points[midIdx].x + edge.points[midIdx-1].x) / 2;
                    midY = (edge.points[midIdx].y + edge.points[midIdx-1].y) / 2;
                }
            } else {
                ctx.moveTo(v.x, v.y);
                ctx.lineTo(w.x, w.y);
                midX = (v.x + w.x) / 2;
                midY = (v.y + w.y) / 2;
            }
            ctx.stroke();
            ctx.globalAlpha = 1.0;

            const srcNode = v;
            if (srcNode && srcNode.info && srcNode.info.tensor_shape) {
                let shapeStr = JSON.stringify(srcNode.info.tensor_shape).replace(/"/g, '');
                let dtypeStr = typeof srcNode.info.dtype === 'string' ? ` [${srcNode.info.dtype.replace('torch.', '')}]` : '';
                let label = `${shapeStr}${dtypeStr}`;
                
                ctx.font = '10px sans-serif';
                const tw = ctx.measureText(label).width;
                const th = 12;
                
                ctx.globalAlpha = Math.max(opacity, 0.8);
                ctx.fillStyle = theme.bg;
                ctx.fillRect(midX - tw/2 - 2, midY - th/2 - 2, tw + 4, th + 4);
                
                ctx.fillStyle = isHovered ? theme.edgeHover : theme.textMuted;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(label, midX, midY);
                ctx.globalAlpha = 1.0;
            }
        });

        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';

        // Helper to lighten/darken hex colors dynamically based on active theme
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

        this.viewer.store.activeNodes.forEach(node => {
            const isHovered = node.id === state.hoveredNodeId;
            const isSelected = node.id === state.selectedNodeId;
            const isPreview = node.id === state.previewNodeId;
            const isInput = inNodes.has(node.id);
            const isOutput = outNodes.has(node.id);

            let opacity = 1.0;
            let isEdgeEndpoint = state.selectedEdge && (state.selectedEdge.v === node.id || state.selectedEdge.w === node.id);

            if (isSelectionMode) {
                const targetNode = state.previewNodeId || state.selectedNodeId;
                if (state.highlightAncestors) {
                    const isAncestors = state.ancestors.has(node.id);
                    const isDescendants = state.descendants.has(node.id);
                    if (!isAncestors && !isDescendants && node.id !== targetNode && !isEdgeEndpoint) {
                        opacity = 0.15;
                    }
                }
            }
            
            ctx.globalAlpha = opacity;
            
            // Determine base fill color (either custom extension color or theme default)
            let baseColor = node.fill_color ? node.fill_color : theme.nodeFill;
            
            // Adjust lightness for Dark Mode to ensure custom colors aren't too bright
            if (state.themeName === 'dark' && node.fill_color) {
                baseColor = shadeColor(baseColor, -20);
            }

            // Apply interaction state coloring dynamically instead of overriding with theme defaults
            if (isSelected || isPreview || isEdgeEndpoint) {
                ctx.fillStyle = shadeColor(baseColor, state.themeName === 'dark' ? 30 : 20);
                ctx.globalAlpha = Math.max(opacity, 0.8);
            } else if (isHovered) {
                ctx.fillStyle = shadeColor(baseColor, state.themeName === 'dark' ? 20 : 20);
            } else if (isInput) {
                ctx.fillStyle = shadeColor(baseColor, state.themeName === 'dark' ? 10 : 10);
            } else if (isOutput) {
                ctx.fillStyle = shadeColor(baseColor, state.themeName === 'dark' ? 10 : 10);
            } else {
                ctx.fillStyle = baseColor;
            }

            ctx.fillRect(node.x - node.width/2, node.y - node.height/2, node.width, node.height);
            
            if (isSelected || isPreview || isHovered) {
                ctx.strokeStyle = theme.edgeHover;
                if (isSelected || isPreview) {
                    ctx.lineWidth = isSelected ? nodeBorderWidths.selected : nodeBorderWidths.preview;
                } else {
                    ctx.lineWidth = nodeBorderWidths.hover;
                }
                if (isHovered && !isSelected && !isPreview) {
                    ctx.setLineDash(hoverDashPattern);
                } else {
                    ctx.setLineDash([]);
                }
                ctx.strokeRect(node.x - node.width/2, node.y - node.height/2, node.width, node.height);
                ctx.setLineDash([]);
            }
            
            ctx.fillStyle = theme.text;
            let allLines = [node.label || node.id];
            if (node.label_append && node.label_append.length > 0) {
                allLines = allLines.concat(node.label_append);
            }
            
            const lineHeight = 16;
            const startY = node.y - ((allLines.length - 1) * lineHeight) / 2;
            
            for (let i = 0; i < allLines.length; i++) {
                if (i === 0) ctx.font = 'bold 14px sans-serif';
                else ctx.font = '12px sans-serif';
                ctx.fillText(allLines[i], node.x, startY + (i * lineHeight));
            }

            ctx.globalAlpha = 1.0;
        });

        // Highlight group overlay pass — drawn after all nodes
        const singleBorderHalf = Math.max(
            nodeBorderWidths.selected,
            nodeBorderWidths.preview,
            nodeBorderWidths.hover
        ) / 2;
        const groupHalf = groupBorderWidth / 2;
        const outerOffset = singleBorderHalf + groupBorderGap + groupHalf;
        if (state.highlightGroups && state.highlightGroups.size > 0) {
            state.highlightGroups.forEach(({ nodeIds, color }) => {
                ctx.strokeStyle = color;
                ctx.lineWidth = groupBorderWidth;
                ctx.setLineDash([]);
                nodeIds.forEach((id) => {
                    const node = this.viewer.store.activeNodeMap.get(id);
                    if (!node) return;
                    ctx.strokeRect(
                        node.x - node.width / 2 - outerOffset,
                        node.y - node.height / 2 - outerOffset,
                        node.width + outerOffset * 2,
                        node.height + outerOffset * 2
                    );
                });
            });
        }

        if (state.hoveredNodeId || state.hoveredEdge) {
            this.drawSmartTooltip(ctx, state.hoveredNodeId, state.hoveredEdge);
        }

        ctx.restore();
    }

    drawSmartTooltip(ctx, hoveredNodeId, hoveredEdge) {
        const theme = THEMES[this.viewer.controller.state.themeName];
        let tooltipLines = [];
        let groupNodes = [];
        let targetX = 0;
        let targetY = 0;

        if (hoveredNodeId) {
            const node = this.viewer.store.activeNodeMap.get(hoveredNodeId);
            if (!node) return;
            targetX = node.x;
            targetY = node.y;
            groupNodes.push(node);
            
            (this.viewer.store.revAdjList.get(hoveredNodeId) || []).forEach(e => {
                const n = this.viewer.store.activeNodeMap.get(e.v);
                if (n) groupNodes.push(n);
            });
            (this.viewer.store.adjList.get(hoveredNodeId) || []).forEach(e => {
                const n = this.viewer.store.activeNodeMap.get(e.w);
                if (n) groupNodes.push(n);
            });

            if (node.tooltip && node.tooltip.length > 0) {
                tooltipLines.push(...node.tooltip);
            }
        } else if (hoveredEdge) {
            const srcNode = this.viewer.store.activeNodeMap.get(hoveredEdge.v);
            const dstNode = this.viewer.store.activeNodeMap.get(hoveredEdge.w);
            if (!srcNode || !dstNode) return;
            groupNodes.push(srcNode, dstNode);
            
            if (hoveredEdge.points && hoveredEdge.points.length > 0) {
                const midIdx = Math.floor(hoveredEdge.points.length / 2);
                if (hoveredEdge.points.length % 2 === 0 && midIdx > 0) {
                    targetX = (hoveredEdge.points[midIdx].x + hoveredEdge.points[midIdx-1].x) / 2;
                    targetY = (hoveredEdge.points[midIdx].y + hoveredEdge.points[midIdx-1].y) / 2;
                } else {
                    targetX = hoveredEdge.points[midIdx].x;
                    targetY = hoveredEdge.points[midIdx].y;
                }
            } else {
                targetX = (srcNode.x + dstNode.x) / 2;
                targetY = (srcNode.y + dstNode.y) / 2;
            }

            if (srcNode.info && srcNode.info.tensor_shape) {
                tooltipLines.push(`Shape: ${JSON.stringify(srcNode.info.tensor_shape).replace(/"/g, '')}`);
            }
            if (srcNode.info && srcNode.info.dtype && typeof srcNode.info.dtype === "string") {
                tooltipLines.push(`Dtype: ${srcNode.info.dtype.replace('torch.', '')}`);
            }
        }

        if (tooltipLines.length === 0 || groupNodes.length === 0) return;

        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
        groupNodes.forEach(n => {
            minX = Math.min(minX, n.x - n.width/2);
            maxX = Math.max(maxX, n.x + n.width/2);
            minY = Math.min(minY, n.y - n.height/2);
            maxY = Math.max(maxY, n.y + n.height/2);
        });

        const transform = this.viewer.controller.transform;
        const dpr = window.devicePixelRatio || 1;

        const fontSize = 12 / transform.k;
        ctx.font = `bold ${fontSize}px sans-serif`;
        let maxW = 0;
        tooltipLines.forEach(line => {
            maxW = Math.max(maxW, ctx.measureText(line).width);
        });
        
        const padding = 8 / transform.k;
        const tw = maxW + padding * 2;
        const lineHeight = 16 / transform.k;
        const th = (tooltipLines.length * lineHeight) + padding * 2;

        const viewLeft = -transform.x / transform.k;
        const viewTop = -transform.y / transform.k;
        const viewRight = viewLeft + (this.canvas.width / dpr) / transform.k;
        const viewBottom = viewTop + (this.canvas.height / dpr) / transform.k;

        const margin = 20 / transform.k;
        
        const candidates = [
            { id: 'up', x: targetX - tw/2, y: minY - margin - th },
            { id: 'left', x: minX - margin - tw, y: targetY - th/2 },
            { id: 'right', x: maxX + margin, y: targetY - th/2 },
            { id: 'down', x: targetX - tw/2, y: maxY + margin }
        ];

        let bestCand = null;
        let minD = Infinity;
        let rightCand = null;

        let validCandidates = candidates.filter(c => 
            c.x >= viewLeft && (c.x + tw) <= viewRight &&
            c.y >= viewTop && (c.y + th) <= viewBottom
        );

        if (validCandidates.length === 0) validCandidates = candidates;

        validCandidates.forEach(c => {
            const cx = c.x + tw/2;
            const cy = c.y + th/2;
            const d = Math.hypot(cx - targetX, cy - targetY);
            c.distance = d;
            
            if (c.id === 'right') {
                rightCand = c;
            }

            if (d < minD) {
                minD = d;
                bestCand = c;
            }
        });

        if (rightCand && rightCand.distance <= minD * 10) {
            bestCand = rightCand;
        }

        let tooltipX = bestCand.x;
        let tooltipY = bestCand.y;

        let lineStartX = targetX;
        let lineStartY = targetY;
        if (hoveredNodeId) {
            const node = this.viewer.store.activeNodeMap.get(hoveredNodeId);
            if (node) {
                if (bestCand.id === 'right') lineStartX = node.x + node.width / 2;
                else if (bestCand.id === 'left') lineStartX = node.x - node.width / 2;
                else if (bestCand.id === 'up') lineStartY = node.y - node.height / 2;
                else if (bestCand.id === 'down') lineStartY = node.y + node.height / 2;
            }
        }

        ctx.strokeStyle = theme.edgeHover;
        ctx.lineWidth = 2 / transform.k;
        ctx.setLineDash([5 / transform.k, 5 / transform.k]);
        ctx.beginPath();
        ctx.moveTo(lineStartX, lineStartY);
        if (bestCand.id === 'up') {
            ctx.lineTo(tooltipX + tw/2, tooltipY + th);
        } else if (bestCand.id === 'down') {
            ctx.lineTo(tooltipX + tw/2, tooltipY);
        } else if (bestCand.id === 'left') {
            ctx.lineTo(tooltipX + tw, tooltipY + th/2);
        } else { // right
            ctx.lineTo(tooltipX, tooltipY + th/2);
        }
        ctx.stroke();
        ctx.setLineDash([]);

        ctx.fillStyle = theme.uiBg;
        ctx.fillRect(tooltipX, tooltipY, tw, th);
        ctx.strokeStyle = theme.uiBorder;
        ctx.lineWidth = 1 / transform.k;
        ctx.strokeRect(tooltipX, tooltipY, tw, th);
        
        ctx.fillStyle = theme.text;
        ctx.textAlign = 'left';
        ctx.textBaseline = 'top';
        tooltipLines.forEach((line, idx) => {
            ctx.fillText(line, tooltipX + padding, tooltipY + padding + idx * lineHeight);
        });
    }
}
