/**
 * ============================================================================
 * CLASS: FXGraphViewer (The Application Facade)
 * ============================================================================
 * This class is the top-level orchestration layer for the entire application.
 * It is responsible for DOM generation, module initialization, layout resizing,
 * and exposing the public API.
 * 
 * USE CASES & METHOD CALLS:
 * 1. Instantiation: `const viewer = new FXGraphViewer('container-id', json_data);`
 * 2. Initialization: `viewer.init();` -> triggers initial drawing and zoom.
 * 3. External Control: `viewer.selectNode('node_name');` -> pans and highlights a node.
 * 4. External Control: `viewer.search('query');` -> fills the search bar and executes search.
 * 5. Re-rendering: `viewer.renderAll();` -> triggers a paint cycle on all canvases.
 * 
 * RELATED VARIABLES & STATE:
 * - `containerId`: String ID of the host HTML element.
 * - `wrapper`, `mainArea`, `sidebar`, `resizer`, `resizerH`: Dynamic HTML DOM 
 *   elements created to establish the layout.
 * - `store`, `searchEngine`, `controller`: Core logical sub-modules.
 * - `canvasRenderer`, `minimapRenderer`, `ui`: Pure view sub-modules.
 * 
 * INFO FLOW & ALGORITHMS:
 * - DOM Building (constructor): It begins by injecting a heavy `<style>` block 
 *   if it doesn't already exist. It then builds a Flexbox layout:
 *   [ Main Area (Flex:1) ] | [ Vertical Resizer (6px) ] | [ Sidebar (300px) ]
 *   It then instantiates the sub-modules, passing them references to their 
 *   respective container elements.
 * 
 * - Resizer Drag Logic (setupResizer):
 *   To allow users to resize the right panel, it listens to `mousedown` on the 
 *   `resizer` div to flip an `isResizing` boolean flag.
 *   On `mousemove` (attached to window to catch fast drags), it calculates the 
 *   new sidebar width: `containerRect.right - e.clientX`.
 *   It clamps this width between 150px and the `container width - 200px` to 
 *   prevent the UI from breaking.
 *   Crucially, every drag tick calls `this.canvasRenderer.resize()`, 
 *   `this.minimapRenderer.resize()`, and `renderAll()` so the canvas resolution 
 *   stays perfectly 1:1 with the DOM pixel size, avoiding any stretching/blur.
 * 
 * - Sidebar Collapse: 
 *   Double-clicking the resizer toggles a `.collapsed` CSS class on the sidebar,
 *   instantly hiding it (display: none). Register a requetsAnimationFrame callback
 *   that forces a canvas resize before next screen redraw that reflect `.collapsed` 
 *   to ensure the main graph expands to fill the newly freed space.
 * 
 * USER EXPERIENCE (UX):
 * - The developer UX is seamless: one class instantiation handles thousands of 
 *   lines of logic without touching external HTML files.
 * - The user experiences a professional split-pane desktop-like interface. They 
 *   can effortlessly drag the partition to see more of the graph or more of the 
 *   metadata, and double click to banish the sidebar completely for fullscreen viewing.
 * ============================================================================
 */
class FXGraphViewer {
    constructor(containerId, data) {
        this.containerId = containerId;
        const container = document.getElementById(containerId);
        if (!container) throw new Error(`Container ${containerId} not found`);
        
        container.innerHTML = '';
        if (!document.getElementById('fx-viewer-styles')) {
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
                
                .fx-taskbar { position: absolute; top: 10px; left: 10px; right: 10px; height: 40px; border-radius: 4px; display: flex; align-items: center; padding: 0 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); z-index: 10; border: 1px solid transparent; overflow: visible; }
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
            `;
            document.head.appendChild(style);
        }
        
        this.wrapper = document.createElement('div');
        this.wrapper.className = 'fx-viewer-wrapper';
        container.appendChild(this.wrapper);
        
        this.mainArea = document.createElement('div');
        this.mainArea.className = 'fx-main-area';
        this.wrapper.appendChild(this.mainArea);
        this.canvasContainer = this.mainArea;
        
        this.resizer = document.createElement('div');
        this.resizer.className = 'fx-resizer';
        this.resizer.title = "Drag to resize sidebar. Double click to toggle.";
        this.wrapper.appendChild(this.resizer);
        
        this.sidebar = document.createElement('div');
        this.sidebar.className = 'fx-sidebar';
        this.wrapper.appendChild(this.sidebar);
        
        this.store = new GraphDataStore(data);
        this.searchEngine = new SearchEngine(this.store);
        this.controller = new ViewerController(this);
        
        this.canvasRenderer = new CanvasRenderer(this.mainArea, this);
        this.ui = new UIManager(this.mainArea, this);
        
        this.resizerH = document.createElement('div');
        this.resizerH.className = 'fx-resizer-h';
        this.resizerH.title = "Drag to resize minimap height.";
        this.sidebar.appendChild(this.resizerH);
        
        this.minimapRenderer = new MinimapRenderer(this.sidebar, this);
        
        this.setupResizer();
    }
    
    setupResizer() {
        let isResizing = false;
        let isResizingH = false;
        
        this.resizer.addEventListener('mousedown', (e) => {
            isResizing = true;
            this.resizer.classList.add('dragging');
            document.body.style.cursor = 'col-resize';
            e.preventDefault();
        });
        
        this.resizerH.addEventListener('mousedown', (e) => {
            isResizingH = true;
            this.resizerH.classList.add('dragging');
            document.body.style.cursor = 'row-resize';
            e.preventDefault();
        });
        
        window.addEventListener('mousemove', (e) => {
            if (isResizing) {
                const containerRect = this.wrapper.getBoundingClientRect();
                let newWidth = containerRect.right - e.clientX;
                newWidth = Math.max(150, Math.min(newWidth, containerRect.width - 200));
                this.sidebar.style.width = newWidth + 'px';
                
                this.canvasRenderer.resize();
                this.minimapRenderer.resize();
                this.minimapRenderer.generateThumbnail();
                this.renderAll();
            } else if (isResizingH) {
                const containerRect = this.wrapper.getBoundingClientRect();
                let newHeight = containerRect.bottom - e.clientY;
                newHeight = Math.max(100, Math.min(newHeight, containerRect.height - 100));
                this.minimapRenderer.container.style.height = newHeight + 'px';
                
                this.minimapRenderer.resize();
                this.minimapRenderer.generateThumbnail();
                this.renderAll();
            }
        });
        
        window.addEventListener('mouseup', () => {
            if (isResizing) {
                isResizing = false;
                this.resizer.classList.remove('dragging');
                document.body.style.cursor = '';
            }
            if (isResizingH) {
                isResizingH = false;
                this.resizerH.classList.remove('dragging');
                document.body.style.cursor = '';
            }
        });
        
        this.resizer.addEventListener('dblclick', () => {
            this.sidebar.classList.toggle('collapsed');
            requestAnimationFrame(() => {
            this.canvasRenderer.resize();
            this.renderAll();
            });
        });
    }
    
    init() {
        this.minimapRenderer.generateThumbnail();
        
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
    
    selectNode(nodeId) {
        this.controller.selectNode(nodeId);
        this.controller.panToNode(nodeId);
    }
    
    search(query) {
        this.ui.searchInput.value = query;
        this.controller.handleSearch(query);
    }
}
