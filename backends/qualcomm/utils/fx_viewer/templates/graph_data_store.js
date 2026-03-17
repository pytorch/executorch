/**
 * ============================================================================
 * CLASS: GraphDataStore
 * ============================================================================
 * Manages the raw JSON graph payload and constructs the "Virtual Node" topology.
 * 
 * USE CASES & METHOD CALLS:
 * - Initialization: `new GraphDataStore(payload)` parses the V3 JSON schema, 
 *   separating the base structural graph from the extension overlays.
 * - Dynamic Composition: `computeActiveGraph(activeExtensionIds, colorById)` is 
 *   called whenever the user toggles a UI layer, flattening the nested data into
 *   a fast array of active Virtual Nodes.
 * - Traversal: `getAncestors(id)` / `getDescendants(id)` provides O(1) BFS 
 *   lookups for the Canvas selection highlighting.
 * 
 * VARIABLES & STATE:
 * - `baseData`: Object. Contains the structural nodes and edges.
 * - `extensions`: Object. Maps extension IDs to their respective annotation data.
 * - `activeNodes`: Array. A pre-computed list of flattened node objects used by 
 *   the rendering engine and search engine.
 * - `activeNodeMap`: Map. O(1) lookup map mapping Node ID -> Virtual Node.
 * - `adjList` / `revAdjList`: Maps tracking outgoing/incoming edges.
 * - `graphBounds`: Object. Bounding box of the entire graph for camera zooming.
 * 
 * ALGORITHM & INFO FLOW:
 * 1. Topology Init (`_initTopology`): Loops over `baseData.nodes` once to calculate 
 *    the global bounds. It normalizes coordinates so the top-left node always 
 *    starts at (50, 50), providing a visual buffer. It builds the adjacency lists.
 * 2. Virtual Node Composition (`computeActiveGraph`):
 *    - Loops over `baseData.nodes`.
 *    - For each node, it creates a flat `info` dictionary starting with base meta.
 *    - It iterates over the provided `activeExtensionIds`. If an extension has data
 *      for this node, it prefixes the keys (e.g. `Profiler.latency: 15`) and 
 *      merges them into the flat `info` dictionary. This enables O(1) searching.
 *    - It concatenates `label_append` and `tooltip` arrays.
 *    - Finally, it resolves the `fill_color` strictly based on the `colorById` argument.
 * 
 * USER EXPERIENCE (UX):
 * - By pre-computing the `activeNodes` array whenever a checkbox is clicked, 
 *   the system avoids creating new objects or arrays during the 60FPS Canvas 
 *   `render()` loop. This prevents Garbage Collection stutters, ensuring buttery
 *   smooth pan and zoom even with tens of thousands of nodes.
 * - The prefixed flattening (e.g., `Profiler.latency`) means users can type 
 *   "Profiler" into the search bar and instantly highlight all nodes that have 
 *   profiling data attached.
 * ============================================================================
 */
class GraphDataStore {
    constructor(payload) {
        this.baseData = payload.base;
        this.extensions = payload.extensions || {};

        // The pre-computed array/map of active Virtual Nodes
        this.activeNodes = [];
        this.activeNodeMap = new Map();
        
        // Structural Topology (never changes when toggling extensions)
        this.adjList = new Map();
        this.revAdjList = new Map();
        this.graphBounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, width: 0, height: 0 };
        
        this._initTopology();
    }
    
    _initTopology() {
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

        this.baseData.nodes.forEach(node => {
            if (!node.width) node.width = 100; // Fallback
            if (!node.height) node.height = 40;
            
            minX = Math.min(minX, node.x - node.width/2);
            maxX = Math.max(maxX, node.x + node.width/2);
            minY = Math.min(minY, node.y - node.height/2);
            maxY = Math.max(maxY, node.y + node.height/2);
            
            this.adjList.set(node.id, []);
            this.revAdjList.set(node.id, []);
        });

        const offsetX = 50 - minX;
        const offsetY = 50 - minY;
        
        this.baseData.nodes.forEach(node => {
            node.x += offsetX;
            node.y += offsetY;
        });
        
        this.baseData.edges.forEach(edge => {
            if (edge.points) {
                edge.points.forEach(p => { p.x += offsetX; p.y += offsetY; });
            }
        });

        minX += offsetX; maxX += offsetX; minY += offsetY; maxY += offsetY;
        
        this.graphBounds = {
            minX, maxX, minY, maxY,
            width: maxX - minX + 100,
            height: maxY - minY + 100
        };

        this.baseData.edges.forEach(edge => {
            let eMinX = Infinity, eMaxX = -Infinity, eMinY = Infinity, eMaxY = -Infinity;
            const v = this.baseData.nodes.find(n => n.id === edge.v);
            const w = this.baseData.nodes.find(n => n.id === edge.w);
            
            if (edge.points && edge.points.length > 0) {
                edge.points.forEach(p => {
                    eMinX = Math.min(eMinX, p.x); eMaxX = Math.max(eMaxX, p.x);
                    eMinY = Math.min(eMinY, p.y); eMaxY = Math.max(eMaxY, p.y);
                });
            } else if (v && w) {
                eMinX = Math.min(v.x, w.x); eMaxX = Math.max(v.x, w.x);
                eMinY = Math.min(v.y, w.y); eMaxY = Math.max(v.y, w.y);
            }
            edge.bounds = { minX: eMinX, maxX: eMaxX, minY: eMinY, maxY: eMaxY };
            
            if (this.adjList.has(edge.v)) this.adjList.get(edge.v).push(edge);
            if (this.revAdjList.has(edge.w)) this.revAdjList.get(edge.w).push(edge);
        });
    }

    /**
     * Called whenever the user toggles Extension checkboxes or Color Radio buttons.
     * Rebuilds `activeNodes` by flattening the enabled JSON hierarchies into single Virtual Nodes.
     */
    computeActiveGraph(activeExtensionIds, colorById) {
        this.activeNodes = [];
        this.activeNodeMap.clear();

        this.baseData.nodes.forEach(baseNode => {
            // 1. Initialize with Base Info
            let flatInfo = { ...baseNode.info }; 
            let label_append = [];
            let tooltip = [...(baseNode.tooltip || [])];
            let fill_color = baseNode.fill_color;

            // 2. Iterate through visible extensions
            activeExtensionIds.forEach(extId => {
                const ext = this.extensions[extId];
                if (!ext) return;
                const extNode = ext.nodes[baseNode.id];
                if (!extNode) return;

                // Merge Info with Prefixes (e.g. "Profiler.latency" = 15)
                if (extNode.info) {
                    for (const [k, v] of Object.entries(extNode.info)) {
                        flatInfo[`${ext.name}.${k}`] = v;
                    }
                }
                
                if (extNode.label_append) {
                    label_append.push(...extNode.label_append);
                }
                
                if (extNode.tooltip) {
                    tooltip.push(`[${ext.name}]`);
                    tooltip.push(...extNode.tooltip);
                }
            });

            // 3. Resolve Node Fill Color
            if (colorById !== 'base' && this.extensions[colorById]) {
                const colorNode = this.extensions[colorById].nodes[baseNode.id];
                if (colorNode && colorNode.fill_color) {
                    fill_color = colorNode.fill_color;
                }
            }

            // 4. Cache Virtual Node
            const virtualNode = {
                ...baseNode,
                info: flatInfo,
                label_append: label_append,
                tooltip: tooltip,
                fill_color: fill_color
            };

            this.activeNodes.push(virtualNode);
            this.activeNodeMap.set(virtualNode.id, virtualNode);
        });
    }

    upsertExtension(extensionId, extensionPayload) {
        if (!extensionId) {
            throw new Error("upsertExtension requires a non-empty extensionId");
        }
        if (!extensionPayload || typeof extensionPayload !== 'object') {
            throw new Error(`upsertExtension('${extensionId}') requires an object payload`);
        }

        const previous = this.extensions[extensionId] || {};
        this.extensions[extensionId] = {
            name: extensionPayload.name || previous.name || extensionId,
            legend: Array.isArray(extensionPayload.legend) ? extensionPayload.legend : (previous.legend || []),
            nodes: extensionPayload.nodes && typeof extensionPayload.nodes === 'object'
                ? extensionPayload.nodes
                : (previous.nodes || {}),
        };
    }

    removeExtension(extensionId) {
        if (!extensionId) return;
        delete this.extensions[extensionId];
    }

    setExtensionLabel(extensionId, label) {
        const ext = this.extensions[extensionId];
        if (!ext) return;
        ext.name = label || ext.name;
    }

    patchExtensionNodes(extensionId, patchByNodeId) {
        if (!extensionId || !patchByNodeId || typeof patchByNodeId !== 'object') return;
        const ext = this.extensions[extensionId];
        if (!ext) {
            this.upsertExtension(extensionId, { name: extensionId, nodes: {} });
        }

        const target = this.extensions[extensionId];
        if (!target.nodes) target.nodes = {};

        Object.entries(patchByNodeId).forEach(([nodeId, patch]) => {
            const prev = target.nodes[nodeId] || {};
            const next = { ...prev, ...patch };
            if (patch && patch.info && typeof patch.info === 'object') {
                next.info = { ...(prev.info || {}), ...patch.info };
            }
            if (patch && patch.tooltip && Array.isArray(patch.tooltip)) {
                next.tooltip = patch.tooltip.slice();
            }
            if (patch && patch.label_append && Array.isArray(patch.label_append)) {
                next.label_append = patch.label_append.slice();
            }
            target.nodes[nodeId] = next;
        });
    }

    setExtensionLegend(extensionId, legend) {
        const ext = this.extensions[extensionId];
        if (!ext) return;
        ext.legend = Array.isArray(legend) ? legend : [];
    }

    getAncestors(nodeId) {
        const visited = new Set();
        const queue = [nodeId];
        visited.add(nodeId);
        while (queue.length > 0) {
            const curr = queue.shift();
            const inEdges = this.revAdjList.get(curr) || [];
            inEdges.forEach(e => {
                if (!visited.has(e.v)) {
                    visited.add(e.v);
                    queue.push(e.v);
                }
            });
        }
        return visited;
    }

    getDescendants(nodeId) {
        const visited = new Set();
        const queue = [nodeId];
        visited.add(nodeId);
        while (queue.length > 0) {
            const curr = queue.shift();
            const outEdges = this.adjList.get(curr) || [];
            outEdges.forEach(e => {
                if (!visited.has(e.w)) {
                    visited.add(e.w);
                    queue.push(e.w);
                }
            });
        }
        return visited;
    }
}
