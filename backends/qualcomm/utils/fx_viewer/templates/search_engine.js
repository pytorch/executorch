
/**
 * ============================================================================
 * CLASS: SearchEngine
 * ============================================================================
 * A dedicated utility class providing fuzzy searching over the graph data.
 * 
 * USE CASES & METHOD CALLS:
 * - Search Querying: `SearchEngine.search(query)` is called whenever the user 
 *   types into the taskbar search input.
 * 
 * VARIABLES & STATE:
 * - `dataStore`: Reference to the `GraphDataStore`. This allows the SearchEngine 
 *   to query against `dataStore.activeNodes`, meaning it inherently searches 
 *   across both base data and ALL currently enabled Extension data.
 * 
 * ALGORITHM & INFO FLOW:
 * 1. Tokenization: Splits the query by spaces into `queryTokens` (e.g., "conv 15ms" 
 *    becomes ["conv", "15ms"]).
 * 2. Active Node Traversal: Iterates over the pre-computed `activeNodes` array.
 *    Because `node.info` is a flattened dictionary containing prefixes (e.g. 
 *    `Profiler.latency: "15ms"`), the engine simply iterates through all key-value
 *    pairs natively without needing to know about extensions.
 * 3. Scoring: 
 *    - `node.id` matches get +10 points.
 *    - `op` matches get +5.
 *    - `target` matches get +3.
 *    - Any other key or value match in `node.info` gets +1 point.
 * 4. Context Highlighting: When a token matches a value in `node.info`, it extracts 
 *    a substring around the match and wraps it in a `<span style="background: yellow">`
 *    tag so the user sees exactly *why* a node matched.
 * 5. Filtering: Calculates `maxMatchedTokensCount`. If the user types 3 keywords, 
 *    it heavily prioritizes nodes that match all 3, filtering out nodes that only 
 *    matched 1 or 2 (acting as a fuzzy AND filter).
 * 
 * USER EXPERIENCE (UX):
 * - By searching against the dynamically prefixed `node.info` dictionary, the 
 *   search engine instantly becomes "Extension-aware". Users can search for a 
 *   specific memory bandwidth value or quantization scale, and the engine will 
 *   highlight the result just like native PyTorch data.
 * - The context highlighting prevents confusion when searching deeply nested 
 *   properties, as the dropdown explicitly shows the matching snippet.
 * ============================================================================
 */
class SearchEngine {
    constructor(dataStore) {
        this.dataStore = dataStore;
    }

    search(query) {
        if (!query || query.trim() === '') return [];
        const queryTokens = query.toLowerCase().split(/\s+/).filter(t => t.length > 0);
        if (queryTokens.length === 0) return [];
        
        const results = [];
        const hOpen = `<span style="background-color: #ffeb3b; color: #000; font-weight: bold; padding: 0 2px; border-radius: 2px;">`;
        const hClose = `</span>`;

        let maxMatchedTokensCount = 0;

        this.dataStore.activeNodes.forEach(node => {
            let totalScore = 0;
            let matchedTokensCount = 0;
            let bestMatchField = null;
            let bestMatchString = null;
            let highestTokenScore = 0;

            queryTokens.forEach(token => {
                let tokenScore = 0;
                let tokenMatchField = null;
                let tokenMatchString = null;

                if (node.id.toLowerCase().includes(token)) {
                    tokenScore += 10;
                    tokenMatchField = 'id';
                    tokenMatchString = node.id;
                } else if (node.info) {
                    for (const [key, value] of Object.entries(node.info)) {
                        const keyStr = String(key).toLowerCase();
                        let valStr = typeof value === 'object' ? JSON.stringify(value).toLowerCase() : String(value).toLowerCase();

                        if (keyStr.includes(token) || valStr.includes(token)) {
                            // Weight base properties slightly higher
                            if (key === 'op') tokenScore += 5;
                            else if (key === 'target') tokenScore += 3;
                            else tokenScore += 1;
                            
                            const actualValStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
                            
                            if (valStr.includes(token)) {
                                // matching value
                                tokenMatchField = String(key);
                                const idx = valStr.indexOf(token);
                                const start = Math.max(0, idx - 15);
                                const end = Math.min(actualValStr.length, idx + token.length + 15);
                                let snippet = actualValStr.substring(start, end);
                                const regex = new RegExp(`(${token})`, 'gi');
                                snippet = snippet.replace(regex, `${hOpen}$1${hClose}`);
                                tokenMatchString = "..." + snippet + "...";
                            } else {
                                // matching key
                                tokenMatchField = String(key);
                                const regex = new RegExp(`(${token})`, 'gi');
                                tokenMatchField = tokenMatchField.replace(regex, `${hOpen}$1${hClose}`);
                                tokenMatchString = actualValStr;
                                if (tokenMatchString.length > 30) {
                                    tokenMatchString = tokenMatchString.substring(0, 30) + "...";
                                }
                            }
                            break;
                        }
                    }
                }

                if (tokenScore > 0) {
                    totalScore += tokenScore;
                    matchedTokensCount += 1;
                    if (tokenScore > highestTokenScore) {
                        highestTokenScore = tokenScore;
                        bestMatchField = tokenMatchField;
                        bestMatchString = tokenMatchString;
                    }
                }
            });

            if (totalScore > 0) {
                if (matchedTokensCount > maxMatchedTokensCount) {
                    maxMatchedTokensCount = matchedTokensCount;
                }

                let highlightedId = node.id;
                queryTokens.forEach(token => {
                    const regex = new RegExp(`(${token})`, 'gi');
                    highlightedId = highlightedId.replace(regex, `${hOpen}$1${hClose}`);
                });

                results.push({ 
                    node, 
                    score: totalScore,
                    matchedTokensCount: matchedTokensCount,
                    matchField: bestMatchField, 
                    matchString: bestMatchString,
                    highlightedId: highlightedId
                });
            }
        });

        const filteredResults = results.filter(r => r.matchedTokensCount === maxMatchedTokensCount);
        filteredResults.sort((a, b) => b.score - a.score);
        return filteredResults;
    }
}
