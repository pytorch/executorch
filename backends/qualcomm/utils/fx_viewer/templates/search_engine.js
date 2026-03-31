// Fuzzy search over active graph nodes with token scoring and context highlighting.
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
