// Fuzzy search over active graph nodes with token scoring and context highlighting.
class SearchEngine {
    constructor(dataStore) {
        this.dataStore = dataStore;
        this._kvScores = {
            exactFieldExactValue: 40,
            exactFieldFuzzyValue: 34,
            fuzzyFieldFuzzyValue: 28,
        };
    }

    search(query) {
        if (!query || query.trim() === '') return [];
        const queryTokens = this._tokenizeQuery(query);
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
                const tokenResult = this._scoreToken(node, token, hOpen, hClose);
                const tokenScore = tokenResult.score;
                const tokenMatchField = tokenResult.matchField;
                const tokenMatchString = tokenResult.matchString;

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
                let stop = false;
                queryTokens.forEach((token) => {
                    const plain = this._stripQuotes(token).toLowerCase();
                    if (!plain) return;
                    if (!stop) highlightedId = this._highlightText(highlightedId, plain, hOpen, hClose);
                    if (highlightedId.includes(hClose)) stop=true;
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

    _tokenizeQuery(query) {
        const tokens = [];
        let current = '';
        let currentQuote = null;

        for (let i = 0; i < query.length; i++) {
            const ch = query[i];
            if (ch === '"' || ch === "'") {
                if(!currentQuote){
                    currentQuote = ch;
                }else if(currentQuote === ch){
                    currentQuote = null;
                }else{
                    current += ch;
                }
                continue;
            }
            if (!currentQuote && /\s/.test(ch)) {
                if (current.trim().length > 0) tokens.push(current.trim());
                current = '';
                continue;
            }
            current += ch;
        }

        if (current.trim().length > 0) tokens.push(current.trim());
        return tokens;
    }

    _stripQuotes(s) {
        return s.replace(/"|'/g, ""); 
    }

    _escapeRegExp(s) {
        return String(s || '').replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
    }

    _escapeHTML(str) {
        return str.replace(/[&<>"']/g, function(m) {
            return {
            '&': '&amp;',
            '<': '&lt;',
            '>': '&gt;',
            '"': '&quot;',
            "'": '&#039;'
            }[m];
        });
    }

    _highlightText(text, pattern, hOpen, hClose) {
        let safeText = this._escapeHTML(text);
        if (!pattern) return safeText;
        const regex = new RegExp(`(${this._escapeRegExp(pattern)})`, 'gi');
        return String(safeText).replace(regex, `${hOpen}$1${hClose}`);
    }

    _valueToString(value) {
        const valueStr = typeof value === 'object' ? JSON.stringify(value) : String(value);
        return valueStr.split(/\r?\n/).join("");
    }

    _formatValueSnippet(actualValStr, pattern, hOpen, hClose) {
        const valStrLower = actualValStr.toLowerCase();
        const idx = valStrLower.indexOf(pattern);
        if (idx === -1) {
            if (actualValStr.length > 30) return `${actualValStr.substring(0, 30)}...`;
            return actualValStr;
        }
        const start = Math.max(0, idx - 15);
        const end = Math.min(actualValStr.length, idx + pattern.length + 15);
        let snippet = actualValStr.substring(start, end);
        snippet = this._highlightText(snippet, pattern, hOpen, hClose);
        return `...${snippet}...`;
    }

    _scoreToken(node, rawToken, hOpen, hClose) {
        const normalizedToken = this._stripQuotes(rawToken).toLowerCase();
        if (!normalizedToken) return { score: 0, matchField: null, matchString: null };

        const firstEq = rawToken.indexOf('=');
        if (firstEq !== -1) {
            const fieldPattern = this._stripQuotes(rawToken.slice(0, firstEq)).toLowerCase();
            const valuePattern = this._stripQuotes(rawToken.slice(firstEq + 1)).toLowerCase();
            if (fieldPattern && valuePattern) {
                const kvResult = this._scoreFieldValueToken(node, fieldPattern, valuePattern, hOpen, hClose);
                if (kvResult.fieldMatched) return kvResult;
            }
            return this._scorePlainToken(node, normalizedToken, hOpen, hClose);
        }

        return this._scorePlainToken(node, normalizedToken, hOpen, hClose);
    }

    _unifyValStr(valStr){
        const valLower = valStr.toLowerCase()
        return this._stripQuotes(valLower.replace(/\s/g, '')); 
    }

    _scoreFieldValueToken(node, fieldPattern, valuePattern, hOpen, hClose) {
        const entries = [['id', node.id], ...Object.entries(node.info || {})];
        let fieldMatched = false;
        let best = { score: 0, matchField: null, matchString: null, fieldMatched: true };

        for (const [key, value] of entries) {
            const keyStr = String(key);
            const keyLower = keyStr.toLowerCase();
            if (!keyLower.includes(fieldPattern)) continue;
            fieldMatched = true;

            const actualValStr = this._valueToString(value);
            const unifiedValStr = this._unifyValStr(actualValStr);
            const unifiedValPattern = this._unifyValStr(valuePattern);
            if (!unifiedValStr.includes(unifiedValPattern)) continue;

            const keyExact = keyLower === fieldPattern;
            const valueExact = unifiedValStr === unifiedValPattern;
            let score = this._kvScores.fuzzyFieldFuzzyValue;
            if (keyExact && valueExact) score = this._kvScores.exactFieldExactValue;
            else if (keyExact) score = this._kvScores.exactFieldFuzzyValue;

            const matchField = this._highlightText(keyStr, fieldPattern, hOpen, hClose);
            const matchString = this._formatValueSnippet(actualValStr, valuePattern, hOpen, hClose);
            if (score > best.score) {
                best = { score, matchField, matchString, fieldMatched: true };
            }
        }

        if (!fieldMatched) return { score: 0, matchField: null, matchString: null, fieldMatched: false };
        return best;
    }

    _scorePlainToken(node, token, hOpen, hClose) {
        if (node.id.toLowerCase().includes(token)) {
            return { score: 10, matchField: 'id', matchString: node.id };
        }

        for (const [key, value] of Object.entries(node.info || {})) {
            const keyStr = String(key);
            const keyLower = keyStr.toLowerCase();
            const actualValStr = this._valueToString(value);
            const unifiedValStr = this._unifyValStr(actualValStr);
            const unifiedToken = this._unifyValStr(token);
            if (!keyLower.includes(unifiedToken) && !unifiedValStr.includes(unifiedToken)) continue;

            let score = 1;
            if (key === 'op') score = 5;
            else if (key === 'target') score = 3;

            if (unifiedValStr.includes(unifiedToken)) {
                return {
                    score,
                    matchField: keyStr,
                    matchString: this._formatValueSnippet(actualValStr, token, hOpen, hClose),
                };
            }

            const highlightedField = this._highlightText(keyStr, token, hOpen, hClose);
            const shortValue = actualValStr.length > 30 ? `${actualValStr.substring(0, 30)}...` : actualValStr;
            return {
                score,
                matchField: highlightedField,
                matchString: shortValue,
            };
        }

        return { score: 0, matchField: null, matchString: null };
    }
}
