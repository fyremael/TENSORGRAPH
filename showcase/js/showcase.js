/**
 * TENSORGRAPH Chrome Metropolis Showcase
 * Interactive Demo Logic
 */

// =============================================================================
// COLLAPSIBLE PANELS
// =============================================================================

function toggleCollapsible(element) {
    element.classList.toggle('open');
}

// =============================================================================
// MODAL SYSTEM
// =============================================================================

function openManifesto() {
    document.getElementById('manifesto-modal').classList.add('open');
}

function closeManifesto(event) {
    if (event.target === event.currentTarget || event.target.classList.contains('modal-close')) {
        document.getElementById('manifesto-modal').classList.remove('open');
    }
}

// =============================================================================
// RULE TOGGLES
// =============================================================================

function toggleRule(element) {
    element.classList.toggle('active');
    const checkbox = element.querySelector('.rule-checkbox');
    checkbox.textContent = element.classList.contains('active') ? '✓' : '';
    updateBoxCount();
}

// =============================================================================
// EXPRESSION PARSER & SIMULATOR
// =============================================================================

/**
 * Parse expression string into a simple AST
 * Supports: f, g, h, id, ;, (, )
 */
function parseExpression(input) {
    const tokens = tokenize(input.trim());
    if (tokens.length === 0) return null;
    return parseSeq(tokens, 0).node;
}

function tokenize(input) {
    const tokens = [];
    let i = 0;
    while (i < input.length) {
        const c = input[i];
        if (c === ' ' || c === '\t' || c === '\n') {
            i++;
        } else if (c === ';' || c === '(' || c === ')') {
            tokens.push({ type: c, value: c });
            i++;
        } else if (/[a-zA-Z_]/.test(c)) {
            let name = '';
            while (i < input.length && /[a-zA-Z0-9_]/.test(input[i])) {
                name += input[i];
                i++;
            }
            tokens.push({ type: 'NAME', value: name });
        } else {
            i++;
        }
    }
    return tokens;
}

function parseSeq(tokens, pos) {
    let result = parseAtom(tokens, pos);
    if (!result.node) return result;

    let node = result.node;
    pos = result.pos;

    while (pos < tokens.length && tokens[pos].type === ';') {
        pos++; // consume ;
        const right = parseAtom(tokens, pos);
        if (!right.node) break;
        node = { type: 'Seq', left: node, right: right.node };
        pos = right.pos;
    }

    return { node, pos };
}

function parseAtom(tokens, pos) {
    if (pos >= tokens.length) return { node: null, pos };

    const token = tokens[pos];

    if (token.type === '(') {
        pos++; // consume (
        const inner = parseSeq(tokens, pos);
        if (inner.pos < tokens.length && tokens[inner.pos].type === ')') {
            return { node: inner.node, pos: inner.pos + 1 };
        }
        return inner;
    }

    if (token.type === 'NAME') {
        if (token.value === 'id') {
            return { node: { type: 'Id' }, pos: pos + 1 };
        }
        return { node: { type: 'Box', op: token.value }, pos: pos + 1 };
    }

    return { node: null, pos };
}

/**
 * Count boxes in AST
 */
function countBoxes(node) {
    if (!node) return 0;
    if (node.type === 'Box') return 1;
    if (node.type === 'Id') return 0;
    if (node.type === 'Seq') return countBoxes(node.left) + countBoxes(node.right);
    if (node.type === 'Par') return countBoxes(node.left) + countBoxes(node.right);
    return 0;
}

/**
 * Pretty print AST
 */
function prettyPrint(node) {
    if (!node) return '(empty)';
    if (node.type === 'Box') return node.op;
    if (node.type === 'Id') return 'id';
    if (node.type === 'Seq') return `${prettyPrint(node.left)} ; ${prettyPrint(node.right)}`;
    if (node.type === 'Par') return `(${prettyPrint(node.left)} ⊗ ${prettyPrint(node.right)})`;
    return '?';
}

// =============================================================================
// SATURATION SIMULATOR
// =============================================================================

/**
 * Simulate saturation with enabled rules
 */
function simulateSaturation(ast, rules, maxIters) {
    const trace = [];
    let current = ast;
    let totalRewrites = 0;

    for (let iter = 0; iter < maxIters; iter++) {
        let changed = false;

        for (const rule of rules) {
            const result = applyRule(current, rule, trace);
            if (result.changed) {
                current = result.node;
                totalRewrites += result.count;
                changed = true;
            }
        }

        if (!changed) {
            trace.push({
                type: 'info',
                message: `Fixed point reached at iteration ${iter + 1}`
            });
            break;
        }
    }

    return { result: current, trace, rewrites: totalRewrites };
}

/**
 * Apply a single rule recursively
 */
function applyRule(node, rule, trace) {
    if (!node) return { node, changed: false, count: 0 };

    let changed = false;
    let count = 0;

    // Try to apply at this node
    const match = rule.match(node);
    if (match) {
        const newNode = rule.apply(node, match);
        trace.push({
            type: 'rewrite',
            rule: rule.name,
            from: prettyPrint(node),
            to: prettyPrint(newNode)
        });
        node = newNode;
        changed = true;
        count++;
    }

    // Recurse into children
    if (node.type === 'Seq') {
        const leftResult = applyRule(node.left, rule, trace);
        const rightResult = applyRule(node.right, rule, trace);
        if (leftResult.changed || rightResult.changed) {
            node = { type: 'Seq', left: leftResult.node, right: rightResult.node };
            changed = true;
            count += leftResult.count + rightResult.count;
        }
    }

    return { node, changed, count };
}

// =============================================================================
// REWRITE RULES
// =============================================================================

const RULES = {
    fuse: {
        name: 'FuseOps',
        match: (node) => {
            // Match: f ; f where both are same Box
            if (node.type === 'Seq' &&
                node.left.type === 'Box' &&
                node.right.type === 'Box' &&
                node.left.op === node.right.op) {
                return { op: node.left.op };
            }
            return null;
        },
        apply: (node, match) => {
            return { type: 'Box', op: match.op };
        }
    },

    assoc: {
        name: 'Assoc',
        match: (node) => {
            // Match: (a ; b) ; c
            if (node.type === 'Seq' && node.left.type === 'Seq') {
                return { a: node.left.left, b: node.left.right, c: node.right };
            }
            return null;
        },
        apply: (node, match) => {
            // Transform to: a ; (b ; c)
            return {
                type: 'Seq',
                left: match.a,
                right: { type: 'Seq', left: match.b, right: match.c }
            };
        }
    },

    identity: {
        name: 'Identity',
        match: (node) => {
            // Match: id ; f or f ; id
            if (node.type === 'Seq') {
                if (node.left.type === 'Id') return { keep: node.right };
                if (node.right.type === 'Id') return { keep: node.left };
            }
            return null;
        },
        apply: (node, match) => {
            return match.keep;
        }
    }
};

// =============================================================================
// UI HANDLERS
// =============================================================================

function updateBoxCount() {
    const input = document.getElementById('input-expr').value;
    const ast = parseExpression(input);
    const boxes = countBoxes(ast);
    document.getElementById('boxes-before').textContent = boxes;
}

function runSaturation() {
    const input = document.getElementById('input-expr').value;
    const iters = parseInt(document.getElementById('param-iters').value) || 10;
    const traceOutput = document.getElementById('trace-output');
    const outputExpr = document.getElementById('output-expr');

    // Parse input
    const ast = parseExpression(input);
    if (!ast) {
        traceOutput.innerHTML = '<div class="trace-entry"><span class="trace-index">[ERR]</span> <span style="color: var(--c-error);">Invalid expression</span></div>';
        return;
    }

    // Get enabled rules
    const enabledRules = [];
    document.querySelectorAll('.rule-item.active').forEach(el => {
        const ruleId = el.dataset.rule;
        if (RULES[ruleId]) enabledRules.push(RULES[ruleId]);
    });

    // Clear trace
    traceOutput.innerHTML = '';

    // Log start
    addTrace(traceOutput, 'info', `Starting saturation with ${enabledRules.length} rules...`);
    addTrace(traceOutput, 'info', `Input: ${prettyPrint(ast)}`);

    const boxesBefore = countBoxes(ast);

    // Simulate saturation with animation
    let currentAST = ast;
    let totalRewrites = 0;
    let iteration = 0;

    function step() {
        if (iteration >= iters) {
            finalize();
            return;
        }

        let changed = false;
        for (const rule of enabledRules) {
            const result = applyRuleOnce(currentAST, rule);
            if (result.changed) {
                currentAST = result.node;
                totalRewrites++;
                changed = true;
                addTrace(traceOutput, 'rewrite', `${rule.name}: ${result.from} → ${result.to}`);

                // Update metrics
                document.getElementById('rewrites-applied').textContent = totalRewrites;
                document.getElementById('iterations').textContent = iteration + 1;

                // Schedule next step
                setTimeout(step, 100);
                return;
            }
        }

        if (!changed) {
            addTrace(traceOutput, 'success', `Fixed point at iteration ${iteration + 1}`);
            finalize();
            return;
        }

        iteration++;
        setTimeout(step, 50);
    }

    function finalize() {
        const boxesAfter = countBoxes(currentAST);
        outputExpr.innerHTML = `<span style="color: var(--c-success);">${prettyPrint(currentAST)}</span>`;
        document.getElementById('boxes-after').textContent = boxesAfter;
        document.getElementById('iterations').textContent = iteration || 1;

        if (boxesAfter < boxesBefore) {
            addTrace(traceOutput, 'success', `✓ Optimized: ${boxesBefore} → ${boxesAfter} boxes`);
        } else {
            addTrace(traceOutput, 'info', `No reduction (${boxesAfter} boxes)`);
        }
    }

    step();
}

function applyRuleOnce(node, rule) {
    if (!node) return { node, changed: false };

    // Try at this node
    const match = rule.match(node);
    if (match) {
        const from = prettyPrint(node);
        const newNode = rule.apply(node, match);
        const to = prettyPrint(newNode);
        return { node: newNode, changed: true, from, to };
    }

    // Try in children
    if (node.type === 'Seq') {
        const leftResult = applyRuleOnce(node.left, rule);
        if (leftResult.changed) {
            return {
                node: { type: 'Seq', left: leftResult.node, right: node.right },
                changed: true,
                from: leftResult.from,
                to: leftResult.to
            };
        }
        const rightResult = applyRuleOnce(node.right, rule);
        if (rightResult.changed) {
            return {
                node: { type: 'Seq', left: node.left, right: rightResult.node },
                changed: true,
                from: rightResult.from,
                to: rightResult.to
            };
        }
    }

    return { node, changed: false };
}

function addTrace(container, type, message) {
    const entry = document.createElement('div');
    entry.className = 'trace-entry';

    let prefix = '[INFO]';
    let color = 'var(--c-silver)';

    if (type === 'rewrite') {
        prefix = '[REWRITE]';
        color = 'var(--c-amber)';
    } else if (type === 'success') {
        prefix = '[OK]';
        color = 'var(--c-success)';
    } else if (type === 'error') {
        prefix = '[ERR]';
        color = 'var(--c-error)';
    }

    entry.innerHTML = `<span class="trace-index">${prefix}</span> <span style="color: ${color};">${message}</span>`;
    container.appendChild(entry);
    container.scrollTop = container.scrollHeight;
}

function resetConsole() {
    document.getElementById('input-expr').value = 'f ; f ; f ; g';
    document.getElementById('trace-output').innerHTML = '<div class="trace-entry"><span class="trace-index">[SYS]</span> <span class="trace-detail">Reset. Ready.</span></div>';
    document.getElementById('output-expr').textContent = 'Optimized expression will appear here...';
    document.getElementById('boxes-before').textContent = '4';
    document.getElementById('boxes-after').textContent = '—';
    document.getElementById('rewrites-applied').textContent = '0';
    document.getElementById('iterations').textContent = '0';

    // Reset all rules to default (fuse and assoc active)
    document.querySelectorAll('.rule-item').forEach(el => {
        const ruleId = el.dataset.rule;
        if (ruleId === 'fuse' || ruleId === 'assoc') {
            el.classList.add('active');
            el.querySelector('.rule-checkbox').textContent = '✓';
        } else {
            el.classList.remove('active');
            el.querySelector('.rule-checkbox').textContent = '';
        }
    });
}

// =============================================================================
// INITIALIZATION
// =============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Update box count on input change
    const inputExpr = document.getElementById('input-expr');
    if (inputExpr) {
        inputExpr.addEventListener('input', updateBoxCount);
        updateBoxCount();
    }

    console.log('TENSORGRAPH Showcase initialized');
});
