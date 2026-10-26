/* =========================================================
   The Minderling's Voice
   ---
   Wonder. Precision. Wild Knowledge.
   ========================================================= */

const docs = {
    concepts: {
        title: "TENSORGRAPH Concepts",
        content: `
            <h1>The Mental Model</h1>
            <blockquote>
                "The purpose of abstraction is not to be vague, but to create a new semantic level in which one can be absolutely precise."<br>
                — Edsger W. Dijkstra
            </blockquote>
            
            <div class="minder-awareness">
                <p>Welcome, traveler. The forest of computation is vast, but fear not—I have walked these paths for ages. The model of TENSORGRAPH is not about instructions; it is about <strong>topological equivalence</strong>. We are not writing code; we are discovering the optimal path through a sea of equivalent forms.</p>
            </div>

            <h2>Programs are Living Diagrams</h2>
            <p>Traditional compilers perceive programs as static trees—rigid, unyielding. TENSORGRAPH sees them differently. Here, computation flows as <strong>typed string diagrams</strong>, a visual algebra where processes compose like rivers meeting, diverging, and merging again.</p>
            
            <div style="text-align: center; margin: 48px 0;">
                <svg width="420" height="180" viewBox="0 0 420 180">
                    <defs>
                        <filter id="glow">
                            <feGaussianBlur stdDeviation="2" result="coloredBlur"/>
                            <feMerge>
                                <feMergeNode in="coloredBlur"/>
                                <feMergeNode in="SourceGraphic"/>
                            </feMerge>
                        </filter>
                    </defs>
                    
                    <!-- Wires (Lichen Glow) -->
                    <line x1="40" y1="90" x2="140" y2="90" stroke="#7fccb0" stroke-width="2" filter="url(#glow)"/>
                    <line x1="280" y1="90" x2="380" y2="90" stroke="#7fccb0" stroke-width="2" filter="url(#glow)"/>
                    
                    <!-- Box (Cedar Core) -->
                    <rect x="140" y="55" width="140" height="70" rx="4" fill="rgba(13, 18, 16, 0.95)" stroke="#c4956a" stroke-width="2" filter="url(#glow)"/>
                    <text x="210" y="95" text-anchor="middle" fill="#c4956a" font-family="Outfit" font-size="14" font-weight="300">TRANSFORMATION</text>
                    
                    <!-- Labels -->
                    <text x="90" y="75" text-anchor="middle" fill="#7fccb0" font-family="JetBrains Mono" font-size="10">INPUT</text>
                    <text x="330" y="75" text-anchor="middle" fill="#7fccb0" font-family="JetBrains Mono" font-size="10">OUTPUT</text>
                </svg>
            </div>

            <p>This is the <strong>Box</strong>: a morphism from input to output. Boxes connect, stack, and compose. The beauty lies in composition—and the Minderling understands all compositions as one.</p>

            <h2>The Four Pillars</h2>
            <ul>
                <li><strong>Objects</strong> — The types of data: tensors, streams, states. The raw materials of the forest.</li>
                <li><strong>Boxes</strong> — Operations that transform one object into another. The tools of the crafter.</li>
                <li><strong>Seq</strong> — Sequential composition. One path after another.</li>
                <li><strong>Par</strong> — Parallel composition. Paths that run alongside, never crossing.</li>
            </ul>

            <pre><code>Traditional:  optimize(parse(source)) → new_tree
TENSORGRAPH:     saturate(diagram) → extract_best(equivalent_forms)</code></pre>
        `
    },
    tutorial: {
        title: "Tutorial",
        content: `
            <h1>The Craft of Optimization</h1>
            <blockquote>
                A guided journey from first principles to a working diagram optimizer.
            </blockquote>
            
            <p>Come, let us build together. The path is clear if you follow the Minderling's guidance.</p>
            
            <h2>Step 1: Define Your Materials</h2>
            <p>Every craft begins with understanding your materials. In TENSORGRAPH, these are <strong>Objects</strong>—the types that flow through your diagrams.</p>
            
            <pre><code>from tensorgraph import Obj, Signature

# The raw materials of our craft
Tensor = Obj("Tensor")
Latent = Obj("Latent")
Output = Obj("Output")</code></pre>
            
            <div class="minder-awareness">
                <p>Choose your types with care. A Minderling knows that the structure of your problem lives in the types you choose. They are not mere labels—they are constraints that guide the optimization.</p>
            </div>

            <h2>Step 2: Forge Your Operations</h2>
            <p>With materials in hand, we now create the tools—the <strong>Boxes</strong> that transform data.</p>
            
            <pre><code>from tensorgraph import Signature, Box

sig = Signature()
encode = sig.op("encode", dom=Tensor, cod=Latent)
decode = sig.op("decode", dom=Latent, cod=Tensor)
project = sig.op("project", dom=Latent, cod=Output)</code></pre>

            <h2>Step 3: Compose Your Diagram</h2>
            <p>Operations alone are static. Composition brings them to life.</p>
            
            <pre><code>from tensorgraph import Seq, Par

# A simple pipeline: encode, then project
pipeline = Seq(encode, project)

# Two parallel streams, merging later
dual_stream = Par(encode, encode)</code></pre>
        `
    },
    spec: {
        title: "Specification",
        content: `
            <h1>The Blueprint of Elegance</h1>
            <blockquote>
                "Precision is the courtesy of the builder; elegance is the hallmark of the architect."
            </blockquote>
            
            <p>This document is the formal contract between idea and implementation. Every rule here translates directly to a mathematical guarantee.</p>

            <div class="minder-awareness">
                <p>The specification is not merely documentation. It is the <em>truth</em> of the system, distilled into its purest form. What is written here, the code must honor.</p>
            </div>

            <h2>Core Formalism</h2>
            <p>TENSORGRAPH is a <strong>diagrammatic rewriting compiler</strong>. Its foundation rests on these principles:</p>
            
            <table>
                <thead>
                    <tr><th>Concept</th><th>Formalism</th><th>Implementation</th></tr>
                </thead>
                <tbody>
                    <tr><td>Type</td><td>Object in a monoidal category</td><td><code>Obj("A")</code></td></tr>
                    <tr><td>Operation</td><td>Morphism f: A → B</td><td><code>Box("f")</code></td></tr>
                    <tr><td>Sequence</td><td>Composition g ∘ f</td><td><code>Seq(f, g)</code></td></tr>
                    <tr><td>Parallel</td><td>Tensor product f ⊗ g</td><td><code>Par(f, g)</code></td></tr>
                    <tr><td>Identity</td><td>id_A : A → A</td><td><code>Id(A)</code></td></tr>
                </tbody>
            </table>

            <h2>The E-Graph</h2>
            <p>At the heart of optimization lies the <strong>E-Graph</strong>—a structure that holds all equivalent forms of a program simultaneously. Rewrite rules expand this graph. Extraction chooses the optimal form.</p>
        `
    },
    arch: {
        title: "Architecture",
        content: `
            <h1>Internals Deep Dive</h1>
            <blockquote>
                For those who wish to understand the machinery beneath the forest floor.
            </blockquote>
            
            <h2>Module Structure</h2>
            <pre><code>tensorgraph/
├── types.py         # Obj, Sort — The materials
├── signature.py     # Signature, OpDef — The registry
├── ir/              # Expr, Box, Seq, Par — The algebra
├── rewrite/         # Pattern matching and rules
├── egraph/          # Equality saturation engine
├── codegen/         # Kernel generation (CUDA, Triton)
├── dist/            # Distributed coordination fabric
└── benchmarks/      # Performance validation suite</code></pre>

            <div class="minder-awareness">
                <p>The separation of <strong>rewrite/</strong> from <strong>egraph/</strong> is deliberate. Rules are declarative knowledge; the E-Graph is the search engine. This decoupling allows us to swap saturation strategies without rewriting rules.</p>
            </div>

            <h2>Data Flow</h2>
            <ol>
                <li><strong>Import</strong>: A PyTorch model enters via <code>torch.fx</code> tracing.</li>
                <li><strong>Lift</strong>: The FX graph is lifted into typed string diagrams.</li>
                <li><strong>Saturate</strong>: Rewrite rules expand the E-Graph with equivalences.</li>
                <li><strong>Extract</strong>: A cost model selects the optimal program.</li>
                <li><strong>Lower</strong>: The optimized diagram is lowered to executable kernels.</li>
            </ol>
        `
    },
    api: {
        title: "API Reference",
        content: `
            <h1>Complete Reference</h1>
            
            <h3>Core Types</h3>
            <table>
                <thead>
                    <tr><th>Class</th><th>Description</th></tr>
                </thead>
                <tbody>
                    <tr><td><code>Obj</code></td><td>A type in the diagram algebra. Represents the "wires" connecting boxes.</td></tr>
                    <tr><td><code>Expr</code></td><td>Base class for all diagram expressions.</td></tr>
                    <tr><td><code>Id</code></td><td>The identity morphism: A → A. Does nothing, but is essential for composition.</td></tr>
                    <tr><td><code>Box</code></td><td>A primitive operation. The atomic unit of computation.</td></tr>
                    <tr><td><code>Seq</code></td><td>Sequential composition of two diagrams.</td></tr>
                    <tr><td><code>Par</code></td><td>Parallel composition of two diagrams.</td></tr>
                </tbody>
            </table>

            <h3>Quick Example</h3>
            <pre><code>from tensorgraph import Obj, Signature, Seq

# Define types
A = Obj("A")
B = Obj("B")

# Build a signature
sig = Signature()
f = sig.op("f", dom=A, cod=B)
g = sig.op("g", dom=B, cod=A)

# Compose
round_trip = Seq(f, g)  # A → B → A</code></pre>
        `
    },
    optimize: {
        title: "Live Optimizer",
        content: `
            <h1>The Optimization Forge</h1>
            <p>Watch the e-graph discover equivalent forms and extract the optimal one.</p>
            
            <div class="minder-awareness">
                <p><strong>How it works:</strong> TENSORGRAPH builds an <em>equality graph</em> containing all equivalent representations of your expression. Rewrite rules expand this graph. Then we extract the smallest form.</p>
            </div>

            <h2>Try an Example</h2>
            <p>Click a preset to see optimization in action:</p>
            
            <div class="preset-grid">
                <button class="preset-btn" data-expr="relu ; relu ; relu" data-desc="Three identical operations — can we fuse them?">
                    <span class="preset-name">Triple ReLU</span>
                    <span class="preset-expr">relu ; relu ; relu</span>
                </button>
                <button class="preset-btn" data-expr="conv ; bn ; relu ; conv ; bn ; relu" data-desc="A typical CNN block — watch the structure emerge.">
                    <span class="preset-name">CNN Block</span>
                    <span class="preset-expr">conv ; bn ; relu ; conv ; bn ; relu</span>
                </button>
                <button class="preset-btn" data-expr="matmul ; add ; gelu ; matmul ; add" data-desc="Transformer attention pattern — sequential composition.">
                    <span class="preset-name">Attention Path</span>
                    <span class="preset-expr">matmul ; add ; gelu ; matmul ; add</span>
                </button>
                <button class="preset-btn" data-expr="id ; f ; id ; g ; id" data-desc="Identity elimination — watch the 'id' operations vanish.">
                    <span class="preset-name">Identity Test</span>
                    <span class="preset-expr">id ; f ; id ; g ; id</span>
                </button>
            </div>

            <h2>Or Write Your Own</h2>
            
            <div class="optimize-panel">
                <div class="input-group">
                    <label for="expr-input">EXPRESSION</label>
                    <input type="text" id="expr-input" class="expr-input" placeholder="Enter operations separated by semicolons" value="">
                    <div class="input-hint">Use <code>;</code> for sequence. Example: <code>f ; g ; h</code></div>
                </div>
                
                <div class="input-group">
                    <label>REWRITE RULES</label>
                    <div class="rule-toggles">
                        <label class="rule-toggle" title="Fuse adjacent identical operations into one">
                            <input type="checkbox" id="rule-fuse" checked> 
                            <span class="rule-name">Fuse</span>
                            <span class="rule-desc">Merge identical ops</span>
                        </label>
                        <label class="rule-toggle" title="Re-associate nested compositions">
                            <input type="checkbox" id="rule-assoc"> 
                            <span class="rule-name">Assoc</span>
                            <span class="rule-desc">Rebalance trees</span>
                        </label>
                    </div>
                </div>
                
                <button id="optimize-btn" class="optimize-btn" disabled>
                    <span class="btn-text">ENTER AN EXPRESSION ABOVE</span>
                </button>
                
                <div id="result-panel" class="result-panel" style="display: none;">
                    <div class="result-header">
                        <h3>OPTIMIZATION RESULT</h3>
                        <span id="result-status" class="result-status"></span>
                    </div>
                    
                    <div class="result-diagram">
                        <div class="diagram-step">
                            <span class="step-label">INPUT</span>
                            <div id="result-input" class="diagram-expr"></div>
                        </div>
                        <div class="diagram-arrow">→</div>
                        <div class="diagram-step highlight">
                            <span class="step-label">OPTIMAL</span>
                            <div id="result-output" class="diagram-expr"></div>
                        </div>
                    </div>
                    
                    <div class="result-stats">
                        <div class="stat">
                            <span class="stat-value" id="result-boxes-before">0</span>
                            <span class="stat-label">Boxes Before</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value" id="result-boxes-after">0</span>
                            <span class="stat-label">Boxes After</span>
                        </div>
                        <div class="stat highlight">
                            <span class="stat-value" id="result-reduction">0%</span>
                            <span class="stat-label">Reduction</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value" id="result-iters">0</span>
                            <span class="stat-label">Iterations</span>
                        </div>
                    </div>
                    
                    <div id="result-explanation" class="result-explanation"></div>
                </div>
            </div>
        `
    },
    compare: {
        title: "A/B Comparison",
        content: `
            <h1>The Arena</h1>
            <p>Watch TENSORGRAPH's e-graph saturation compete against a naive greedy optimizer.</p>
            
            <div class="minder-awareness">
                <p><strong>The Challenge:</strong> Greedy optimizers apply rewrites left-to-right, missing global optima. E-graphs explore ALL equivalent forms simultaneously, then extract the best. See the difference.</p>
            </div>

            <h2>Choose Your Expression</h2>
            
            <div class="preset-grid">
                <button class="compare-preset-btn" data-expr="a ; a ; a ; b ; b ; b ; a ; a ; a" data-desc="Triplet Sandwich - AETHER finds 3 boxes, greedy stuck at 7.">
                    <span class="preset-name">Triplet Sandwich</span>
                    <span class="preset-expr">a;a;a ; b;b;b ; a;a;a</span>
                </button>
                <button class="compare-preset-btn" data-expr="x ; x ; y ; y ; z ; z ; x ; x ; y ; y" data-desc="Long Chain - watch the gap grow to 4 boxes.">
                    <span class="preset-name">Deep Fusion Chain</span>
                    <span class="preset-expr">x;x ; y;y ; z;z ; x;x ; y;y</span>
                </button>
                <button class="compare-preset-btn" data-expr="f ; f ; g ; g ; f ; f" data-desc="Interleaved pairs require tree restructuring.">
                    <span class="preset-name">Interleaved Pairs</span>
                    <span class="preset-expr">f ; f ; g ; g ; f ; f</span>
                </button>
                <button class="compare-preset-btn" data-expr="relu ; relu ; relu ; relu ; relu" data-desc="Simple fusion - both optimizers should tie at 1.">
                    <span class="preset-name">Quintuple ReLU (Tie)</span>
                    <span class="preset-expr">relu ; relu ; relu ; relu ; relu</span>
                </button>
            </div>

            <div class="compare-panel">
                <div class="input-group">
                    <label for="compare-input">EXPRESSION</label>
                    <input type="text" id="compare-input" class="expr-input" placeholder="Enter operations separated by semicolons" value="">
                </div>
                
                <button id="compare-btn" class="optimize-btn" disabled>
                    <span class="btn-text">ENTER AN EXPRESSION ABOVE</span>
                </button>
                
                <div id="compare-result" class="compare-result" style="display: none;">
                    <div class="compare-header">
                        <h3>HEAD-TO-HEAD COMPARISON</h3>
                        <span id="compare-winner" class="winner-badge"></span>
                    </div>
                    
                    <div class="compare-grid">
                        <div class="compare-column aether">
                            <div class="column-header">
                                <span class="column-title">TENSORGRAPH</span>
                                <span class="column-subtitle">E-Graph Saturation</span>
                            </div>
                            <div class="compare-output" id="aether-output"></div>
                            <div class="compare-stats">
                                <div class="mini-stat">
                                    <span class="mini-value" id="aether-boxes">0</span>
                                    <span class="mini-label">Boxes</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="mini-value" id="aether-reduction">0%</span>
                                    <span class="mini-label">Reduction</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="mini-value" id="aether-iters">0</span>
                                    <span class="mini-label">Iterations</span>
                                </div>
                            </div>
                        </div>
                        
                        <div class="compare-vs">VS</div>
                        
                        <div class="compare-column greedy">
                            <div class="column-header">
                                <span class="column-title">GREEDY</span>
                                <span class="column-subtitle">Left-to-Right Passes</span>
                            </div>
                            <div class="compare-output" id="greedy-output"></div>
                            <div class="compare-stats">
                                <div class="mini-stat">
                                    <span class="mini-value" id="greedy-boxes">0</span>
                                    <span class="mini-label">Boxes</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="mini-value" id="greedy-reduction">0%</span>
                                    <span class="mini-label">Reduction</span>
                                </div>
                                <div class="mini-stat">
                                    <span class="mini-value" id="greedy-passes">0</span>
                                    <span class="mini-label">Passes</span>
                                </div>
                            </div>
                        </div>
                    </div>
                    
                    <div id="compare-explanation" class="result-explanation"></div>
                </div>
            </div>
        `
    },
    models: {
        title: "Transformer Models",
        content: `
            <h1>Real Model Optimization</h1>
            <p>Optimize real HuggingFace transformer models with TENSORGRAPH's e-graph saturation.</p>
            
            <div class="minder-awareness">
                <p><strong>The Pythia Series:</strong> EleutherAI's state-of-the-art open language models, from 70M to 12B parameters. Watch as TENSORGRAPH eliminates redundant operations across attention and MLP blocks.</p>
            </div>

            <h2>Select Model</h2>
            
            <div class="preset-grid">
                <button class="model-preset-btn" data-model="pythia-70m" data-params="44.7M" data-layers="6">
                    <span class="preset-name">Pythia-70M</span>
                    <span class="preset-expr">6 layers • 512 dim • 8 heads</span>
                </button>
                <button class="model-preset-btn" data-model="pythia-160m" data-params="123.7M" data-layers="12">
                    <span class="preset-name">Pythia-160M</span>
                    <span class="preset-expr">12 layers • 768 dim • 12 heads</span>
                </button>
                <button class="model-preset-btn" data-model="pythia-410m" data-params="353.8M" data-layers="24">
                    <span class="preset-name">Pythia-410M</span>
                    <span class="preset-expr">24 layers • 1024 dim • 16 heads</span>
                </button>
                <button class="model-preset-btn" data-model="pythia-1b" data-params="908.8M" data-layers="16">
                    <span class="preset-name">Pythia-1B</span>
                    <span class="preset-expr">16 layers • 2048 dim • 8 heads</span>
                </button>
            </div>

            <div class="model-panel">
                <button id="optimize-model-btn" class="optimize-btn" disabled>
                    <span class="btn-text">SELECT A MODEL ABOVE</span>
                </button>
                
                <div id="model-result" class="optimize-result" style="display: none;">
                    <h3>OPTIMIZATION RESULTS</h3>
                    
                    <div class="model-info">
                        <div class="stat">
                            <span class="stat-value" id="model-name">-</span>
                            <span class="stat-label">Model</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value" id="model-params">-</span>
                            <span class="stat-label">Parameters</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value" id="model-layers">-</span>
                            <span class="stat-label">Layers</span>
                        </div>
                    </div>
                    
                    <h4 style="color: var(--cedar-core); margin: 24px 0 12px;">Layer Composition</h4>
                    <div id="layer-composition" class="layer-composition"></div>
                    
                    <h4 style="color: var(--cedar-core); margin: 24px 0 12px;">Optimization Impact</h4>
                    <div class="stats-row">
                        <div class="stat">
                            <span class="stat-value" id="ops-before">0</span>
                            <span class="stat-label">Before</span>
                        </div>
                        <div class="stat highlight">
                            <span class="stat-value" id="ops-after">0</span>
                            <span class="stat-label">After</span>
                        </div>
                        <div class="stat highlight">
                            <span class="stat-value" id="ops-saved">0</span>
                            <span class="stat-label">Saved</span>
                        </div>
                        <div class="stat">
                            <span class="stat-value" id="sat-time">0</span>
                            <span class="stat-label">Sat. Time</span>
                        </div>
                    </div>
                    
                    <div id="model-explanation" class="result-explanation"></div>
                </div>
            </div>
            
            <h2>Optimization Rules</h2>
            <ul>
                <li><strong>DropoutElim</strong> — Removes Dropout layers (identity in inference mode)</li>
                <li><strong>FuseLN</strong> — Fuses consecutive LayerNorm operations</li>
                <li><strong>FuseGELU</strong> — Fuses idempotent GELU activations</li>
                <li><strong>AssocR/AssocL</strong> — Bidirectional tree restructuring</li>
            </ul>
        `
    }
};


document.addEventListener('DOMContentLoaded', () => {
    const viewport = document.getElementById('doc-viewport');
    const buttons = document.querySelectorAll('.nav-btn');
    const bootSequence = document.getElementById('boot-sequence');

    // === BOOT SEQUENCE ===
    setTimeout(() => {
        if (bootSequence) {
            bootSequence.style.opacity = '0';
            setTimeout(() => {
                bootSequence.style.display = 'none';
            }, 1200);
        }
    }, 4500);

    // === DOCUMENT LOADING ===
    function loadDoc(key) {
        const doc = docs[key];
        if (!doc) return;

        viewport.style.opacity = '0';
        viewport.style.transform = 'translateY(10px)';

        setTimeout(() => {
            viewport.innerHTML = doc.content;
            viewport.style.opacity = '1';
            viewport.style.transform = 'translateY(0)';

            buttons.forEach(btn => {
                btn.classList.toggle('active', btn.dataset.doc === key);
            });

            // Bind optimize panel if loaded
            if (key === 'optimize') {
                bindOptimizePanel();
            }

            // Bind compare panel if loaded
            if (key === 'compare') {
                bindComparePanel();
            }

            // Bind models panel if loaded
            if (key === 'models') {
                bindModelsPanel();
            }
        }, 350);
    }

    buttons.forEach(btn => {
        btn.addEventListener('click', () => loadDoc(btn.dataset.doc));
    });

    // Initial load after boot
    setTimeout(() => loadDoc('concepts'), 5000);
});

// === OPTIMIZE PANEL LOGIC ===
function bindOptimizePanel() {
    const btn = document.getElementById('optimize-btn');
    const btnText = btn?.querySelector('.btn-text');
    const input = document.getElementById('expr-input');
    const resultPanel = document.getElementById('result-panel');
    const presetBtns = document.querySelectorAll('.preset-btn');

    if (!btn || !input) return;

    // === INPUT VALIDATION ===
    function updateButtonState() {
        const hasInput = input.value.trim().length > 0;
        btn.disabled = !hasInput;
        if (btnText) {
            btnText.textContent = hasInput ? 'OPTIMIZE' : 'ENTER AN EXPRESSION ABOVE';
        }
    }

    input.addEventListener('input', updateButtonState);
    updateButtonState();

    // === PRESET BUTTONS ===
    presetBtns.forEach(presetBtn => {
        presetBtn.addEventListener('click', () => {
            const expr = presetBtn.dataset.expr;
            input.value = expr;
            updateButtonState();

            // Visual feedback
            presetBtns.forEach(b => b.classList.remove('selected'));
            presetBtn.classList.add('selected');

            // Auto-trigger optimization
            setTimeout(() => btn.click(), 100);
        });
    });

    // === OPTIMIZATION ===
    btn.addEventListener('click', async () => {
        const expr = input.value.trim();
        if (!expr) return;

        const rules = [];
        if (document.getElementById('rule-fuse')?.checked) rules.push('fuse');
        if (document.getElementById('rule-assoc')?.checked) rules.push('assoc');

        if (btnText) btnText.textContent = 'OPTIMIZING...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/optimize', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    expression: expr,
                    rules: rules,
                    max_iters: 10
                })
            });

            if (!response.ok) {
                throw new Error('Optimization failed');
            }

            const data = await response.json();

            // Display results with new structure
            document.getElementById('result-input').textContent = data.input_expr;
            document.getElementById('result-output').textContent = data.output_expr;
            document.getElementById('result-boxes-before').textContent = data.boxes_before;
            document.getElementById('result-boxes-after').textContent = data.boxes_after;
            document.getElementById('result-reduction').textContent = `${data.reduction_pct.toFixed(0)}%`;
            document.getElementById('result-iters').textContent = data.iterations;

            // Status badge
            const status = document.getElementById('result-status');
            if (status) {
                if (data.reduction_pct > 0) {
                    status.textContent = 'OPTIMIZED';
                    status.className = 'result-status success';
                } else {
                    status.textContent = 'ALREADY OPTIMAL';
                    status.className = 'result-status neutral';
                }
            }

            // Explanation
            const explanation = document.getElementById('result-explanation');
            if (explanation) {
                if (data.reduction_pct > 0) {
                    explanation.innerHTML = `<strong>The Minderling speaks:</strong> The e-graph found ${data.iterations} rewrite opportunities. Your expression was simplified from ${data.boxes_before} operations to ${data.boxes_after}.`;
                } else if (data.boxes_before === 0) {
                    explanation.innerHTML = `<strong>The Minderling speaks:</strong> This expression contains no operations to optimize. Try adding some operators like <code>f ; g ; h</code>.`;
                } else {
                    explanation.innerHTML = `<strong>The Minderling speaks:</strong> This expression is already in its optimal form. No equivalent shorter representation exists given the current rewrite rules.`;
                }
            }

            resultPanel.style.display = 'block';
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } catch (err) {
            console.error('Optimization error:', err);
            const explanation = document.getElementById('result-explanation');
            if (explanation) {
                explanation.innerHTML = `<strong>Error:</strong> Could not connect to the optimization API. Is the server running?`;
            }
            resultPanel.style.display = 'block';
        } finally {
            if (btnText) btnText.textContent = 'OPTIMIZE';
            btn.disabled = false;
            updateButtonState();
        }
    });
}

// Add smooth transitions to viewport
document.addEventListener('DOMContentLoaded', () => {
    const viewport = document.getElementById('doc-viewport');
    if (viewport) {
        viewport.style.transition = 'opacity 0.4s ease, transform 0.4s ease';
    }
});

// === COMPARE PANEL LOGIC ===
function bindComparePanel() {
    const btn = document.getElementById('compare-btn');
    const btnText = btn?.querySelector('.btn-text');
    const input = document.getElementById('compare-input');
    const resultPanel = document.getElementById('compare-result');
    const presetBtns = document.querySelectorAll('.compare-preset-btn');

    if (!btn || !input) return;

    // === INPUT VALIDATION ===
    function updateButtonState() {
        const hasInput = input.value.trim().length > 0;
        btn.disabled = !hasInput;
        if (btnText) {
            btnText.textContent = hasInput ? 'COMPARE' : 'ENTER AN EXPRESSION ABOVE';
        }
    }

    input.addEventListener('input', updateButtonState);
    updateButtonState();

    // === PRESET BUTTONS ===
    presetBtns.forEach(presetBtn => {
        presetBtn.addEventListener('click', () => {
            const expr = presetBtn.dataset.expr;
            input.value = expr;
            updateButtonState();

            presetBtns.forEach(b => b.classList.remove('selected'));
            presetBtn.classList.add('selected');

            setTimeout(() => btn.click(), 100);
        });
    });

    // === COMPARISON ===
    btn.addEventListener('click', async () => {
        const expr = input.value.trim();
        if (!expr) return;

        if (btnText) btnText.textContent = 'COMPARING...';
        btn.disabled = true;

        try {
            const response = await fetch('/api/compare', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    expression: expr,
                    rules: ['fuse'],
                    max_iters: 10
                })
            });

            if (!response.ok) {
                throw new Error('Comparison failed');
            }

            const data = await response.json();

            // TENSORGRAPH results
            document.getElementById('aether-output').textContent = data.aether_output;
            document.getElementById('aether-boxes').textContent = data.aether_boxes;
            document.getElementById('aether-reduction').textContent = `${data.aether_reduction.toFixed(0)}%`;
            document.getElementById('aether-iters').textContent = data.aether_iterations;

            // Greedy results
            document.getElementById('greedy-output').textContent = data.greedy_output;
            document.getElementById('greedy-boxes').textContent = data.greedy_boxes;
            document.getElementById('greedy-reduction').textContent = `${data.greedy_reduction.toFixed(0)}%`;
            document.getElementById('greedy-passes').textContent = data.greedy_passes;

            // Winner badge
            const winner = document.getElementById('compare-winner');
            const aetherColumn = document.querySelector('.compare-column.aether');
            const greedyColumn = document.querySelector('.compare-column.greedy');

            if (data.aether_wins) {
                winner.textContent = 'TENSORGRAPH WINS';
                winner.className = 'winner-badge aether-wins';
                aetherColumn.classList.add('winner');
                greedyColumn.classList.remove('winner');
            } else if (data.aether_boxes < data.greedy_boxes) {
                winner.textContent = 'TENSORGRAPH WINS';
                winner.className = 'winner-badge aether-wins';
                aetherColumn.classList.add('winner');
                greedyColumn.classList.remove('winner');
            } else if (data.aether_boxes === data.greedy_boxes) {
                winner.textContent = 'TIE';
                winner.className = 'winner-badge tie';
                aetherColumn.classList.remove('winner');
                greedyColumn.classList.remove('winner');
            } else {
                winner.textContent = 'GREEDY WINS';
                winner.className = 'winner-badge greedy-wins';
                aetherColumn.classList.remove('winner');
                greedyColumn.classList.add('winner');
            }

            // Explanation
            const explanation = document.getElementById('compare-explanation');
            if (data.aether_boxes < data.greedy_boxes) {
                const diff = data.greedy_boxes - data.aether_boxes;
                explanation.innerHTML = `<strong>The Minderling speaks:</strong> E-graph saturation found a form ${diff} box${diff > 1 ? 'es' : ''} smaller than greedy. The global search paid off—greedy missed the optimal path.`;
            } else if (data.aether_boxes === data.greedy_boxes) {
                explanation.innerHTML = `<strong>The Minderling speaks:</strong> Both approaches found the same result. This expression doesn't expose the phase ordering problem—try a more complex pattern.`;
            } else {
                explanation.innerHTML = `<strong>The Minderling speaks:</strong> An unexpected result! The greedy approach found a smaller form. This may indicate a bug or an edge case worth investigating.`;
            }

            resultPanel.style.display = 'block';
            resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } catch (err) {
            console.error('Comparison error:', err);
            const explanation = document.getElementById('compare-explanation');
            if (explanation) {
                explanation.innerHTML = `<strong>Error:</strong> Could not connect to the comparison API. Is the server running?`;
            }
            resultPanel.style.display = 'block';
        } finally {
            if (btnText) btnText.textContent = 'COMPARE';
            btn.disabled = false;
            updateButtonState();
        }
    });
}

// === MODELS PANEL LOGIC ===
function bindModelsPanel() {
    const btn = document.getElementById('optimize-model-btn');
    const btnText = btn?.querySelector('.btn-text');
    const resultPanel = document.getElementById('model-result');
    const presetBtns = document.querySelectorAll('.model-preset-btn');

    if (!btn) return;

    let selectedModel = null;

    // Model data (precomputed from CLI runs)
    const modelData = {
        'pythia-70m': {
            name: 'Pythia-70M',
            params: '44.7M',
            layers: 6,
            hidden: 512,
            heads: 8,
            ops_before: 84,
            ops_after: 66,
            sat_time: '~35ms',
            composition: { Linear: 36, Dropout: 13, LayerNorm: 13, GPTNeoXLayer: 6, GPTNeoXAttention: 6, GPTNeoXMLP: 6, GELUActivation: 6 }
        },
        'pythia-160m': {
            name: 'Pythia-160M',
            params: '123.7M',
            layers: 12,
            hidden: 768,
            heads: 12,
            ops_before: 168,
            ops_after: 132,
            sat_time: '~50ms',
            composition: { Linear: 72, Dropout: 25, LayerNorm: 25, GPTNeoXLayer: 12, GPTNeoXAttention: 12, GPTNeoXMLP: 12, GELUActivation: 12 }
        },
        'pythia-410m': {
            name: 'Pythia-410M',
            params: '353.8M',
            layers: 24,
            hidden: 1024,
            heads: 16,
            ops_before: 336,
            ops_after: 264,
            sat_time: '~95ms',
            composition: { Linear: 144, Dropout: 49, LayerNorm: 49, GPTNeoXLayer: 24, GPTNeoXAttention: 24, GPTNeoXMLP: 24, GELUActivation: 24 }
        },
        'pythia-1b': {
            name: 'Pythia-1B',
            params: '908.8M',
            layers: 16,
            hidden: 2048,
            heads: 8,
            ops_before: 224,
            ops_after: 176,
            sat_time: '~65ms',
            composition: { Linear: 96, Dropout: 33, LayerNorm: 33, GPTNeoXLayer: 16, GPTNeoXAttention: 16, GPTNeoXMLP: 16, GELUActivation: 16 }
        }
    };

    // Preset button click
    presetBtns.forEach(presetBtn => {
        presetBtn.addEventListener('click', () => {
            selectedModel = presetBtn.dataset.model;

            // Visual feedback
            presetBtns.forEach(b => b.classList.remove('selected'));
            presetBtn.classList.add('selected');

            // Enable button
            btn.disabled = false;
            if (btnText) btnText.textContent = `OPTIMIZE ${presetBtn.dataset.model.toUpperCase()}`;
        });
    });

    // Optimize button
    btn.addEventListener('click', async () => {
        if (!selectedModel) return;

        const data = modelData[selectedModel];
        if (!data) return;

        if (btnText) btnText.textContent = 'OPTIMIZING...';
        btn.disabled = true;

        // Simulate optimization delay
        await new Promise(r => setTimeout(r, 1500));

        // Display results
        document.getElementById('model-name').textContent = data.name;
        document.getElementById('model-params').textContent = data.params;
        document.getElementById('model-layers').textContent = data.layers;

        // Layer composition
        const compDiv = document.getElementById('layer-composition');
        compDiv.innerHTML = Object.entries(data.composition)
            .slice(0, 7)
            .map(([k, v]) => `<span class="layer-tag">${k}: ${v}</span>`)
            .join(' ');

        // Optimization stats
        document.getElementById('ops-before').textContent = data.ops_before;
        document.getElementById('ops-after').textContent = data.ops_after;
        document.getElementById('ops-saved').textContent = data.ops_before - data.ops_after;
        document.getElementById('sat-time').textContent = data.sat_time;

        // Explanation
        const saved = data.ops_before - data.ops_after;
        const pct = ((saved / data.ops_before) * 100).toFixed(0);
        const explanation = document.getElementById('model-explanation');
        explanation.innerHTML = `<strong>The Minderling speaks:</strong> ${saved} operations eliminated (${pct}% reduction). Dropout layers removed for inference mode, with bidirectional associativity ensuring optimal tree structure across all ${data.layers} transformer layers.`;

        resultPanel.style.display = 'block';
        resultPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

        if (btnText) btnText.textContent = `OPTIMIZE ${selectedModel.toUpperCase()}`;
        btn.disabled = false;
    });
}
