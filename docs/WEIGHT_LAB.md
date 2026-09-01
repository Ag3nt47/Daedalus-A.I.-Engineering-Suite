# Weight Lab

Weight Lab is a bounded numerical design studio inside **4 · Plan → Calculator
Lab**. It implements six ideas from the AI-weights training brief as inspectable
NumPy prototypes. Each run returns output arrays plus a deterministic evidence
record containing the declared assurance level, configuration, diagnostics,
actionable hints, and SHA-256 array descriptors.

The tools do not claim to create knowledge without training data, priors, or
evaluation. “Exact” means exact only for the tool's explicitly stated contract.

## Choose a tool

| Goal | Tool | Assurance | Important boundary |
|---|---|---|---|
| Produce a small adapter from numeric context | Meta-Weight Synthesizer | Initialized only | The seeded controller is untrained |
| Reproduce a complete small binary truth table | Direct Logic Compiler | Exact for declared binary domain | Cost grows exponentially with input count |
| Explore a stable input-selective recurrence | Recurrent Kernel Engine | Approximation | This is not a trained or hardware-fused Mamba block |
| Fit sparse data under a known 1-D linear differential equation | Constraint Optimizer | Numerical fit | This is a finite polynomial basis, not a higher-order-autodiff neural PINN |
| Build a fast random-feature regression baseline | Matrix Inverter | Numerical fit | Output weights use stable least squares, never an explicit inverse |
| Decide which rows in an explicit pool to label next | Uncertainty Sampler | Approximation | Uncertainty is conditional on one RBF-GP model and candidate pool |

## Mathematical contracts

### Meta-Weight Synthesizer

For numeric context `c`, a seeded controller produces two factors and a bounded
low-rank update:

```text
z = tanh(c W_h + b_h)
ΔW = scale / sqrt(r) · A(z) B(z)
W = W_base + clip_norm(ΔW)
```

The low-rank form controls allocation and makes the generated delta inspectable.
The result stays `initialized_only` until a controller is trained against task
losses or target weights and then evaluated on held-out contexts. See
[HyperNetworks](https://arxiv.org/abs/1609.09106).

### Direct Logic Compiler

For every binary pattern `s`, Weight Lab builds a threshold pattern detector:

```text
h_s(x) = 1[(2s − 1)ᵀx + 0.5 − ||s||₁ > 0]
y(x) = Σ_s h_s(x)y_s
```

The input must contain every binary combination exactly once. Weight Lab
canonicalizes the rows and exhaustively verifies the result before returning it.
It does not parse or execute program text. For broader program-to-network
research, see [Learning to Compile Programs to Neural Networks](https://proceedings.mlr.press/v235/weber24b.html).

### Recurrent Kernel Engine

The prototype uses an input-selective diagonal recurrence:

```text
h_t = a_t ⊙ h_(t−1) + g_t ⊙ (B x_t),  with |a_t| < ρ < 1
y_t = C h_t + D x_t
```

It also reports a context-frozen reference impulse response. The contraction
bound controls the autonomous transition; it is not a promise that every driven
output is small. Mamba adds trained input-dependent state-space parameters and
specialized parallel kernels; see the [Mamba paper](https://arxiv.org/abs/2312.00752).

### Constraint Optimizer

For a typed one-dimensional operator

```text
L(u) = a₂u″ + a₁u′ + a₀u = source
```

Weight Lab solves polynomial coefficients with the visible objective

```text
loss = data_MSE + λphysics · residual_MSE
       + λboundary · boundary_MSE + ridge
```

Observation, residual, and boundary losses remain separate in the result so an
apparently small total cannot hide a poor balance. This implementation is a
finite-basis physics-informed solver because Daedalus's current teaching
autograd does not build higher-order derivative graphs. See the original
[physics-informed neural networks formulation](https://doi.org/10.1016/j.jcp.2018.10.045).

### Matrix Inverter / ELM

The user-facing name comes from the brief, but the implementation deliberately
does not invert a matrix. It freezes a seeded hidden feature map and solves:

```text
H = activation(standardize(X) W_in + b)
β = argmin ||Hβ − Y||² + λ||β||²
```

NumPy's stable least-squares solver supplies `β`; diagnostics report rank,
conditioning, and training RMSE. Always reserve held-out rows before treating a
low training error as evidence. See [Extreme learning machine: Theory and applications](https://doi.org/10.1016/j.neucom.2005.12.126).

### Uncertainty Sampler

The sampler fits a small RBF Gaussian process by Cholesky factorization and
ranks unobserved candidate inputs by posterior standard deviation:

```text
μ* = K*ᵀ K⁻¹ y
variance* = k(x*, x*) − K*ᵀ K⁻¹ K*
```

Already-labeled candidate rows are excluded, ties are stable, and bounded jitter
is recorded. Multi-row requests greedily condition the remaining variance after
each choice so a batch is less likely to contain redundant neighbors. The
selected rows still need real labels; append them and recompute the posterior.
See [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/chapters/).

## Guided, Sandbox, and More Info

Every launcher opens through the same point → line → window animation and closes
by reversing it. Reduced-motion mode skips directly to the same accessible end
states. The real controls remain hidden and disabled during the transition;
focus enters the revealed tool and returns to its launcher on close.

- **Guided** provides safe presets, typed/bounded inputs, result hashes, separate
  diagnostics, and suggestions for the next validation step.
- **Sandbox** shows a tool-specific NumPy starter in memory. **Create private
  draft** is the first write and uses exclusive creation. **Save** is explicit
  and atomic. Only a saved file can run.
- **More Info** shows the current scope, formula, use/avoid guidance, and primary
  source. **Search YouTube** generates a live pre-filled search at activation
  time. Both buttons validate the exact packaged HTTPS destination before
  opening a browser.

The tool explanations and source anchors were reviewed **2026-08-30**. The date
does not claim that every new paper or video has been indexed; the YouTube link
is a current search rather than a frozen recommendation list.

## Private extension layout

One draft is available per project and tool:

```text
<private project>/
└── experiments/
    └── weight_lab/
        ├── meta_weight.py
        ├── logic_compiler.py
        ├── recurrent_kernel.py
        ├── constraint_optimizer.py
        ├── matrix_inverter.py
        └── uncertainty_sampler.py
```

Verified built-ins are application code and are never overwritten by these
files. Drafts are visible from Code Workshop and execute through the existing
isolated, time- and path-confined subprocess. This protects against ordinary
mistakes, not malicious code. Use a disposable virtual machine for hostile or
unknown programs.

## Adding a future tool

Keep the extension path predictable:

1. Add a pure engine function with shape, finiteness, allocation, and deterministic
   seed checks.
2. Return raw arrays only on a typed result and summarize them in a
   `WeightToolRecord` with assurance, hashes, diagnostics, and hints.
3. Add exactly one catalog record and a policy-clean sandbox starter using the
   same key.
4. Add one guided form page and one result preview; reuse the shared editor,
   reveal host, and More Info renderer.
5. Test a hand-computed oracle, deterministic replay, invalid inputs, resource
   caps, sandbox path policy, keyboard/focus behavior, and reduced motion.
