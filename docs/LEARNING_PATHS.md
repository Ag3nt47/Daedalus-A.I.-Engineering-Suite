# Learning paths

Daedalus teaches AI engineering by turning lessons into working pieces of the
suite. Every module has measurable objectives, a local lab, a saved artifact,
deterministic checks, relevant glossary/error cards, and authoritative sources.
The required engine work uses Python and NumPy rather than PyTorch or TensorFlow.

## Tracks

| Order | Track | Level | Artifact and completion focus |
|---:|---|---|---|
| 0 | Launch Pad | Beginner | Tested Python calculator; safe Git-state exercise |
| 1 | Math Forge | Beginner | Shape-annotated linear algebra and finite-difference lab |
| 2 | Array Lab | Beginner | Vectorized array work and exact controlled memory estimates |
| 3 | Learning Fundamentals | Builder | Leak-free linear and logistic learners |
| 4 | Autograd Foundry | Builder | Scalar tape and NumPy Tensor reverse-mode engine |
| 5 | Neural Workshop | Builder | Layers, losses, optimizers, XOR, diagnostics, checkpoints |
| 6 | Deep Architecture Lab | Advanced | Convolution, normalization, dropout, tiny transformer |
| 7 | Daedalus Systems Lab | Builder | Private workspace, constrained runner, UI/service slice, restore |
| 8 | Secure Delivery | Advanced | Push Guardian, dependency findings, CI, SBOM, release rehearsal |
| 9 | Capstone | Capstone | Installable, auditable local AI tool and evidence bundle |

The exact dependency graph, module text, and checkpoint IDs live in
`learning_paths.json`. The UI should read that file rather than duplicating
curriculum rules in widgets.

## Suggested routes

### New to programming

Follow Launch Pad -> Math Forge -> Array Lab -> Learning Fundamentals. Build
Shape Detective before writing autograd. Shape literacy prevents hours of
guessing later.

### Programmer new to neural networks

Take the Launch Pad Git safety checkpoint, then Math Forge and Array Lab
checkpoints. Continue through Autograd Foundry -> Neural Workshop. Do not skip
the scalar tape: it isolates graph logic from array-shape logic.

### Experienced ML user learning internals

Demonstrate the Math Forge and Array Lab gates, then begin Autograd Foundry.
Complete every finite-difference gate even if a high-level framework result is
available for comparison. Framework output is an optional oracle, not the core
implementation.

### Operator or maintainer

Complete Launch Pad -> Systems Lab -> Secure Delivery. Add Neural Workshop when
responsible for model-state or training diagnostics. A release maintainer must
also complete the backup restore drill and untrusted-checkpoint exercise.

## Checkpoint conventions

- Float64 analytical-versus-centered-finite-difference relative error:
  `max(|g - g_fd| / (1 + |g| + |g_fd|)) < 1e-5` for tensor primitives; scalar
  exercises target `< 1e-6`.
- Stable softmax at logits `[-1000, 0, 1000]` must remain finite and sum to one
  within `1e-12` along the declared class axis.
- Vectorized and loop-oracle results use declared dtype-aware `rtol` and `atol`;
  tolerance is never hidden in a global default.
- Reproducibility records code/data identity, complete configuration, explicit
  RNG seed/state, Python version, NumPy version, and platform assumptions. NumPy
  does not promise identical random streams across every future version, so the
  environment is part of the result.
- Checkpoint round-trip uses `np.array_equal` for saved parameter arrays and
  validates metadata, checksum, key order, count, shapes, and dtypes.
- Security, privacy, path-boundary, and restore gates are hard gates. A high
  average score cannot compensate for failing one.

## Capstone rubric

| Dimension | Weight |
|---|---:|
| Numerical correctness | 30% |
| Tests and diagnostics | 20% |
| Explainability | 15% |
| Reproducibility | 10% |
| Security and privacy | 10% |
| Packaging and operator UX | 10% |
| Documentation and attribution | 5% |

Passing requires at least 80% overall and zero failed hard security or recovery
gates. A capstone must state its problem, data boundary, metric, threat model,
non-goals, and known limitations.

## Glossary and error learning

Glossary cards expose the same concept at four depths:

- **Plain**: one clear mental model.
- **Math**: notation or equation and its assumptions.
- **Code**: a minimal inspectable example.
- **Diagnostic**: evidence to collect when the concept fails in practice.

Error cards are deterministic first: a known exception, warning, diagnostic
code, or Git condition selects a reviewed card. Each card shows the observed
evidence, likely causes, ordered checks, safe fixes, and actions never to take.
An AI-generated explanation may add context but must not replace the reviewed
card, reveal raw private data, or silently edit code.

## Source and offline policy

The source catalog links to Python, NumPy, Git, GitHub, NIST, MIT OpenCourseWare,
and primary research publications. Offline summaries are original Daedalus
prose, not copies of those works. Each release runs schema, internal-link, topic
coverage, and HTTPS checks. A source version or availability change is reviewed
as content work rather than silently redirecting learners to an unrelated page.
