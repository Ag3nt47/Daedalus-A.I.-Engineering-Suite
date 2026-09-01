# Project recipes

Recipes turn curriculum modules into reusable projects. Start with the smallest
fixture, record assumptions and versions, make the failure visible, and only
then add capability or optimization. Full machine-readable steps and gates are
in `src/daedalus/resources/project_recipes.json`.

## Common project record

Every recipe should save:

- objective and non-goals;
- code and input identity;
- shapes, dtypes, units, and axis meanings;
- seed plus Python and NumPy versions;
- complete hyperparameters and loss reductions;
- test and checkpoint results;
- known failure modes and next experiment;
- private/public classification for every artifact.

## 1. Shape Detective

**Level:** Beginner  
**Outcome:** Predict elementwise broadcasting, matrix multiplication, output
shape, element count, and bytes before execution.

Build an input parser for shapes, dtypes, and a named operation. Right-align
elementwise shapes and label each pair as equal, expanded, or conflicting. Treat
matrix multiplication separately as `(..., m, k) @ (..., k, n)`. Verify each
prediction by generating safe arrays and comparing the result shape and
`ndarray.nbytes`.

Pass when scalar, vector, batch-bias, multi-axis, incompatible, matrix, and
batched-matrix fixtures all produce deterministic explanations. Never flatten,
tile, or transpose inputs merely to silence an error.

## 2. Derivative Detective

**Level:** Builder  
**Outcome:** Display the path from a composed expression to each leaf gradient.

Implement scalar nodes for add, multiply, power, exp, log, and ReLU. Store
parents and local backward rules, topologically order the graph, seed the scalar
output with gradient one, and accumulate all parent contributions. Compare
reverse-mode results with centered finite differences over several epsilon
values.

Pass when `x*x + x` includes both paths, smooth fixtures stay below `1e-6`
relative error in float64, invalid log domains produce a typed explanation, and
repeated-backward semantics are tested.

## 3. First Learner

**Level:** Beginner  
**Outcome:** Fit seeded `y = 3x + 2 + noise` from equations.

Use an explicit NumPy `Generator`. Implement prediction, mean-squared error, and
analytical slope/intercept gradients. Train while recording loss and parameter
trajectories. Compare the final solution with `numpy.linalg.lstsq` as an oracle,
not as the learner itself.

Pass when gradients agree below `1e-6`, known parameters are recovered within
declared noise-aware tolerances, a locked-environment rerun reproduces history,
and held-out data never influences fitted preprocessing.

## 4. XOR Ladder

**Level:** Builder  
**Outcome:** Explain why one linear unit fails, then train a small nonlinear
network with observable gradients and safe checkpoints.

Keep the linear baseline as evidence. Add `Linear -> ReLU -> Linear`, verify all
forward shapes and gradients, then compare SGD and a hand-verified Adam update.
Record predictions, loss, parameter/gradient norms, seed, and versions. Save
numeric arrays to NPZ with validated JSON metadata.

Pass when the fixed-seed multilayer model classifies all four XOR inputs, layer
gradients stay below `1e-5` relative error, checkpoint arrays round-trip exactly,
and loading uses `allow_pickle=False`.

## 5. Tiny Character Transformer

**Level:** Advanced  
**Outcome:** Train and inspect one causal attention block.

Create a character vocabulary and deterministic next-token batches. Implement
embeddings, positional information, scaled dot-product attention, a causal mask,
residual paths, normalization, a feed-forward block, and stable cross-entropy.
Overfit one batch before expanding to a small local text sample.

Pass when attention rows are finite and sum to one, all future-token weights are
zero, targeted primitive gradients pass, fixed-corpus loss falls at least 50%,
and the recorded environment/checkpoint reproduces the reference generation.
Do not package copyrighted corpora with the project.

## 6. Push Guardian and Restore Drill

**Level:** Advanced operator  
**Outcome:** Prove that unsafe publication stops and private work can be restored.

Create disposable repositories for clean content, a fake secret, an ignored
untracked file, an ignored tracked file, a failing test, an unresolved advisory,
and a remote-ahead conflict. Scan the exact outgoing commit range, redact
evidence, and distinguish local audit data from authenticated GitHub alerts.

Back up source and private workspace to a marked destination. Corrupt the local
fixture, restore into a new directory, and compare hashes. Pass when every unsafe
fixture blocks for the expected reason, the clean fixture passes, reports reveal
neither the fake secret nor private absolute paths, and restore never overwrites
the active workspace.

## Safe extension rule

Add one extension per experiment and preserve the prior passing record. An
optimization must match a slow clear oracle before replacing it. A dependency
update must be reviewable, tested, and reversible. A UI improvement must not
weaken path, privacy, checkpoint, runner, or release boundaries.
