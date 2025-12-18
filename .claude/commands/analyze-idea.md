# Analyze Research Idea

When user proposes a new modification, feature, or research idea, follow this rigorous analysis workflow:

## Step 1: Deep Analysis

1. **Understand the idea thoroughly**
   - What is being proposed?
   - What problem does it solve?
   - How does it fit into the current architecture?

2. **Search for related research**
   ```
   Use WebSearch to find:
   - Related papers on arXiv, CVPR, NeurIPS, ICLR
   - Similar implementations in other projects
   - Known issues or criticisms of this approach
   ```

3. **Fetch and analyze key papers**
   ```
   Use WebFetch on arxiv.org to read abstracts and methods
   ```

## Step 2: Critical Evaluation

1. **Mathematical justification**
   - Does the idea have sound mathematical basis?
   - What assumptions does it make?
   - Are there edge cases where it fails?

2. **Fit with current system**
   - How does it interact with SIGReg loss?
   - Does it conflict with existing components?
   - What are the computational costs?

3. **Potential issues**
   - What could go wrong?
   - Are there known failure modes?
   - What are the alternatives?

## Step 3: Formulate Hypotheses

Create testable hypotheses that can be verified during training:

### Hypothesis Template
```
If [idea/modification], then [expected metric change], because [mathematical/theoretical reason].

Verification:
- Metric: [specific WandB metric path]
- Expected range: [quantitative expectation]
- Comparison: [baseline vs modified]
- Timeline: [when to check - early/mid/late training]
```

### Example Hypotheses
```
H1: If we add attention entropy regularization, then attn/head_entropy_std should decrease by >20%, because regularization encourages uniform head behavior.
- Metric: attn/head_entropy_std
- Expected: < 0.08 (from current ~0.10)
- Compare: baseline run vs regularized run
- Check: after epoch 20

H2: If we increase sigreg_lambda, then rep/effective_dim should increase, because stronger regularization prevents dimensional collapse.
- Metric: rep/effective_dim
- Expected: > 100 (from current ~50)
- Compare: lambda=0.05 vs lambda=0.1
- Check: continuous monitoring
```

## Step 4: WandB Verification Plan

1. **Define metrics to track**
   - List all relevant existing metrics
   - Identify if new metrics need to be added

2. **Create comparison strategy**
   - Baseline run configuration
   - Modified run configuration
   - Statistical significance criteria

3. **Set up alerts/thresholds**
   - What values indicate success?
   - What values indicate failure?
   - Early stopping criteria if idea is harmful

## Step 5: Summary Report

Present findings in structured format:

```markdown
## Idea Analysis: [Name]

### Summary
[1-2 sentence description]

### Related Research
- [Paper 1]: [key finding]
- [Paper 2]: [key finding]

### Mathematical Basis
[Equations or theoretical justification]

### Critical Assessment
**Pros:**
- ...

**Cons:**
- ...

**Risks:**
- ...

### Hypotheses
1. H1: [hypothesis with verification plan]
2. H2: [hypothesis with verification plan]

### Recommendation
[Implement / Modify / Reject] because [reason]

### Implementation Plan (if recommended)
1. ...
2. ...
```

## When to Use This Skill

Use `/analyze-idea` when user:
- Proposes a new loss function or regularization
- Suggests architectural changes
- Wants to try a technique from a paper
- Has an intuition about improving training
- Asks "what if we tried..."
