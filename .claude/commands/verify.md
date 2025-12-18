# Verify Training Code

Run a quick verification to ensure training code works correctly before committing changes.

## Steps

1. **Run ruff linting and formatting:**
   ```bash
   source .venv/bin/activate && ruff check . --fix && ruff format .
   ```

2. **Run verification script with WandB:**
   ```bash
   source .venv/bin/activate && python verify.py
   ```

   Or without WandB for quick syntax/import check:
   ```bash
   source .venv/bin/activate && python verify.py --no_wandb
   ```

3. **Check for errors** in the output

4. **Review visualization quality** (if WandB enabled) - see Presentation Quality section below

## What This Tests

- All imports work (no missing modules)
- Training loop runs without errors
- All diagnostics and visualizations generate correctly
- Dataset loading with subset works
- Model forward/backward passes work
- Loss computation works

## Options

- `--samples N`: Use N training samples (default: 100)
- `--val_samples N`: Use N validation samples (default: 50)
- `--no_wandb`: Disable WandB logging
- `--encoder jit|vit`: Test specific encoder

## When to Use

Use this before committing changes to:
- `train.py`
- `utils/visualization.py`
- `utils/metrics.py`
- `utils/dashboards.py`
- `utils/health_checks.py`
- `models/*.py`
- `losses/*.py`

---

## Presentation Quality Review

After running with WandB, check visualizations in `wandb/latest-run/files/media/images/vis/`

### Core Principles

Every visualization should be **self-explanatory** - a reviewer should understand:
1. What the plot shows (clear title)
2. What the axes/colors represent (labels, legends)
3. What "good" vs "bad" looks like (interpretation aids, thresholds)
4. What action to take if values are abnormal (recommendations)

### Checklist for Every Visualization

- [ ] **Title**: Descriptive, explains what the plot shows
- [ ] **Axis Labels**: Both axes labeled; integer indices where applicable (not floats like 0.0, 0.5, 1.0)
- [ ] **Legend**: If multiple colors/series, explain each
- [ ] **Interpretation**: Annotation or subtitle indicating healthy ranges or thresholds
- [ ] **Statistics**: Show computed values (mean, entropy, etc.) not just raw data
- [ ] **Readable**: Sufficient font size, contrast, no truncation

### Red Flags to Fix Before Committing

- Float indices on axes that should be integers (sample 0, 1, 2...)
- Missing legends for color-coded plots
- No interpretation text (user must guess what's good/bad)
- Truncated titles or overlapping labels
- Raw tensor values without normalization
- Heatmaps without colorbars

### Quick Check Commands

```bash
# Find latest run
ls -lt wandb/ | head -3

# List visualization files
ls wandb/latest-run/files/media/images/vis/

# View image (macOS)
open wandb/latest-run/files/media/images/vis/*.png
```

---

## HTML Health Report Quality Check

After WandB run, verify the HTML health report at `health/recommendation` meets these UX criteria:

### Rendering Check
- [ ] Report renders without HTML errors in WandB panel
- [ ] All sections visible (Status, Metrics, Analysis, Recommendations)
- [ ] Color coding works (green/yellow/red visible and distinguishable)
- [ ] Font is readable, no truncation

### Content Check
- [ ] Status icons display correctly (checkmark/warning/X)
- [ ] Metric values populated (not showing "None" or empty)
- [ ] Good ranges shown for each metric
- [ ] Automatic analysis text is present and specific

### Usefulness Check
- [ ] Can determine training health at a glance (without reading details)
- [ ] Recommendations are actionable (say what to do, not just what's wrong)
- [ ] Quick reference footer helps interpret values

### Common Issues to Fix
- Missing metric values → Check if health_check_interval triggers on epoch 1
- Blank analysis → Ensure current_epoch_metrics dict has expected keys
- No recommendation → Check if health checks are returning proper HealthResult objects
