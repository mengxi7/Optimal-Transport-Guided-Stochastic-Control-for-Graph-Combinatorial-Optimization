# Optimal Transport-Guided Stochastic Control for Graph Combinatorial Optimization

This repository contains the implementation for **Optimal Transport-Guided
Stochastic Control for Graph Combinatorial Optimization**. The code supports
MaxCut, Maximum Independent Set (MIS), and Maximum Clique (MaxClique).

## Paper

- [Optimal Transport-Guided Stochastic Control for Graph Combinatorial Optimization](ot_guided_stochastic_control_preprint.pdf)

## Structure

- `ot_gco/data.py`: dataset loading and Jraph-to-PyG conversion.
- `ot_gco/objectives.py`: MaxCut, MIS, and MaxClique objectives and constraints.
- `ot_gco/models.py`: GNN model components.
- `ot_gco/sampler.py`: sampler, Euler-Maruyama update, and step-mask noise.
- `ot_gco/decoding.py`: discrete decoding and projected evaluation loss.
- `ot_gco/training.py`: training loss, optimizer setup, checkpoint writing, and train loop.
- `ot_gco/evaluation.py`: validation and test evaluation.
- `scripts/train.py`: training entrypoint.
- `scripts/evaluate.py`: evaluation entrypoint.

## Data

By default, the scripts look for datasets under `data/RB_iid_small`. Override this
with `--data_path /path/to/dataset`.

Expected layout:

```text
data/RB_iid_small/
  train/*.pickle
  val/*.pickle
  test/*.pickle
```

Each pickle file should contain a dictionary with key `H_graphs`. The graph object
is expected to expose `senders` and `receivers`; `nodes` or `n_node` is used to
infer the number of nodes. For `--P MaxClique`, the complement graph is built from
`H_graphs` inside the data conversion path.

## Usage

Train:

```bash
python scripts/train.py \
  --data_path data/RB_iid_small \
  --P MaxCut \
  --device cuda:0
```

Evaluate:

```bash
python scripts/evaluate.py \
  --data_path data/RB_iid_small \
  --P MaxCut \
  --checkpoint model_path/ot_gco/.../epoch0.pth \
  --device cuda:0
```

## Noise

The sampler keeps the Euler-Maruyama Gaussian term controlled by `--epsilon`:

```text
x <- clamp(x + ((y_theta - x) / (1 - t)) * dt + sqrt(2 * epsilon * dt) * z, 0, 1)
```

It also supports step-scheduled mask perturbation. At step `k`, entries are masked
with probability

```text
p_k = noise * exp(-noise_decay * k / T)
```

Masked entries are set to `--noise_value`, which defaults to `0.0`. This behavior is
disabled by default because `--noise` defaults to `0.0`.

Useful arguments:

- `--epsilon`: Gaussian noise scale in the Euler-Maruyama step.
- `--delta`: terminal time truncation; inference uses `[0, 1 - delta]`.
- `--noise`: initial probability for step-mask noise.
- `--noise_decay`: decay rate for step-mask noise.
- `--noise_value`: value assigned to masked entries.

## Verification

Run a syntax check:

```bash
python -m py_compile \
  ot_gco/*.py \
  scripts/*.py
```
