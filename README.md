# Overlap-Aware Sub-hypergraph Pattern Frequencies for Hypergraph-Level Classification

Code and data for the paper of the same title. The method mines frequent
sub-hypergraph patterns whose hyperedge pairs carry overlap-size labels,
encodes each hypergraph by the relative occurrence frequencies of those
patterns, and classifies it with a learned pattern encoder.

In the main, fully inductive protocol, patterns are mined from the training
hypergraphs of each repeat only and the vocabulary is then frozen; validation
and test hypergraphs are only matched against it. Feature scaling and the
classifier are fitted on training hypergraphs, and the validation set is used
only for model selection. A secondary protocol also lets the unlabeled
validation and test hypergraphs take part in mining.

## Requirements

- Linux (the experiments ran on Ubuntu 24.04), on the CPU.
- Python 3.12 and [uv](https://docs.astral.sh/uv/).
- `tmux`, only for the optional `launch` commands that start one session per
  dataset.
- A git checkout of this repository. Runs record the commit they ran from,
  and the commands below that freeze a run or a suite refuse uncommitted
  changes to the source.

## Setup

```bash
git clone https://github.com/NafiHasan/overlap-fhgm.git
cd overlap-fhgm
uv sync
```

## Data

`data/` holds the six hypergraph-classification datasets of DHG-Bench
([Li et al., 2025](https://arxiv.org/abs/2508.12244)), introduced by Feng et
al. (TPAMI 2024): IMDB-Dir-Form, IMDB-Dir-Genre, Steam-Player, Twitter-Friend,
RHG-10 and RHG-3. For each dataset:

- `composite/<name>.txt`: the hypergraphs, with vertex degree and hyperedge
  cardinality as labels, in the same order as the DHG-Bench files;
- `hg_label.txt`: the class labels;
- `splits/dhgbench_torch_2_2_2/composite/repeat_{0..4}.json`: the five
  training/validation/test partitions (80/10/10).

The partitions reproduce DHG-Bench's `random_split` with seed `42 + r` for
repeat `r`; `tools/dhgbench_split_generator/` regenerates them in its own
environment (PyTorch 2.2.2). `scripts/convert_dataset.py` converts raw
DHG-Bench files into this format.

The datasets were released by Feng et al. (TPAMI 2024) in
[iMoonLab/HIC](https://github.com/iMoonLab/HIC) under the Apache License 2.0
(copy in [`data/LICENSE-APACHE-2.0`](data/LICENSE-APACHE-2.0)) and
redistributed by DHG-Bench. The files here are modified: vertices are
relabeled with their degree, hyperedges with their cardinality, and vertex
IDs are 1-based. The MIT licence below covers the code only.

## Repository layout

| Path | Contents |
|---|---|
| `mining/`, `hypergraph/` | Overlap-aware frequent sub-hypergraph mining and pattern matching |
| `classification/` | Occurrence counts, the pattern encoder and classifier, evaluation |
| `pipeline/` | Inductive and transductive runners, settings profile, splits, provenance |
| `config.py` | Datasets and file locations |
| `scripts/run_campaign.*` | Runs of the final settings over all datasets and repeats |
| `scripts/grid_search/` | Configuration search |
| `scripts/ablation/` | Ablation suites and their reports |
| `scripts/baselines/` | Hypergraph WL kernel baseline |
| `scripts/sanity/` | Leakage and variance checks |
| `scripts/analysis/` | Dataset statistics and presence saturation |
| `scripts/convert_dataset.py` | Conversion of raw DHG-Bench files |
| `scripts/validate_artifacts.py` | Integrity check of a completed run or mining cache |
| `tools/dhgbench_split_generator/` | Regeneration of the DHG-Bench partitions |

The method's settings are the profile `FHGM_OVERLAP_COUNTS_V1` in
`pipeline/profiles.py`.

## Reproducing the paper

Run the main campaign first: the configuration search, the ablations and
the sanity checks all start from its record. Everything is written under
`results/`. Mining results are cached there by configuration and reused
automatically when the same settings run again, so later steps that change
only the classifier do not mine again.

### 1. Main results

```bash
bash scripts/run_campaign.sh init MAIN inductive --require-clean-source
for d in imdb_dir_form imdb_dir_genre steam_player twitter_friend rhg_10 rhg_3; do
  bash scripts/run_campaign.sh run MAIN "$d"
done
bash scripts/run_campaign.sh analyze MAIN
```

Each dataset can also run in its own terminal. Per-repeat scores are in
`results/<dataset>/composite/inductive/MAIN/` (`repeats.csv`,
`inductive_report.md`), and the summary over datasets in
`results/campaigns/MAIN/analysis/analysis.md`.

### 2. Transductive vocabulary access

```bash
bash scripts/run_campaign.sh init TRANSDUCTIVE transductive --require-clean-source
for d in imdb_dir_form imdb_dir_genre steam_player twitter_friend rhg_10 rhg_3; do
  bash scripts/run_campaign.sh run TRANSDUCTIVE "$d"
done
```

Records are in `results/<dataset>/composite/transductive/TRANSDUCTIVE/`, with
the same partitions and settings as `MAIN`; only the vocabulary differs. The
paired comparison is computed from the two runs' `repeats.csv` files, matched
on `repeat`: test scores are `accuracy` and `macro_f1`, and the mined
vocabulary size is `patterns`.

### 3. Configuration search

```bash
bash scripts/grid_search/run_grid.sh init GRID
bash scripts/grid_search/run_grid.sh launch GRID     # or: run GRID <dataset>
bash scripts/grid_search/run_grid.sh report GRID
```

The search covers 24 configurations: overlap labels, maximum pattern size and
support, count encoding, and regularization. One configuration is chosen for
all datasets by mean validation Macro-F1. The chosen one,
`GRID_O_D3S10_REL_S1E`, has the settings of `MAIN`.

### 4. Ablations

```bash
for s in ABL_COMPONENTS ABL_SELECTION ABL_ENCODER ABL_DEPTH4; do
  bash scripts/ablation/run_ablations.sh init "$s"
  bash scripts/ablation/run_ablations.sh launch "$s"   # or: run "$s" <dataset>
done
# when the runs have finished:
for s in ABL_COMPONENTS ABL_SELECTION ABL_ENCODER ABL_DEPTH4; do
  bash scripts/ablation/run_ablations.sh report "$s"
done
```

| Suite | Studies |
|---|---|
| `ABL_COMPONENTS` | Overlap labels and counts, label resolution, maximum pattern size, one-incidence patterns, support schedule |
| `ABL_SELECTION` | Pattern selection before the encoder |
| `ABL_ENCODER` | Pattern encoder, embedding dimension, regularization, encoder input, classifier head |
| `ABL_DEPTH4` | Maximum pattern size 4 |

Reports, with paired differences from the full model per repeat, are in
`results/ablation/<suite>/report/`. `ABL_DEPTH4` mines far longer than the
other suites.

### 5. Hypergraph WL kernel baseline

```bash
bash scripts/baselines/run_hypergraph_wl.sh launch   # or: run <dataset>
```

The main variant is `HGWL_SUBTREE_EXACT`; the other kernel, label and norm
variants are written alongside it in
`results/<dataset>/composite/baselines/`.

### 6. Sanity checks

```bash
uv run python scripts/sanity/cross_split_duplicates.py   # isomorphic train/held-out pairs
uv run python scripts/sanity/shuffled_labels.py          # permuted training labels
uv run python scripts/sanity/training_seeds.py           # four further classifier seeds
uv run python scripts/sanity/training_seeds.py --report
```

Outputs are in `results/sanity/`.

### 7. Dataset statistics and presence saturation

```bash
uv run python scripts/analysis/dataset_stats.py
uv run python scripts/analysis/presence_saturation.py --run MAIN
```

The first gives the dataset table and the majority-class reference; the
second needs the mining caches of `MAIN`.

### Determinism

Each repeat has a fixed model seed and TensorFlow's deterministic operations
are enabled, so a rerun on the same hardware and software, with the same
thread settings, gives the same results. Other hardware or library versions
may give slightly different scores.

## Citation

```bibtex
@misc{overlapfhgm2026,
  title         = {Overlap-Aware Sub-hypergraph Pattern Frequencies for Hypergraph-Level Classification},
  author        = {Nafiul Hasan Anik and Nazira Jesmin Lina and Tanvir Alam and Fahim Arefin},
  year          = {2026},
  eprint        = {TODO},
  archivePrefix = {arXiv}
}
```

## License

The code is released under the [MIT License](LICENSE). The datasets in
`data/` are distributed under the [Apache License 2.0](data/LICENSE-APACHE-2.0)
of their original release (see [Data](#data)).
