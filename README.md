# Distill first, then quantize: knowledge distillation before and during 8-bit quantization

Code and measurements for a factorial study of knowledge distillation and 8-bit (INT8) quantization in compact
pill retrieval models on ePillID. Seven convolutional students are distilled from six teachers (42 pairs). Each
pair is taken to INT8 by post-training quantization (PTQ) and by quantization-aware fine-tuning that starts from
the undistilled student A or the distilled student B, with the teacher switched off (QAT) or on (QAKD). Every INT8
accuracy is measured on the graph that ONNX Runtime deploys, on a held-out fold whose pill types no model saw in
training.

The repository contains the code that produced every number of the article, the measurements behind every table
and figure (`reports_qj/`), and the plan with its pre-declared criteria and a dated decision log
(`docs/PLAN_AND_LOG.md`). Trained weights are not included because of their size. Parts of the code come from
an earlier version that also served other experiments; what this study does not run was left out, and two
inherited modules are reduced to what is used here: `pipeline/_j3_common.py` (the path anchor every script
imports) and `pipeline/_j3_topo_losses.py` (a placeholder for optional structural distillation terms, which the
pointwise cosine objective of this study never calls).

## Layout

| Path | Content |
|---|---|
| `pipeline/` | the study's scripts, numbered in run order (table below), and their shared modules (`_qj_*.py`, `_qakd_*.py`, `_j3_*.py`) |
| `models/`, `pooling/`, `utils/` | the retrieval network wrapper (backbones from torchvision and timm, GeM pooling, projection head), the training recipe of each backbone family, structural reparameterization of RepVGG, data loading and the ePillID evaluator |
| `journal2_pipeline/` | quantization, ONNX export, evaluation and benchmarking utilities, and the fingerprints of runs and checkpoints |
| `android/` | the on-device latency benchmark (C++ program, build script and runner) |
| `reports_qj/` | the measurements (CSV and JSON) that the table and figure generators read |
| `docs/PLAN_AND_LOG.md` | questions, hypotheses and criteria fixed before any test measurement, and the dated decision log |
| `run_*.sh` | the entry points used on the GPU server, meant to run inside tmux |

## Requirements

* One NVIDIA GPU with 10 GB of memory for training (the study used a GeForce RTX 3080) and a CPU for INT8
  evaluation. On a card shared with other jobs, the FP32 phase (67 runs) took about 23 hours of training and the
  313 quantization-aware fine-tuning runs about 90 hours, between 3 and 46 minutes each.
* Python 3.12 with the packages in `requirements.txt` (the versions the study used). The table and figure
  generators need only numpy, scipy, matplotlib and pillow, and also run with Python 3.13.

```bash
pip install -r requirements.txt
```

## Data

The ePillID benchmark (Usuyama et al., 2020) is available from its authors at
<https://github.com/usuyama/ePillID-benchmark>. It is not redistributed here. Place it so that the code finds:

```
data/raw/ePillID/
  all_labels.csv
  classification_data/                                          (the images, as distributed)
  folds/pilltypeid_nih_sidelbls0.01_metric_5folds/base/         (fold CSV files and pill_classes.txt)
```

The split of the study (train on consumer folds 1 to 3 with the reference images of their pill types, select on
fold 0, test on fold 4) is built by `pipeline/_qj_split.py`, which checks it every time a script loads it;
`python pipeline/_qj_split.py` runs the same checks on their own.

## Regenerating the tables and figures (no GPU)

The generators read only `reports_qj/` and write to `paper/`; run them in this order. They reproduce every number,
table and figure of the article:

```bash
python pipeline/29_paper_tables.py       # LaTeX tables and every in-text number (paper/tables, paper/numbers.tex)
python pipeline/29b_paper_figures.py     # figures (paper/figures)
python pipeline/29c_graphical_abstract.py
python pipeline/26_analysis.py --fold test --rule minmax   # the declared analysis, as a Markdown report
```

## Running the study

Each step is resumable, records a fingerprint of every checkpoint it uses, and refuses to overwrite a finished
result unless asked. Commands are run from the repository root.

| Step | Command | Output |
|---|---|---|
| Self-test of the fused RepVGG-A0 deployment (CPU) | `python pipeline/07_sanity_fused.py` | none (exit code) |
| FP32 phase: 6 teachers, 7 A, 42 B and 12 seed repeats (67 runs) | `python pipeline/10_train_fp32.py` | `weights/qj/fp32/` |
| FP32 and post-training INT8 on the test fold | `python pipeline/11_bench_fp32_ptq.py` | `reports_qj/bench/main/fp32_ptq.csv` |
| Calibration noise (five calibration seeds) | `python pipeline/12_bench_calib_noise.py` | `reports_qj/bench/main/calib_noise.csv` |
| FP32 gate (validation) | `python pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs 40 --fp32-refs --kinds fp32_A --seeds 42`, then `python pipeline/13_check_fp32.py` | `reports_qj/fp32/gate_f.json` |
| Pilot and its gate (validation) | `python pipeline/21_train_part2.py --plan pilot`, then `python pipeline/22_eval_part2.py --fold val --plan pilot --fp32-refs` and `python pipeline/23_gate_g0.py` | `reports_qj/part2/g0.json` |
| Fine-tuning grid (QAT and QAKD from A and B, 40 epochs) | `python pipeline/21_train_part2.py --plan grid --ft-epochs 40 --phase all` | `weights/qj/runs/` |
| Its measurement | `python pipeline/22_eval_part2.py --fold val --plan grid --ft-epochs 40 --fp32-refs`, then `--fold test`; `python pipeline/25_check_sim_real.py --ft-epochs 40` flags a run whose deployed accuracy falls far below its simulated one | `reports_qj/part2/main/` |
| Anchoring control on B | `--plan anchor --ft-epochs 40` for training, `--plan anchor --tag anchor` for measurement | `reports_qj/part2/anchor/` |
| Equal-dose control and anchoring on A | `--plan ed --ft-epochs 40`, measured with `--plan ed --tag ed` | `reports_qj/part2/ed/` |
| Sensitivity runs (ANC40, extra QAT(A) seeds) | `--plan sens --ft-epochs 40`, measured with `--plan sens --tag sens` | `reports_qj/part2/sens/` |
| Stronger post-training settings and calibration noise of fine-tuned models | `python pipeline/39_ptq_sensitivity.py --part strong`, then `--part calib` | `reports_qj/bench/sens/` |
| Parameters and desktop latency | `python pipeline/28_count_params.py`, `taskset -c 0,2,4,6 python pipeline/27_bench_latency.py --rounds 5` | `reports_qj/bench/main/` |
| Phone latency | `python pipeline/38_export_latency_graphs.py`, then `android/README.md` | `reports_qj/bench/main/latency_android*.csv` |
| Mechanism (validation images) | `python pipeline/36_mech_measure.py`, `python pipeline/36b_mech_accum.py`, `python pipeline/37_mech_analysis.py` | `reports_qj/mech/`, `reports_qj/analysis/` |
| Checks reported in the article | `python pipeline/41a_pilot_summary.py` (where the runs are), `python pipeline/41_paper_checks.py` | `reports_qj/analysis/checks_main/` |
| Tables and figures | see the previous section | `paper/` |

The `run_*.sh` scripts chain these steps as they were run on the GPU server (for example
`tmux new -s qj_sens -d "bash run_sens_server.sh train"`).

RepVGG-A0 is deployed with its training branches folded into single convolutions (`models/reparam.py`), a
decision recorded in the log on 2026-10-11. The rows measured before that decision, with the branches unfolded,
are kept in `reports_qj/bench/main/fp32_ptq_repvgg_unfused.csv` and `calib_noise_repvgg_unfused.csv` for the
article's comparison. To measure them again, empty `FUSED_STUDENTS` in `pipeline/_qj_common.py` and rerun
scripts 11 and 12 with another `--tag`.

## Determinism

Runs are seeded (seed 42 for the main grid; 123 and 223 for the repeats), A and B of a student share their
initialization and batch order, and the calibration draw uses seed 123. GPU training is not bit-exact across
hardware and library versions, so a rerun reproduces the study's numbers within the training noise reported in
the article, not to the last digit. The table and figure generators are deterministic given `reports_qj/`.

## License

MIT, see `LICENSE`. The ePillID data keep their own terms.
