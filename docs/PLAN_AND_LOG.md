# Pre-declared plan and decision log

This document records the study's questions, hypotheses and criteria as they were fixed in a dated plan on
2026-10-08, before any measurement on the test fold, together with the design and a dated log of every later
decision and of the data each decision used. It is a condensed English version of the plan kept with the code.
Entries that concerned other projects or only the wording of the manuscript are left out. File and script names
refer to this repository.

## 1. Questions, hypotheses and criteria (fixed 2026-10-08)

**Central question.** For the teacher's benefit to reach the INT8 model on the device, must the teacher act
before quantization, during quantization, or both, and how does the answer depend on the student's INT8 floor?

**Notation.** A is the undistilled FP32 student and B the distilled FP32 student, both trained in this study on
the split of section 2.1. gF = B − A in FP32. After INT8, each route is compared with the undistilled model taken
through the same route.

| Route | Gain after INT8 | Teacher acts |
|---|---|---|
| PTQ | PTQ(B) − PTQ(A) | before |
| QAT | QAT(B) − QAT(A) | before |
| QAKD from A | QAKD(A) − QAT(A) | during |
| QAKD from B | QAKD(B) − QAT(A) | before and during |

| RQ | Hypothesis | Counted as supported when |
|---|---|---|
| 1. Under PTQ, does the gain reach the INT8 model, and what decides it? | H1: two regimes, decided by the floor (seen in preliminary data of an earlier, unpublished study) | Both regimes are seen on this study's 7 × 6 grid |
| 2. Does quantization-aware fine-tuning of the distilled weights keep the gain where PTQ loses it? | H2: for the three students with the lowest floors, the QAT gain is positive where the PTQ gain is not | The interval clustered by student excludes 0, and fewer pairs flip their sign (paired McNemar) |
| 3. Does a teacher during quantization add to a distilled start? | H3: QAKD(B) − QAT(B) > 0, adding more at low floors | The interval excludes 0, and ρ(floor, Added) ≤ −0.71 (one-sided 5% critical value for seven students) |
| 4. Can distillation during quantization replace distillation in FP32? | No direction declared | Equivalence test (TOST) of QAKD(A) against QAKD(B), margin twice the training noise of one QAKD(B) model |
| 5. Does the choice of teacher matter more after quantization? | H5: the teacher's share of variance is larger under QAKD than under PTQ on the same 42 pairs, and the best FP32 teacher is also the best INT8 teacher for more students under QAKD than under PTQ | Direct comparison on the same pairs |

## 2. Design

### 2.1 Data split (`pipeline/_qj_split.py`)

ePillID has five consumer folds that share no pill type (192 types each) and 9,804 reference images. The study
follows the first run of the benchmark's own cross-validation protocol.

| Subset | Content | Images |
|---|---|---|
| Train | consumer folds 1 to 3, with the reference images of their 576 pill types | 2,237 + 1,152 = 3,389 |
| Validation | consumer queries of fold 0, gallery of all reference images | 746 queries |
| Test | consumer queries of fold 4, gallery of all reference images | 745 queries |

* Every decision (checkpoint selection, the pilot gate, the number of fine-tuning epochs, every design change)
  uses validation only. The test fold is used for measurement only.
* No model, the teachers included, sees a pill type of fold 4 during training (zero-shot at the type level).
* Post-training calibration uses 256 images drawn with seed 123 from all images except the test queries.
* One image size, 224 × 224, for every model and every step (training, validation, export, calibration, test),
  the size at which the ePillID images are stored. The teachers resize their input to 384 × 384 inside the
  network wrapper, as their recipe does; the students run at 224 × 224 throughout.

### 2.2 Students and teachers

Seven convolutional students: ResNet-18, SqueezeNet-1.0, RepVGG-A0, MobileNetV3-L, GhostNet, EfficientNet-B0 and
EfficientNet-B1, chosen in a preliminary study on another split of the same data to span a wide range of INT8
floors. Six teachers: EfficientNet-B5, SE-ResNeXt-101, ConvNeXt-B, ConvNeXt-S, ResNet-101 and DenseNet-161, all
of which improved the students in FP32 in that preliminary study. Three larger teachers (more than 100 million
parameters) that harmed some students there are left out; teachers of that size would not fit beside a student
during quantization-aware fine-tuning on a 10 GB GPU. Students are convolutional because the training-time
quantization simulator covers only convolutions and linear layers.

### 2.3 FP32 phase (`pipeline/10_train_fp32.py`)

| Group | Content | Runs |
|---|---|---|
| Teachers | 6 teachers, seed 42 | 6 |
| A | 7 students without a teacher, seed 42 | 7 |
| B | 7 × 6 pairs, pointwise cosine distillation with α = 10, seed 42 | 42 |
| Seed repeats | EfficientNet-B1, MobileNetV3-L, ResNet-18 × {A, B with ConvNeXt-S} × seeds 123 and 223 | 12 |
| **Total** | | **67** |

A and B are trained by the same trainer with the teacher switched off for A, so they share their recipe, seed,
initialization and batch order and differ only by the teacher term. One process per run (`pipeline/_qj_run.py`)
writes one directory `weights/qj/fp32/<role>/<model>/<teacher>/seed<s>/` with `model.pth` and `run.json` (split,
trainer hash, teacher fingerprint, every validation point, selected epoch).

### 2.4 Test-fold benchmark of the FP32 models

* `pipeline/11_bench_fp32_ptq.py`: FP32 mAP, INT8 under MinMax (primary) and percentile 99.9 (secondary) for
  every FP32 model; the teachers in FP32 only.
* `pipeline/12_bench_calib_noise.py`: A and B (ConvNeXt-S) of the seven students with calibration seeds 123 to
  523 under both rules, for the calibration noise.

### 2.5 Quantization-aware fine-tuning grid

| Cell | Runs |
|---|---|
| QAT(A) | 7 |
| QAT(B) | 42 |
| QAKD(A) | 42 |
| QAKD(B) | 42 |
| Seed repeats: 3 students × {QAT(A), QAKD(B) with ConvNeXt-S} × 2 extra seeds | 12 |
| Pilot | 4 |
| **Total** | **149** |

Each run starts from the A or B of its own seed (`pipeline/_qj_common.py`, `start_checkpoint`), so a seed repeat
repeats the whole chain (FP32 training, then fine-tuning). The teacher is always the seed 42 teacher. Runner:
`pipeline/_qj_run_qat.py`; orchestrator: `pipeline/21_train_part2.py`; measurement: `pipeline/22_eval_part2.py`
(`--fold val` for decisions, `--fold test` for results); pilot gate: `pipeline/23_gate_g0.py`.

### 2.6 Fine-tuning recipe

* Start: A or B, wrapped by a simulated quantizer with learned clipping ranges; every layer quantized from the
  first epoch.
* Batch normalization: weights quantized as folded with the running statistics while batch statistics are used;
  exact folding and frozen statistics in the last five epochs.
* Learning rate: the recipe's value at epoch 41 of its 80-epoch schedule, cosine decay to 1e-6, no warm-up; the
  clipping ranges have their own optimizer group (5e-3, no weight decay), initialized from a moving average over
  the first 20 batches.
* QAKD keeps the distillation term (α = 10) with the same frozen FP32 teacher.
* Number of epochs: the pilot compares 20 and 40 epochs; the smallest that passes the pilot gate is used.
* Checkpoint selection on validation fold 0.

### 2.7 Deployment and measurement

FP32 ONNX export at 224 × 224 (learned ranges written as `Clip` nodes); ONNX Runtime 1.26.0 static post-training
quantization in QDQ form with 256 calibration images (seed 123), per-channel int8 weights and per-tensor uint8
activations; two-sided mAP on fold 4. INT8 runs on the CPU execution provider. MinMax is the primary rule and
percentile 99.9 the secondary one. Every measurement row records whether ONNX Runtime pre-processing ran.

### 2.8 Noise

* Calibration noise: five calibration seeds of the FP32 benchmark.
* Training noise of the whole chain: the seed repeats of three students.
* A sign change of a pair counts only beyond twice the relevant noise, for pairs with a clear FP32 gain.

### 2.9 Gates

* FP32 gate: each teacher above every student on validation; after the FP32 benchmark, the students still span
  low to high floors on validation.
* Pilot gate G0 (validation only): EfficientNet-B1 with ConvNeXt-S, QAT(A) and QAKD(B) at 20 and 40 epochs.
  Thresholds: deployed INT8 MinMax of QAT(A) at least 0.59 × A and of QAKD(B) at least 0.67 × B. The smallest
  passing number of epochs is used.
* G1: after the first student's core cells, measure on validation and check the analysis before the other
  students run.

### 2.10 Safeguards

Each run owns one directory and saves only through a claim object (`pipeline/_qj_io.py`): atomic saves, no
overwrite of a finished run unless requested, locks for running runs. On the shared GPU the orchestrator starts a
run only when the measured memory peak of that student plus 1 GiB is free; a run that fails for lack of memory is
restarted from the beginning at most three times.

## 3. Analysis plan (fixed 2026-10-08)

* Quantities: gF, the gain after INT8 per route, the contrasts of the factorial design, the transfer coefficient
  per student and route (regression through the origin of the INT8 gain on gF over the six teachers), sign
  changes beyond the noise, the shares of variance of teacher and student on the 7 × 6 grid, and the agreement
  between the best FP32 teacher and the best INT8 teacher.
* Statistics: the student is the independent unit (n = 7). Pooled quantities with a cluster bootstrap over
  students (10,000 resamples, seed 42); Spearman's ρ over students for the dependence on the floor; paired
  McNemar for sign changes; TOST for RQ4. Intervals and effect sizes are reported, not p values alone.
* Mechanism (no training): per-tensor quantization noise and neighbor preservation between FP32 and INT8
  embeddings, on validation images.

## 4. Decision log

* **2026-10-08.** Plan approved. One split for every experiment (section 2.1); 224 × 224 for every model; A trained
  by the distillation trainer with the teacher switched off, so that A and B differ only by the teacher.
* **2026-10-09.** FP32 phase done (67 of 67 runs, no error). The FP32 gate passed on validation.
* **2026-10-10.** Pilot gate G0 passed at 40 epochs on validation (deployed INT8 MinMax: QAT(A) 0.546 against a
  threshold of 0.428; QAKD(B) 0.605 against 0.525). 20 epochs failed (0.171 and 0.162) although the simulated
  validation mAP was 0.561 and 0.589. The grid runs in priority phases (QAT(A), QAT(B) and QAKD(B) first, then
  QAKD(A), then the seed repeats). Seed repeats start from the A and B of their own seed. After an audit of the
  run safeguards: per-student memory thresholds, automatic restarts after memory failures, and a run counts as
  finished only when its own attempt saved a valid checkpoint; nothing measured before changed. G1 passed (13
  runs of EfficientNet-B1; deployed over simulated validation mAP 0.77 to 0.89).
* **2026-10-11.** RepVGG-A0 is deployed with its branches folded (structural reparameterization,
  `models/reparam.py`). Reason, on validation: three unfolded QAT runs reached 0.28 to 0.43 deployed INT8 mAP
  against 0.68 to 0.73 simulated, because ONNX Runtime quantizes the identity branch and the sum of the branches,
  which the simulator does not see. The fold is exact (FP32 embedding change below 1e-6). The unfolded runs and
  their measurements are kept apart (`*_repvgg_unfused.csv`).
* **2026-10-12.** Seed 42 grid done (133 runs, no failure; deployed over simulated validation mAP 0.77 to 1.01).
  The test fold measurement of the grid started. The seed repeats were run exactly as designed (12 runs), a
  choice made before the test numbers of the grid were seen.
* **2026-10-13.** Deployment cost measured on a desktop CPU (`pipeline/27_bench_latency.py`, five round-robin
  rounds). Mechanism measurements on a validation analysis set that excludes the calibration images
  (`pipeline/36_mech_measure.py`, `pipeline/36b_mech_accum.py`, `pipeline/37_mech_analysis.py`).
  **Anchoring control on B declared before running** (exploratory): arm `qanc_b`, 40 epochs of
  quantization-aware fine-tuning from B (ConvNeXt-S) with B's own frozen FP32 model in the cosine term in place
  of the teacher, 7 runs. Reading, with a margin of twice the noise of one QAKD(B) model (MinMax 0.046,
  percentile 0.022), on Gap = QAKD(B) − QANC(B): equivalence if its 90% t interval lies within the margin; the
  teacher adds beyond anchoring if that interval lies above 0 and the mean exceeds the margin; undecided
  otherwise. Result: equivalence under MinMax and undecided under percentile calibration; since the teacher's
  whole contribution on B lies within the margin, the control cannot discriminate.
* **2026-10-14.** **Equal-dose control and anchoring on A declared before running** (exploratory, three seeds,
  126 runs, teacher ConvNeXt-S, every run from A of seed 42). Phase 1: 40 FP32 epochs without the teacher (FT40)
  or with it (KD40). Phase 2: 40 epochs of quantization-aware fine-tuning, QAT(FT40), QAT(KD40) and QAKD(FT40).
  Also QANC(A), 40 epochs of quantization-aware fine-tuning from A with A's own frozen FP32 model in the cosine
  term. Readings (MinMax primary, seed means per student, then t intervals over the seven students): Timing =
  QAT(KD40) − QAKD(FT40): the stage matters if the 95% t interval excludes 0, negligible at this dose if the 90%
  t interval lies within ±0.02 mAP, undecided otherwise. Secondary: the teacher's effect at each stage; further
  training (FT40 − A, PTQ(FT40) − PTQ(A)); retention at equal length, PTQ(KD40)/KD40 − PTQ(FT40)/FT40. QANC(A):
  with Gap_A = QAKD(A) (mean over the six teachers) − QANC(A), anchoring accounts for During if the 90% t
  interval of Gap_A lies within ±0.02 and anchoring moves QAT(A) (95% t interval of QANC(A) − QAT(A) excludes 0);
  the teacher adds beyond anchoring if the 95% t interval of Gap_A lies above 0; undecided otherwise.
  CURE is not used, not even for a trial, because it was built for few-shot recognition with more images per
  pill type and its consumer images are not paired by side, so it would need a different protocol; running a
  trial before deciding whether to report it would be selective reporting.
* **2026-10-15.** Equal-dose results read as declared: Timing +0.001 under MinMax (90% t interval
  [−0.013, +0.015]) and +0.009 under percentile calibration ([−0.001, +0.018]), negligible at this dose under
  both rules and on validation; retention at equal length +0.045 (95% t interval [+0.018, +0.072]); QANC(A):
  under MinMax the teacher adds beyond anchoring when QAKD(A) is averaged over the six teachers (+0.020,
  [+0.002, +0.038]); undecided under percentile calibration.
* **2026-10-16.** **Four sensitivity analyses declared before running** (exploratory; teacher ConvNeXt-S; MinMax
  primary, percentile 99.9 secondary; the student is the unit).
  (1) Stronger post-training quantization of A and B: ONNX Runtime's entropy calibrator with 2048 histogram bins,
  percentiles 99.99 and 99.999, and both main rules with the retrieval head (from the GeM clamp onward) kept in
  FP32; test and validation. Reading: per configuration the floor, the PTQ benefit and the kept shares; the two
  regimes belong to per-tensor MinMax alone if no other configuration leaves a floor below 0.1; B's kept share
  against A's counted per student, differences below 0.01 being ties.
  (2) Calibration noise of QAT(A), QAT(B) and QAKD(B) with four more calibration seeds (test); the total noise of
  a difference between two fine-tuned models, training and calibration taken as independent; descriptive.
  (3) Self-anchored FP32 control ANC40: 40 FP32 epochs from A with A's own frozen FP32 model in the cosine term
  (α = 10, the schedule of FT40 and KD40), three seeds. With Ret(X) = PTQ(X)/X, D1 = Ret(KD40) − Ret(ANC40) and
  D0 = Ret(ANC40) − Ret(FT40): the teacher's knowledge if the 95% t interval of D1 lies above 0; the cosine term
  itself if the 90% t interval of D1 lies within ±0.02 and the 95% t interval of D0 lies above 0; undecided
  otherwise.
  (4) Two more fine-tuning seeds of QAT(A) from A of seed 42 (14 runs): the fine-tuning noise of QAT(A) per
  student, and Rescue, the QAT benefit, During and Both recomputed with QAT(A) averaged over the three seeds.
  (5) Checks from existing data: the change of basis that cosine distillation from A requires, the number of
  quantization points and learned ranges in the deployed graphs, the composition of the calibration draw,
  simulated against deployed validation accuracy, and the 20-epoch pilot (`pipeline/41_paper_checks.py`).
  Phone latency measured with the same protocol on a Samsung Galaxy A17 5G (`android/`).
  Results, read as declared: (1) the 99.99th and 99.999th percentiles left EfficientNet-B1 collapsed and MinMax
  with the head in FP32 left both EfficientNets collapsed, so the two regimes are not a property of MinMax alone;
  the PTQ benefit stayed positive under every setting. (2) Pooled calibration noise per fine-tuned model 0.0064
  (MinMax) and 0.0094 (percentile). (3) D1 = +0.027 (95% t interval [+0.002, +0.053]) under MinMax on the test
  fold, the teacher's knowledge; undecided under percentile calibration and on validation. (4) Pooled contrasts
  moved by at most 0.005 (rounded values) and kept intervals excluding 0; EfficientNet-B1's QAT benefit became
  positive (+0.026), so its negative value in the grid rests on one fine-tuning run.
* **2026-10-16, after reading the results** (from an independent read of the manuscript; exploratory additions,
  labelled as such in the article, the declared readings unchanged). (a) The Interaction also contains QAT(A)
  and was left out of the list of (4) by oversight; with QAT(A) averaged over three seeds it is negative for
  every student, −0.018 (95% t interval [−0.026, −0.009], MinMax) and −0.010 ([−0.015, −0.006], percentile),
  against [−0.050, +0.014] and [−0.041, +0.012] in the grid. (b) The fine-tuning noise of QAT(A) is larger than
  the chain estimate for the same three students (0.046 against 0.023 under MinMax, 0.012 against 0.009 under
  percentile calibration); the article keeps the declared margin of RQ4 and adds that with twice the pooled
  fine-tuning noise (±0.075 and ±0.055) both 90% intervals would lie inside it. (c) The QAT kept share, the
  rank correlation of QAT(A) with the floor, the ranks of the QAT benefit and the sign change of H2 are reported
  with the three-seed QAT(A) beside the grid's values. (d) The supplementary table of these contrasts bootstraps
  in each rule's floor order, as the pooled table does, so its grid column reproduces the pooled table.
