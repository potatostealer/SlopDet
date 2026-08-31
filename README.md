# doomscrollers

Real vs. AI-generated image detection on top of a frozen SigLIP2 vision tower
(`siglip2-so400m-patch16-naflex`): LoRA adapters on the SigLIP2 encoder + a small QFormer + MLP head,
trained with Lightning on one GPU or with DDP, optionally with on-the-fly augmentation.

Labels everywhere: `0 = real`, `1 = AI generated` (the positive class for precision / recall / F1).

## Setup

Conda environment:

```bash
conda create -n slop_det python=3.11
conda activate slop_det
```

Pip install the dependencies and make `src` importable:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu132 # Refer to official PyTorch site for your CUDA version
pip install -r requirements.txt
pip install -e .          # makes `src` importable
```

## Downloads

Everything needed to run the demo and the training comes from one [Google Drive folder](https://drive.google.com/drive/folders/1O8TzR2C4LMrlKhntE9p9wa5ozeeT2Tfp?usp=sharing):

1. **Model checkpoint** — download `final.ckpt` and place it at `model_data/final.ckpt`. This is the trained
   LoRA + QFormer + MLP checkpoint that the demo and the eval configs point at.
2. **SigLIP2 base model** — download the contents of the `siglip` folder and place them in `model_data/siglip/`
   (so that `model_data/siglip/config.json`, `model.safetensors`, `preprocessor_config.json`, ... exist).
   This is the frozen SigLIP2 base model + image processor; both inference and training need it.
3. **Dataset** (needed for training / evaluation only, not for the demo) — download the dataset archive(s) from
   the same folder, unzip them and place the resulting image directories under `data/` in the repo root:

   | Directory | Contents | Pointed at by |
   |---|---|---|
   | `data/real_train` | real images, train split (label 0) | `dataset.yml: real_img_train_ds_path` |
   | `data/ai_gen_train` | AI generated images, train split (label 1) | `dataset.yml: aigen_img_train_ds_path` |
   | `data/real_val` | real images, val split | `dataset.yml: real_img_val_ds_path` |
   | `data/ai_gen_val` | AI generated images, val split | `dataset.yml: aigen_img_val_ds_path`, `calibration.yml` |
   | `data/real_test` | real images, test split | `eval.yml`, `comprehensive_eval.yml` |
   | `data/ai_gen_test` | AI generated images, test split | `eval.yml`, `comprehensive_eval.yml` |

   If the unzipped directories carry different names, either rename them to the above or point the config keys at
   wherever you placed them (paths relative to the repo root are fine). A split is always a pair of flat image
   directories (`.png .jpg .jpeg .webp`): real = label 0, AI generated = label 1.

The demo only needs items 1 and 2 plus the committed `model_data/calibration.json`.

## Inference

```bash
python demo.py path/to/image.jpg
```

`src/configs/demo.yml` points at:

| Key | What it is |
|---|---|
| `model.ckpt_path` | Lightning checkpoint of a `LoraQFormerDetector`; the architecture is rebuilt from the hyper-parameters stored inside it |
| `model.siglip_checkpoint_path` | `null` = the SigLIP2 base model / image processor directory recorded in the checkpoint; set it only to relocate that directory |
| `calibration_path` | JSON bucket calibration table for this checkpoint: `num_buckets` (K) and `table`, K rows of `{predicted_label, confidence}` indexed by `min(floor(c * K), K - 1)` |
| `device` | `cuda:N`, a bare index `N`, or `cpu` |
| `precision` | autocast for the forward pass: `bf16-mixed` \| `16-mixed` \| `32` |
| `output_path` | where the prediction JSON is written |

`c = sigmoid(logit)` is looked up in the bucket table at `calibration_path`
(bucket index `min(floor(c * K), K - 1)`); the prediction is the class with the larger
posterior, the confidence that posterior. The result — `{"image", "prediction", "confidence"}` —
is printed and written to `output_path`.

## Training Quickstart

The released checkpoint was trained with `train_aug_ddp.py` (multi-GPU DDP with on-the-fly augmentation);
`train_aug_classical_ddp.py` is the variant that additionally fuses classical forensic features into the
QFormer input. To run either:

1. Download the dataset and the SigLIP2 base model (see [Downloads](#downloads)); the training configs already
   point at `model_data/siglip` and the `data/` directories above.
2. Check `src/configs/dataset.yml` (the four split directories) and `src/configs/training.yml`
   (`run_name`, `ddp.cuda_visible_devices` = the physical GPUs to train on, hyper-parameters).
3. Smoke test, then launch from the repo root:

```bash
# smoke tests (5 batches, 1 epoch)
CUDA_VISIBLE_DEVICES=0,1 python -m src.experiments.train_aug_ddp --limit-batches 5 --epochs 1
CUDA_VISIBLE_DEVICES=0,1 python -m src.experiments.train_aug_classical_ddp --limit-batches 5 --epochs 1 --num-workers 16

# full runs (GPUs from ddp.cuda_visible_devices in training.yml)
python -m src.experiments.train_aug_ddp
python -m src.experiments.train_aug_classical_ddp --num-workers 32   # classical extraction is CPU-bound: use many workers
```

Checkpoints land in `logs/checkpoints/<run_name>/` (best `val/acc` per epoch + `last.ckpt`), TensorBoard logs in
`logs/tb/<run_name>/`. See [Training](#training) below for the architecture, the augmentation mixture, the
classical-fusion details and the single-GPU variants.

## Layout

```
src/configs/        dataset.yml  training.yml  augment.yml  eval.yml  comprehensive_eval.yml
src/dataset/        image_dataset.py (dir -> (path, label)), dataloader.py, collate.py (SigLIP2 NaFlex processor),
                    augment.py (offline augmentation CLI), online_augment.py (same augmentations, in the collate)
src/experiments/    train_single.py / train_ddp.py             clean training, 1 GPU / multi-GPU
                    train_aug_single.py / train_aug_ddp.py     training with on-the-fly augmentation
                    train_aug_classical_single.py / _ddp.py   the same, plus classical forensic features fused
                                                               into the QFormer input
                    eval.py                                    evaluate a checkpoint on the test set
                    comprehensive_eval.py                      evaluate a checkpoint per augmentation and option value
src/modules/        lora_adapter.py, attention.py (QFormer), classifier.py (MLP head)
classical_forensics.py   ClassicalFeatureExtractor: per-patch spectral / DCT / residual / wavelet / colour features
fusion_tokenizer.py      ClassicalTokenizer (classical features -> tokens), FeatureStandardizer, stats file I/O
src/dataset/classical_collate.py   the Siglip2 collates + the extractor run on every (augmented) image
src/utils/          one-off dataset helpers (arrow shard export, val split, SigLIP inspection)
dataset_generation/ captioner.py (image -> prompt JSON, OpenRouter), image_generator.py (prompt -> image, fal.ai),
                    img2txt_prompt_simplified.md (the captioning system prompt)
logs/               tb/ (TensorBoard), checkpoints/, eval/ (metrics.json + failures.csv, comprehensive_eval.json)
```

## Eval Quickstart

`src/configs/comprehensive_eval.yml` already points at the released checkpoint (`model_data/final.ckpt`) and the
clean test directories `data/ai_gen_test` / `data/real_test` — adjust these if yours live elsewhere — then run:

```bash
python src/experiments/comprehensive_eval.py
```

## Configs

Every script takes `--config <yml>` (default: its file in `src/configs/`) plus a few CLI flags that
override single values. The usual workflow: edit the yml, or copy it and pass the copy.

| File | Read by | What it sets |
|---|---|---|
| `dataset.yml` | all `train_*` scripts | the four image directories `real_img_{train,val}_ds_path` / `aigen_img_{train,val}_ds_path` and `batch_size` |
| `training.yml` | `train_*` scripts | `run_name`, SigLIP2 `model.checkpoint_path`, `lora` / `qformer` / `classifier` / `optimizer` hyper-parameters, `data` (points at `dataset.yml`, workers), `trainer` (epochs, precision, GPU, log / ckpt dirs); `online_augment` is read only by the `train_aug_*` scripts, `classical` only by the `train_aug_classical_single*` scripts and `ddp` only by the `*_ddp` scripts |
| `augment.yml` | `augment.py`; its `params` + `multi` blocks also by `train_aug_*` | offline run settings (`input_dir`, `output_dir`, `num_multi`, `workers`, `overwrite`, ...) and the augmentation parameter lists |
| `eval.yml` | `eval.py` | `model.ckpt_path`, `data.{real,aigen}_img_test_ds_path`, `device`, `precision`, `threshold`, `output.dir` |
| `comprehensive_eval.yml` | `comprehensive_eval.py` | `model.ckpt_path`, clean `data.ai_gen_dir` (label 1) / `data.real_dir` (label 0), `augment.params_config` (the grid = every option in its `params` block), `augment.jitter_random`, `augment.seed`, `device`, `precision`, `threshold`, `output.dir` |

Conventions:

- **A split is a pair of flat image directories**: `real_img_<split>_ds_path` (label 0) and
  `aigen_img_<split>_ds_path` (label 1). `.png .jpg .jpeg .webp` files are picked up.
- **Configs reference each other by path** (relative to the repo root): `training.yml` →
  `data.dataset_config: src/configs/dataset.yml` and `online_augment.params_config: src/configs/augment.yml`.
- **`run_name` decides where outputs land**: `logs/tb/<run_name>/`, `logs/checkpoints/<run_name>/`,
  `logs/eval/<run_name>/`. `online_augment.run_name` replaces it for augmented runs so they do not
  collide with the clean baseline.
- **GPU selection**: single-device scripts take the GPU from the config (`trainer.devices: [N]` in
  `training.yml`, `device: cuda:N` in `eval.yml`) — do *not* also set
  `CUDA_VISIBLE_DEVICES`, the indices would shift. DDP uses `ddp.cuda_visible_devices` (one rank per GPU);
  an externally set `CUDA_VISIBLE_DEVICES` takes precedence over it.
- `model.checkpoint_path` must hold the HF SigLIP2 NaFlex model **and** its image processor.

## Augmenting images (offline)

`src/dataset/augment.py` writes degraded copies of every image in a directory: six single augmentations
with fixed suffixes, plus `num_multi` random chains of two or more of them.

| Suffix | Augmentation | Parameter (one value drawn uniformly per output) |
|---|---|---|
| `_a1` | jpeg | quality 90 / 70 / 50 / 30 (in-memory round trip, saved as PNG) |
| `_a2` | blur | Gaussian sigma 0.5 / 1.0 / 2.0 |
| `_a3` | resize | downscale ×0.5 / ×0.25, then back to the original size |
| `_a4` | noise | additive Gaussian sigma 0.02 / 0.05 / 0.10 |
| `_a5` | jitter | brightness / contrast / saturation, each in [0.8, 1.2] |
| `_a6` | crop | centre crop to 80 % per side (output keeps the smaller size) |
| `_a7`… | chain | random subset of 2–6 of the above, random order |

```bash
# 1. point augment.input_dir (and optionally output_dir) in src/configs/augment.yml at your images
# 2. run
python -m src.dataset.augment --config src/configs/augment.yml --dry-run      # print the plan, write nothing
python -m src.dataset.augment --config src/configs/augment.yml --limit 4      # smoke test on 4 images
python -m src.dataset.augment --config src/configs/augment.yml                # full run, all cores

# or override on the command line
python -m src.dataset.augment --config src/configs/augment.yml \
    --input-dir /data/real_test --output-dir /data/real_with_aug_test --num-multi 3 --workers 32
python -m src.dataset.augment --config src/configs/augment.yml --set augment.params.blur.sigma=[3.0]
```

`--config` is required (the script only looks for `augment.yml` next to itself by default). Outputs are
`{stem}_a{n}.png` (`output_format`), written next to the originals when `output_dir` is null. Everything
is seeded from `blake2b(seed:filename)`: re-running reproduces identical pixels and an interrupted run
resumes where it stopped (`--overwrite` re-renders). Files already named `*_aN` are never used as sources,
so an in-place run can safely be repeated.

The `params` and `multi` blocks of this file are also the single source of truth for on-the-fly
augmentation during training (below).

## Training

Frozen SigLIP2 vision tower + LoRA on the attention projections (`lora.targets`) → QFormer (`qformer.m`
latent queries attending over the patch tokens) → MLP head → one logit, BCE loss. Only the LoRA, QFormer
and MLP parameters are trained.

| Script | Devices | Augmentation |
|---|---|---|
| `train_single.py` | one GPU, `trainer.devices` | none, datasets used as-is |
| `train_ddp.py` | one rank per GPU in `ddp.cuda_visible_devices` | none |
| `train_aug_single.py` | one GPU | on the fly, `online_augment` block |
| `train_aug_ddp.py` | multi-GPU | on the fly |
| `train_aug_classical_single.py` | one GPU | on the fly + classical forensic features, `classical` block |
| `train_aug_classical_ddp.py` | multi-GPU | on the fly + classical forensic features |

```bash
# smoke tests first (each checkpoint is ~2.2 GB: point trainer.ckpt_dir at scratch in a config copy)
python -m src.experiments.train_single --fast-dev-run --accelerator cpu --num-workers 0
python -m src.experiments.train_aug_single --limit-batches 5 --epochs 1
CUDA_VISIBLE_DEVICES=0,1 python -m src.experiments.train_aug_ddp --limit-batches 5 --epochs 1

# full runs
python -m src.experiments.train_single       # clean, single GPU
python -m src.experiments.train_aug_ddp      # augmented, GPUs from ddp.cuda_visible_devices
```

CLI overrides: `--config`, `--epochs`, `--batch-size`, `--num-workers`, `--precision`,
`--accelerator cpu|gpu`, `--fast-dev-run`, `--limit-batches N`. Under DDP `batch_size` is per device
(global = batch × GPUs) and each rank sees a distinct shard of both splits.

**On-the-fly augmentation** (`online_augment` in `training.yml`): each training image is, independently,
left untouched with `p_none`, given one random single augmentation with `p_single`, or one random chain
with `p_multi` (parameters from `params_config`, i.e. `augment.yml`). Train images are re-drawn every
epoch; validation follows `val_mode` — `deterministic` (fixed per file, so `val/acc` is comparable across
epochs and runs), `none` (clean) or `stream` (random like train).

**Classical forensic fusion** (`train_aug_classical_single.py` / `_ddp.py`, `classical` block): the
same streaming augmentation, but every augmented image — at its native resolution, before the SigLIP2
processor resizes it — is also run through `ClassicalFeatureExtractor` (`classical_forensics.py`) inside the
DataLoader workers: `n_rich + n_poor` texture-stratified, JPEG-grid-aligned crops of `patch` px, each with a
spectral / block-DCT / noise-residual / wavelet / colour-CFA descriptor, plus one global degradation
descriptor. In the model SigLIP2 is untouched up to `last_hidden_state`; those 256 tokens are projected by a
shallow MLP (`siglip_proj`) to `d_model` (null = 1152, the SigLIP2 width), the classical features become
`(n_rich + n_poor) × families + 1` tokens of the same width via `ClassicalTokenizer` (`fusion_tokenizer.py`:
per-family standardisation and projection MLPs, family / position embeddings, family dropout), and the
QFormer + MLP read the concatenation. The standardisation mean / std are estimated on `stats.num_images`
augmented train images before fitting and cached in `stats.dir/<run_name>.npz` (rank 0 computes, the file is
validated against the extractor config; they also travel inside the checkpoint). The extraction is CPU-bound
(~0.3 s per 1024 px image), so use many workers (`--num-workers 32`). An image whose extraction raises is
not fatal: it is warned with its path, gets `valid = 0` and its classical tokens are replaced by the
tokenizer's mask token (the missing-modality case family dropout trains). Stats files written by an older
`FEATURE_VERSION` of the extractor are recomputed automatically. `classical.run_name` replaces `run_name`,
like `online_augment.run_name`.

```bash
python -m src.experiments.train_aug_classical_single --limit-batches 5 --epochs 1 --num-workers 16   # smoke
python -m src.experiments.train_aug_classical_single --num-workers 32                                # one GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.experiments.train_aug_classical_ddp --num-workers 32
```

Outputs:

- TensorBoard: `logs/tb/<run_name>/version_N/` — `tensorboard --logdir logs/tb`
- Checkpoints: `logs/checkpoints/<run_name>/epochNN.ckpt` (best `val/acc`) and `last.ckpt`

### Evaluating a checkpoint

```bash
python -m src.experiments.eval                                                  # ckpt + test dirs from eval.yml
python -m src.experiments.eval --ckpt logs/checkpoints/<run>/last.ckpt --device 0 --limit-batches 5
```

The architecture is rebuilt from the hyper-parameters stored in the checkpoint, so `eval.yml` only needs
the checkpoint path, the two test directories, `device` and `threshold`. Results: a printed report plus
`logs/eval/<run_name>/<ckpt stem>/{metrics.json,failures.csv}`. Checkpoints of the classical-fusion scripts
are recognised by their weights and evaluated with the classical collate (`comprehensive_eval.py` and
`calibration.py` do not support them yet).

### Comprehensive evaluation per augmentation

```bash
python -m src.experiments.comprehensive_eval                                   # ckpt + clean dirs from comprehensive_eval.yml
python -m src.experiments.comprehensive_eval --device 0 --augments none,jpeg --limit-per-class 8   # smoke test
```

Takes two directories of **clean** images (`data.ai_gen_dir` = label 1, `data.real_dir` = label 0) and runs
one round per single augmentation and option value of `augment.params_config` (default `augment.yml`), plus an
un-augmented round — 21 rounds with the default lists: none, jpeg × 4 qualities, blur × 3 sigmas, resize × 2
scales, noise × 3 sigmas, jitter × 7 (each of brightness / contrast / saturation at 1 ± x with the other two at
1.0, plus one round drawing all three per file as training does), crop × 1. Each round applies its augmentation
on the fly to every image (no chains) and records the confusion matrix and accuracy / precision / recall / F1;
the noise field and the random jitter factors are seeded per file from `augment.seed`, so re-runs reproduce
the pixels. Results: a table on stdout plus `logs/eval/<run_name>/<ckpt stem>/comprehensive_eval.json` with the
rounds split by augmentation (and a per-augmentation mean), rewritten after every round. `--limit-per-class N`
takes the first N images of each class (the loader lists real before AI generated, so a batch limit would see
real images only).

## Other tools

| Command | Purpose |
|---|---|
| `python -m src.dataset.dataset_sanity_check` | plot the first images of a few train / val batches with their labels to `logs/` |
| `python leakage_check.py`, `python leak_removal.py` | find (and delete) train images that leak into val / test, by byte / pixel hash or SigLIP cosine > 0.9 |
| `python benchmark_dataload.py` | dataloader throughput sweep over worker counts |
| `python profile_dims.py` | image size / format statistics per dataset split |
| `python sample_dims.py` | copy K images (default 2) from each of the most common resolutions (those covering 90 % of the images) of one directory into `dim_samples/`, with `manifest.csv` / `buckets.csv` |
| `python sample_outofdim.py` | build the out-of-dimension real validation set: one header-only pass over the Open Images parquet shards on NFS (~70 min, resumable, cached per shard in `--scan-dir`) keeps every photo whose resolution is far from `real_all_toremove_train`'s (default: at most 0.5 % of the training images within ±15 % on both sides), drops stems already used by any dataset under `--exclude-root`, samples 20 k of them and, with `--execute`, copies them into `real_outofdimsample_val/`; other thresholds re-select in seconds without a rescan |
| `python inspect_siglip.py` | how the collate + SigLIP2 NaFlex processor + model handle varying resolutions: resize rule, padding, batch independence, augment collate, position embeddings — each claim checked on the `dim_samples/` images; writes the model's-eye canvases to `dim_samples/_siglip_view/` |
| `python qformer_attention.py` | per-head attention of the QFormer's final cross-attention layer (checkpoint from `eval.yml` or `--ckpt`) drawn on the rescaled + padded token canvas and un-scaled onto the original image for every `dim_samples/` file → `dim_samples/_attention/` (`*__tokens.png`, `*__original.png`, `attention.npz`) |
| `python qformer_attention_pair.py` | N real/AI pairs sharing a stem in `real_all_val` / `ai_gen_all_val`: head-mean QFormer attention on the token canvas for the clean pair and for 3 augmentation plans applied identically to both → `comparisons/` (`<stem>__clean.png`, `<stem>__augK.png`, `summary.csv`) |
| `python self_attention_pair.py` | the same pairs / plans / figures, but the heat is the attention each token *receives* in the SigLIP2 vision encoder's self-attention (all 27 layers × 16 heads, LoRA applied; `--agg rollout` for attention rollout) → `comparisons_self_attention/`; `summary.csv` adds the outer-ring share per layer |
| `python self_attention_noborder_pair.py` | `self_attention_pair.py` with the `--rings 2` outermost rings of patches dropped from the *visualisation* softmax (inference untouched), to look past border/register tokens; `--queries interior` also drops them as queries → `comparisons_self_attention_noborder/` |

## Dataset generation (API image synthesis)

Publicly available AI-generated corpora are narrow in resolution and subject matter, which a detector can
shortcut on. To make the AI-generated half of the dataset more diverse in dimension and content, part of it is
generated by us: a subset of the real photos (drawn from the Open Images subset) is captioned and then
re-generated through text-to-image APIs, so every real photo gets an AI counterpart with matched content, a
near-native resolution and the same filename stem — a higher quality, harder dataset. `dataset_generation/`
holds the two-stage pipeline:

1. **`captioner.py`** — captions every image in a directory with the Qwen3.8-27B vision model via the OpenRouter
   API. Each image is sent (downscaled to at most `--max-pixels`) together with the system prompt of
   `img2txt_prompt_simplified.md`, which demands a single 150–250-word reconstruction prompt (an `<analysis>`
   block then a `<prompt>` block; no real people or artist names). The parsed answer becomes one JSON per image —
   `{image_id, prompt, analysis, status, width, height, ...}`, with the ORIGINAL resolution recorded, not the
   downscaled one. Failed parses are retried (`--retries`); re-runs skip images already captioned `ok` /
   `truncated` and retry only `failed` / `error` ones.
2. **`image_generator.py`** — generates one synthetic image per usable caption JSON with HiDream-O1-Image via the
   fal.ai API. The generation size is planned from the original photo's recorded dimensions (aspect ratio kept,
   sides floored to multiples of 32, short side >= 256, long side <= 2048 — the endpoint's grid), so the
   synthetic images inherit the real photos' resolution diversity instead of a fixed 1024x1024. Outputs are saved
   as `{image_id}.<ext>`: real / generated pairs share a stem, which `make_val_split.py` and the pair
   explainability tools rely on.

```bash
pip install openrouter fal-client       # only needed for this pipeline
export OPENROUTER_API_KEY=...           # https://openrouter.ai/keys
export FAL_KEY=...                      # https://fal.ai/dashboard/keys

python dataset_generation/captioner.py ./real_subset -o ./prompts        # stage 1: image -> prompt JSON
python dataset_generation/image_generator.py ./prompts -o ./generated    # stage 2: prompt JSON -> image

# no-network checks
python dataset_generation/captioner.py ./real_subset --dry-run           # show what would be sent
python dataset_generation/captioner.py --selftest                        # output parser
python dataset_generation/image_generator.py --selftest              # resolution planner
```

Both stages are resumable (existing outputs are skipped; `--force` redoes them) and print a per-file status plus
a final count.
