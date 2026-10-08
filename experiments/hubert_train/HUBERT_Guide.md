# HuBERT

4 GPUs, Nvidia V100 32GB, precision fp32.

```shell
    cd /home/vrdauer/VK-SSL/experiments/hubert_train
```

## 0. Datasets and duration caches

`/home/vrdauer/VK-SSL/experiments/ctc_train/librispeech`

`LibriSpeech: train-500, traint-360, train-clean-100,dev-clean,dev-other`

`Libri-Light: librispeech_finetuning`


```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m src.data.duration_cache \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --subset 100h ll-10h \
        --scan-workers 32
```

## 1. Baseline: fine-tune on 10 hours from random init (no pre-training)

Same data, preset, batch and model size as the pre-trained fine-tunes in section 5, only the
initialisation differs. `--random-init` never reads a checkpoint (passing `--pretrained-path`
as well is an error), trains the CNN feature extractor end to end and sets `--freeze-steps 0`.
Look for `[init] encoder: random init` in the log.

Pairs with the `small` pipeline (sections 3–5):

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 finetune_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --random-init \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h_scratch_small \
        --preset 10h-pt100h \
        --model-size small \
        --gpus 4
```

Pairs with the old `base` + fairseq `10h` fine-tune (`lightning_logs/fine`, dev WER 0.775):

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 finetune_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --random-init \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h_scratch_base \
        --preset 10h \
        --model-size base \
        --gpus 4
```

The preset LR (2e-5 / 5e-5) is tuned for pre-trained encoders, so this is the strict
"same recipe" control. For a scratch-tuned control add `--lr 5e-4 --gradient-clip-val 10.0`
(not validated yet).

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python eval_hubert.py \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h_scratch_small/checkpoints/<best>.ckpt \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-type char \
        --use-cuda
```

## 2. Teacher labels, iteration 1

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h \
        --feature-type mfcc \
        --subsets train-clean-100 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/km_100.bin \
        --n-clusters 100 \
        --percent 1.0 \
        --max-frames 50000000 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/km_100.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/all.km
```

## 3. Pre-train on 100 hours

`small` (8 layers × 512, FFN 2048, 8 heads, 31.5M params) instead of `base` (94.4M): on 100 h
`base` over-fits the MFCC targets (see `REPORT.md`). Effective batch 4 × 87.5 s × 4 = 1400 s,
100k steps ≈ 390 epochs of 100 h. `small` is only ~1.4× cheaper per step than `base` (the CNN
dominates), so this takes ~40 h on 4 × V100: plan for at least one resume from `last.ckpt`.

GER and RankMe-t (`compute_ger.py` metrics) are computed at the end of every epoch on the first
`--ger-max-seconds` of dev audio, last layer by default (`--ger-layer`). They are logged as
`Metrics/val_ger` / `Metrics/val_rankme_t` to TensorBoard and `metrics.csv`, and printed as
`[GER] epoch=... step=... GER=... RankMe-t=...`. `--ger-max-seconds 0` disables it.

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc_100h/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h_small \
        --train-subsets train-clean-100 \
        --model-size small \
        --num-classes 100 \
        --label-rate 100 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 100000 \
        --accumulate-grad-batches 4 \
        --ger-max-seconds 3600 \
        --gpus 4
```

Resume after a job time limit (same command plus):

```shell
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h_small/checkpoints/last.ckpt
```

The previous `base` recipe (`--model-size base --max-steps 250000 --accumulate-grad-batches 8`,
`exp_hubert_it1_100h`) still works; its only run (`lightning_logs/hubert_12`) stopped at step
21349 of 250000.

## 4. Iteration 2

Layer 4 of 8 for `small` (half depth, as layer 6 of 12 for `base`).

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small \
        --feature-type hubert \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h_small/checkpoints/<best>.ckpt \
        --model-size small \
        --num-classes 100 \
        --label-rate 100 \
        --layer 4 \
        --use-cuda \
        --subsets train-clean-100 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/km_500.bin \
        --n-clusters 500 \
        --percent 0.1 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/km_500.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/all.km
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/hubert4_100h_small/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it2_100h_small \
        --train-subsets train-clean-100 \
        --model-size small \
        --num-classes 500 \
        --label-rate 50 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 100000 \
        --accumulate-grad-batches 4 \
        --gpus 4
```

## 5. Fine-tune on 10 hours

`--preset 10h-pt100h`: LR 5e-5, tri-stage 2500 / 10000 / 12500 (fully decayed at step 25000),
head-only for the first 2000 steps. The log prints `[init] encoder: loaded N/N tensors ...
step=...`, and a size or key mismatch is an error. `--num-classes` / `--label-rate` must match
the pre-train run (`100` / `100` for iteration 1, `500` / `50` for iteration 2).

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 finetune_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --pretrained-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h_small/checkpoints/<best>.ckpt \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h_small \
        --preset 10h-pt100h \
        --model-size small \
        --num-classes 100 \
        --label-rate 100 \
        --gpus 4
```

Optional gradual unfreeze (top layer first, all layers released by step 2000 + 4000):

```shell
        --unfreeze-steps 4000
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python eval_hubert.py \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_ft10h_small/checkpoints/<best>.ckpt \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-type char \
        --use-cuda
```

## Reference

### Pre-train settings vs fairseq

| Item | fairseq | Repo |
|---|---|---|
| Transformer | 12 / 768 / **12 heads** / 3072 | same for `base`; `small` = 8 / 512 / 8 heads / 2048 |
| mask | `mask_prob 0.80` = 0.08·T starts, span 10, `min_masks 2` | `--mask-prob`, our formula counts starts directly, so fairseq value ÷ 10 |
| loss | masked-only + `10 · mean(conv_out²)` | `mask_alpha=1.0`, `feature_penalty_weight=10.0` |
| `feature_grad_mult` | 0.1 | same |
| `clip_norm` | 10.0 | `gradient_clip_val=10.0` |
| Adam | lr `5e-4`, β `(0.9,0.98)`, eps `1e-6`, wd `0.01` (**decoupled**, fairseq Adam ≡ AdamW) | `torch.optim.AdamW`, same values |
| schedule | `polynomial_decay`, warmup **8%**, decay to 0 | `warmup_ratio=0.08` |
| steps | it1 **250k**, it2 **400k** (960 h) | `--max-steps` |
| batch | 32 GPUs × 87.5 s = 2800 s | `--gpus × --max-batch-duration × --accumulate-grad-batches` |
| precision | fp16 | **fp32 hardcoded** |
| labels | it1 MFCC → 100 @ 100 Hz, it2 layer 6 → 500 @ 50 Hz | `--num-classes`, `--label-rate` |

### Fine-tune presets

| Preset | data | lr | steps | tri-stage warmup/hold/decay | freeze | mask | effective batch |
|---|---|---|---|---|---|---|---|
| `10h` | Libri-Light 10h (`ll-10h`) | 2e-5 | 25000 | 8000 / 0 / 72000 | 10000 | 0.075 | 200 s |
| `10h-pt100h` | Libri-Light 10h (`ll-10h`) | 5e-5 | 25000 | 2500 / 10000 / 12500 | 2000 | 0.075 | 200 s |
| `100h` | `train-clean-100` | 3e-5 | 80000 | 8000 / 32000 / 40000 | 10000 | 0.065 | 1600 s |

`10h` keeps the fairseq `base_10h` values, tuned for an encoder pre-trained on 960 h; its decay
is cut at step 25000 with LR still at ~49% of peak. `10h-pt100h` is for encoders pre-trained on
100 h: higher LR (wav2vec 2.0 `base_10h` value), phases 10/40/50% of `max_steps` so the LR
actually reaches `final_lr_scale`, and a short head-only phase.

| Flag | Meaning |
|---|---|
| `--random-init` | no checkpoint, CNN trained, `--freeze-steps 0` unless given |
| `--freeze-steps N` | only the CTC head trains for N steps (encoder under `no_grad`, DDP-safe) |
| `--unfreeze-steps N` | after the freeze, release layers top-down over N steps (default 0 = at once) |
| `--gradient-clip-val V` | grad-norm clipping, default 0 |

| Item | fairseq | Repo |
|---|---|---|
| targets | `labels: ["ltr"]` — 26 letters + `'` + `\|` | `--label-type char`, 28 + blank |
| criterion | ctc, `zero_infinity`, `sentence_avg` | same |
| optimizer | Adam, β `(0.9,0.98)`, eps **1e-8**, no wd | same |
| `final_lr_scale` | 0.05 | same |
| freeze | `freeze_finetune_updates 10000` | `--freeze-steps` (preset) |
| channel mask | `mask_channel_prob 0.5`, length 64 | same |
| dropouts | `dropout 0`, `attention 0`, `activation 0.1`, `layerdrop 0.1` | same |
| `feature_grad_mult` | 0.0 | CNN frozen |

## Full 960 hours

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m src.data.duration_cache \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --subset all \
        --scan-workers 32
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python extract_features.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --out-dir /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc \
        --feature-type mfcc \
        --subsets train-clean-100 train-clean-360 train-other-500 dev-clean dev-other
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python learn_kmeans.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/km_100.bin \
        --n-clusters 100 \
        --percent 1.0 \
        --max-frames 50000000 \
        --batch-size 10000 \
        --n-init 20 \
        --max-iter 100
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python dump_labels.py \
        --features-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/features.npy \
        --index-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/index.json \
        --km-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/km_100.bin \
        --out-path /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/all.km
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python -m torch.distributed.run --nproc_per_node=4 train_hubert.py \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --label-paths /home/vrdauer/VK-SSL/experiments/hubert_train/labels/mfcc/all.km \
        --exp-dir /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1 \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --mask-alpha 1.0 \
        --mask-prob 0.08 \
        --lr 5e-4 \
        --weight-decay 0.01 \
        --warmup-ratio 0.08 \
        --max-batch-duration 87.5 \
        --max-steps 250000 \
        --accumulate-grad-batches 8 \
        --gpus 4
```

```shell
    PYTHONPATH=/home/vrdauer/VK-SSL python compute_ger.py \
        --checkpoint-path /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/checkpoints/<best>.ckpt \
        --librispeech-path /home/vrdauer/VK-SSL/experiments/ctc_train/librispeech \
        --model-size base \
        --num-classes 100 \
        --label-rate 100 \
        --layer 12 \
        --subsets train-clean-100 \
        --max-seconds 3600 \
        --use-cuda \
        --out-json /home/vrdauer/VK-SSL/experiments/hubert_train/exp_hubert_it1_100h/ger.json
```

`--layer` defaults to the last layer of `--model-size` (12 for `base`, 8 for `small`). The
per-epoch `Metrics/val_ger` in pre-training uses the first `--ger-max-seconds` of
`dev-clean dev-other`, padded batches; use `--subsets dev-clean dev-other` here for the closest
offline match.