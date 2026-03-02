# Latent-Flow-Based-Optimization

# CHASE: Conditional High-fitness Amino acid Sequence Enhancement

Reproduction of: **"Repurposing Protein Language Models for Latent Flow–Based Fitness Optimization"** (Caceres Arroyo et al., 2026, arXiv:2602.02425)

## Overview

CHASE is a framework for protein fitness optimization that:
1. Encodes protein sequences using a pretrained ESM2 protein language model
2. Compresses embeddings into a compact latent manifold via a β-VAE (compressor/decompressor)
3. Trains a conditional flow matching model with classifier-free guidance
4. Generates high-fitness variants without predictor-based guidance during ODE sampling

## Architecture

```
Sequence → ESM2 Encoder → Compressor → Latent z
                                          ↕  (Flow Matching with fitness conditioning)
Sequence ← ESM2 Decoder ← Decompressor ← Latent z'
```

## Setup

```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare Data

Download the GFP/AAV benchmark datasets from:
- Kirjner et al. (2023): https://github.com/kirjner/GGS

```bash
python scripts/prepare_data.py --dataset gfp --split medium --output_dir data/
```

### 2. Train VAE (Stage 1: Decoder pretraining)

```bash
python scripts/train_vae.py \
    --dataset gfp_medium \
    --data_dir data/ \
    --output_dir checkpoints/vae/ \
    --stage 1
```

### 3. Train VAE (Stage 2: Compressor/Decompressor)

```bash
python scripts/train_vae.py \
    --dataset gfp_medium \
    --data_dir data/ \
    --output_dir checkpoints/vae/ \
    --stage 2 \
    --vae_checkpoint checkpoints/vae/stage1/best.pt
```

### 4. Train Flow Matching Model

```bash
python scripts/train_flow.py \
    --dataset gfp_medium \
    --data_dir data/ \
    --vae_checkpoint checkpoints/vae/stage2/best.pt \
    --output_dir checkpoints/flow/ \
    --score_dropout 0.0 \
    --train_steps 600000
```

### 5. Sample High-Fitness Sequences

```bash
python scripts/sample.py \
    --dataset gfp_medium \
    --vae_checkpoint checkpoints/vae/stage2/best.pt \
    --flow_checkpoint checkpoints/flow/best.pt \
    --target_fitness 0.8 \
    --guidance_scale -0.08 \
    --n_samples 512 \
    --output sequences.fasta
```

### 6. Bootstrapping (Optional)

```bash
python scripts/bootstrap.py \
    --dataset gfp_medium \
    --vae_checkpoint checkpoints/vae/stage2/best.pt \
    --flow_checkpoint checkpoints/flow/best.pt \
    --output_dir checkpoints/flow_bootstrapped/
```

## Configuration

Pre-set configs for all 4 benchmarks are in `configs/`.

## Benchmarks

| Dataset    | CHASE Fitness | CHASE Bootstrapped |
|------------|---------------|--------------------|
| AAV Medium | 0.62          | 0.65               |
| AAV Hard   | 0.61          | 0.63               |
| GFP Medium | 0.91          | 0.93               |
| GFP Hard   | 0.92          | 0.87               |

## Citation

```bibtex
@article{caceresarroyo2026chase,
  title={Repurposing Protein Language Models for Latent Flow-Based Fitness Optimization},
  author={Caceres Arroyo, Amaru and Bogensperger, Lea and Allam, Ahmed and Krauthammer, Michael and Schindler, Konrad and Narnhofer, Dominik},
  journal={arXiv preprint arXiv:2602.02425},
  year={2026}
}
```
