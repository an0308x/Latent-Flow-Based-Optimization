#!/usr/bin/env python3
"""
CHASE Bootstrapping: Augment training data with synthetic sequences and retrain.

Algorithm 1 from the paper (Section 3.4):
1. Generate synthetic sequences across evenly-spaced fitness targets
2. Perturb labels with Gaussian noise (Eq. 7)
3. Combine with original data and retrain flow model

Usage:
    python scripts/bootstrap.py \
        --dataset aav_medium \
        --checkpoint_dir checkpoints/ \
        --data_dir data/ \
        --output_dir checkpoints/
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import esm

from models.vae import ProteinVAE
from models.flow_matching import UNetVelocityField, ConditionalFlowMatcher
from data.dataset import get_dataloaders, ProteinFitnessDataset, BootstrappedDataset, load_benchmark
from utils.metrics import bootstrap_dataset
from utils.training import train_flow_matching
from configs.benchmark_configs import CONFIGS


def main():
    parser = argparse.ArgumentParser(description="Bootstrap CHASE with synthetic data")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["aav_medium", "aav_hard", "gfp_medium", "gfp_hard"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(args.seed)

    # Load ESM2
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model = esm_model.eval()

    # Build and load VAE
    vae = ProteinVAE(
        esm_model=esm_model,
        esm_dim=cfg["esm_dim"],
        latent_dim=cfg["latent_dim"],
        compression=cfg["compression"],
        beta=cfg["beta_vae"],
        n_transformer_layers=cfg["n_transformer_layers"],
        n_attn_heads=cfg["n_attn_heads"],
    )
    vae_ckpt = os.path.join(args.checkpoint_dir, args.dataset, "stage2", "vae_stage2.pt")
    ckpt = torch.load(vae_ckpt, map_location="cpu")
    vae.compressor.load_state_dict(ckpt["compressor"])
    vae.decompressor.load_state_dict(ckpt["decompressor"])
    vae.decoder.load_state_dict(ckpt["decoder"])
    vae = vae.to(device).eval()

    # Build and load flow model (trained without bootstrapping)
    velocity_field = UNetVelocityField(
        latent_dim=cfg["latent_dim"], base_channels=128, cond_dim=256, n_downup_blocks=2
    )
    flow_matcher_base = ConditionalFlowMatcher(
        velocity_field=velocity_field,
        score_dropout=cfg.get("bootstrap_score_dropout", 0.1),
    )
    flow_ckpt = os.path.join(args.checkpoint_dir, args.dataset, "flow", "flow_best.pt")
    flow_matcher_base.load_state_dict(torch.load(flow_ckpt, map_location="cpu"))
    flow_matcher_base = flow_matcher_base.to(device)

    # Original data loaders
    train_loader, val_loader, (f_min, f_max) = get_dataloaders(
        args.data_dir, args.dataset, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
    )

    n_base = len(train_loader.dataset)
    seq_len = cfg["seq_len"]
    latent_len = seq_len // cfg["compression"]
    d = vae.compressor.to_mean.out_features

    # Bootstrap: generate synthetic pairs
    print("\n=== Generating Synthetic Data ===")
    syn_sequences, syn_fitness = bootstrap_dataset(
        flow_matcher=flow_matcher_base,
        vae=vae,
        target_interval=cfg["bootstrap_interval"],
        n_targets=cfg["bootstrap_n_targets"],
        expand_factor=cfg.get("bootstrap_expand_factor", 0.25),
        n_base=n_base,
        label_noise_scale=cfg["bootstrap_label_noise"],
        guidance_scale=0.0,  # w=0 for bootstrap generation (Appendix B.3)
        n_ode_steps=cfg["n_ode_steps"],
        device=device,
    )
    print(f"Generated {len(syn_sequences)} synthetic sequences")

    # Build bootstrapped dataset and retrain flow model
    original_sequences, original_fitness = load_benchmark(args.data_dir, args.dataset)
    # Combine: train on A_aug = A ∪ A_syn
    aug_sequences = original_sequences + syn_sequences
    aug_fitness = np.concatenate([
        np.array(original_fitness, dtype=np.float32),
        syn_fitness,
    ])

    # Normalize synthetic fitness to [0,1]
    aug_fitness_norm = (aug_fitness - f_min) / (f_max - f_min + 1e-8)
    aug_fitness_norm = np.clip(aug_fitness_norm, 0, 1.5)  # allow extrapolation

    batch_converter = alphabet.get_batch_converter()
    aug_ds = ProteinFitnessDataset(aug_sequences, aug_fitness_norm, batch_converter)

    from torch.utils.data import DataLoader, random_split
    n_train = int(len(aug_ds) * 0.8)
    n_val = len(aug_ds) - n_train
    gen = torch.Generator().manual_seed(args.seed)
    train_aug, val_aug = random_split(aug_ds, [n_train, n_val], generator=gen)

    aug_train_loader = DataLoader(
        train_aug, batch_size=args.batch_size, shuffle=True,
        collate_fn=aug_ds.collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    aug_val_loader = DataLoader(
        val_aug, batch_size=args.batch_size, shuffle=False,
        collate_fn=aug_ds.collate_fn, num_workers=args.num_workers, pin_memory=True
    )

    # Retrain flow model on augmented data
    print("\n=== Retraining Flow Model on Augmented Data ===")
    velocity_field_bs = UNetVelocityField(
        latent_dim=cfg["latent_dim"], base_channels=128, cond_dim=256, n_downup_blocks=2
    )
    flow_matcher_bs = ConditionalFlowMatcher(
        velocity_field=velocity_field_bs,
        score_dropout=cfg.get("bootstrap_score_dropout", 0.1),
    )
    # Initialize from base model weights
    flow_matcher_bs.load_state_dict(flow_matcher_base.state_dict())

    bootstrap_out = os.path.join(args.output_dir, args.dataset, "flow_bootstrapped")
    train_flow_matching(
        flow_matcher=flow_matcher_bs,
        vae=vae,
        train_loader=aug_train_loader,
        val_loader=aug_val_loader,
        output_dir=bootstrap_out,
        lr=cfg["flow_lr"],
        warmup_steps=cfg["flow_warmup"],
        train_steps=cfg["flow_train_steps"],
        device=device,
        use_wandb=args.use_wandb,
    )

    print(f"\nBootstrapped model saved to {bootstrap_out}")


if __name__ == "__main__":
    main()
