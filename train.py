#!/usr/bin/env python3
"""
CHASE: Main training script.

Runs the full two-stage VAE training followed by conditional flow matching.

Usage:
    python scripts/train.py --dataset gfp_medium --data_dir data/ --output_dir checkpoints/

For all options:
    python scripts/train.py --help
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch
import esm

from models.vae import ProteinVAE, ESM2Decoder, Compressor, Decompressor
from models.flow_matching import UNetVelocityField, ConditionalFlowMatcher
from data.dataset import get_dataloaders
from utils.training import train_decoder_stage1, train_vae_stage2, train_flow_matching
from configs.benchmark_configs import CONFIGS


def build_model(cfg: dict, esm_model):
    """Construct ProteinVAE and ConditionalFlowMatcher from config."""
    vae = ProteinVAE(
        esm_model=esm_model,
        esm_dim=cfg["esm_dim"],
        latent_dim=cfg["latent_dim"],
        compression=cfg["compression"],
        beta=cfg["beta_vae"],
        n_transformer_layers=cfg["n_transformer_layers"],
        n_attn_heads=cfg["n_attn_heads"],
    )

    velocity_field = UNetVelocityField(
        latent_dim=cfg["latent_dim"],
        base_channels=128,
        cond_dim=256,
        n_downup_blocks=2,
    )
    flow_matcher = ConditionalFlowMatcher(
        velocity_field=velocity_field,
        score_dropout=cfg["score_dropout"],
    )

    return vae, flow_matcher


def main():
    parser = argparse.ArgumentParser(description="Train CHASE protein fitness optimizer")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["aav_medium", "aav_hard", "gfp_medium", "gfp_hard"],
                        help="Benchmark dataset to train on")
    parser.add_argument("--data_dir", type=str, default="data/",
                        help="Directory containing benchmark CSV files")
    parser.add_argument("--output_dir", type=str, default="checkpoints/",
                        help="Where to save model checkpoints")
    parser.add_argument("--stage", type=str, default="all",
                        choices=["1", "2", "flow", "all"],
                        help="Which training stage to run")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="chase")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cuda/cpu). Defaults to cuda if available.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    cfg = CONFIGS[args.dataset]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    print(f"Dataset: {args.dataset}")

    if args.use_wandb:
        import wandb
        wandb.init(project=args.wandb_project, config={**cfg, **vars(args)})

    # Load ESM2
    print("Loading ESM2-8M...")
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model = esm_model.eval()

    # Data
    print("Loading data...")
    train_loader, val_loader, (f_min, f_max) = get_dataloaders(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"Train: {len(train_loader.dataset)} | Val: {len(val_loader.dataset)}")
    print(f"Fitness range: [{f_min:.3f}, {f_max:.3f}]")

    # Build models
    vae, flow_matcher = build_model(cfg, esm_model)

    stage1_dir = os.path.join(args.output_dir, args.dataset, "stage1")
    stage2_dir = os.path.join(args.output_dir, args.dataset, "stage2")
    flow_dir   = os.path.join(args.output_dir, args.dataset, "flow")

    # -----------------------------------------------------------------------
    # Stage 1: Decoder pretraining
    # -----------------------------------------------------------------------
    if args.stage in ("1", "all"):
        print("\n=== Stage 1: Decoder Pretraining ===")
        train_decoder_stage1(
            esm_model=esm_model,
            decoder=vae.decoder,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=stage1_dir,
            lr=cfg["stage1_lr"],
            warmup_steps=cfg["stage1_warmup"],
            max_epochs=cfg["stage1_epochs"],
            patience=cfg["stage1_patience"],
            eval_every=cfg["stage1_eval_every"],
            device=device,
            use_wandb=args.use_wandb,
        )

    # Load Stage 1 decoder checkpoint
    decoder_ckpt = os.path.join(stage1_dir, "decoder_stage1.pt")
    if os.path.exists(decoder_ckpt):
        vae.decoder.load_state_dict(torch.load(decoder_ckpt, map_location="cpu"))
        print(f"Loaded Stage 1 decoder from {decoder_ckpt}")

    # -----------------------------------------------------------------------
    # Stage 2: Compressor/Decompressor training
    # -----------------------------------------------------------------------
    if args.stage in ("2", "all"):
        print("\n=== Stage 2: VAE Compressor/Decompressor Training ===")
        train_vae_stage2(
            vae=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=stage2_dir,
            lr=cfg["stage2_lr"],
            warmup_steps=cfg["stage2_warmup"],
            max_epochs=cfg["stage2_epochs"],
            patience=cfg["stage2_patience"],
            eval_every=cfg["stage2_eval_every"],
            device=device,
            use_wandb=args.use_wandb,
        )

    # Load Stage 2 checkpoint
    vae_ckpt = os.path.join(stage2_dir, "vae_stage2.pt")
    if os.path.exists(vae_ckpt):
        ckpt = torch.load(vae_ckpt, map_location="cpu")
        vae.compressor.load_state_dict(ckpt["compressor"])
        vae.decompressor.load_state_dict(ckpt["decompressor"])
        vae.decoder.load_state_dict(ckpt["decoder"])
        print(f"Loaded Stage 2 VAE from {vae_ckpt}")

    # -----------------------------------------------------------------------
    # Flow Matching training
    # -----------------------------------------------------------------------
    if args.stage in ("flow", "all"):
        print("\n=== Flow Matching Training ===")
        # Update score_dropout from config (may differ from default)
        flow_matcher.score_dropout = cfg["score_dropout"]
        train_flow_matching(
            flow_matcher=flow_matcher,
            vae=vae,
            train_loader=train_loader,
            val_loader=val_loader,
            output_dir=flow_dir,
            lr=cfg["flow_lr"],
            warmup_steps=cfg["flow_warmup"],
            train_steps=cfg["flow_train_steps"],
            device=device,
            use_wandb=args.use_wandb,
        )

    print("\n=== Training Complete ===")
    print(f"Checkpoints saved to: {args.output_dir}/{args.dataset}/")


if __name__ == "__main__":
    main()
