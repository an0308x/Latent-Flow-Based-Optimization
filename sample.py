#!/usr/bin/env python3
"""
CHASE: Sample high-fitness protein sequences.

Loads trained VAE and flow matching model, generates sequences,
applies top-k ranking, and evaluates against oracle.

Usage:
    python scripts/sample.py \
        --dataset gfp_medium \
        --checkpoint_dir checkpoints/ \
        --target_fitness 0.8 \
        --guidance_scale -0.08 \
        --n_samples 512 \
        --n_seeds 5 \
        --output results/gfp_medium.csv
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd
import torch
import esm

from models.vae import ProteinVAE
from models.flow_matching import UNetVelocityField, ConditionalFlowMatcher
from data.dataset import load_benchmark
from utils.metrics import tokens_to_sequences, compute_diversity, compute_novelty
from configs.benchmark_configs import CONFIGS


def load_chase(cfg: dict, checkpoint_dir: str, dataset: str, device):
    """Load a trained CHASE model from checkpoints."""
    esm_model, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
    esm_model = esm_model.eval()

    vae = ProteinVAE(
        esm_model=esm_model,
        esm_dim=cfg["esm_dim"],
        latent_dim=cfg["latent_dim"],
        compression=cfg["compression"],
        beta=cfg["beta_vae"],
        n_transformer_layers=cfg["n_transformer_layers"],
        n_attn_heads=cfg["n_attn_heads"],
    )

    vae_ckpt = os.path.join(checkpoint_dir, dataset, "stage2", "vae_stage2.pt")
    ckpt = torch.load(vae_ckpt, map_location="cpu")
    vae.compressor.load_state_dict(ckpt["compressor"])
    vae.decompressor.load_state_dict(ckpt["decompressor"])
    vae.decoder.load_state_dict(ckpt["decoder"])
    vae = vae.to(device).eval()

    velocity_field = UNetVelocityField(
        latent_dim=cfg["latent_dim"],
        base_channels=128,
        cond_dim=256,
        n_downup_blocks=2,
    )
    flow_matcher = ConditionalFlowMatcher(velocity_field=velocity_field, score_dropout=0.0)

    flow_ckpt = os.path.join(checkpoint_dir, dataset, "flow", "flow_best.pt")
    flow_matcher.load_state_dict(torch.load(flow_ckpt, map_location="cpu"))
    flow_matcher = flow_matcher.to(device).eval()

    return vae, flow_matcher, alphabet


@torch.no_grad()
def generate_sequences(
    vae,
    flow_matcher,
    alphabet,
    target_fitness: float,
    guidance_scale: float,
    n_samples: int,
    n_ode_steps: int,
    latent_len: int,
    device,
) -> list:
    """Generate amino acid sequences via ODE sampling."""
    d = vae.compressor.to_mean.out_features
    shape = (n_samples, latent_len, d)

    z_gen = flow_matcher.sample(
        shape=shape,
        f=target_fitness,
        guidance_scale=guidance_scale,
        n_steps=n_ode_steps,
        device=device,
    )

    h_prime = vae.decompressor(z_gen)
    logits = vae.decoder(h_prime)
    token_ids = logits.argmax(dim=-1)
    sequences = tokens_to_sequences(token_ids, alphabet)
    return sequences


def main():
    parser = argparse.ArgumentParser(description="Sample from trained CHASE model")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["aav_medium", "aav_hard", "gfp_medium", "gfp_hard"])
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/")
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--target_fitness", type=float, default=None,
                        help="Target fitness for conditional sampling. Defaults to config value.")
    parser.add_argument("--guidance_scale", type=float, default=None,
                        help="CFG weight w. Defaults to config value.")
    parser.add_argument("--n_samples", type=int, default=512)
    parser.add_argument("--top_k", type=int, default=128)
    parser.add_argument("--n_seeds", type=int, default=5,
                        help="Number of independent runs")
    parser.add_argument("--n_ode_steps", type=int, default=40)
    parser.add_argument("--output", type=str, default="results.csv")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    cfg = CONFIGS[args.dataset]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    target_fitness = args.target_fitness or cfg["target_fitness"]
    guidance_scale = args.guidance_scale or cfg["guidance_scale"]

    print(f"Loading CHASE for {args.dataset}...")
    vae, flow_matcher, alphabet = load_chase(cfg, args.checkpoint_dir, args.dataset, device)

    # Load training sequences for novelty computation
    train_sequences, train_fitness = load_benchmark(args.data_dir, args.dataset)

    # Infer latent sequence length
    seq_len = cfg["seq_len"]
    latent_len = seq_len // cfg["compression"]

    all_results = []

    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        print(f"\n=== Seed {seed+1}/{args.n_seeds} ===")

        sequences = generate_sequences(
            vae=vae,
            flow_matcher=flow_matcher,
            alphabet=alphabet,
            target_fitness=target_fitness,
            guidance_scale=guidance_scale,
            n_samples=args.n_samples,
            n_ode_steps=args.n_ode_steps,
            latent_len=latent_len,
            device=device,
        )

        # Filter empty sequences
        sequences = [s for s in sequences if len(s) > 0]
        print(f"Generated {len(sequences)} valid sequences")

        # NOTE: In the full pipeline, sequences would be ranked here using
        # the external predictor from Kirjner et al. (2023) and top-k selected.
        # Since we don't have that predictor here, we select randomly as placeholder.
        selected = sequences[:args.top_k] if len(sequences) >= args.top_k else sequences

        # Compute diversity and novelty (fitness requires oracle)
        diversity = compute_diversity(selected)
        novelty = compute_novelty(selected, train_sequences)

        result = {
            "seed": seed,
            "n_generated": len(sequences),
            "n_selected": len(selected),
            "diversity": diversity,
            "novelty": novelty,
        }
        all_results.append(result)
        print(f"Diversity: {diversity:.2f} | Novelty: {novelty:.2f}")

        # Save sequences for this seed
        seed_output = args.output.replace(".csv", f"_seed{seed}.fasta")
        os.makedirs(os.path.dirname(seed_output) or ".", exist_ok=True)
        with open(seed_output, "w") as f:
            for i, seq in enumerate(selected):
                f.write(f">generated_{seed}_{i}\n{seq}\n")
        print(f"Saved to {seed_output}")

    # Summary
    df = pd.DataFrame(all_results)
    print("\n=== Summary ===")
    print(df[["diversity", "novelty"]].describe())

    df.to_csv(args.output, index=False)
    print(f"\nResults saved to {args.output}")
    print("\nNote: Compute fitness using the oracle from Kirjner et al. (2023)")
    print("      https://github.com/kirjner/GGS")


if __name__ == "__main__":
    main()
