"""
Training routines for CHASE.

Two-stage VAE training:
  Stage 1: Pretrain ESM2 decoder (cross-entropy loss on sequence reconstruction)
  Stage 2: Train compressor/decompressor (full VAE objective, Eq. 2)

Then: Train conditional flow matching model (Eq. 4).
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR
from tqdm import tqdm
import wandb
import esm

from models.vae import ProteinVAE, ESM2Decoder, Compressor, Decompressor
from models.flow_matching import UNetVelocityField, ConditionalFlowMatcher
from data.dataset import get_dataloaders


# ---------------------------------------------------------------------------
# Stage 1: Decoder pretraining
# ---------------------------------------------------------------------------

def train_decoder_stage1(
    esm_model,
    decoder: ESM2Decoder,
    train_loader,
    val_loader,
    output_dir: str,
    lr: float = 5e-5,
    warmup_steps: int = 200,
    max_epochs: int = 30,
    patience: int = 8,
    eval_every: int = 50,
    device: torch.device = None,
    use_wandb: bool = False,
):
    """
    Train the ESM2 decoder to reconstruct sequences from frozen ESM2 embeddings.
    Loss: cross-entropy between predicted and true token sequences.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    esm_model = esm_model.to(device).eval()
    decoder = decoder.to(device)

    optimizer = AdamW(decoder.parameters(), lr=lr)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup], milestones=[warmup_steps])

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(max_epochs):
        decoder.train()
        for tokens, _ in tqdm(train_loader, desc=f"Stage1 Epoch {epoch+1}"):
            tokens = tokens.to(device)

            with torch.no_grad():
                results = esm_model(tokens, repr_layers=[esm_model.num_layers])
                h = results["representations"][esm_model.num_layers]

            logits = decoder(h)
            B, L, V = logits.shape
            loss = nn.functional.cross_entropy(
                logits.reshape(B * L, V),
                tokens.reshape(B * L),
                ignore_index=1,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(decoder.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if use_wandb:
                wandb.log({"stage1/train_loss": loss.item(), "step": step})

            if step % eval_every == 0:
                val_loss = evaluate_decoder(esm_model, decoder, val_loader, device)
                print(f"Step {step} | Val CE: {val_loss:.4f}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save(decoder.state_dict(), os.path.join(output_dir, "decoder_stage1.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping (Stage 1)")
                        return

    print(f"Stage 1 complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate_decoder(esm_model, decoder, val_loader, device):
    decoder.eval()
    total_loss, total_tokens = 0.0, 0
    for tokens, _ in val_loader:
        tokens = tokens.to(device)
        results = esm_model(tokens, repr_layers=[esm_model.num_layers])
        h = results["representations"][esm_model.num_layers]
        logits = decoder(h)
        B, L, V = logits.shape
        loss = nn.functional.cross_entropy(
            logits.reshape(B * L, V),
            tokens.reshape(B * L),
            ignore_index=1,
            reduction="sum",
        )
        total_loss += loss.item()
        total_tokens += B * L
    decoder.train()
    return total_loss / total_tokens


# ---------------------------------------------------------------------------
# Stage 2: Compressor + Decompressor training
# ---------------------------------------------------------------------------

def train_vae_stage2(
    vae: ProteinVAE,
    train_loader,
    val_loader,
    output_dir: str,
    lr: float = 5e-4,
    warmup_steps: int = 200,
    max_epochs: int = 400,
    patience: int = 8,
    eval_every: int = 400,
    device: torch.device = None,
    use_wandb: bool = False,
):
    """
    Train compressor and decompressor with the full VAE objective (Eq. 2).
    ESM2 encoder and decoder remain frozen.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vae = vae.to(device)

    # Only optimize compressor + decompressor
    params = list(vae.compressor.parameters()) + list(vae.decompressor.parameters())
    optimizer = AdamW(params, lr=lr)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup], milestones=[warmup_steps])

    best_val_loss = float("inf")
    patience_counter = 0
    step = 0

    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(max_epochs):
        vae.train()
        vae.esm.eval()  # Keep ESM frozen
        vae.decoder.eval()

        for tokens, _ in tqdm(train_loader, desc=f"Stage2 Epoch {epoch+1}"):
            tokens = tokens.to(device)

            logits, h, h_prime, mean, logvar = vae(tokens)
            loss, loss_dict = vae.compute_loss(tokens, logits, h, h_prime, mean, logvar)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            step += 1

            if use_wandb:
                wandb.log({"stage2/" + k: v for k, v in loss_dict.items()} | {"step": step})

            if step % eval_every == 0:
                val_loss = evaluate_vae(vae, val_loader, device)
                print(f"Step {step} | Val Loss: {val_loss:.4f} | {loss_dict}")

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    torch.save({
                        "compressor": vae.compressor.state_dict(),
                        "decompressor": vae.decompressor.state_dict(),
                        "decoder": vae.decoder.state_dict(),
                    }, os.path.join(output_dir, "vae_stage2.pt"))
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print("Early stopping (Stage 2)")
                        return

    print(f"Stage 2 complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate_vae(vae, val_loader, device):
    vae.eval()
    total_loss = 0.0
    n = 0
    for tokens, _ in val_loader:
        tokens = tokens.to(device)
        logits, h, h_prime, mean, logvar = vae(tokens)
        loss, _ = vae.compute_loss(tokens, logits, h, h_prime, mean, logvar)
        total_loss += loss.item()
        n += 1
    vae.train()
    vae.esm.eval()
    vae.decoder.eval()
    return total_loss / n


# ---------------------------------------------------------------------------
# Flow Matching training
# ---------------------------------------------------------------------------

def train_flow_matching(
    flow_matcher: ConditionalFlowMatcher,
    vae: ProteinVAE,
    train_loader,
    val_loader,
    output_dir: str,
    lr: float = 2e-4,
    warmup_steps: int = 400,
    train_steps: int = 600_000,
    device: torch.device = None,
    use_wandb: bool = False,
    log_every: int = 500,
    save_every: int = 10_000,
):
    """
    Train the conditional flow matching model.
    VAE components are fully frozen.
    Objective: CFM loss (Eq. 4)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    flow_matcher = flow_matcher.to(device)
    vae = vae.to(device).eval()
    for p in vae.parameters():
        p.requires_grad = False

    optimizer = AdamW(flow_matcher.v_theta.parameters(), lr=lr)
    warmup = LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_steps)
    cosine = CosineAnnealingLR(optimizer, T_max=train_steps - warmup_steps)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[warmup_steps])

    os.makedirs(output_dir, exist_ok=True)

    step = 0
    best_val_loss = float("inf")
    loader_iter = iter(train_loader)

    pbar = tqdm(total=train_steps, desc="Flow Matching Training")
    while step < train_steps:
        flow_matcher.train()

        try:
            tokens, fitness = next(loader_iter)
        except StopIteration:
            loader_iter = iter(train_loader)
            tokens, fitness = next(loader_iter)

        tokens = tokens.to(device)
        fitness = fitness.to(device)

        # Encode to latent z
        with torch.no_grad():
            mean, logvar, _ = vae.encode(tokens)
            z = vae.compressor.reparameterize(mean, logvar)

        # CFM loss
        loss = flow_matcher(z, fitness)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(flow_matcher.v_theta.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        step += 1
        pbar.update(1)
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

        if use_wandb and step % log_every == 0:
            wandb.log({"flow/train_loss": loss.item(), "step": step})

        if step % save_every == 0:
            val_loss = evaluate_flow(flow_matcher, vae, val_loader, device)
            print(f"\nStep {step} | Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(flow_matcher.state_dict(), os.path.join(output_dir, "flow_best.pt"))

            torch.save(flow_matcher.state_dict(), os.path.join(output_dir, f"flow_step{step}.pt"))

    pbar.close()
    print(f"Flow training complete. Best val loss: {best_val_loss:.4f}")


@torch.no_grad()
def evaluate_flow(flow_matcher, vae, val_loader, device):
    flow_matcher.eval()
    total_loss = 0.0
    n = 0
    for tokens, fitness in val_loader:
        tokens = tokens.to(device)
        fitness = fitness.to(device)
        mean, logvar, _ = vae.encode(tokens)
        z = mean  # use mean at eval time
        loss = flow_matcher(z, fitness)
        total_loss += loss.item()
        n += 1
    flow_matcher.train()
    return total_loss / n
