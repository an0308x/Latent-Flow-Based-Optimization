"""
Evaluation metrics for CHASE (Appendix C, Kirjner et al. 2023):

  - Median Fitness:  normalized oracle fitness (Eq. 8)
  - Diversity:       median pairwise Levenshtein distance (Eq. 9)
  - Novelty:         median min-distance to training set (Eq. 10)

Also includes bootstrapping (data augmentation) logic from Section 3.4.
"""

import numpy as np
import torch
from tqdm import tqdm
from Levenshtein import distance as levenshtein_distance


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_fitness(sequences: list, oracle, f_min: float, f_max: float, device=None) -> float:
    """
    Median normalized oracle fitness.

    Args:
        sequences: list of amino acid strings
        oracle:    callable nn.Module; takes tokenized sequences, returns fitness scores
        f_min/max: normalization bounds from the ground-truth dataset A_GT
    Returns:
        median normalized fitness (float)
    """
    scores = oracle(sequences, device=device)
    normalized = (scores - f_min) / (f_max - f_min + 1e-8)
    return float(np.median(normalized))


def compute_diversity(sequences: list, sample_size: int = None) -> float:
    """
    Median pairwise Levenshtein distance (Eq. 9).

    Args:
        sequences:   list of strings
        sample_size: if given, subsample pairs for speed
    Returns:
        median pairwise distance (float)
    """
    n = len(sequences)
    distances = []
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    if sample_size is not None and len(pairs) > sample_size:
        idx = np.random.choice(len(pairs), sample_size, replace=False)
        pairs = [pairs[i] for i in idx]

    for i, j in pairs:
        distances.append(levenshtein_distance(sequences[i], sequences[j]))

    return float(np.median(distances)) if distances else 0.0


def compute_novelty(generated: list, training_set: list) -> float:
    """
    Median min-distance from generated sequences to training set (Eq. 10).

    Args:
        generated:    list of generated sequences
        training_set: list of training sequences
    Returns:
        median min distance (float)
    """
    min_dists = []
    for seq in tqdm(generated, desc="Computing novelty", leave=False):
        min_d = min(levenshtein_distance(seq, train_seq) for train_seq in training_set)
        min_dists.append(min_d)
    return float(np.median(min_dists))


def evaluate_generated(
    sequences: list,
    oracle,
    training_sequences: list,
    f_min: float,
    f_max: float,
    device=None,
    diversity_sample_size: int = 5000,
) -> dict:
    """
    Compute all three metrics for a set of generated sequences.

    Returns:
        dict with keys 'fitness', 'diversity', 'novelty'
    """
    fitness = compute_fitness(sequences, oracle, f_min, f_max, device)
    diversity = compute_diversity(sequences, sample_size=diversity_sample_size)
    novelty = compute_novelty(sequences, training_sequences)
    return {"fitness": fitness, "diversity": diversity, "novelty": novelty}


# ---------------------------------------------------------------------------
# Bootstrapping / Data Augmentation (Section 3.4, Algorithm 1)
# ---------------------------------------------------------------------------

@torch.no_grad()
def bootstrap_dataset(
    flow_matcher,
    vae,
    target_interval: tuple,
    n_targets: int = 20,
    expand_factor: float = 0.25,
    n_base: int = None,
    label_noise_scale: float = 0.01,
    guidance_scale: float = 0.0,
    n_ode_steps: int = 40,
    device=None,
) -> tuple:
    """
    Generate synthetic (sequence, fitness) pairs to augment training data.

    Algorithm 1 from the paper:
    1. Evenly space |F| target fitness values across interval I
    2. For each f ∈ F, sample sequences using the flow model
    3. Perturb labels: f_hat = f + q * η,  η ~ N(0,1)

    Args:
        flow_matcher:    trained ConditionalFlowMatcher
        vae:             trained ProteinVAE
        target_interval: (f_min, f_max) for synthetic targets
        n_targets:       |F|, number of evenly spaced target values
        expand_factor:   fraction to expand training set by
        n_base:          size of original dataset (to compute expansion budget)
        label_noise_scale: q in Eq. 7
        guidance_scale:  CFG weight w
        n_ode_steps:     K
        device:          torch device

    Returns:
        syn_sequences: list of amino acid strings
        syn_fitness:   numpy array of perturbed fitness labels
    """
    if device is None:
        device = next(flow_matcher.parameters()).device

    flow_matcher.eval()
    vae.eval()

    f_min, f_max = target_interval
    target_values = np.linspace(f_min, f_max, n_targets)

    # Budget: expand by expand_factor of base dataset
    total_budget = max(1, int((n_base or 1000) * expand_factor))
    per_target = max(1, total_budget // n_targets)

    syn_sequences = []
    syn_fitness = []

    # Get latent shape from a dummy forward
    # We need to know (l, d) -- infer from compressor
    l = getattr(vae.compressor, "_cached_l", None)
    d = vae.compressor.to_mean.out_features

    for f_target in tqdm(target_values, desc="Bootstrapping"):
        # Sample latents
        # l (compressed seq len) needs to be inferred; use typical value
        # In practice this is seq_len // compression
        # We'll use the stored value or default 16
        latent_l = l if l is not None else 16
        shape = (per_target, latent_l, d)

        z_syn = flow_matcher.sample(
            shape=shape,
            f=float(f_target),
            guidance_scale=guidance_scale,
            n_steps=n_ode_steps,
            device=device,
        )

        # Decode to sequences
        h_prime = vae.decompressor(z_syn)
        logits = vae.decoder(h_prime)  # (B, L, vocab)
        token_ids = logits.argmax(dim=-1)  # greedy decode

        # Convert token IDs to amino acid strings
        alphabet = vae.esm.alphabet if hasattr(vae.esm, "alphabet") else None
        if alphabet is not None:
            seqs = tokens_to_sequences(token_ids, alphabet)
        else:
            seqs = [f"SYNTHETIC_{i}" for i in range(per_target)]

        # Perturb labels (Eq. 7)
        noise = np.random.randn(per_target)
        labels = f_target + label_noise_scale * noise

        syn_sequences.extend(seqs)
        syn_fitness.extend(labels.tolist())

    return syn_sequences, np.array(syn_fitness, dtype=np.float32)


def tokens_to_sequences(token_ids: torch.Tensor, alphabet) -> list:
    """
    Convert ESM2 token ID tensor to list of amino acid strings.

    Args:
        token_ids: (B, L) integer tensor
        alphabet:  ESM2 Alphabet object
    Returns:
        list of strings, one per sequence
    """
    # Standard amino acids
    AA = set("ACDEFGHIKLMNPQRSTVWY")
    seqs = []
    for row in token_ids:
        chars = []
        for tok_id in row.tolist():
            tok = alphabet.get_tok(tok_id)
            if tok in AA:
                chars.append(tok)
        seqs.append("".join(chars))
    return seqs


# ---------------------------------------------------------------------------
# Ranking / top-k selection (for post-hoc evaluation, Sec 4.2)
# ---------------------------------------------------------------------------

def select_top_k(sequences: list, predictor, k: int = 128, device=None) -> list:
    """
    Rank sequences by predictor score and return top-k.
    Mirrors the benchmark protocol from Kirjner et al. (2023).

    Args:
        sequences: candidate sequences
        predictor: ranking predictor (same as used by baselines)
        k:         number to select
    Returns:
        top-k sequences
    """
    scores = predictor(sequences, device=device)
    idx = np.argsort(scores)[::-1][:k]
    return [sequences[i] for i in idx]
