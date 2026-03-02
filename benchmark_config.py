# CHASE Benchmark Configurations
# Hyperparameters from Appendix B (Caceres Arroyo et al., 2026)
#
# Usage:
#   from configs.benchmark_configs import CONFIGS
#   cfg = CONFIGS["gfp_medium"]

CONFIGS = {
    "aav_medium": {
        # Dataset
        "dataset": "aav_medium",
        "seq_len": 735,
        "fitness_range_train": [0.29, 0.38],  # Table 1

        # VAE
        "esm_model": "esm2_t6_8M_UR50D",
        "esm_dim": 320,
        "latent_dim": 64,
        "compression": 20,        # CHASE/20 (Table 5)
        "n_transformer_layers": 2,
        "n_attn_heads": 8,
        "beta_vae": 1e-4,

        # Stage 1 training
        "stage1_lr": 5e-5,
        "stage1_warmup": 200,
        "stage1_epochs": 30,
        "stage1_patience": 8,
        "stage1_eval_every": 50,

        # Stage 2 training
        "stage2_lr": 5e-4,
        "stage2_warmup": 200,
        "stage2_epochs": 400,
        "stage2_patience": 8,
        "stage2_eval_every": 400,

        # Flow matching
        "flow_lr": 2e-4,
        "flow_warmup": 400,
        "flow_train_steps": 800_000,
        "flow_batch_size": 256,
        "score_dropout": 0.0,      # Appendix B.2

        # Sampling
        "target_fitness": 0.5,     # Appendix B.2
        "guidance_scale": 0.2,
        "n_ode_steps": 40,
        "n_samples": 512,
        "top_k": 128,

        # Bootstrapping
        "bootstrap_interval": [0.05, 0.5],
        "bootstrap_n_targets": 20,
        "bootstrap_label_noise": 0.0075,
        "bootstrap_expand_factor": 0.25,
        "bootstrap_score_dropout": 0.1,
        "bootstrap_target_fitness": 0.52,
        "bootstrap_guidance_scale": -0.1,
    },

    "aav_hard": {
        "dataset": "aav_hard",
        "seq_len": 735,
        "fitness_range_train": [0.0, 0.33],

        "esm_model": "esm2_t6_8M_UR50D",
        "esm_dim": 320,
        "latent_dim": 64,
        "compression": 20,
        "n_transformer_layers": 2,
        "n_attn_heads": 8,
        "beta_vae": 1e-4,

        "stage1_lr": 5e-5,
        "stage1_warmup": 200,
        "stage1_epochs": 30,
        "stage1_patience": 8,
        "stage1_eval_every": 50,

        "stage2_lr": 5e-4,
        "stage2_warmup": 200,
        "stage2_epochs": 400,
        "stage2_patience": 8,
        "stage2_eval_every": 400,

        "flow_lr": 2e-4,
        "flow_warmup": 400,
        "flow_train_steps": 800_000,
        "flow_batch_size": 256,
        "score_dropout": 0.0,

        "target_fitness": 0.55,
        "guidance_scale": 0.1,
        "n_ode_steps": 40,
        "n_samples": 512,
        "top_k": 128,

        "bootstrap_interval": [0.05, 0.5],
        "bootstrap_n_targets": 20,
        "bootstrap_label_noise": 0.0075,
        "bootstrap_expand_factor": 0.25,
        "bootstrap_score_dropout": 0.1,
        "bootstrap_target_fitness": 0.53,
        "bootstrap_guidance_scale": 0.1,
    },

    "gfp_medium": {
        "dataset": "gfp_medium",
        "seq_len": 239,
        "fitness_range_train": [0.01, 0.62],

        "esm_model": "esm2_t6_8M_UR50D",
        "esm_dim": 320,
        "latent_dim": 64,
        "compression": 20,
        "n_transformer_layers": 2,
        "n_attn_heads": 8,
        "beta_vae": 1e-4,

        "stage1_lr": 5e-5,
        "stage1_warmup": 200,
        "stage1_epochs": 30,
        "stage1_patience": 8,
        "stage1_eval_every": 50,

        "stage2_lr": 5e-4,
        "stage2_warmup": 200,
        "stage2_epochs": 400,
        "stage2_patience": 8,
        "stage2_eval_every": 400,

        "flow_lr": 2e-4,
        "flow_warmup": 400,
        "flow_train_steps": 600_000,
        "flow_batch_size": 256,
        "score_dropout": 0.0,

        "target_fitness": 0.8,
        "guidance_scale": -0.08,
        "n_ode_steps": 40,
        "n_samples": 512,
        "top_k": 128,

        "bootstrap_interval": [0.05, 0.8],
        "bootstrap_n_targets": 20,
        "bootstrap_label_noise": 0.01,
        "bootstrap_expand_factor": 0.25,
        "bootstrap_score_dropout": 0.0,
        "bootstrap_target_fitness": 1.05,
        "bootstrap_guidance_scale": 0.0,
    },

    "gfp_hard": {
        "dataset": "gfp_hard",
        "seq_len": 239,
        "fitness_range_train": [0.0, 0.10],

        "esm_model": "esm2_t6_8M_UR50D",
        "esm_dim": 320,
        "latent_dim": 64,
        "compression": 20,
        "n_transformer_layers": 2,
        "n_attn_heads": 8,
        "beta_vae": 1e-4,

        "stage1_lr": 5e-5,
        "stage1_warmup": 200,
        "stage1_epochs": 30,
        "stage1_patience": 8,
        "stage1_eval_every": 50,

        "stage2_lr": 5e-4,
        "stage2_warmup": 200,
        "stage2_epochs": 400,
        "stage2_patience": 8,
        "stage2_eval_every": 400,

        "flow_lr": 2e-4,
        "flow_warmup": 400,
        "flow_train_steps": 500_000,
        "flow_batch_size": 256,
        "score_dropout": 0.0,

        "target_fitness": 1.4,
        "guidance_scale": 0.1,
        "n_ode_steps": 40,
        "n_samples": 512,
        "top_k": 128,

        "bootstrap_interval": [0.05, 0.8],
        "bootstrap_n_targets": 20,
        "bootstrap_label_noise": 0.01,
        "bootstrap_expand_factor": 0.25,
        "bootstrap_score_dropout": 0.0,
        "bootstrap_target_fitness": 1.3,
        "bootstrap_guidance_scale": 0.2,
    },
}
