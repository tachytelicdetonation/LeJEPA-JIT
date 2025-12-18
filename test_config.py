"""Minimal test configuration for quick verification runs."""

from config import Config


def get_test_config(**overrides) -> Config:
    """
    Returns a config optimized for quick local testing.

    Features:
    - ~100 images only (via max_train_samples)
    - batch_size=2
    - num_workers=0 (avoids multiprocessing issues on macOS)
    - epochs=1
    - All diagnostic intervals set to 1 (trigger everything)
    - WandB enabled for visualization verification

    Args:
        **overrides: Any config parameters to override

    Returns:
        Config with test-friendly settings
    """
    test_settings = dict(
        # Minimal data loading
        batch_size=2,
        num_workers=0,
        # Single epoch
        epochs=1,
        # Force all diagnostics to run on epoch 1
        pca_vis_interval=1,
        attn_rollout_interval=1,
        embedding_pca_scatter_interval=1,
        layer_attention_interval=1,
        per_head_attention_interval=1,
        token_similarity_interval=1,
        rsm_interval=1,
        collapse_monitor_interval=1,
        gradient_flow_interval=1,
        drift_interval=1,
        knn_interval=1,
        lid_interval=1,
        transformer_diag_interval=1,
        block_diag_interval=1,
        gns_interval=1,
        sharpness_interval=1,
        landscape_interval=1,
        attn_distance_headmap_interval=1,
        attn_entropy_headmap_interval=1,
        pocp_interval=1,
        attn_logits_interval=1,
        mlp_output_stats_interval=1,
        landscape2d_interval=1,
        attribution_vis_interval=1,
        faithfulness_eval_interval=1,
        head_specialization_interval=1,
        summary_dashboard_interval=1,
        epoch_delta_interval=1,
        training_radar_interval=1,
        health_check_interval=1,
        pattern_classification_interval=1,
        qscore_interval=1,
        head_specialization_analysis_interval=1,
        layer_importance_interval=1,
        rope_correlation_interval=1,
        swiglu_sparsity_interval=1,
        # WandB enabled for verification
        wandb_mode="online",
        use_wandb=True,
        # Logging frequently
        log_interval=10,
        # Smaller diagnostic batches for speed
        diagnostic_batch_size=8,
        heavy_diagnostic_batch_size=4,
        # Reduce expensive computation sizes
        knn_max_samples=100,
        lid_max_samples=100,
        qscore_max_samples=100,
        faithfulness_num_samples=4,
        faithfulness_num_steps=10,
        pocp_num_pairs=64,
        sharpness_power_iters=3,
        landscape_points=5,
        landscape2d_points=5,
    )

    # Apply overrides
    test_settings.update(overrides)

    return Config(**test_settings)
