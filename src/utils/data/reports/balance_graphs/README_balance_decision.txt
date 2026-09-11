Diagnostics for class balancing decisions:
- class_counts.csv: train/val counts per class
- imbalance_summary.json: entropy, gini, CV, max/min ratio
- inverse_freq_weights.csv: weights for WeightedRandomSampler
- class_balanced_weights_beta_sweep.csv: CB weights across betas; pick beta so minority weights are ~2-5x major-class
- focal_gamma_guideline.csv: heuristic importance vs gamma in [1.0, 2.5]
- plot_class_counts.png: long-tail visualization
- plot_weights_sweep.png: inverse-freq vs CB across betas
- plot_recall_vs_count.png: (if preds provided) recall vs frequency
