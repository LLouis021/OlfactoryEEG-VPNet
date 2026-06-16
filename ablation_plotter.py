import matplotlib.pyplot as plt
import os

def plot_ablation_results(results_dict,
                         save_path="runs/final_ablation.png"):
    """
    v9.0: VPNet-EEG ablation experiments
    Full model: VPNet-EEG (CBAM + Multi-scale Residual + EMA + Cosine Warm Restart)
    Ablation: No_VPNet, No_Residual, No_Spatial, Baseline
    """

    if not results_dict:
        print("No results to plot.")
        return


    # =========================================================
    # v9.0 fixed order
    # =========================================================
    ordered_keys = [
        "baseline",
        "no_vpnet",
        "no_residual",
        "no_spatial",
        "vpnet_eeg"
    ]

    names = []
    values = []

    for k in ordered_keys:
        if k in results_dict:
            names.append(k)
            values.append(results_dict[k] * 100)

    # Add extra non-standard keys
    for k, v in results_dict.items():
        if k not in ordered_keys:
            names.append(k)
            values.append(v * 100)

    if not names:
        print("No results to plot.")
        return

    # =========================================================
    # v9.0 name display mapping
    # =========================================================
    display_names = {
        "baseline": "Baseline\n(MLP Only)",
        "no_vpnet": "w/o VPNet\n(Freq. Branch)",
        "no_residual": "w/o Residual\n(Time Branch)",
        "no_spatial": "w/o Spatial\n(Channel Conv)",
        "vpnet_eeg": "VPNet-EEG\n(Ours)"
    }

    display = [display_names.get(n, n) for n in names]

    # =========================================================
    # colour
    # =========================================================
    colors = [
        "#95a5a6",  # baseline (gray)
        "#e67e22",  # no_vpnet (orange)
        "#3498db",  # no_residual (blue)
        "#9b59b6",  # no_spatial (purple)
        "#2ecc71"   # VPNet-EEG (green - highlight)
    ]

    # If there are more experiments, cycle through colors
    extended_colors = colors + ["#e74c3c", "#1abc9c", "#f39c12", "#16a085", "#8e44ad"]
    bar_colors = [extended_colors[i % len(extended_colors)] for i in range(len(names))]

    # =========================================================
    # Draw picture
    # =========================================================
    plt.figure(figsize=(max(14, len(names) * 2), 6))

    bars = plt.bar(display, values, color=bar_colors, width=0.5)

    # Value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2,
                 yval + 0.5,
                 f'{yval:.2f}%',
                 ha='center',
                 fontsize=11,
                 fontweight='bold')

    plt.title("Ablation Study of VPNet-EEG Components for EEG Odor Classification",
              fontsize=15, pad=20, fontweight='bold')

    plt.ylabel("Accuracy (%)", fontsize=13)
    plt.xlabel("Model Configuration", fontsize=13)

    plt.ylim(0, 100)
    plt.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()

    # =========================================================
    # save
    # =========================================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)

    print(f"\n Saved to: {save_path}")

    plt.close()
