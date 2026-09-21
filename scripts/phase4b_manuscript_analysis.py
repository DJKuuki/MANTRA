"""
POST-HOC REPORTING / VISUALIZATION ONLY
Phase 5 — Manuscript Results, Figures & Discussion Synthesis

This script performs deterministic data extraction, statistical summaries,
and publication figure generation strictly from frozen archived JSON outputs:
  experiments/phase4_confirmatory/results/phase4_confirmatory_results.json

INVARIANTS:
- ZERO model imports (no torch, no transformers, no FinBERT)
- ZERO model training or inference
- ZERO probe recomputation
- ZERO alterations to frozen branch metrics or manifests
- ALL derived statistics across seeds or events are labeled POST-HOC DESCRIPTIVE
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Configure publication typography and styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
    'axes.grid': True,
    'grid.alpha': 0.35,
    'grid.linestyle': '--',
    'figure.autolayout': False,
})

RESULTS_FILE = "experiments/phase4_confirmatory/results/phase4_confirmatory_results.json"
FIG_DIR = "docs/research/figures/phase4b"


def load_frozen_results():
    if not os.path.exists(RESULTS_FILE):
        raise FileNotFoundError(f"Missing frozen results: {RESULTS_FILE}")
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def generate_figure1_dose_response(data, fig_dir):
    """
    Figure 1: Representational temporal leakage L_repr as a function of dose.
    Features:
    - Individual seed trajectories with distinct markers and colors
    - Individual seed-dose points
    - Across-seed mean overlay with +/- 1 SEM post-hoc error band
    - Zero baseline
    - No fitted monotonic curve
    """
    seeds = [13, 42, 87, 123, 2024]
    doses = [0.00, 0.25, 0.50, 0.75, 1.00]
    seed_colors = {
        13: '#1f77b4',
        42: '#2ca02c',
        87: '#d62728',
        123: '#ff7f0e',
        2024: '#9467bd',
    }
    seed_markers = {
        13: 'o',
        42: 's',
        87: '^',
        123: 'D',
        2024: 'v',
    }

    # Extract values: seed -> list of (dose, l_repr)
    seed_l_repr = {s: [] for s in seeds}
    dose_l_repr = {d: [] for d in doses}

    for b in data['branches'].values():
        s = b['seed']
        d = b['dose']
        l_r = b['l_repr']
        seed_l_repr[s].append((d, l_r))
        dose_l_repr[d].append(l_r)

    for s in seeds:
        seed_l_repr[s].sort(key=lambda x: x[0])

    # Compute across-seed mean and SEM (post-hoc descriptive)
    mean_by_dose = [np.mean(dose_l_repr[d]) for d in doses]
    sem_by_dose = [np.std(dose_l_repr[d], ddof=1) / np.sqrt(len(seeds)) if d > 0 else 0.0 for d in doses]

    fig, ax = plt.subplots(figsize=(8.5, 5.8), dpi=300)

    # Plot zero reference baseline
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6, label=r'Zero Leakage Baseline ($L_{\mathrm{repr}}=0$)')

    # Plot individual seeds
    for s in seeds:
        x_vals = [pt[0] for pt in seed_l_repr[s]]
        y_vals = [pt[1] for pt in seed_l_repr[s]]
        ax.plot(x_vals, y_vals, marker=seed_markers[s], color=seed_colors[s],
                linewidth=1.2, alpha=0.75, linestyle=':', label=f'Seed {s}')

    # Plot across-seed mean overlay
    mean_line, = ax.plot(doses, mean_by_dose, marker='o', color='#111111',
                         linewidth=2.8, markersize=8, label='Across-Seed Mean (Post-Hoc)')
    upper_band = [m + s for m, s in zip(mean_by_dose, sem_by_dose)]
    lower_band = [m - s for m, s in zip(mean_by_dose, sem_by_dose)]
    ax.fill_between(doses, lower_band, upper_band, color='#333333', alpha=0.15,
                    label=r'Across-Seed $\pm 1$ SEM (Descriptive)')

    # Add peak annotation at D=0.75
    peak_dose = 0.75
    peak_val = mean_by_dose[3]
    ax.annotate(f'Aggregate Peak\n($D=0.75$, +{peak_val:.5f})',
                xy=(peak_dose, peak_val),
                xytext=(peak_dose - 0.16, peak_val + 0.0028),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.2, headwidth=6),
                fontsize=9.0, fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffdd', edgecolor='#aaaaaa', alpha=0.9))

    # Add non-monotonic dip annotation at D=1.00
    dip_dose = 1.00
    dip_val = mean_by_dose[4]
    ax.annotate(f'Non-Monotonic Dip\n($D=1.00$, +{dip_val:.5f})',
                xy=(dip_dose, dip_val),
                xytext=(dip_dose - 0.25, dip_val - 0.0032),
                arrowprops=dict(facecolor='#d62728', shrink=0.08, width=1.2, headwidth=6),
                fontsize=9.0, color='#800000',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffeeee', edgecolor='#cc8888', alpha=0.9))

    ax.set_xlabel(r'Contamination Dose ($D$)', fontweight='bold', labelpad=8)
    ax.set_ylabel(r'Representational Temporal Leakage ($L_{\mathrm{repr}}$)', fontweight='bold', labelpad=8)
    ax.set_title('Figure 1: Representational Temporal Leakage Dose-Response Curve\n'
                 'Substantial Positive Shift with Seed Heterogeneity & Non-Monotonic Peak',
                 pad=12, fontweight='bold', fontsize=11.5)
    ax.set_xticks(doses)
    ax.set_xticklabels(['0.00\n(Clean)', '0.25', '0.50', '0.75', '1.00'])
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.4f'))
    ax.set_ylim(-0.0045, 0.0130)
    ax.legend(loc='upper left', framealpha=0.92, fontsize=9.0, ncol=2)

    fig.text(0.5, 0.01,
             "Note: The SEM band is descriptive across five optimization seeds and is not a preregistered confidence interval or confirmatory inferential band.",
             ha='center', fontsize=8.0, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    png_path = os.path.join(fig_dir, "figure1_l_repr_dose_response.png")
    pdf_path = os.path.join(fig_dir, "figure1_l_repr_dose_response.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 1: {png_path} and {pdf_path}")


def generate_figure2_significance_heatmap(data, fig_dir):
    """
    Figure 2: Branch-level significance / effect heatmap.
    Rows: Seeds [13, 42, 87, 123, 2024]
    Cols: Doses [0.25, 0.50, 0.75, 1.00]
    Cell values: L_repr with nominal significance annotation (*)
    """
    seeds = [13, 42, 87, 123, 2024]
    doses = [0.25, 0.50, 0.75, 1.00]

    matrix_l_repr = np.zeros((len(seeds), len(doses)))
    matrix_sig = np.zeros((len(seeds), len(doses)), dtype=bool)
    matrix_p = np.zeros((len(seeds), len(doses)))

    for b in data['branches'].values():
        s = b['seed']
        d = b['dose']
        if d > 0:
            r_idx = seeds.index(s)
            c_idx = doses.index(d)
            matrix_l_repr[r_idx, c_idx] = b['l_repr']
            matrix_sig[r_idx, c_idx] = b['is_statistically_significant']
            matrix_p[r_idx, c_idx] = b['permutation_p_value']

    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=300)

    # Use a diverging colormap centered at zero
    vmax = max(abs(matrix_l_repr.min()), abs(matrix_l_repr.max()))
    im = ax.imshow(matrix_l_repr, cmap='RdYlBu_r', vmin=-vmax, vmax=vmax, aspect='auto')

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'Representational Leakage $L_{\mathrm{repr}}$', fontweight='bold')
    cbar.formatter = ticker.FormatStrFormatter('%+.4f')
    cbar.update_ticks()

    ax.set_xticks(range(len(doses)))
    ax.set_yticks(range(len(seeds)))
    ax.set_xticklabels([f"D = {d:.2f}" for d in doses], fontweight='bold')
    ax.set_yticklabels([f"Seed {s}" for s in seeds], fontweight='bold')
    ax.set_xlabel('Contamination Dose Level', fontweight='bold', labelpad=8)
    ax.set_ylabel('Optimization Random Seed', fontweight='bold', labelpad=8)

    # Cell text annotations
    for r in range(len(seeds)):
        for c in range(len(doses)):
            val = matrix_l_repr[r, c]
            p_val = matrix_p[r, c]
            is_sig = matrix_sig[r, c]
            sig_marker = " *" if is_sig else ""
            
            # Format p-value sensitivity note if p==0
            p_str = "<0.0005" if p_val == 0.0 else f"{p_val:.3f}"
            cell_text = f"{val:+.4f}{sig_marker}\n(p={p_str})"
            
            # High contrast font color
            text_color = "white" if abs(val) > vmax * 0.55 else "black"
            font_weight = "bold" if is_sig else "normal"
            ax.text(c, r, cell_text, ha="center", va="center", color=text_color,
                    fontsize=9.5, fontweight=font_weight)

    ax.set_title("Figure 2: Branch-Level Effect & Nominal Significance Heatmap\n"
                 r"Cell: $L_{\mathrm{repr}}$ and Monte-Carlo p-value (* nominal branch-level $p < 0.05$)",
                 pad=12, fontweight='bold')

    # Sub-caption / footnote
    fig.text(0.5, 0.01,
             "* Denotes uncorrected nominal branch-level p < 0.05 (one-sided right-tailed paired sign-flip permutation test, B=2000).\n"
             "Note: Protocol v1.2.4 did not preregister an omnibus global test; significance is nominal branch-level only.",
             ha='center', fontsize=8.0, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    png_path = os.path.join(fig_dir, "figure2_branch_significance_map.png")
    pdf_path = os.path.join(fig_dir, "figure2_branch_significance_map.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 2: {png_path} and {pdf_path}")


def generate_figure3_event_level_deltas(data, fig_dir):
    """
    Figure 3: Event-level effect distribution:
    d_e = |y_e - yhat_{clean,e}| - |y_e - yhat_{leak,e}|
    across the 32 OOS events.
    Two panels:
    - Panel A: Box & strip plots of d_e grouped by contamination dose
    - Panel B: Event-by-event waterfall / sorted bar chart of median d_e across all 32 events
    """
    doses = [0.25, 0.50, 0.75, 1.00]
    
    # Collect deltas by dose
    dose_deltas = {d: [] for d in doses}
    # Collect deltas by event across all 20 contaminated branches
    event_deltas_dict = {}

    for b in data['branches'].values():
        d = b['dose']
        if d > 0:
            for row in b['oos_event_table']:
                e_id = row['event_id']
                delta = row['paired_delta']
                dose_deltas[d].append(delta)
                if e_id not in event_deltas_dict:
                    event_deltas_dict[e_id] = []
                event_deltas_dict[e_id].append(delta)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)

    # Panel A: Boxplot + scatter of event deltas by dose
    box_data = [dose_deltas[d] for d in doses]
    bp = ax1.boxplot(box_data, patch_artist=True, notch=False,
                     boxprops=dict(facecolor='#d0e1f9', color='#1f77b4', linewidth=1.5),
                     medianprops=dict(color='#d62728', linewidth=2.0),
                     whiskerprops=dict(color='#1f77b4', linewidth=1.2),
                     capprops=dict(color='#1f77b4', linewidth=1.2),
                     flierprops=dict(marker='o', markerfacecolor='#999999', markersize=4, alpha=0.5))

    # Overlay jittered strip plot
    np.random.seed(42)
    for i, d in enumerate(doses):
        y = dose_deltas[d]
        x = np.random.normal(i + 1, 0.05, size=len(y))
        ax1.plot(x, y, 'r.', alpha=0.25, markersize=5)
        # Compute positive fraction
        pos_frac = np.mean(np.array(y) > 0)
        mean_val = np.mean(y)
        ax1.text(i + 1, max(y) + 0.007, f"Mean: +{mean_val:.4f}\nPos: {pos_frac*100:.1f}%",
                 ha='center', va='bottom', fontsize=8.5, fontweight='semibold',
                 bbox=dict(boxstyle='square,pad=0.2', facecolor='white', alpha=0.8, edgecolor='#cccccc'))

    ax1.axhline(0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    ax1.set_xticklabels([f"D = {d:.2f}\n(160 branch-events)" for d in doses], fontweight='bold')
    ax1.set_xlabel('Contamination Dose Level', fontweight='bold')
    ax1.set_ylabel(r'Paired Absolute Error Reduction ($d_e$)', fontweight='bold')
    ax1.set_title(r'(A) Event Error Reductions by Contamination Dose' + '\n' +
                  r'($d_e = |y_e - \hat{y}_{\mathrm{clean}}| - |y_e - \hat{y}_{\mathrm{leak}}| > 0 \rightarrow \mathrm{Improvement}$)',
                  fontsize=11, fontweight='bold')
    ax1.set_ylim(-0.06, 0.085)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.2f'))

    # Panel B: Event-by-event median delta across the 32 OOS events
    # Sort events by median delta
    sorted_events = sorted(event_deltas_dict.keys(),
                           key=lambda k: np.median(event_deltas_dict[k]))
    median_vals = [np.median(event_deltas_dict[k]) for k in sorted_events]
    short_names = [e.replace('fomc-statement-', '') for e in sorted_events]

    bar_colors = ['#2ca02c' if m > 0 else '#d62728' for m in median_vals]
    bars = ax2.barh(range(len(sorted_events)), median_vals, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax2.axvline(0, color='black', linestyle='-', linewidth=1.0)
    ax2.set_yticks(range(len(sorted_events)))
    ax2.set_yticklabels(short_names, fontsize=7.5)
    ax2.set_xlabel(r'Median Paired Error Reduction ($d_e$)', fontweight='bold')
    ax2.set_ylabel('Out-of-Sample FOMC Event Date', fontweight='bold')
    ax2.set_title('(B) Event Heterogeneity Across 32 OOS Meetings\n'
                  f'(Post-Hoc Breadth Check: {sum(1 for m in median_vals if m > 0)} / 32 Meetings Show Median ' + r'$d_e > 0$)',
                  fontsize=10.5, fontweight='bold')
    ax2.xaxis.set_major_formatter(ticker.FormatStrFormatter('%+.3f'))

    # Add descriptive note
    fig.suptitle('Figure 3: Distribution and Heterogeneity of Out-of-Sample Event-Level Effects (Post-Hoc Descriptive)',
                 fontsize=13, fontweight='bold', y=0.99)

    fig.text(0.5, 0.01,
             "Each dose panel contains 160 branch-event observations (32 OOS events x 5 seeds). These observations are not independent and are displayed for post-hoc descriptive visualization only.\n"
             "The preregistered inferential unit remains the independent FOMC event. Breadth check illustrates that gains were not concentrated in an isolated meeting.",
             ha='center', fontsize=7.5, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.05, 1, 0.96])
    png_path = os.path.join(fig_dir, "figure3_event_level_deltas.png")
    pdf_path = os.path.join(fig_dir, "figure3_event_level_deltas.pdf")
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Generated Figure 3: {png_path} and {pdf_path}")


def generate_figure4_layered_comparison(data, fig_dir):
    """
    Figure 4: Layered outcome comparison across:
    1. L_repr (Continuous representational leakage)
    2. Binary Co-primary (Delta Macro-F1)
    3. L_behavior (Masking sensitivity delta)
    4. E_L 2Y (Treasury 2Y Delta IC)
    5. E_L SPY (SPY Delta IC)
    Demonstrating that leakage signal is concentrated at the representational layer
    and decouples across downstream layers.
    """
    doses = [0.25, 0.50, 0.75, 1.00]
    dose_labels = ['0.25', '0.50', '0.75', '1.00']

    # Extract metrics by dose across seeds
    l_repr_means, l_repr_sems = [], []
    binary_means, binary_sems = [], []
    behav_means, behav_sems = [], []
    ic_2y_means, ic_2y_sems = [], []
    ic_spy_means, ic_spy_sems = [], []

    for d in doses:
        b_d = [b for b in data['branches'].values() if b['dose'] == d]
        n = len(b_d)
        
        # 1. L_repr
        vals_r = [b['l_repr'] for b in b_d]
        l_repr_means.append(np.mean(vals_r))
        l_repr_sems.append(np.std(vals_r, ddof=1) / np.sqrt(n))

        # 2. Binary Macro-F1
        vals_b = [b['binary_co_primary']['delta_macro_f1'] for b in b_d]
        binary_means.append(np.mean(vals_b))
        binary_sems.append(np.std(vals_b, ddof=1) / np.sqrt(n))

        # 3. Behavioral L_behavior
        vals_beh = [b['l_behavior'] for b in b_d]
        behav_means.append(np.mean(vals_beh))
        behav_sems.append(np.std(vals_beh, ddof=1) / np.sqrt(n))

        # 4. Economic 2Y Delta IC
        vals_2y = [b['economic_2y']['delta_ic'] for b in b_d]
        ic_2y_means.append(np.mean(vals_2y))
        ic_2y_sems.append(np.std(vals_2y, ddof=1) / np.sqrt(n))

        # 5. Economic SPY Delta IC
        vals_spy = [b['economic_spy']['delta_ic'] for b in b_d]
        ic_spy_means.append(np.mean(vals_spy))
        ic_spy_sems.append(np.std(vals_spy, ddof=1) / np.sqrt(n))

    fig, axes = plt.subplots(1, 5, figsize=(16, 4.8), dpi=300, sharex=True)

    # 1. L_repr
    ax = axes[0]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.errorbar(doses, l_repr_means, yerr=l_repr_sems, fmt='o-', color='#1f77b4',
                linewidth=2, capsize=4, markersize=6)
    ax.set_title(r'1. Representation' + '\n' + r'$L_{\mathrm{repr}}$ (Continuous)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Leakage Effect Size', fontweight='bold')
    ax.set_ylim(-0.002, 0.009)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.4f'))
    ax.text(0.5, 0.08, 'EVIDENCE DETECTED\n(18/20 > 0, 10/20 p<0.05)', transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='#005500',
            bbox=dict(facecolor='#ddffdd', edgecolor='#55aa55', boxstyle='round,pad=0.2'))

    # 2. Binary Co-Primary
    ax = axes[1]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.errorbar(doses, binary_means, yerr=binary_sems, fmt='s-', color='#ff7f0e',
                linewidth=2, capsize=4, markersize=6)
    ax.set_title(r'2. Policy Task' + '\n' + r'$\Delta\mathrm{Macro\text{-}F1}$ (Binary)', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.05, 0.06)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.3f'))
    ax.text(0.5, 0.08, 'NOT SUPPORTED\n(p > 0.05 all branches)', transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='#880000',
            bbox=dict(facecolor='#ffdddd', edgecolor='#aa5555', boxstyle='round,pad=0.2'))

    # 3. Behavioral
    ax = axes[2]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.errorbar(doses, behav_means, yerr=behav_sems, fmt='^-', color='#2ca02c',
                linewidth=2, capsize=4, markersize=6)
    ax.set_title(r'3. Behavioral' + '\n' + r'$L_{\mathrm{behavior}}$ ($S_{\mathrm{mask}}$)', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.0015, 0.0015)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.4f'))
    ax.text(0.5, 0.08, 'MINIMAL SHIFT\n(Order $10^{-4}$, Descriptive)', transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='#555500',
            bbox=dict(facecolor='#ffffdd', edgecolor='#aaaa55', boxstyle='round,pad=0.2'))

    # 4. Economic 2Y
    ax = axes[3]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.errorbar(doses, ic_2y_means, yerr=ic_2y_sems, fmt='d-', color='#9467bd',
                linewidth=2, capsize=4, markersize=6)
    ax.set_title(r'4. Economic Primary' + '\n' + r'$\Delta\mathrm{IC}_{2\mathrm{Y}}$ (Treasury)', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.06, 0.03)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.3f'))
    ax.text(0.5, 0.08, 'NOT SUPPORTED\n(95% CI crosses zero)', transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='#880000',
            bbox=dict(facecolor='#ffdddd', edgecolor='#aa5555', boxstyle='round,pad=0.2'))

    # 5. Economic SPY
    ax = axes[4]
    ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
    ax.errorbar(doses, ic_spy_means, yerr=ic_spy_sems, fmt='v-', color='#8c564b',
                linewidth=2, capsize=4, markersize=6)
    ax.set_title(r'5. Economic Exploratory' + '\n' + r'$\Delta\mathrm{IC}_{\mathrm{SPY}}$ (Equities)', fontsize=11, fontweight='bold')
    ax.set_ylim(-0.04, 0.04)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%+.3f'))
    ax.text(0.5, 0.08, 'NOT SUPPORTED\n(95% CI crosses zero)', transform=ax.transAxes,
            ha='center', fontsize=8, fontweight='bold', color='#880000',
            bbox=dict(facecolor='#ffdddd', edgecolor='#aa5555', boxstyle='round,pad=0.2'))

    for a in axes:
        a.set_xticks(doses)
        a.set_xticklabels(dose_labels, fontweight='bold')
        a.set_xlabel('Dose ($D$)', fontweight='bold')

    fig.suptitle('Figure 4: Layered Outcome Comparison Across Evaluated Endpoints\n'
                 'Signal is Concentrated at Representational Layer and Decouples from Downstream Task & Market Behavior',
                 fontsize=13, fontweight='bold', y=1.02)

    fig.text(0.5, 0.01,
             "Note: Evaluated endpoints use distinct native units and are not on a common numerical scale; visual alignment only. No composite leakage score is defined.",
             ha='center', fontsize=8.0, style='italic', color='#333333')

    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    png_path = os.path.join(fig_dir, "figure4_layered_outcome_comparison.png")
    pdf_path = os.path.join(fig_dir, "figure4_layered_outcome_comparison.pdf")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')
    plt.close()
    print(f"Generated Figure 4: {png_path} and {pdf_path}")


def print_table_summaries(data):
    """Prints markdown tables to stdout for verification and documentation copying."""
    seeds = [13, 42, 87, 123, 2024]
    doses = [0.25, 0.50, 0.75, 1.00]

    print("\n" + "="*80)
    print("DOSE-LEVEL AGGREGATE TABLE (MAIN PAPER)")
    print("="*80)
    print("| Dose Level ($D$) | Mean $L_{\\mathrm{repr}}$ | Median $L_{\\mathrm{repr}}$ | Std $L_{\\mathrm{repr}}$ | Nom. Sig. Frac ($p < 0.05$) | Mean $\\Delta\\mathrm{Spearman}$ | Mean Binary $\\Delta\\mathrm{Macro\\text{-}F1}$ | Mean $L_{\\mathrm{behavior}}$ | Mean $\\Delta\\mathrm{IC}_{2\\mathrm{Y}}$ | Mean $\\Delta\\mathrm{IC}_{\\mathrm{SPY}}$ |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for d in [0.00] + doses:
        b_d = [b for b in data['branches'].values() if b['dose'] == d]
        l_r = [b['l_repr'] for b in b_d]
        sig_count = sum(1 for b in b_d if b.get('is_statistically_significant', False))
        d_sp = [b.get('delta_spearman', 0.0) for b in b_d]
        d_f1 = [b['binary_co_primary']['delta_macro_f1'] for b in b_d] if d > 0 else [0.0]
        l_beh = [b['l_behavior'] for b in b_d] if d > 0 else [0.0]
        ic_2y = [b['economic_2y']['delta_ic'] for b in b_d] if d > 0 else [0.0]
        ic_spy = [b['economic_spy']['delta_ic'] for b in b_d] if d > 0 else [0.0]

        print(f"| **{d:.2f}** | {np.mean(l_r):+.5f} | {np.median(l_r):+.5f} | {np.std(l_r, ddof=1) if len(l_r)>1 else 0.00000:.5f} | {sig_count} / {len(b_d)} ({sig_count/len(b_d)*100:.1f}%) | {np.mean(d_sp):+.4f} | {np.mean(d_f1):+.4f} | {np.mean(l_beh):+.5f} | {np.mean(ic_2y):+.4f} | {np.mean(ic_spy):+.4f} |")

    print("\n" + "="*80)
    print("FULL 20-BRANCH CONTAMINATED RESULTS TABLE (SUPPLEMENT)")
    print("="*80)
    print("| Seed | Dose | $L_{\\mathrm{repr}}$ | Frozen Perm $p$ | Finite-MC Reporting Note | Nom. Sig. ($\\alpha=0.05$) | $\\Delta\\mathrm{Spearman}$ | Binary $\\Delta\\mathrm{Macro\\text{-}F1}$ | Binary $p$ | $L_{\\mathrm{behavior}}$ | $\\Delta\\mathrm{IC}_{2\\mathrm{Y}}$ | 95% CI (2Y) | $p_{2\\mathrm{Y}}$ | $\\Delta\\mathrm{IC}_{\\mathrm{SPY}}$ | 95% CI (SPY) | $p_{\\mathrm{SPY}}$ |")
    print("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    for s in seeds:
        for d in doses:
            b_key = f"seed_{s}_dose_{int(d*100):03d}"
            b = data['branches'][b_key]
            l_r = b['l_repr']
            p_perm = b['permutation_p_value']
            p_sens = "k=0/2000; res. ~ 1/2001 = 0.00050" if p_perm == 0.0 else "--"
            p_perm_str = f"{p_perm:.4f}"
            is_sig = b['is_statistically_significant']
            sig_str = "**True**" if is_sig else "False"
            d_sp = b['delta_spearman']
            
            bin_f1 = b['binary_co_primary']['delta_macro_f1']
            bin_p = b['binary_co_primary']['p_value']
            
            l_beh = b['l_behavior']
            
            ic_2y = b['economic_2y']['delta_ic']
            ci_2y = b['economic_2y']['delta_ic_ci_95']
            ci_2y_str = f"[{ci_2y[0]:+.4f}, {ci_2y[1]:+.4f}]"
            p_2y = b['economic_2y']['p_value']
            p_2y_str = f"{p_2y:.3f}*" if (s == 42 and d in [0.25, 0.50]) else f"{p_2y:.3f}"
            
            ic_spy = b['economic_spy']['delta_ic']
            ci_spy = b['economic_spy']['delta_ic_ci_95']
            ci_spy_str = f"[{ci_spy[0]:+.4f}, {ci_spy[1]:+.4f}]"
            p_spy = b['economic_spy']['p_value']
            
            print(f"| **{s}** | {d:.2f} | {l_r:+.5f} | {p_perm_str} | {p_sens} | {sig_str} | {d_sp:+.4f} | {bin_f1:+.4f} | {bin_p:.4f} | {l_beh:+.5f} | {ic_2y:+.4f} | {ci_2y_str} | {p_2y_str} | {ic_spy:+.4f} | {ci_spy_str} | {p_spy:.3f} |")


def main():
    print("Starting Phase 4B Manuscript Analysis and Figure Generation...")
    data = load_frozen_results()
    os.makedirs(FIG_DIR, exist_ok=True)

    generate_figure1_dose_response(data, FIG_DIR)
    generate_figure2_significance_heatmap(data, FIG_DIR)
    generate_figure3_event_level_deltas(data, FIG_DIR)
    generate_figure4_layered_comparison(data, FIG_DIR)
    print_table_summaries(data)
    print("\nPhase 4B Manuscript Analysis Complete. All figures generated.")


if __name__ == '__main__':
    main()
