"""
Figure 14: Weight Matrix Spectral Structure (RLVR-inspired heatmap style)

Visualizes the singular value distribution across layers for Pythia models,
comparing random initialization vs trained state.

*** REQUIRES downloading Pythia checkpoints ***
Run with: python scripts/figures_v2/fig14_weight_structure.py --download

Without --download: generates a synthetic demonstration version using
available concentration/alpha data to create a pseudo-spectral heatmap.

Data (synthetic mode): results/pythia_v2/*.jsonl
Data (full mode): HuggingFace Pythia checkpoints
"""
import sys
import argparse
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))

from style import *
from matplotlib.colors import LogNorm, PowerNorm

def generate_synthetic():
    """Generate a pseudo-spectral heatmap from existing concentration data."""
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 3.5))

    # We'll create a synthetic eigenvalue-like distribution based on
    # the alpha and concentration metrics we already have.
    # This approximates what a real per-layer SVD would look like.

    n_layers = 16  # Pythia-1B has 16 transformer layers
    n_ranks = 128  # Show top-128 singular values

    # Load Pythia-1B data (step 0 and final)
    path = RESULTS / 'pythia_v2/pythia_1b.jsonl'
    if not path.exists():
        print('  ⚠ Missing pythia_1b data')
        plt.close(fig)
        return

    records = load_jsonl(path)
    init_record = records[0]  # step 0
    final_record = records[-1]  # step 143000

    # Simulate singular value distributions based on alpha
    # Power law: P(σ) ~ σ^{-α}, so σ_i ~ i^{-1/(α-1)}
    def make_spectrum(alpha, n_ranks, noise_level=0.02):
        """Generate synthetic singular value spectrum from alpha."""
        ranks = np.arange(1, n_ranks + 1, dtype=float)
        # Zipf-like distribution parameterized by alpha
        if alpha > 1.5:
            spectrum = ranks ** (-1.0 / (alpha - 1))
        else:
            spectrum = ranks ** (-1.0)
        # Add small noise
        spectrum += np.random.randn(n_ranks) * noise_level * spectrum.mean()
        spectrum = np.maximum(spectrum, 0.001)
        # Normalize to sum to 1 (like probabilities)
        spectrum /= spectrum.sum()
        return spectrum

    np.random.seed(42)

    # Generate per-layer spectra for init and final
    # Init: all layers have similar random-matrix spectra (alpha ~ 4)
    alpha_init = init_record['alpha_mean']
    # Final: layers have different alpha (attention vs MLP)
    alpha_final = final_record['alpha_mean']
    alpha_attn_final = final_record['alpha_attn']
    alpha_mlp_final = final_record['alpha_mlp']

    init_matrix = np.zeros((n_layers, n_ranks))
    final_matrix = np.zeros((n_layers, n_ranks))

    for layer in range(n_layers):
        # Init: all layers similar (random)
        init_matrix[layer] = make_spectrum(alpha_init + np.random.randn()*0.3, n_ranks)

        # Final: alternate attn/mlp with slight depth gradient
        if layer % 2 == 0:  # attention-like
            a = alpha_attn_final + (layer / n_layers) * 0.3
        else:  # MLP-like
            a = alpha_mlp_final + (layer / n_layers) * 0.2
        final_matrix[layer] = make_spectrum(a, n_ranks)

    # Panel A: Initial (random) — fairly uniform
    im1 = axes[0].imshow(init_matrix, aspect='auto',
                          cmap='magma', interpolation='bilinear',
                          norm=PowerNorm(gamma=0.5, vmin=0, vmax=0.08))
    axes[0].set_xlabel('Singular Value Rank')
    axes[0].set_ylabel('Layer Index')
    axes[0].set_title(f'(a) Random Init (step 0, α={alpha_init:.1f})',
                      fontsize=8.5, pad=6)
    axes[0].set_xticks([0, 32, 64, 96, 128])
    axes[0].set_xticklabels(['1', '32', '64', '96', '128'])

    # Panel B: Trained — concentrated in top ranks
    im2 = axes[1].imshow(final_matrix, aspect='auto',
                          cmap='magma', interpolation='bilinear',
                          norm=PowerNorm(gamma=0.5, vmin=0, vmax=0.08))
    axes[1].set_xlabel('Singular Value Rank')
    axes[1].set_title(f'(b) Trained (step 143K, α={alpha_final:.1f})',
                      fontsize=8.5, pad=6)
    axes[1].set_xticks([0, 32, 64, 96, 128])
    axes[1].set_xticklabels(['1', '32', '64', '96', '128'])

    # Colorbars
    cbar = plt.colorbar(im2, ax=axes, fraction=0.02, pad=0.03)
    cbar.set_label(r'$p_i = \sigma_i^2 / \|\mathbf{W}\|_F^2$', fontsize=7)
    cbar.ax.tick_params(labelsize=6)

    # Annotations
    axes[0].text(64, -1.5, 'Uniform: energy spread across all ranks',
                fontsize=6, color=C['gray'], ha='center', style='italic')
    axes[1].text(64, -1.5, 'Concentrated: energy in top few ranks',
                fontsize=6, color=C['gray'], ha='center', style='italic')

    # Layer type annotations on right side
    for layer in range(n_layers):
        label = 'A' if layer % 2 == 0 else 'M'
        color = C['blue'] if layer % 2 == 0 else C['red']
        axes[1].text(n_ranks + 3, layer, label, fontsize=4.5, color=color,
                    va='center', ha='left', fontweight='bold')

    fig.suptitle('Pythia-1B: Spectral Energy Distribution (simulated from measured α)',
                 fontsize=9, y=0.98, color=C['navy'])

    save_fig(fig, 'fig14_weight_structure')


def generate_from_checkpoint():
    """Generate real spectral heatmap from downloaded checkpoint."""
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError:
        print('  ⚠ torch/transformers not available. Using synthetic mode.')
        generate_synthetic()
        return

    print('  Loading Pythia-1B step 0...')
    model_init = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-1b-deduped', revision='step0',
        torch_dtype=torch.float16
    )

    print('  Loading Pythia-1B final...')
    model_final = AutoModelForCausalLM.from_pretrained(
        'EleutherAI/pythia-1b-deduped',
        torch_dtype=torch.float16
    )

    n_ranks = 128
    init_spectra = []
    final_spectra = []

    # Extract weight matrices and compute SVD
    for name, param in model_init.named_parameters():
        if 'weight' in name and param.dim() == 2 and param.shape[0] >= 256:
            W = param.data.float()
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            spectrum = (S[:n_ranks] ** 2 / (S ** 2).sum()).cpu().numpy()
            init_spectra.append(spectrum)

    for name, param in model_final.named_parameters():
        if 'weight' in name and param.dim() == 2 and param.shape[0] >= 256:
            W = param.data.float()
            U, S, V = torch.linalg.svd(W, full_matrices=False)
            spectrum = (S[:n_ranks] ** 2 / (S ** 2).sum()).cpu().numpy()
            final_spectra.append(spectrum)

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_W, 4.0))

    init_matrix = np.array(init_spectra[:32])  # First 32 layers
    final_matrix = np.array(final_spectra[:32])

    im1 = axes[0].imshow(init_matrix, aspect='auto', cmap='magma',
                          interpolation='bilinear',
                          norm=PowerNorm(gamma=0.5, vmin=0, vmax=0.1))
    im2 = axes[1].imshow(final_matrix, aspect='auto', cmap='magma',
                          interpolation='bilinear',
                          norm=PowerNorm(gamma=0.5, vmin=0, vmax=0.1))

    axes[0].set_title('(a) Random Init (step 0)', fontsize=8.5, pad=6)
    axes[1].set_title('(b) Trained (step 143K)', fontsize=8.5, pad=6)

    for ax in axes:
        ax.set_xlabel('Singular Value Rank')
        ax.set_ylabel('Layer Index')

    cbar = plt.colorbar(im2, ax=axes, fraction=0.02, pad=0.03)
    cbar.set_label(r'$\sigma_i^2 / \|W\|_F^2$', fontsize=7)

    fig.suptitle('Pythia-1B: Real Spectral Energy Distribution',
                 fontsize=9, y=0.98, color=C['navy'])

    save_fig(fig, 'fig14_weight_structure')
    del model_init, model_final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', action='store_true',
                       help='Download checkpoints and compute real SVD')
    args = parser.parse_args()

    if args.download:
        generate_from_checkpoint()
    else:
        generate_synthetic()


if __name__ == '__main__':
    main()
