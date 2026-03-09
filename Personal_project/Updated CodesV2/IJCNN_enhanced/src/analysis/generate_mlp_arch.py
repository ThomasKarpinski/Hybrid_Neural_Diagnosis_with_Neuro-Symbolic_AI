import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURES_DIR = "paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

def draw_mlp_arch():
    print("Generating MLP Architecture Diagram...")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Define layers
    layers = [
        {'name': 'Input\n(21 or 8)', 'x': 2, 'y': 3, 'h': 4, 'c': '#a1c9f4'}, # Light blue
        {'name': 'Dense 1\n(32)\nReLU', 'x': 5, 'y': 3, 'h': 5, 'c': '#ffb482'}, # Orange
        {'name': 'Dense 2\n(16)\nReLU', 'x': 8, 'y': 3, 'h': 3, 'c': '#ffb482'}, # Orange
        {'name': 'Output\n(1)\nSigmoid', 'x': 10.5, 'y': 3, 'h': 1, 'c': '#8de5a1'}  # Green
    ]

    # Draw Boxes
    for i, l in enumerate(layers):
        rect = patches.FancyBboxPatch((l['x'] - 0.5, l['y'] - l['h']/2), 1.0, l['h'], 
                                 boxstyle="round,pad=0.1", linewidth=1.5, edgecolor='black', facecolor=l['c'], alpha=0.9)
        ax.add_patch(rect)
        ax.text(l['x'], l['y'], l['name'], ha='center', va='center', fontsize=11, fontweight='bold')
        
        # Connections (Arrows)
        if i < len(layers) - 1:
            next_l = layers[i+1]
            # Draw arrow from right edge to left edge
            start_x = l['x'] + 0.5 + 0.1 # box width/2 + pad
            end_x = next_l['x'] - 0.5 - 0.1
            
            # Draw multiple lines to suggest full connectivity? No, simpler is cleaner. 
            # Just one thick arrow.
            ax.annotate("", xy=(end_x, 3), xytext=(start_x, 3),
                        arrowprops=dict(arrowstyle="->", lw=2, color='gray'))
            
            # Add text for weights
            if i < len(layers) - 1:
                 ax.text((start_x+end_x)/2, 3.2, "Full Conn.", ha='center', fontsize=9, color='gray')

    plt.title("Baseline MLP Classifier Architecture")
    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "mlp_architecture.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved MLP Architecture to {save_path}")

if __name__ == "__main__":
    draw_mlp_arch()
