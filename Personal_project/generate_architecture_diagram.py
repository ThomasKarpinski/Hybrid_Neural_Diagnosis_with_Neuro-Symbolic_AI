import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyArrowPatch

def draw_diagram():
    # Increased figure size for better spacing
    fig, ax = plt.subplots(figsize=(16, 20))
    # Scale: x=0..28, y=0..36
    ax.set_xlim(0, 28)
    ax.set_ylim(0, 36)
    ax.axis('off')

    # Styles
    bbox_style = dict(boxstyle="round,pad=0.5", fc="white", ec="black", lw=1.5)
    neural_style = dict(boxstyle="round,pad=0.5", fc="#e6f2ff", ec="#0000ff", lw=1.5) # Light Blue
    symbolic_style = dict(boxstyle="round,pad=0.5", fc="#ffe6e6", ec="#cc0000", lw=1.5) # Light Red
    fusion_style = dict(boxstyle="round,pad=0.5", fc="#e6ffe6", ec="#00cc00", lw=1.5) # Light Green
    
    # Helper to draw box
    def draw_box(x, y, width, height, text, style=bbox_style, fontsize=10):
        rect = patches.FancyBboxPatch((x, y), width, height, **style)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=fontsize, fontweight='bold', wrap=True)
        return (x + width/2, y) # Return bottom center
    
    def draw_diamond(cx, cy, size, text, style=bbox_style, fontsize=9):
        pts = [[cx, cy+size], [cx+size*1.5, cy], [cx, cy-size], [cx-size*1.5, cy]]
        poly = patches.Polygon(pts, **style)
        ax.add_patch(poly)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
        return (cx, cy-size)

    def draw_ellipse(cx, cy, width, height, text, style=bbox_style, fontsize=10):
        ell = patches.Ellipse((cx, cy), width, height, **style)
        ax.add_patch(ell)
        ax.text(cx, cy, text, ha='center', va='center', fontsize=fontsize, fontweight='bold')
        return (cx, cy - height/2)

    def connect(p1, p2, text=None, connectionstyle="arc3,rad=0", color='black'):
        arrow = FancyArrowPatch(p1, p2, connectionstyle=connectionstyle, 
                                arrowstyle='-|>', mutation_scale=15, color=color, lw=1.5)
        ax.add_patch(arrow)
        if text:
            # Simple text placement logic, can be improved for curves
            mid_x = (p1[0] + p2[0]) / 2
            mid_y = (p1[1] + p2[1]) / 2
            # Adjust label position slightly based on direction
            ax.text(mid_x, mid_y, text, fontsize=9, ha='center', va='center', 
                    bbox=dict(facecolor='white', edgecolor='none', pad=2))

    # ================= LAYOUT GRID =================
    # Columns (Centers):
    # Col 1 (Neural): x=4.5
    # Col 2 (Fuzzy): x=11.5
    # Col 3 (GNB): x=17.5
    # Col 4 (Rules): x=23.5
    
    col1_c = 4.5
    col2_c = 11.5
    col3_c = 17.5
    col4_c = 23.5
    
    row_input = 33
    row_modules = 26
    row_probs = 21
    row_fusion_top = 13
    row_fusion_mid = 9
    row_final = 4

    # ================= INPUT =================
    # Spanning top
    in_w, in_h = 8, 2
    in_x = 10 # Center at 14
    input_pts = [[in_x, row_input], [in_x+in_w, row_input], [in_x+in_w-1, row_input-in_h], [in_x+1, row_input-in_h]]
    input_patch = patches.Polygon(input_pts, fc="#e0e0e0", ec="black", lw=1.5)
    ax.add_patch(input_patch)
    ax.text(in_x + in_w/2, row_input - in_h/2, "Input Data (x)\n[Isolation Forest + Oversampling]", 
            ha='center', va='center', fontsize=11, fontweight='bold')
    
    input_bottom = (in_x + in_w/2, row_input - in_h)

    # ================= FRAMES =================
    # Neural Frame (Left)
    rect_neural = patches.Rectangle((0.5, 17), 8, 12, linewidth=2, edgecolor='blue', facecolor='none', linestyle='--')
    ax.add_patch(rect_neural)
    ax.text(4.5, 29.5, "Neural Module", ha='center', color='blue', fontweight='bold', fontsize=12)

    # Symbolic Frame (Right)
    rect_symbolic = patches.Rectangle((9, 17), 18, 12, linewidth=2, edgecolor='#cc4400', facecolor='none', linestyle='--')
    ax.add_patch(rect_symbolic)
    ax.text(18, 29.5, "Symbolic Expert Module", ha='center', color='#cc4400', fontweight='bold', fontsize=12)
    
    # Fusion Frame (Bottom)
    rect_fusion = patches.Rectangle((2, 1), 24, 15, linewidth=2, edgecolor='green', facecolor='none', linestyle='--')
    ax.add_patch(rect_fusion)
    ax.text(14, 1.5, "Neuro-Symbolic Fusion Layer", ha='center', color='green', fontweight='bold', fontsize=12)


    # ================= NEURAL MODULE =================
    # MLP
    mlp_w, mlp_h = 6, 3
    mlp_bottom = draw_box(col1_c - mlp_w/2, row_modules, mlp_w, mlp_h, "MLP Network\n(Optimized)\n[Lion, PCA]", style=dict(boxstyle="round,pad=0.3", fc="#cce6ff", ec="blue"))
    
    # Prob
    p_nn_bottom = draw_ellipse(col1_c, row_probs, 4, 2, "Neural Prob\n($P_{nn}$)", style=dict(fc="#cce6ff", ec="blue"))
    
    connect(mlp_bottom, (col1_c, row_probs + 1))


    # ================= SYMBOLIC MODULE =================
    # Reordered to avoid crossing: Fuzzy, GNB, Rules
    
    # Fuzzy
    fuzzy_w, fuzzy_h = 4, 2.5
    fuzzy_bottom = draw_box(col2_c - fuzzy_w/2, row_modules, fuzzy_w, fuzzy_h, "Fuzzy Logic\n(Age, BMI)", style=dict(boxstyle="round", fc="#ffe6cc", ec="#cc4400"))
    p_fuzzy_bottom = draw_ellipse(col2_c, row_probs, 3.5, 1.5, "$P_{fuzzy}$", style=dict(fc="#ffe6cc", ec="#cc4400"))
    connect(fuzzy_bottom, (col2_c, row_probs + 0.75))

    # GNB
    gnb_w, gnb_h = 4, 2.5
    gnb_bottom = draw_box(col3_c - gnb_w/2, row_modules, gnb_w, gnb_h, "Naive Bayes\n(Prior)", style=dict(boxstyle="round", fc="#ffe6cc", ec="#cc4400"))
    p_gnb_bottom = draw_ellipse(col3_c, row_probs, 3.5, 1.5, "$P_{gnb}$", style=dict(fc="#ffe6cc", ec="#cc4400"))
    connect(gnb_bottom, (col3_c, row_probs + 0.75))

    # Rules
    rule_w, rule_h = 4, 2.5
    rule_bottom = draw_box(col4_c - rule_w/2, row_modules, rule_w, rule_h, "Clinical Rules\n(5 Heuristics)", style=dict(boxstyle="round", fc="#ffe6cc", ec="#cc4400"))
    # Rules don't have a prob output, they go to flag
    
    
    # ================= INPUT DISTRIBUTION =================
    # Connecting input to all 4 columns top
    split_y = 30.5
    connect(input_bottom, (input_bottom[0], split_y))
    
    # Horizontal bus
    ax.plot([col1_c, col4_c], [split_y, split_y], color='black', lw=1.5)
    
    # Dropdowns
    connect((col1_c, split_y), (col1_c, row_modules + mlp_h))
    connect((col2_c, split_y), (col2_c, row_modules + fuzzy_h))
    connect((col3_c, split_y), (col3_c, row_modules + gnb_h))
    connect((col4_c, split_y), (col4_c, row_modules + rule_h))


    # ================= FUSION LAYER =================
    
    # 1. Consensus (Aggregates Cols 1, 2, 3)
    # Position centered between 1 and 3 -> col2_c seems good? 
    # Or slightly shifted left if GNB is far?
    # Let's put Consensus at x=11 (near col2)
    cons_x, cons_y = 8, row_fusion_top
    cons_w, cons_h = 6, 2.5
    cons_center = (cons_x + cons_w/2, cons_y + cons_h/2)
    cons_bottom = draw_box(cons_x, cons_y, cons_w, cons_h, "Consensus\nAvg($P_{nn}, P_{fz}, P_{gnb}$)", style=dict(boxstyle="round", fc="#ccffcc", ec="green"))
    
    # Connections to Consensus
    # P_nn -> Consensus
    # Start from Right of P_nn ellipse (x=4.5 + 2 = 6.5)
    connect((col1_c + 2, row_probs), (cons_x + 1, cons_y + cons_h), connectionstyle="angle,angleA=0,angleB=90,rad=5")
    
    # P_fuzzy -> Consensus
    # Direct vertical
    connect((col2_c, row_probs - 0.75), (cons_x + 3, cons_y + cons_h), connectionstyle="arc3,rad=0")
    
    # P_gnb -> Consensus
    # Start from Left of P_gnb ellipse (x=17.5 - 1.75 = 15.75)
    connect((col3_c - 1.75, row_probs), (cons_x + 5, cons_y + cons_h), connectionstyle="angle,angleA=180,angleB=90,rad=5")
    
    # 2. Rule Check (Aggregates Col 4)
    # Position under Col 4
    safe_cx, safe_cy = col4_c, row_fusion_top + 1
    safe_bottom_pt = draw_diamond(safe_cx, safe_cy, 1.5, "Risk Rule\nTriggered?", style=dict(fc="#ffebcc", ec="#e69500"))
    
    # Connect Rule -> Safety
    connect(rule_bottom, (safe_cx, safe_cy + 1.5), text="Flags")

    # 3. Override Logic
    # If Yes -> Force Prob
    # If No -> Use Consensus
    # Let's verify flow.
    # Consensus provides a Base Score.
    # Rule Check determines if we IGNORE Base Score or USE it.
    
    # So we need a "Gate" or "Switch" node.
    # Let's put a "Final Logic" box centrally below.
    
    final_logic_x, final_logic_y = 10, row_fusion_mid - 2
    final_logic_w, final_logic_h = 8, 3
    # final_logic_bottom = draw_box(final_logic_x, final_logic_y, final_logic_w, final_logic_h, "Decision Gate\nApply Overrides", style=dict(boxstyle="round", fc="#e6ffe6", ec="green"))
    
    # Let's draw arrows explicitly instead of a box for the logic flow
    
    # Override Box (Result of Rule=Yes)
    override_x, override_y = col4_c - 2.5, row_fusion_mid
    override_w, override_h = 5, 2.5
    override_bottom = draw_box(override_x, override_y, override_w, override_h, "Safety Override\nSet $P \in \{0.05, 0.95\}$", style=dict(boxstyle="round", fc="#ffcccb", ec="red"))
    
    # Connect Safety -> Override (Yes)
    connect((safe_cx, safe_bottom_pt[1]), (safe_cx, override_y + override_h), text="Yes")
    
    # Final Output
    final_cx, final_cy = 14, row_final
    final_w, final_h = 5, 2
    draw_ellipse(final_cx, final_cy, final_w, final_h, "Final Diagnosis\nOutput Class", style=dict(fc="white", ec="black", lw=2))
    
    # Connect Override -> Final
    # Curved connection
    connect((override_x + override_w/2, override_y), (final_cx + 1, final_cy + 2), connectionstyle="arc3,rad=-0.2")
    
    # Connect Consensus -> Final
    connect((cons_x + cons_w/2, cons_y), (final_cx - 1, final_cy + 2), connectionstyle="arc3,rad=0.2")
    
    # Add a visual "Gate" annotation
    ax.text(14, 7.5, "Priority Merge\n(Rules > Stats)", ha='center', fontsize=9, style='italic', backgroundcolor='white', bbox=dict(facecolor='white', edgecolor='lightgray'))


    plt.tight_layout()
    plt.savefig('figures/architecture_diagram.png', dpi=300, bbox_inches='tight')
    print("Diagram generated at figures/architecture_diagram.png")

if __name__ == "__main__":
    draw_diagram()
