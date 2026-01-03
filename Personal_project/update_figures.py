import os
import shutil
import subprocess
import sys

def run_script(script_path):
    print(f"\n--- Running {script_path} ---")
    result = subprocess.run([sys.executable, script_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error running {script_path}:")
        print(result.stderr)
    else:
        print(result.stdout)
        print(f"Successfully ran {script_path}")

def copy_file(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Copied {src} to {dst}")
    else:
        print(f"Warning: Source file {src} does not exist.")

def main():
    # Ensure paper/figures directory exists
    paper_figs_dir = "paper/figures"
    os.makedirs(paper_figs_dir, exist_ok=True)

    # 1. Run generate_all_figures.py (Updates figures/)
    run_script("generate_all_figures.py")

    # 2. Run src/analysis/generate_missing_figures.py (Updates paper/figures/)
    run_script("src/analysis/generate_missing_figures.py")

    # 3. Run src/analysis/plot_hpo.py (Updates experiments/)
    run_script("src/analysis/plot_hpo.py")

    # 4. Copy figures to paper/figures/
    
    # From figures/
    figs_source = "figures"
    if os.path.exists(figs_source):
        for f in os.listdir(figs_source):
            if f.endswith(".png"):
                copy_file(os.path.join(figs_source, f), os.path.join(paper_figs_dir, f))
    
    # From experiments/
    experiments_source = "experiments"
    exp_figures = [
        "hpo_convergence.png",
        "hpo_distributions.png",
        "confusion_matrix.png",
        "evo_comparison.png",
        "feature_importance.png",
        "unsupervised_pca_labels.png",
        "unsupervised_pca_clusters.png"
    ]
    
    for f in exp_figures:
        copy_file(os.path.join(experiments_source, f), os.path.join(paper_figs_dir, f))

    print("\nAll figures updated and copied to paper/figures/")

if __name__ == "__main__":
    main()
