import json
import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import pandas as pd
import seaborn as sns
from typing import List

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def get_hpo_data_from_json(filepath, method_name):
    """Loads HPO results from a JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return pd.DataFrame() # Return empty DataFrame
        
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    results = data.get("all_results", [])
    
    # Extract roc_auc and hparams
    processed_results = []
    for i, r in enumerate(results): # Added enumerate
        roc_auc = r.get("roc_auc")
        hparams = r.get("hparams", {})
        
        # Ensure roc_auc is a float and not NaN
        if roc_auc is None or (isinstance(roc_auc, (float, np.float64)) and np.isnan(roc_auc)):
            roc_auc = -np.inf # Treat invalid ROC-AUCs as very bad
        
        entry = {"roc_auc": roc_auc, "method": method_name, "trial_number": r.get("trial_number", i)} # Added trial_number
        # Flatten hparams into the main dictionary
        for k, v in hparams.items():
            # Convert numpy types to native Python types for pandas
            if isinstance(v, (np.float64, np.int64, np.bool_)):
                entry[k] = v.item()
            else:
                entry[k] = v
        processed_results.append(entry)
            
    return pd.DataFrame(processed_results)

def get_manual_optuna_data():
    # Manually reconstructed data based on previous console output (only scores)
    # This function cannot provide hparams for scatter plots without manual parsing of huge logs.
    # For now, it will return empty df for hparams if not explicitly available.
    return pd.DataFrame() 

def get_manual_genetic_data():
    # Manually reconstructed data based on previous console output (only scores)
    return pd.DataFrame()


def plot_convergence():
    print("Generating HPO Convergence Plot...")
    
    # 1. Random Search (Actual Data)
    rs_df = get_hpo_data_from_json("experiments/hpo_results/random_search.json", "Random Search")
    
    # 2. Optuna (Actual Data)
    opt_df = get_hpo_data_from_json("experiments/hpo_results/optuna.json", "Optuna")
    
    # 3. Genetic (Actual Data)
    gen_df = get_hpo_data_from_json("experiments/hpo_results/genetic.json", "Genetic Algo")

    # 4. DE (Actual Data)
    de_df = get_hpo_data_from_json("experiments/hpo_results/differential_evolution.json", "Differential Evolution")

    # 5. PSO (Actual Data)
    pso_df = get_hpo_data_from_json("experiments/hpo_results/pso.json", "Particle Swarm (PSO)")
    
    all_hpo_df = pd.concat([rs_df, opt_df, gen_df, de_df, pso_df], ignore_index=True)

    # Process into "Best So Far" for convergence lines
    def get_best_so_far_df(df_method):
        # Assume df_method has 'trial_number' column now
        df_method = df_method.sort_values(by="trial_number")
        df_method["best_roc_auc_so_far"] = df_method["roc_auc"].cummax()
        return df_method

    # Apply to each method (they should all have 'trial_number' now)
    rs_df = get_best_so_far_df(rs_df) if not rs_df.empty else pd.DataFrame()
    opt_df = get_best_so_far_df(opt_df) if not opt_df.empty else pd.DataFrame()
    gen_df = get_best_so_far_df(gen_df) if not gen_df.empty else pd.DataFrame()
    de_df = get_best_so_far_df(de_df) if not de_df.empty else pd.DataFrame()
    pso_df = get_best_so_far_df(pso_df) if not pso_df.empty else pd.DataFrame()

    plt.figure(figsize=(10, 6))
    
    if not rs_df.empty:
        plt.plot(rs_df["trial_number"] + 1, rs_df["best_roc_auc_so_far"], label='Random Search', marker='o', linestyle='--')
    if not opt_df.empty:
        plt.plot(opt_df["trial_number"] + 1, opt_df["best_roc_auc_so_far"], label='Optuna (Bayesian)', marker='s', linestyle='-')
    if not gen_df.empty:
        plt.plot(gen_df["trial_number"] + 1, gen_df["best_roc_auc_so_far"], label='Genetic Algo', marker='^', linestyle=':')
    if not de_df.empty:
        plt.plot(de_df["trial_number"] + 1, de_df["best_roc_auc_so_far"], label='Differential Evolution', marker='x', linestyle='-.')
    if not pso_df.empty:
        plt.plot(pso_df["trial_number"] + 1, pso_df["best_roc_auc_so_far"], label='Particle Swarm (PSO)', marker='*', linestyle=':')
    
    plt.title("HPO Convergence: Best ROC-AUC Found vs. Number of Evaluations")
    plt.xlabel("Evaluation Trial #")
    plt.ylabel("Best ROC-AUC Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = "experiments/hpo_convergence.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")


def plot_hyperparameter_distributions(all_hpo_df: pd.DataFrame, hparams_to_plot: List[str]):
    """
    Generates scatter plots of hyperparameter pairs colored by ROC-AUC.
    """
    if all_hpo_df.empty:
        print("No HPO data available for plotting distributions.")
        return
        
    print("Generating Hyperparameter Distribution Plots...")

    # Identify numerical hyperparameters
    numerical_hparams = [hp for hp in hparams_to_plot if hp in all_hpo_df.columns and pd.api.types.is_numeric_dtype(all_hpo_df[hp])]
    
    if len(numerical_hparams) < 2:
        print("Not enough numerical hyperparameters to create pair-wise distribution plots.")
        return
        
    num_plots = len(numerical_hparams) * (len(numerical_hparams) - 1) // 2
    if num_plots == 0:
        return
        
    # Create subplots dynamically
    fig, axes = plt.subplots(ncols=len(numerical_hparams)-1, nrows=len(numerical_hparams)-1, figsize=(15, 12))
    
    if len(numerical_hparams) == 2: # Handle case with only two numerical hparams
        axes = np.array([axes]) # Make it 2D for consistent indexing

    axes = axes.flatten() # Flatten for easy iteration

    plot_idx = 0
    for i in range(len(numerical_hparams)):
        for j in range(i + 1, len(numerical_hparams)):
            hp1 = numerical_hparams[i]
            hp2 = numerical_hparams[j]
            
            ax = axes[plot_idx]
            
            sns.scatterplot(
                data=all_hpo_df,
                x=hp1,
                y=hp2,
                hue="roc_auc",
                size="roc_auc",
                palette="viridis",
                sizes=(20, 200),
                ax=ax,
                legend="brief" if plot_idx == 0 else False
            )
            ax.set_title(f"{hp1} vs {hp2}")
            ax.set_xscale("log" if "lr" in (hp1,hp2) or "weight_decay" in (hp1,hp2) else "linear")
            ax.set_yscale("log" if "lr" in (hp1,hp2) or "weight_decay" in (hp1,hp2) else "linear")

            plot_idx += 1
            
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        fig.delaxes(axes[i])
            
    fig.suptitle("Hyperparameter Distributions vs. ROC-AUC", y=1.02, fontsize=16)
    plt.tight_layout()
    save_path = "experiments/hpo_distributions.png"
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")

if __name__ == "__main__":
    plot_convergence()
    
    # Collect data for distribution plots
    rs_df = get_hpo_data_from_json("experiments/hpo_results/random_search.json", "Random Search")
    opt_df = get_hpo_data_from_json("experiments/hpo_results/optuna.json", "Optuna")
    gen_df = get_hpo_data_from_json("experiments/hpo_results/genetic.json", "Genetic Algo")
    de_df = get_hpo_data_from_json("experiments/hpo_results/differential_evolution.json", "Differential Evolution")
    pso_df = get_hpo_data_from_json("experiments/hpo_results/pso.json", "Particle Swarm (PSO)")
    
    # Concatenate all non-empty DataFrames
    all_hpo_df = pd.concat([df for df in [rs_df, opt_df, gen_df, de_df, pso_df] if not df.empty], ignore_index=True)
    
    # Define which hyperparameters to plot. Exclude epochs and batch_size (often categorical/integer)
    # Focus on continuous ones that tend to have log scales for better visualization.
    hparams_for_scatter = ["lr", "weight_decay", "dropout", "beta1", "beta2"]
    plot_hyperparameter_distributions(all_hpo_df, hparams_for_scatter)