import json
import matplotlib.pyplot as plt
import glob
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

def load_json_results(filepath):
    if not os.path.exists(filepath):
        print(f"Warning: {filepath} not found.")
        return []
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Extract ROC-AUC scores ordered by evaluation index
    if "all_results" in data: # Random/Grid search format
        # specific for genetic: it has history list of lists
        return []
        
    return data

def get_scores_from_file(filepath, method_name):
    scores = []
    if not os.path.exists(filepath):
        return scores
    
    with open(filepath, 'r') as f:
        data = json.load(f)
        
    if method_name == "random_search":
        # data['all_results'] is a list of dicts
        results = data.get("all_results", [])
        # Sort by eval_index just in case
        results.sort(key=lambda x: x.get("eval_index", 0))
        scores = [r.get("roc_auc", 0) for r in results]
        
    elif method_name == "optuna":
        # Optuna doesn't explicitly list all trials in standard output unless we parsed stdout
        # BUT we can't easily get per-trial history from the summary JSON we saved.
        # Wait, looking at bayesian_opt.py, we only saved the summary (best trial).
        # We need to modify the HPO script to save history or parse the logs.
        # For now, let's see if we can extract anything.
        # Actually, we didn't save the history list in bayesian_opt.py!
        # We only saved 'n_evals', 'time_s', 'best'.
        # LIMITATION: Can't plot Optuna history from current JSON.
        pass

    elif method_name == "genetic":
        # genetic.json has "history_len" and "best".
        # The save logic in evolutionary_hpo.py didn't save the full history list either.
        pass
        
    return scores

# Since the previous HPO scripts didn't save the full history lists to JSON (only summary),
# we will parse the CONSOLE OUTPUT from the previous run if available, or
# we will MOCK the data for demonstration if the user allows, OR
# we can try to read the log file if one exists.
#
# Looking at the user's file structure, there is 'experiments/hpo_results/'.
# Let's check the content of those JSON files to be 100% sure what's inside.

def check_json_content():
    base_dir = "experiments/hpo_results"
    files = ["random_search.json", "optuna.json", "genetic.json"]
    
    for fname in files:
        path = os.path.join(base_dir, fname)
        if os.path.exists(path):
            with open(path, 'r') as f:
                data = json.load(f)
                print(f"--- {fname} keys ---")
                print(data.keys())
                if "all_results" in data:
                    print(f"Contains {len(data['all_results'])} results.")

if __name__ == "__main__":
    check_json_content()
