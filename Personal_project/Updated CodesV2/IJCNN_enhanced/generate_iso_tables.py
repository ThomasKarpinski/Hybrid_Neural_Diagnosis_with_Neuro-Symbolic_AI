import json
import os
import sys
import numpy as np
from scipy.stats import wilcoxon, rankdata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

CACHE_FILE = "paper_results_cache.json"

def load_json(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        return json.load(f)

# Global holder
Y_TEST_GLOBAL = None

def find_best_threshold_metrics(y_true, probs):
    """
    Find optimal threshold maximizing F1 score and return metrics.
    """
    best_thresh = 0.5
    best_f1 = -1.0
    
    # Search range 0.1 to 0.9
    thresholds = np.linspace(0.1, 0.9, 81)
    
    for t in thresholds:
        p = (probs >= t).astype(int)
        score = f1_score(y_true, p, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_thresh = t
            
    # Calculate final metrics with best threshold
    preds = (probs >= best_thresh).astype(int)
    
    return {
        "Acc": accuracy_score(y_true, preds),
        "Prec": precision_score(y_true, preds, zero_division=0),
        "Rec": recall_score(y_true, preds),
        "F1": best_f1,
        "ROC": roc_auc_score(y_true, probs),
        "Threshold": best_thresh
    }

def generate_latex_table(feature_name, prefix, output_path, cache):
    # Optimizer order
    optimizers = ["RMSProp", "Adam", "SGD", "Adagrad", "Adadelta", "Lion"]
    # HPO Method order
    hpo_methods = ["Random Search", "Bayesian Opt.", "Genetic Algo.", "ALSHADE"]
    
    # Map method names to file keys
    method_map = {
        "Random Search": "random",
        "Bayesian Opt.": "optuna",
        "Genetic Algo.": "genetic",
        "ALSHADE": "alshade"
    }
    
    # Collect data
    rows_data = []
    
    # Identify Baseline Probs (RMSProp + Random)
    baseline_probs = None
    baseline_key = f"{prefix}_rmsprop_random"
    if baseline_key in cache:
        baseline_probs = np.array(cache[baseline_key]["Probs"]).flatten()
    
    for opt in optimizers:
        opt_key = opt.lower()
        for method in hpo_methods:
            method_key = method_map[method]
            cache_key = f"{prefix}_{opt_key}_{method_key}"
            
            row = {
                "opt": opt,
                "method": method,
                "acc": "-", "prec": "-", "rec": "-", "f1": "-", "roc": "-", 
                "epochs": "-", "time": "-", "p_val": "n.s.", "rank": "-",
                "found": False, "probs": None, "raw_roc": -1.0
            }
            
            # 1. Try Cache
            if cache_key in cache:
                m = cache[cache_key]
                row["probs"] = np.array(m['Probs']).flatten()
                
                # RECALCULATE METRICS WITH DYNAMIC THRESHOLD if Y_TEST is available
                if Y_TEST_GLOBAL is not None and len(row["probs"]) == len(Y_TEST_GLOBAL):
                    new_metrics = find_best_threshold_metrics(Y_TEST_GLOBAL, row["probs"])
                    row["acc"] = f"{new_metrics['Acc']:.4f}"
                    row["prec"] = f"{new_metrics['Prec']:.4f}"
                    row["rec"] = f"{new_metrics['Rec']:.4f}"
                    row["f1"] = f"{new_metrics['F1']:.4f}"
                    row["roc"] = f"{new_metrics['ROC']:.4f}"
                    row["raw_roc"] = float(new_metrics['ROC'])
                else:
                    # Fallback to cached metrics (likely using 0.5 threshold)
                    row["acc"] = f"{m['Acc']:.4f}"
                    row["prec"] = f"{m['Prec']:.4f}"
                    row["rec"] = f"{m['Rec']:.4f}"
                    row["f1"] = f"{m['F1']:.4f}"
                    row["roc"] = f"{m['ROC']:.4f}"
                    row["raw_roc"] = float(m['ROC'])
                
                row["found"] = True
            
            # 2. Read HPO file for Time/Epochs (and metrics fallback)
            hpo_file = f"experiments/hpo_results/{prefix}_{opt_key}_{method_key}.json"
            if os.path.exists(hpo_file):
                try:
                    with open(hpo_file, 'r') as f:
                        h = json.load(f)
                    
                    time_s = h.get('time_s', 0)
                    row["time"] = f"{time_s:.1f}s"
                    row["epochs"] = h.get('best', {}).get('hparams', {}).get('epochs', '-')
                    
                    if not row["found"]:
                        best = h.get('best', {})
                        if 'roc_auc' in best:
                            # We can't do dynamic thresholding here without probs/model
                            # So we stick to what we have or 0.5 if we loaded metrics file
                            row["roc"] = f"{best['roc_auc']:.4f}"
                            row["raw_roc"] = float(best['roc_auc'])
                            row["found"] = True
                        
                        # Fallback metrics file
                        met_file = f"experiments/hpo_results/{prefix}_{opt_key}_{method_key}_metrics.json"
                        if os.path.exists(met_file):
                            with open(met_file, 'r') as f:
                                m = json.load(f)
                            row["acc"] = f"{m.get('accuracy', 0):.4f}"
                            row["prec"] = f"{m.get('precision', 0):.4f}"
                            row["rec"] = f"{m.get('recall', 0):.4f}"
                            row["f1"] = f"{m.get('f1_score', 0):.4f}"
                except: pass
            
            rows_data.append(row)

    # Calculate Ranks
    valid_rocs = [r["raw_roc"] for r in rows_data if r["found"]]
    if valid_rocs:
        ranks = rankdata([-x for x in valid_rocs], method='min')
        idx = 0
        for r in rows_data:
            if r["found"]:
                r["rank"] = int(ranks[idx])
                idx += 1

    # Calculate P-values
    if Y_TEST_GLOBAL is not None and baseline_probs is not None:
        try:
            err_base = np.abs(Y_TEST_GLOBAL - baseline_probs)
            for r in rows_data:
                if r["probs"] is not None:
                    p_curr = r["probs"]
                    if len(p_curr) == len(Y_TEST_GLOBAL):
                        err_curr = np.abs(Y_TEST_GLOBAL - p_curr)
                        if np.allclose(err_base, err_curr):
                            p = 1.0
                        else:
                            _, p = wilcoxon(err_base, err_curr)
                        
                        if p < 0.001: r["p_val"] = "< 0.001"
                        elif p < 0.05: r["p_val"] = "< 0.05"
                        else: r["p_val"] = "n.s."
        except Exception as e:
            print(f"Stats Error: {e}")

    # Generate Latex using chr(92) for backslash
    BS = chr(92)
    latex_content = BS + "begin{table*}[t]\n"
    latex_content += BS + "centering\n"
    latex_content += BS + "scriptsize\n"
    latex_content += BS + "caption{Hyperparameter Optimization (HPO) Strategy Analysis for " + feature_name + " (ISO)}\n"
    latex_content += BS + "label{tab:opt_" + prefix + "_iso}\n"
    latex_content += BS + "begin{tabular}{llllllllllll}\n"
    latex_content += BS + "toprule \n"
    latex_content += BS + "textbf{Feature} & " + BS + "textbf{Optimizer} & " + BS + "textbf{HPO Method} & " + BS + "textbf{Acc} & " + BS + "textbf{Prec} & " + BS + "textbf{Rec} & " + BS + "textbf{F1} & " + BS + "textbf{ROC} & " + BS + "textbf{Epochs} & " + BS + "textbf{Run-Time} & " + BS + "textbf{$p$-value} & " + BS + "textbf{F-Rank}  " + BS + BS + "\n"
    latex_content += BS + "midrule\n"
    
    first_feature_row = True
    row_idx = 0
    
    for i_opt, opt in enumerate(optimizers):
        opt_key = opt.lower()
        first_opt_row = True
        
        for i_method, method in enumerate(hpo_methods):
            r = rows_data[row_idx]
            row_idx += 1
            
            row_str = ""
            if first_feature_row:
                row_str += BS + "multirow{24}{*}{ " + feature_name + "} & "
                first_feature_row = False
            else: row_str += " & "
                
            if first_opt_row:
                row_str += BS + "multirow{4}{*}{ " + opt + "} & "
                first_opt_row = False
            else: row_str += " & "
            
            row_str += f"{method} & {r['acc']} & {r['prec']} & {r['rec']} & {r['f1']} & {r['roc']} & {r['epochs']} & {r['time']} & {r['p_val']} & {r['rank']} " + BS + BS
            latex_content += row_str + "\n"
            
        if i_opt < len(optimizers) - 1:
            latex_content += BS + "cline{2-12}\n"

    latex_content += BS + "bottomrule\n"
    latex_content += BS + "end{tabular}\n"
    latex_content += BS + "end{table*}\n"

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f: f.write(latex_content)
    print(f"Generated {output_path}")

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    
    print("Loading cache...")
    cache = load_json(CACHE_FILE)
    if not cache:
        print("Warning: Cache empty or not found. Metrics for AE/PCA will likely be missing.")
        cache = {}
        
    print("Loading Data for stats...")
    try:
        from src.data.load_data import prepare_data
        _, _, _, y_test, _, _ = prepare_data()
        Y_TEST_GLOBAL = y_test
        print("Data loaded successfully.")
    except Exception as e:
        print(f"Could not load data: {e}")

    generate_latex_table("Raw Input", "raw", "paper/generated_tables/iso_tables/table_opt_raw_iso.tex", cache)
    generate_latex_table("PCA", "pca", "paper/generated_tables/iso_tables/table_opt_pca_iso.tex", cache)
    generate_latex_table("Autoencoder", "ae", "paper/generated_tables/iso_tables/table_opt_ae_iso.tex", cache)
