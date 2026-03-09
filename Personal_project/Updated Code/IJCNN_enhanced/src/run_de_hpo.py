import sys
import os
# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.hpo.differential_evolution import run_differential_evolution
from sklearn.model_selection import train_test_split

def run():
    print("=== Loading Data for DE HPO ===")
    # Fast load
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
    
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean,
        test_size=0.1,
        stratify=y_train_clean,
        random_state=42
    )
    
    input_dim = X_tr.shape[1]
    
    print("\n=== Starting Differential Evolution (DE) HPO ===")
    # Small run for demo: pop=5, gen=3
    res = run_differential_evolution(
        X_tr, y_tr, X_val, y_val,
        input_dim,
        pop_size=5,
        generations=3,
        seed=42
    )
    
    print("\n=== DE Finished ===")
    print(f"Best ROC-AUC: {res['best']['roc_auc']:.4f}")
    print(f"Best Params: {res['best']['hparams']}")
    print(f"Results saved to experiments/hpo_results/differential_evolution.json")

if __name__ == "__main__":
    run()
