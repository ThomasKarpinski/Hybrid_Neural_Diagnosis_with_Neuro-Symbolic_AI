import os
import numpy as np
import pandas as pd
import json
import time
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers
from src.models.autoencoder import train_autoencoder, get_embeddings

# HPO Runners
from src.hpo.grid_random_search import run_random_search
from src.hpo.bayesian_opt import run_optuna
from src.hpo.evolutionary_hpo import run_genetic
from src.hpo.enhanced_alshade import run_enhanced_alshade

def run_classical_baselines(X_train, y_train, X_test, y_test, save_path="experiments/hpo_results/classical_baselines.json"):
    print("\n>>> RUNNING CLASSICAL BASELINES")
    results = []
    
    # Models definition
    models = {
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    }
    
    try:
        from xgboost import XGBClassifier
        models["XGBoost"] = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    except ImportError:
        print("XGBoost not installed, skipping...")

    for name, clf in models.items():
        print(f"   Training {name}...")
        start = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - start
        
        # Predictions
        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)[:, 1]
        
        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred),
            "F1-Score": f1_score(y_test, y_pred),
            "AUC-ROC": roc_auc_score(y_test, y_prob),
            "Time": train_time
        }
        results.append(metrics)
        print(f"     {name}: F1={metrics['F1-Score']:.4f}, AUC={metrics['AUC-ROC']:.4f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    return results

def get_pca_features(X_train, X_val, X_test, variance_threshold=0.95):
    print(f"Training PCA (variance={variance_threshold})...")
    pca = PCA(n_components=variance_threshold)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    print(f"PCA retained {pca.n_components_} components.")
    return X_train_pca, X_val_pca, X_test_pca, pca.n_components_

def get_ae_features(X_train, X_val, X_test, input_dim, encoding_dim=10):
    print(f"Training Autoencoder (dim={encoding_dim})...")
    model, _, _ = train_autoencoder(X_train, input_dim, encoding_dim=encoding_dim, epochs=20, verbose=True)
    
    X_train_ae = get_embeddings(model, X_train)
    X_val_ae = get_embeddings(model, X_val)
    X_test_ae = get_embeddings(model, X_test)
    return X_train_ae, X_val_ae, X_test_ae, encoding_dim

def run_experiments():
    # 1. Load and Preprocess Data
    print("Loading Data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
    
    # Run Classical Baselines first (on clean data)
    run_classical_baselines(X_train_clean, y_train_clean, X_test, y_test)
    
    # Train/Val split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_clean, y_train_clean,
        test_size=0.1,
        stratify=y_train_clean,
        random_state=42
    )
    
    input_dim_raw = X_tr.shape[1]
    
    # 2. Prepare Representations
    datasets = {}
    
    # A. Raw
    datasets["raw"] = (X_tr, X_val, y_tr, y_val, input_dim_raw)
    
    # B. PCA
    X_tr_pca, X_val_pca, _, dim_pca = get_pca_features(X_tr, X_val, X_test, variance_threshold=0.95)
    datasets["pca"] = (X_tr_pca, X_val_pca, y_tr, y_val, dim_pca)
    
    # C. Autoencoder
    X_tr_ae, X_val_ae, _, dim_ae = get_ae_features(X_tr, X_val, X_test, input_dim_raw, encoding_dim=10)
    datasets["ae"] = (X_tr_ae, X_val_ae, y_tr, y_val, dim_ae)
    
    # 3. Define Experiment Grid
    representations = ["raw", "pca", "ae"]
    optimizers = ["adam", "rmsprop", "sgd", "adagrad", "adadelta", "lion"]
    
    # Reduced budget for demonstration/CLI speed
    # Increase these for full paper reproduction
    N_TRIALS_RANDOM = 5
    N_TRIALS_OPTUNA = 5
    POP_SIZE_EVO = 5
    GEN_EVO = 3
    
    results_log = []

    for rep_name in representations:
        X_t, X_v, y_t, y_v, in_dim = datasets[rep_name]
        print(f"\n\n>>> STARTING REPRESENTATION: {rep_name.upper()} (dim={in_dim})")
        
        for opt_name in optimizers:
            print(f"\n   >>> OPTIMIZER: {opt_name.upper()}")
            fixed_hparams = {"optimizer_name": opt_name}
            base_path = f"experiments/hpo_results/{rep_name}_{opt_name}"
            
            # 1. Random Search
            print("      Running Random Search...")
            run_random_search(
                X_t, y_t, X_v, y_v, in_dim,
                search_space={
                    "lr": ("loguniform", 1e-4, 1e-1),
                    "batch_size": [32, 64],
                    "epochs": ("uniform", 10, 30)
                },
                n_iter=N_TRIALS_RANDOM,
                save_path=f"{base_path}_random.json",
                fixed_hparams=fixed_hparams
            )
            
            # 2. Optuna
            print("      Running Optuna...")
            run_optuna(
                X_t, y_t, X_v, y_v, in_dim,
                n_trials=N_TRIALS_OPTUNA,
                save_path=f"{base_path}_optuna.json",
                fixed_hparams=fixed_hparams
            )
            
            # 3. Genetic
            print("      Running Genetic Algorithm...")
            run_genetic(
                X_t, y_t, X_v, y_v, in_dim,
                pop_size=POP_SIZE_EVO, generations=GEN_EVO,
                save_path=f"{base_path}_genetic.json",
                fixed_hparams=fixed_hparams
            )
            
            # 4. Enhanced ALSHADE
            print("      Running Enhanced ALSHADE...")
            alshade_hparams = fixed_hparams.copy()
            alshade_hparams["use_fuzzy"] = True
            run_enhanced_alshade(
                X_t, y_t, X_v, y_v, in_dim,
                pop_size=POP_SIZE_EVO, generations=GEN_EVO,
                save_path=f"{base_path}_alshade.json",
                fixed_hparams=alshade_hparams
            )

if __name__ == "__main__":
    run_experiments()
