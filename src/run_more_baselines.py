import os
import time
import json
import numpy as np
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    GradientBoostingClassifier, 
    AdaBoostClassifier, 
    ExtraTreesClassifier, 
    BaggingClassifier,
    RandomForestClassifier
)
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef

from src.data.load_data import prepare_data
from src.data.outlier_detection import remove_outliers

def run_extended_baselines(save_path="experiments/hpo_results/extended_baselines.json"):
    print("Loading Data...")
    X_train, X_test, y_train, y_test, scaler, feature_names = prepare_data()
    X_train_clean, y_train_clean = remove_outliers(X_train, y_train)
    
    print("\n>>> RUNNING EXTENDED BASELINES")
    results = []
    
    # Models definition
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=1000, random_state=42)),
        ("Ridge Classifier", RidgeClassifier(random_state=42)),
        ("k-Nearest Neighbors", KNeighborsClassifier(n_neighbors=5)),
        ("Decision Tree", DecisionTreeClassifier(random_state=42)),
        ("Extra Tree", ExtraTreeClassifier(random_state=42)),
        ("Gaussian NB", GaussianNB()),
        ("Bernoulli NB", BernoulliNB()),
        ("LDA", LinearDiscriminantAnalysis()),
        ("QDA", QuadraticDiscriminantAnalysis()),
        ("AdaBoost", AdaBoostClassifier(algorithm='SAMME', random_state=42)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=42)),
        ("Extra Trees", ExtraTreesClassifier(n_estimators=100, random_state=42)),
        ("Bagging Classifier", BaggingClassifier(random_state=42)),
        # Re-running these for consistency
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=42)), 
        ("SVM (RBF)", SVC(kernel='rbf', probability=True, random_state=42)),
        ("Linear SVC", LinearSVC(dual=False, random_state=42)),
        ("SGD Classifier", SGDClassifier(loss='log_loss', random_state=42))
    ]
    
    for name, clf in models:
        print(f"   Training {name}...")
        start = time.time()
        
        try:
            clf.fit(X_train_clean, y_train_clean)
            train_time = time.time() - start
            
            # Predictions
            y_pred = clf.predict(X_test)
            
            # Probability for AUC
            if hasattr(clf, "predict_proba"):
                y_prob = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                # For models like Ridge/LinearSVC, use decision function
                d = clf.decision_function(X_test)
                # Sigmoid for approximation
                y_prob = 1 / (1 + np.exp(-d)) 
            else:
                y_prob = y_pred # Fallback
            
            metrics = {
                "Model": name,
                "Accuracy": accuracy_score(y_test, y_pred),
                "Precision": precision_score(y_test, y_pred, zero_division=0),
                "Recall": recall_score(y_test, y_pred),
                "F1-Score": f1_score(y_test, y_pred),
                "AUC-ROC": roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else 0.0,
                "MCC": matthews_corrcoef(y_test, y_pred),
                "Time": train_time
            }
            results.append(metrics)
            print(f"     {name}: F1={{metrics['F1-Score']:.4f}}, AUC={{metrics['AUC-ROC']:.4f}}, MCC={{metrics['MCC']:.4f}}")
            
        except Exception as e:
            print(f"    Failed to run {name}: {e}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    run_extended_baselines()
