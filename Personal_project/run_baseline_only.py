from src.pipeline import run_pipeline
import os

if __name__ == "__main__":
    print("Running baseline pipeline only...")
    model, metrics, data, scaler = run_pipeline()
    print("Baseline metrics obtained:")
    print(metrics)
