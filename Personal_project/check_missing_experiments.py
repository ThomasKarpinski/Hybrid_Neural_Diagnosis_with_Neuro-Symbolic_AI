
import os
import itertools

results_dir = "experiments/hpo_results"
dim_reductions = ["raw", "pca", "ae"]
optimizers = ["adam", "sgd", "rmsprop", "adagrad", "adadelta", "lion"]
hpo_methods = ["random", "optuna", "genetic", "alshade"]

existing_files = set(os.listdir(results_dir))
missing = []
found_count = 0

for dim, opt, hpo in itertools.product(dim_reductions, optimizers, hpo_methods):
    filename = f"{dim}_{opt}_{hpo}.json"
    if filename in existing_files:
        found_count += 1
    else:
        missing.append(filename)

print(f"Total expected: {len(dim_reductions) * len(optimizers) * len(hpo_methods)}")
print(f"Found: {found_count}")
print(f"Missing: {len(missing)}")
if missing:
    print("Missing files:")
    for f in missing:
        print(f" - {f}")
