import time
import json
import numpy as np
import random
import os
from typing import Dict, Any, List

from src.hpo.common import train_and_eval_on_val

class Particle:
    def __init__(self, bounds: Dict[str, Any], initial_hparams: Dict[str, Any]):
        self.position = initial_hparams # dict of actual hparams
        self.velocity = {k: random.uniform(-1, 1) for k in bounds.keys()} # numerical representation
        
        self.pbest_position = self.position
        self.pbest_score = -np.inf
        
        self.score = -np.inf # current score

def hparams_to_vector(hparams: Dict[str, Any], bounds_info: Dict[str, Any]) -> Dict[str, float]:
    """Converts dictionary of hparams to numerical vector representation."""
    vec = {}
    for k, v_info in bounds_info.items():
        val = hparams[k]
        if v_info[2] == "cat": # Categorical
            vec[k] = v_info[3].index(val) # Map to index
        elif v_info[2] == "int": # Integer
            vec[k] = float(val)
        else: # Float
            vec[k] = val
    return vec

def vector_to_hparams(vector: Dict[str, float], bounds_info: Dict[str, Any]) -> Dict[str, Any]:
    """Converts numerical vector representation back to hparams dict, applying bounds and types."""
    hparams = {}
    for k, v_info in bounds_info.items():
        val = vector[k]
        if v_info[2] == "cat": # Categorical
            idx = int(np.clip(round(val), 0, len(v_info[3]) - 1))
            hparams[k] = v_info[3][idx]
        elif v_info[2] == "int": # Integer
            hparams[k] = int(np.clip(round(val), v_info[0], v_info[1]))
        else: # Float
            hparams[k] = np.clip(val, v_info[0], v_info[1])
    return hparams

def run_particle_swarm_optimization(
    X_train, y_train, X_val, y_val,
    input_dim: int,
    pop_size: int = 10,
    generations: int = 5,
    w: float = 0.5,  # Inertia weight
    c1: float = 1.5, # Cognitive coefficient
    c2: float = 1.5, # Social coefficient
    seed: int = 42,
    save_path: str = "experiments/hpo_results/pso.json"
):
    """
    Particle Swarm Optimization (PSO) for hyperparameter optimization.
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Define search space bounds (low, high, type, [choices for cat])
    bounds_info = {
        "lr": (1e-5, 1e-1, "float"),
        "batch_size": (16, 128, "cat", [16, 32, 64, 128]), # 4 options
        "weight_decay": (1e-7, 1e-2, "float"),
        "dropout": (0.0, 0.5, "float"),
        "epochs": (10, 60, "int"),
        "beta1": (0.8, 0.99, "float"),
        "beta2": (0.9, 0.9999, "float"),
    }

    # Initialize Swarm
    swarm: List[Particle] = []
    gbest_score = -np.inf
    gbest_position_vec = None # numerical vector
    
    for _ in range(pop_size):
        initial_hparams_vec = {}
        for k, v_info in bounds_info.items():
            if v_info[2] == "cat":
                initial_hparams_vec[k] = random.choice(v_info[3])
            elif v_info[2] == "int":
                initial_hparams_vec[k] = random.randint(v_info[0], v_info[1])
            else:
                initial_hparams_vec[k] = random.uniform(v_info[0], v_info[1])
        swarm.append(Particle(bounds_info, initial_hparams_vec))

    history = [] # Store all evaluations
    start_time = time.time()

    for gen in range(generations):
        print(f"[PSO] Generation {gen + 1}/{generations}")
        
        for i, particle in enumerate(swarm):
            # Evaluate current position
            hparams_to_eval = vector_to_hparams(hparams_to_vector(particle.position, bounds_info), bounds_info)
            res = train_and_eval_on_val(
                {**hparams_to_eval, "save_dir": "experiments/best_models/pso"}, 
                X_train, y_train, X_val, y_val, input_dim, seed=seed + gen * pop_size + i
            )
            particle.score = res["roc_auc"] if res["roc_auc"] is not None and not np.isnan(res["roc_auc"]) else -np.inf
            
            history.append({"gen": gen, "particle": i, "roc_auc": particle.score, "hparams": particle.position})

            # Update pbest
            if particle.score > particle.pbest_score:
                particle.pbest_score = particle.score
                particle.pbest_position = particle.position

            # Update gbest
            if particle.score > gbest_score:
                gbest_score = particle.score
                gbest_position_vec = hparams_to_vector(particle.position, bounds_info)

        # Update velocities and positions
        for particle in swarm:
            pos_vec = hparams_to_vector(particle.position, bounds_info)
            pbest_pos_vec = hparams_to_vector(particle.pbest_position, bounds_info)
            
            for k in bounds_info.keys():
                r1, r2 = random.random(), random.random()
                
                # Update velocity
                cognitive_component = c1 * r1 * (pbest_pos_vec[k] - pos_vec[k])
                social_component = c2 * r2 * (gbest_position_vec[k] - pos_vec[k])
                particle.velocity[k] = w * particle.velocity[k] + cognitive_component + social_component
                
                # Update position
                pos_vec[k] += particle.velocity[k]
            
            particle.position = vector_to_hparams(pos_vec, bounds_info)
            
    elapsed = time.time() - start_time
    
    # Final evaluation of gbest
    final_gbest_hparams = vector_to_hparams(gbest_position_vec, bounds_info)
    best_summary = {"roc_auc": gbest_score, "hparams": final_gbest_hparams}

    summary = {
        "method": "pso",
        "n_evals": len(history),
        "time_s": elapsed,
        "best": best_summary,
        "all_results": history
    }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    
    return summary
