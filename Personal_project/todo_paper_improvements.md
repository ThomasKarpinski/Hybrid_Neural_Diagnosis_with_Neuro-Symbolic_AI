# TODO: Paper Improvements for IJCNN Credibility

Based on the feedback in `paper/professor_tips.txt`, the following tasks must be completed:

- [x] **1. Concrete Auditing Table (Issue 5):**
    - Generate a table showing exact counts of TP, FP, TN, FN for the best MLP (PCA+Lion) vs. the Full Hybrid System.
    - Explicitly calculate and show the FN reduction rate.
- [x] **2. Formalize Evaluation Protocol (Issue 3):**
    - Update the Methodology section to explicitly state the use of a single Stratified 80/20 split.
    - Clarify that Wilcoxon tests are performed on individual patient error vectors ($|y - \hat{p}|$) to establish significance.
- [x] **3. Bayesian & Fusion Reproducibility (Issue 4):**
    - Add the formal fusion equation: $\hat{p}_{hybrid} = \frac{p_{nn} + p_{fuzzy} + p_{bayes}}{3}$.
    - Explicitly describe the Rule-based override logic ($p > 0.95$ for high risk, $p < 0.05$ for low risk).
- [x] **4. Reproducibility Appendix (Issue C):**
    - List the IF-THEN rules used.
    - Define fuzzy membership triangular parameters for Age and BMI.
    - State HPO budgets (Trials: 15 for Optuna/Random, Population: 8, Generations: 4 for GA/DE).
- [x] **5. Final Consistency Pass (Issue 2):**
    - Ensure all text strictly refers to the Hybrid system as a "Safety/Recall improvement" rather than a "Discrimination improvement."
