import numpy as np

class FuzzyController:
    """
    A simple Fuzzy Logic Controller for dynamic hyperparameter adjustment.
    Inputs:
      - Current Loss (Low, Medium, High)
      - Loss Change (Negative/Improving, Zero/Stable, Positive/Worsening)
    Output:
      - Learning Rate Adjustment Factor (Decrease, Maintain, Increase)
    """
    def __init__(self):
        pass

    def _triangle(self, x, a, b, c):
        return max(0, min((x - a) / (b - a), (c - x) / (c - b))) if b != a and c != b else 0

    def _trapezoid(self, x, a, b, c, d):
        return max(0, min((x - a) / (b - a), 1, (d - x) / (d - c))) if b != a and d != c else 0

    def fuzzify_loss(self, loss):
        # Assuming loss is normalized or in a predictable range [0, 1+]
        # For BCE, it's usually 0 to ~0.7 (log(2)) or higher.
        # Let's define relative to 0.5 for binary classification
        return {
            "low": self._trapezoid(loss, -0.1, 0.0, 0.3, 0.4),
            "medium": self._triangle(loss, 0.3, 0.5, 0.7),
            "high": self._trapezoid(loss, 0.6, 0.8, 10.0, 10.1)
        }

    def fuzzify_delta(self, delta):
        # Delta = current_loss - prev_loss
        # Negative means improving, Positive means worsening
        return {
            "improving": self._trapezoid(delta, -1.0, -0.1, -0.001, 0.0),
            "stable": self._triangle(delta, -0.01, 0.0, 0.01),
            "worsening": self._trapezoid(delta, 0.0, 0.001, 0.1, 1.0)
        }

    def defuzzify(self, rules_output):
        # Center of Gravity method
        # Outputs: Decrease (0.5), Maintain (1.0), Increase (1.2)
        # We'll use singleton outputs for simplicity
        
        numerator = 0.0
        denominator = 0.0
        
        # Rule definitions and their consequent centers
        # Structure: {'output_name': aggregate_strength}
        
        centers = {
            "decrease_strong": 0.5,
            "decrease_slight": 0.8,
            "maintain": 1.0,
            "increase_slight": 1.1,
        }
        
        for name, strength in rules_output.items():
            if strength > 0:
                numerator += strength * centers.get(name, 1.0)
                denominator += strength
                
        if denominator == 0:
            return 1.0
            
        return numerator / denominator

    def compute_update(self, current_loss, prev_loss):
        if prev_loss is None:
            return 1.0
            
        delta = current_loss - prev_loss
        
        f_loss = self.fuzzify_loss(current_loss)
        f_delta = self.fuzzify_delta(delta)
        
        # Rules:
        # 1. If Loss is High AND Improving -> Maintain (let it converge)
        # 2. If Loss is High AND Stable -> Increase Slight (stuck in local minima?)
        # 3. If Loss is High AND Worsening -> Decrease Strong (diverging)
        # 4. If Loss is Low AND Improving -> Maintain
        # 5. If Loss is Low AND Stable -> Decrease Slight (fine tuning)
        # 6. If Loss is Low AND Worsening -> Decrease Slight
        
        rules = {}
        
        # Rule 1
        rules["maintain"] = min(f_loss["high"], f_delta["improving"])
        # Rule 2
        rules["increase_slight"] = min(f_loss["high"], f_delta["stable"])
        # Rule 3
        rules["decrease_strong"] = min(f_loss["high"], f_delta["worsening"])
        
        # Rule 4 (incorporating Medium loss here for simplicity)
        r4_a = min(f_loss["low"], f_delta["improving"])
        r4_b = min(f_loss["medium"], f_delta["improving"])
        current_maintain = rules.get("maintain", 0.0)
        rules["maintain"] = max(current_maintain, r4_a, r4_b)
        
        # Rule 5
        rules["decrease_slight"] = min(f_loss["low"], f_delta["stable"])
        
        # Rule 6
        r6 = min(f_loss["low"], f_delta["worsening"])
        current_dec_slight = rules.get("decrease_slight", 0.0)
        rules["decrease_slight"] = max(current_dec_slight, r6)

        # Catch-all for medium stable/worsening -> maintain or decrease slight
        r_med_stab = min(f_loss["medium"], f_delta["stable"])
        rules["maintain"] = max(rules["maintain"], r_med_stab)
        
        return self.defuzzify(rules)
