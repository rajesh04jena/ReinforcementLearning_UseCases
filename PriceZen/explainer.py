################################################################################
# Name: explainer.py
# Purpose: Model Explainability
# Date                          Version                Created By
# 24-Apr-2023                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import shap
import numpy as np
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class PricingExplainer:
    """
    Provides natural language explanations for the Deep Q-Learning optimizer's price recommendations.
    Uses SHAP values to explain the model's decision-making process.
    """

    def __init__(self, agent, feature_names: List[str], config: Dict[str, Any]):
        """
        Initialize the explainer.

        Args:
            agent: The trained DQN agent.
            feature_names (List[str]): Names of the features used by the model.
            config (Dict[str, Any]): Configuration dictionary containing business rules and thresholds.
        """
        self.agent = agent
        self.feature_names = feature_names
        self.config = config
        self.explainer = shap.DeepExplainer(agent.q_network, np.zeros((1, len(feature_names))))
        logger.info("PricingExplainer initialized")

    def explain_decision(self, state: np.ndarray, action: int) -> Dict[str, Any]:
        """
        Generate a natural language explanation for the recommended price.

        Args:
            state (np.ndarray): Current state of the environment.
            action (int): Recommended action (price) from the agent.

        Returns:
            Dict[str, Any]: Explanation containing SHAP values, feature importance, and natural language reasoning.
        """
        # Convert state to a 2D array for SHAP explanation
        state_2d = state[np.newaxis]

        # Calculate SHAP values
        shap_values = self.explainer.shap_values(state_2d)[0]
        shap_summary = dict(zip(self.feature_names, shap_values[0]))

        # Generate natural language explanation
        explanation = self._generate_natural_language_explanation(shap_summary, action)

        return {
            "recommended_price": self._action_to_price(action),
            "shap_values": shap_summary,
            "explanation": explanation
        }

    def _generate_natural_language_explanation(self, shap_summary: Dict[str, float], action: int) -> str:
        """
        Generate a natural language explanation for the recommended price.

        Args:
            shap_summary (Dict[str, float]): SHAP values for each feature.
            action (int): Recommended action (price).

        Returns:
            str: Natural language explanation.
        """
        recommended_price = self._action_to_price(action)
        explanation = f"The recommended price is **${recommended_price:.2f}**. Here's why:\n\n"

        # Explain key factors influencing the decision
        explanation += "### Key Factors Influencing the Decision:\n"
        for feature, value in shap_summary.items():
            if abs(value) > 0.1:  # Only highlight significant features
                explanation += f"- **{feature}**: Contributed {value:.2f} to the decision.\n"

        # Add business context
        explanation += "\n### Business Context:\n"
        if recommended_price < self.config['price_bounds'][0] * 1.1:
            explanation += "The price is set lower to **boost demand** and **clear inventory**, especially for products in the **End-of-Life (EOL)** stage.\n"
        elif recommended_price > self.config['price_bounds'][1] * 0.9:
            explanation += "The price is set higher to **maximize profit margins**, typically for products in the **New Product Introduction (NPI)** stage.\n"
        else:
            explanation += "The price is set at a **balanced level** to optimize both **profit** and **demand**, suitable for products in the **Mature** stage.\n"

        # Add competitive context
        if "log_comp_price" in shap_summary:
            comp_price_effect = shap_summary["log_comp_price"]
            if comp_price_effect > 0:
                explanation += "The competitor's price is **higher than usual**, allowing us to **increase our price** without losing market share.\n"
            else:
                explanation += "The competitor's price is **lower than usual**, prompting us to **lower our price** to remain competitive.\n"

        # Add inventory context
        if "inventory_level" in shap_summary:
            inventory_effect = shap_summary["inventory_level"]
            if inventory_effect > 0:
                explanation += "Our **inventory levels are high**, so the model recommends a **lower price** to accelerate sales.\n"
            else:
                explanation += "Our **inventory levels are low**, so the model recommends a **higher price** to maximize revenue.\n"

        return explanation

    def _action_to_price(self, action: int) -> float:
        """
        Convert a discrete action to a continuous price.

        Args:
            action (int): Discrete action selected by the agent.

        Returns:
            float: Continuous price corresponding to the action.
        """
        min_price = self.config['price_bounds'][0]
        max_price = self.config['price_bounds'][1]
        return min_price + (max_price - min_price) * (action / (self.agent.action_size - 1))