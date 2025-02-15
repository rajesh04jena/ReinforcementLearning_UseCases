# Deep Q-Learning Based Retail Pricing Optimization

## Overview
This repository contains a Deep Q-Learning (DQN) based reinforcement learning model to optimize pricing decisions in a retail environment. The model aims to maximize profit, revenue, and inventory turnover through a multi-objective optimization framework.

## Features
### Multi-Objective Optimization
- Weighted reward function combining **profit**, **revenue**, and **inventory turnover**.
- Configurable weights to prioritize business objectives.

### Research-Grade Implementation
- **Dual neural networks** (Q-network and Target network) with periodic synchronization.
- **Experience replay** with prioritized sampling.
- **Epsilon decay** for an effective exploration-exploitation balance.
- **SHAP-based explainability** for model decision transparency.

### Economic Realism
- **Log-linear demand model** ensuring realistic price elasticity behavior.
- **Inventory constraints** incorporated into the reward function.
- **Lifecycle-aware pricing** (New Product Introduction (NPI), Mature, End-of-Life (EOL)).
- **Competitor response modeling** to account for market dynamics.
- **Marketing spend lag effects** included in demand modeling.

## Components
### 1. Environment Simulator (`environment.py`)
The environment models the retail business scenario, including:
- **Demand estimation using a linear regression model** based on log-transformed price and competitor pricing.
- **Lifecycle stages impact on price elasticity** (NPI, Mature, EOL).
- **Marketing lag effects** to model delayed impact of advertising.
- **Multi-objective reward function** combining profit, revenue, and inventory turnover.
- **Inventory constraints and stock depletion effects**.

### 2. DQN Agent (`agent.py`)
The Deep Q-Learning agent is responsible for learning an optimal pricing policy.
- **Dual neural networks** for stable learning.
- **Experience replay** to improve sample efficiency.
- **Epsilon-greedy action selection** with decay.
- **Target network synchronization** for training stability.
- **Dynamic adjustment of pricing strategies** based on inventory, lifecycle, and competition.

### 3. Training Pipeline (`trainer.py`)
Handles the end-to-end training process.
- **Episodes-based reinforcement learning** with an adjustable number of iterations.
- **Interaction between the agent and environment** through action selection and reward feedback.
- **Periodic target network updates** to stabilize training.
- **Early stopping** based on reward improvements.

### 4. Model Explainability (`explainer.py`)
Provides interpretability of pricing recommendations.
- **SHAP (SHapley Additive exPlanations) values** to explain how features impact pricing decisions.
- **Natural language explanations** for business understanding.
- **Competitive price comparisons** to provide market insights.
- **Inventory-driven price recommendations**.

```sh
The recommended price is **$125.50**. Here's why:

### Key Factors Influencing the Decision:
- **log_comp_price**: Contributed 0.25 to the decision.
- **inventory_level**: Contributed -0.15 to the decision.

### Business Context:
The price is set at a **balanced level** to optimize both **profit** and **demand**, suitable for products in the **Mature** stage.

### Competitive Context:
The competitor's price is **higher than usual**, allowing us to **increase our price** without losing market share.

### Inventory Context:
Our **inventory levels are high**, so the model recommends a **lower price** to accelerate sales.
```

## Usage Example

To be launched soon