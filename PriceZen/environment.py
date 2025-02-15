################################################################################
# Name: environment.py
# Purpose: Core Environment Simulation
# Date                          Version                Created By
# 24-Apr-2023                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import logging
from typing import Tuple, Optional, List

logger = logging.getLogger(__name__)

class RetailPricingEnvironment:
    """Simulates retail pricing environment with economic constraints"""
    
    def __init__(self, sku_data: pd.DataFrame, config: dict):
        """
        Args:
            sku_data: DataFrame containing historical data
            config: Environment configuration
        """
        self.sku_data = sku_data
        self.config = config
        self.current_step = 0
        self.current_inventory = config['initial_inventory']
        self.lifecycle_stage = self._determine_lifecycle()
        self.elasticity_model = self._build_elasticity_model()
        logger.info("Environment initialized")

    def _determine_lifecycle(self) -> str:
        """Determine product lifecycle stage using sales trend analysis"""
        sales = self.sku_data['quantity'].values[-self.config['lookback_window']:]
        if len(sales) < 2:
            logger.warning("Insufficient data, defaulting to Mature")
            return "Mature"
            
        slope = linregress(np.arange(len(sales)), sales).slope
        if slope > self.config['npi_threshold']:
            return "NPI"
        elif slope < self.config['eol_threshold']:
            return "EOL"
        return "Mature"

    def _build_elasticity_model(self) -> LinearRegression:
        """Build price elasticity model with hierarchy borrowing"""
        features = self._prepare_features()
        target = np.log(self.sku_data['quantity'] + 1e-10)
        
        model = LinearRegression()
        model.fit(features, target)
        logger.info(f"Elasticity model R²: {model.score(features, target):.2f}")
        return model

    def _prepare_features(self) -> pd.DataFrame:
        """Prepare features with lagged marketing effects"""
        features = self.sku_data[[
            'log_price', 
            'log_comp_price',
            'product_attribute'
        ]].copy()
        
        # Add lagged marketing effects
        for lag in range(1, self.config['max_marketing_lag'] + 1):
            features[f'marketing_lag_{lag}'] = self.sku_data['marketing_spend'].shift(lag)
        
        # Add lifecycle encoding
        features = pd.get_dummies(features, columns=['lifecycle'], prefix='lifecycle')
        return features.dropna()

    def calculate_price_elasticity(self) -> dict:
        """Calculate stage-specific price elasticities"""
        elasticities = {}
        for stage in ["NPI", "Mature", "EOL"]:
            mask = self.sku_data['lifecycle'] == stage
            stage_data = self.sku_data[mask]
            X = np.log(stage_data[['price', 'comp_price']])
            y = np.log(stage_data['quantity'])
            
            model = LinearRegression().fit(X, y)
            elasticities[stage] = {
                'self_elasticity': model.coef_[0],
                'cross_elasticity': model.coef_[1]
            }
        return elasticities

    def step(self, price: float) -> Tuple[np.ndarray, float, bool, dict]:
        """Execute pricing action"""
        if price <= 0:
            logger.error("Invalid price")
            raise ValueError("Price must be positive")
            
        # Predict demand
        log_price = np.log(price)
        features = self._get_current_features(log_price)
        predicted_log_demand = self.elasticity_model.predict(features)
        demand = np.exp(predicted_log_demand)
        
        # Apply inventory constraints
        sold_units = min(demand, self.current_inventory)
        self.current_inventory -= sold_units
        
        # Calculate rewards
        reward = self._calculate_reward(price, sold_units)
        done = self.current_inventory <= 0
        
        # Update state
        self.current_step += 1
        next_state = self._get_state()
        
        logger.debug(f"Step {self.current_step}: Price {price}, Sold {sold_units}, Reward {reward}")
        return next_state, reward, done, {'demand': demand}

    def _calculate_reward(self, price: float, sold_units: float) -> float:
        """Multi-objective reward function"""
        profit = (price - self.config['unit_cost']) * sold_units
        revenue = price * sold_units
        inventory_turn = sold_units / self.config['initial_inventory']
        
        return (
            self.config['profit_weight'] * profit +
            self.config['revenue_weight'] * revenue +
            self.config['inventory_weight'] * inventory_turn
        )

    def reset(self) -> np.ndarray:
        """Reset environment"""
        self.current_step = 0
        self.current_inventory = self.config['initial_inventory']
        return self._get_state()






