################################################################################
# Name: trainer.py
# Purpose: Training Pipeline
# Date                          Version                Created By
# 24-Apr-2023                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import numpy as np
import logging
from tqdm import tqdm
from typing import List
from .environment import RetailPricingEnvironment  # Import the environment class
from .agent import DQNAgent  # Import the DQNAgent class

logger = logging.getLogger(__name__)

class RetailPricingTrainer:
    """
    End-to-end training pipeline for the retail pricing optimization model.
    Handles the training loop, interaction between the agent and environment,
    and logging of training progress.
    """

    def __init__(self, env: RetailPricingEnvironment, agent: DQNAgent, config: dict):
        """
        Initialize the trainer.

        Args:
            env (RetailPricingEnvironment): The pricing environment simulator.
            agent (DQNAgent): The Deep Q-Learning agent.
            config (dict): Configuration dictionary containing hyperparameters.
        """
        self.env = env
        self.agent = agent
        self.config = config
        logger.info("RetailPricingTrainer initialized")

    def train(self, episodes: int = 1000) -> List[float]:
        """
        Main training loop for the DQN agent.

        Args:
            episodes (int): Number of training episodes. Defaults to 1000.

        Returns:
            List[float]: History of rewards for each episode.
        """
        rewards_history = []  # Store rewards for each episode
        best_avg_reward = -np.inf  # Track the best average reward for early stopping

        logger.info(f"Starting training for {episodes} episodes...")

        for episode in tqdm(range(episodes)):
            # Reset the environment for a new episode
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                # Agent selects an action (price) based on the current state
                action = self.agent.act(state)
                price = self._action_to_price(action)

                # Execute the action in the environment
                next_state, reward, done, _ = self.env.step(price)

                # Store the experience in the replay buffer
                self.agent.remember(state, action, reward, next_state, done)

                # Train the agent using experience replay
                self.agent.replay()

                # Accumulate the total reward for the episode
                total_reward += reward
                state = next_state

            # Log the episode reward
            rewards_history.append(total_reward)
            logger.debug(f"Episode {episode + 1}/{episodes}, Total Reward: {total_reward}")

            # Update the target network periodically
            if episode % self.config['target_update_freq'] == 0:
                self.agent.update_target_network()
                logger.debug("Target network updated")

            # Early stopping based on average reward
            avg_reward = np.mean(rewards_history[-100:])  # Rolling average of last 100 episodes
            if avg_reward > best_avg_reward:
                best_avg_reward = avg_reward
            elif episode > 100:  # Allow some warm-up episodes
                logger.info("Early stopping triggered: No improvement in average reward")
                break

        logger.info("Training completed")
        return rewards_history

    def _action_to_price(self, action: int) -> float:
        """
        Convert a discrete action to a continuous price within the configured bounds.

        Args:
            action (int): Discrete action selected by the agent.

        Returns:
            float: Continuous price corresponding to the action.
        """
        min_price = self.config['price_bounds'][0]
        max_price = self.config['price_bounds'][1]
        return min_price + (max_price - min_price) * (action / (self.agent.action_size - 1))


