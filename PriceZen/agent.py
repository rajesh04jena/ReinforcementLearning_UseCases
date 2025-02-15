################################################################################
# Name: agent.py
# Purpose: DQN Agent Implementation
# Date                          Version                Created By
# 24-Apr-2023                   1.0         Rajesh Kumar Jena(Initial Version)
################################################################################

import tensorflow as tf
import numpy as np
from collections import deque
import random
import logging

logger = logging.getLogger(__name__)

class DQNAgent:
    """Deep Q-Learning Agent with Experience Replay"""
    
    def __init__(self, state_size: int, action_size: int, config: dict):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=config['memory_capacity'])
        self.gamma = config['gamma']
        self.epsilon = config['epsilon_max']
        self.epsilon_min = config['epsilon_min']
        self.epsilon_decay = config['epsilon_decay']
        self.batch_size = config['batch_size']
        self.update_freq = config['target_update_freq']
        
        self.q_network = self._build_network()
        self.target_network = self._build_network()
        self._update_target_network()
        
        logger.info("DQN Agent initialized")

    def _build_network(self) -> tf.keras.Model:
        """Build dual-stream Q-network architecture"""
        state_input = tf.keras.Input(shape=(self.state_size,))
        x = tf.keras.layers.Dense(64, activation='relu')(state_input)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        q_values = tf.keras.layers.Dense(self.action_size)(x)
        
        return tf.keras.Model(inputs=state_input, outputs=q_values)

    def _update_target_network(self):
        """Synchronize target network weights"""
        self.target_network.set_weights(self.q_network.get_weights())
        logger.debug("Target network updated")

    def remember(self, state: np.ndarray, action: int, reward: float, 
                next_state: np.ndarray, done: bool):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        q_values = self.q_network.predict(state[np.newaxis], verbose=0)
        return np.argmax(q_values[0])

    def replay(self):
        """Experience replay training"""
        if len(self.memory) < self.batch_size:
            return
            
        minibatch = random.sample(self.memory, self.batch_size)
        states, targets = [], []
        
        for state, action, reward, next_state, done in minibatch:
            target = self.q_network.predict(state[np.newaxis], verbose=0)[0]
            
            if done:
                target[action] = reward
            else:
                t = self.target_network.predict(next_state[np.newaxis], verbose=0)[0]
                target[action] = reward + self.gamma * np.amax(t)
            
            states.append(state)
            targets.append(target)
        
        # Train Q-network
        self.q_network.fit(np.array(states), np.array(targets), 
                          batch_size=self.batch_size, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            
        logger.debug(f"Experience replay completed. Epsilon: {self.epsilon:.3f}")

    def update_target_network(self):
        """Periodic target network update"""
        self._update_target_network()