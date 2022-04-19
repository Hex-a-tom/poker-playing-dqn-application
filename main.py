#!/usr/bin/env python

"""A neural network for playing poker"""

__author__ = "Felix Sondhauss"
__credits__ = ["Felix Sondhauss"]
__version__ = "1.0.0"

import random
import os
from collections import deque
import time

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.callbacks import TensorBoard
from keras.optimizers import adam_v2
from tqdm import tqdm

from env import PokerEnv

# import customProfile as prof


epsilon = 1
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

DISCOUNT = 0.99

REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1000  # minimum number of steps in memory to start training

MODEL_NAME = "256x2"
MIN_REWARD = -200  # For model save

MINIBATCH_SIZE = 64  # How many steps to use for training

UPDATE_TARGET_EVERY = 5  # Termianl states (end of episodes)

EPISODES = 20_000
# EPISODES = 100

AGGREGATE_STATS_EVERY = 1000  # episodes

SHOW_PREVIEW = False

# profile = prof.Profiler()

# Own Tensorboard class
class ModifiedTensorBoard(TensorBoard):

    # Overriding init to set initial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = os.path.join(self.log_dir, MODEL_NAME)

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number
    # (otherwise every .fit() will start writing from 0th step)
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided
    # We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        with self.writer.as_default():
            for name, value in stats.items():
                tf.summary.scalar(name, value, self.step)
                self.writer.flush()


class DQNAgent:
    def __init__(self) -> None:

        # Main model (gets trained every step)
        self.model = self.create_model()

        # Target model (.predict is on this)
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f"logs/{MODEL_NAME}-{int(time.time())}",
        )

        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()
        model.add(Input(shape=env.OBSERVATION_SPACE_VALUES))

        model.add(Dense(128, activation="relu"))

        model.add(Dense(128, activation="relu"))

        model.add(Dense(64))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))
        model.compile(
            optimizer=adam_v2.Adam(learning_rate=0.001),
            loss="mse",
            metrics=["accuracy"],
        )
        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state):
        return self.model.predict(np.array([state]))[0]

    def train(self, terminal_state):
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        current_states = np.array([transition[0] for transition in minibatch])
        current_qs_list = self.model.predict(current_states)

        new_current_states = np.array([transition[3] for transition in minibatch])
        future_qs_list = self.target_model.predict(new_current_states)

        X = []
        Y = []

        for index, (
            current_state,
            action,
            reward,
            new_current_state,
            done,
        ) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)
            Y.append(current_qs)

        self.model.fit(
            np.array(X),
            np.array(Y),
            batch_size=MINIBATCH_SIZE,
            verbose=0,
            shuffle=False,
            callbacks=[self.tensorboard] if terminal_state else None,
        )

        # updateing to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0


tf.compat.v1.disable_eager_execution()


env = PokerEnv()

agent = DQNAgent()


# For stats
ep_rewards = [-200]

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

# Create models folder
if not os.path.isdir("models"):
    os.makedirs("models")


# Main training loop
for episode in tqdm(range(1, EPISODES), unit="episodes"):
    # Update tensorboard
    agent.tensorboard.step = episode

    # restart episode
    step = 1

    # env
    current_states, current_player = env.reset()
    new_states = [None] * env.PLAYER_N
    previous_player = (current_player - 1) % env.PLAYER_N

    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_states[current_player]))
        else:
            # Random action
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        previous_player = current_player

        (
            new_state,
            current_player,
            reward,
            done,
        ) = env.step(action, current_player)

        new_states[current_player] = new_state

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # Update memory and train network
        if new_states[previous_player] != None:
            agent.update_replay_memory(
                (
                    current_states[previous_player],
                    action,
                    reward,
                    new_states[previous_player],
                    done,
                )
            )
        agent.train(done)
        current_states[current_player] = new_state
        step += 1

    ep_rewards.append(max(env.getRewards()))
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:]) / len(
            ep_rewards[-AGGREGATE_STATS_EVERY:]
        )
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(
            reward_avg=average_reward,
            reward_min=min_reward,
            reward_max=max_reward,
            epsilon=epsilon,
        )

        if average_reward >= MIN_REWARD:
            agent.model.save(
                f"models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model"
            )

    # decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
