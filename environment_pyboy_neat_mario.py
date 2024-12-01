from mss import mss
import cv2
import numpy as np
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
# Import the sb3 monitor for logging
from stable_baselines3.common.monitor import Monitor
import os
from collections import deque
from pyboy import PyBoy
import random


class GbaGame(Env):
    def __init__(self):
        super().__init__()
        self.frame_stack = deque(maxlen=3)
        self.frame_skip = 1  # Number of frames to skip
        # Adjust observation space to 3D for CNN compatibility
        self.observation_space = Box(low=0, high=255, shape=(80, 40, 3), dtype=np.uint8)
        self.action_space = Discrete(4)
        self.cap = mss()
        self.pyboy = PyBoy('ROMs/Super_Mario_Bros_Delux.gbc', window='null')
        self.game_location = {'top': 53, 'left': 0, 'width': 318, 'height': 339}
        self.score_location = {'top': 53, 'left': 65, 'width': 70, 'height': 25}
        self.done_location = {'top': 28, 'left': 21, 'width': 100, 'height': 79}
        self.score_cap = False
        self.penalty_cap = False
        self.reset_game_state()
        self.total_reward = 0
        self.current_step = 0
        self.truncated = False
        self.episode_length = 0
        self.current_score = 0
        self.max_episodes = 100000
        self.wait_frames = 1
        self.initial_observation = True
        self.pyboy_counter = 0
        self.level_progress = 0
        self.level_progress_pct = 0.0
        self.mario_is_moving = 0
        self.mario_stuck_counter = 0
        self.best_progress = 0.0
        self.best_progress_counter = 0
        self.best_reward = 0.0
        self.flag_reached = False
        self.agent_id = random.randint(1, 1000000)
        print('STARTED AGENT: ', self.agent_id)

    def reset_game_state(self):
        self.total_reward = 0
        self.episode_length = 0
        self.current_score = 0
        self.level_progress = 0
        self.level_progress_pct = 0.0
        self.mario_is_moving = 0
        self.mario_stuck_counter = 0
        self.penalty_cap = False
        self.score_cap = False
        self.truncated = False
        self.flag_reached = False

    def step(self, action):
        total_reward = 0
        done = False
        info = {}
        for _ in range(self.frame_skip):
            if not done:
                self.execute_action(action)
                self.update_frame_stack()
                observation = self.get_stacked_observation()
                # self.render()
                self.current_step += 1
                self.episode_length += 1
                reward, done = self.calculate_reward_and_done(action)
                truncated = self.truncated
                if truncated:
                    if self.total_reward > self.best_reward:
                        self.best_reward = self.total_reward
                        print("Agent Number: ", self.agent_id)
                        print("Best reward is now: ", self.best_reward)
                    if self.level_progress_pct > self.best_progress:
                        self.best_progress = self.level_progress_pct
                        self.best_progress_counter = 0
                        print("Agent Number: ", self.agent_id)
                        print("Best progress is now: %", round(self.best_progress))
                    if self.level_progress_pct == self.best_progress:
                        self.best_progress_counter += 1
                        print("Agent Number: ", self.agent_id)
                        print("Best progress is now: %", round(self.best_progress))
                        print("Best progress count is now: ", self.best_progress_counter)
                    if self.flag_reached:
                        done = True
                # You can aggregate or choose the last 'info' as needed
                info = {}
        # Line below for training
        return observation, reward, done, truncated, info
        # Line below for running
        # return observation, total_reward, done, info

    def reset(self, seed=None, options=None):
        if self.pyboy_counter == 10000:
            self.pyboy.stop()
            del self.pyboy
            self.pyboy_counter = 0
            self.pyboy = PyBoy('ROMs/Super_Mario_Bros_Delux.gbc', window='null')
        if seed:
            np.random.seed(seed)
        self.reset_game_state()
        self.reset_game_in_gui()
        if self.initial_observation:
            l_obs = self.get_observation()
            self.frame_stack.extend([l_obs] * self.frame_stack.maxlen)
        self.pyboy_counter += 1
        return self.get_stacked_observation(), {}

    def render(self):
        raw = np.array(self.pyboy.screen.image)[:, :, :3].astype(np.uint8)
        rgb = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        cv2.imshow('Game', rgb)  # Display the cropped image
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.close()

    def close(self):
        cv2.destroyAllWindows()
        self.pyboy.stop()

    def get_observation(self):
        raw = np.array(self.pyboy.screen.image)[:, :, :3].astype(np.uint8)
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        # Define the top-left corner and the size of the crop area
        x_start = 25  # Starting x-coordinate of the crop area
        y_start = 50  # Starting y-coordinate of the crop area
        width = 60  # Width of the crop area
        height = 120  # Height of the crop area
        # Crop the image
        cropped_image = gray[y_start:y_start + height, x_start:x_start + width]
        edges = cv2.Canny(cropped_image, threshold1=100, threshold2=200)
        resized = cv2.resize(edges, (40, 80))
        # resized = cv2.resize(gray, (40, 36))
        return resized[:, :, np.newaxis]

    def get_stacked_observation(self):
        # The shape of each frame should be (80, 72, 1)
        # After stacking along the last axis, the shape should be (80, 72, 4)
        return np.concatenate(self.frame_stack, axis=-1)

    def execute_action(self, action):
        action_map = {'0': 'a,right', '1': 'right', '2': 'a', '3': 'no_op'}
        if action == 3:
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 1:
            self.release_all_keys()
            self.pyboy.button_press('right')
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 2:
            self.release_all_keys()
            self.pyboy.button_press('a')
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()
        if action == 0:
            self.release_all_keys()
            self.pyboy.button_press('a')
            self.pyboy.button_press('right')
            for _ in range(self.wait_frames):  # tick for wait frames
                self.pyboy.tick()

    def update_frame_stack(self):
        observation = self.get_observation()
        self.frame_stack.append(observation)  # Each frame should have shape (80, 72, 1)

    def calculate_reward_and_done(self, action):
        reward = self.calculate_reward(action)
        self.total_reward += reward
        if self.truncated:
            done = True
        else:
            done = False
        if done:
            pass
            # print('episode_length = ', self.episode_length)
            # print('current_step = ', self.current_step)
            # print('total_reward = ', self.total_reward)
            # print('Score recorded = ', self.current_score)
        return reward, done

    def calculate_reward(self, action):
        # Set reward variable
        reward = 0
        # Call get_score to see if a points sprite is detected in the obs
        current_score = self.get_score()
        # current_score = 0
        # If there is no score detected then it will return 0
        if current_score > 0:
            # If there is a score detected then it will return the reward that matches
            # reward += current_score  # Setting reward equal to the score difference
            reward += 0
            # print("Reward is now =", reward)
        else:
            pass
            # print("No new points scored, reward remains 0")
        was_flag_reached = self.found_flag()
        if was_flag_reached:
            reward += 2000
            print('flag reached')
        is_episode_finished = self.timed_out()
        if is_episode_finished:
            # print('Timed Out at episode', self.episode_length)
            reward += -100
            self.truncated = True
        # Call the function to detect if Mario is alive
        did_mario_fall = self.detect_mario_fall()
        # print("did_mario_fall = ",did_mario_fall)
        if did_mario_fall:
            # Apply the penalty for falling and set penalty_cap to True to avoid repeated penalties
            reward += -100  # Assigning a negative reward
            self.truncated = True
            # print("Mario is dead")
        did_mario_get_stuck = self.is_mario_stuck()
        if did_mario_get_stuck:
            reward += -0.01
            if self.mario_stuck_counter >= 1000:
                self.truncated = True
                reward += -100
            # print("Mario is stuck")
        did_level_progress = self.level_did_progress()

        if did_level_progress > 0:
            mu = 0.80  # Shifts the peak towards the end
            sigma = 0.2  # Adjusts how quickly the rewards increase/decrease
            gaussian_reward = np.exp(-np.power(did_level_progress - mu, 2.) / (2 * np.power(sigma, 2.)))
            reward += gaussian_reward * 1000  # Scale the reward
            # reward += 1000

        return reward

    def detect_mario_fall(self):
        value = self.pyboy.memory[0xc17f]
        if value == 4:
            # print('life lost')
            return True
        else:
            return False

    def reset_game_in_gui(self):
        for _ in range(self.wait_frames):  # tick for wait frames
            self.pyboy.tick()
        # List of specific filenames
        gamestate_filenames = [
            "ROMs/Super_Mario_Bros_Delux_start_2_1.gbc.state"
        ]
        # Select a random filename from the list
        selected_filename = random.choice(gamestate_filenames)
        file_like_object = open(selected_filename, "rb")
        self.pyboy.load_state(file_like_object)

    def get_score(self):
        d_1 = self.value_map(self.pyboy.memory[0xc103])
        d_2 = self.value_map(self.pyboy.memory[0xc104])
        d_3 = self.value_map(self.pyboy.memory[0xc105])
        d_4 = self.value_map(self.pyboy.memory[0xc106])
        d_5 = self.value_map(self.pyboy.memory[0xc107])
        values = [d_1, d_2, d_3, d_4, d_5]
        score = 0

        # Find the first non-zero digit and concatenate the rest
        for i, val in enumerate(values):
            if val > 0:
                score = int(''.join(map(str, values[i:])))
                break
        if self.current_score == 0:
            self.current_score += score
        else:
            score = score - self.current_score
            self.current_score += score
        # print("score based reward =", score)
        return score

    def found_flag(self):
        found = False
        value = self.pyboy.memory[0xc02e]
        if value == 80:
            found = True
            self.flag_reached = True
        return found

    def timed_out(self):
        finish = False
        v_1 = self.pyboy.memory[0x9811]
        v_2 = self.pyboy.memory[0x9812]
        v_3 = self.pyboy.memory[0x9813]
        if v_1 == 208:
            if v_2 == 208:
                if v_3 == 209:
                    finish = True
        return finish

    def is_mario_stuck(self):
        stuck = False
        v_1 = self.pyboy.memory[0xc175]
        if self.mario_is_moving == 0:
            self.mario_is_moving = v_1
        if self.mario_is_moving == v_1:
            stuck = True
            self.mario_stuck_counter += 1
            return stuck
        self.mario_stuck_counter = 0
        self.mario_is_moving = v_1
        return stuck

    def level_did_progress(self):
        progress = True
        v_1 = self.pyboy.memory[0xc24b]
        if self.level_progress == 0:
            self.level_progress = v_1
            return 0
        elif v_1 <= self.level_progress:
            progress = False
            return 0
        elif v_1 > self.level_progress:
            progress = True
            v_2 = self.value_map(v_1)
            v_3 = v_2 / 12
            self.level_progress = v_1
            self.level_progress_pct = (v_3 * 100)
            # print("progress %", (v_3 * 100))
            return v_3
        else:
            return 0

    def value_map(self, digit):
        if digit == 208:
            return 0
        elif digit == 209:
            return 1
        elif digit == 210:
            return 2
        elif digit == 211:
            return 3
        elif digit == 212:
            return 4
        elif digit == 213:
            return 5
        elif digit == 214:
            return 6
        elif digit == 215:
            return 7
        elif digit == 216:
            return 8
        elif digit == 217:
            return 9
        elif digit == 218:
            return 10
        elif digit == 219:
            return 11
        elif digit == 220:
            return 12
        else:
            return 0

    def release_all_keys(self):
        self.pyboy.button_release('left')
        self.pyboy.button_release('right')
        self.pyboy.button_release('a')
