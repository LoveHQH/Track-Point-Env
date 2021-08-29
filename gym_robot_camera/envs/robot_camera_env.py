import gym
import math 
import random
import pygame
import numpy as np
from gym import utils
from gym import error, spaces
from gym.utils import seeding
from scipy.spatial.distance import euclidean

class RobotCameraEnvV0(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.set_window_size([960,720])
        # self.set_link_properties([100,100])
        self.link = 100
        self.min_theta = math.radians(0)
        self.max_theta = math.radians(180)
        self.min_accelerate = -0.01
        self.max_accelerate =  0.01
        self.rate = 0.15
        self.min_theta_target = math.radians(0)
        self.max_theta_target = math.radians(180)
        #Time to rotete 1' = pi/180 
        # self.rotate_in_1s = math.radians(100)
        # 
        self.theta_target= self.generate_random_angle(self.min_theta_target, self.max_theta_target)
        self.target_pos = self.generate_target_pos(self.theta_target)
        self.theta = self.generate_random_angle(self.min_theta, self.max_theta)
        self.accelerate = self.generate_random_accelerate()
        self.timesteps = 0
        # self.action = {0: "HOLD",
        #                1: "INC_J",
        #                2: "DEC_J"}
        
        self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(3,), dtype=np.float32)
        # self.action_space = spaces.Discrete(len(self.action))
        self.action_space = spaces.Box(self.min_theta, self.max_theta, shape=(2,), dtype=np.float32)

        self.current_error = -math.inf
        self.seed()
        self.viewer = None

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]

    def rotate_z(self, theta):
        rz = np.array([[np.cos(theta), - np.sin(theta), 0, 0],
                       [np.sin(theta), np.cos(theta), 0, 0],
                       [0, 0, 1, 0],
                       [0, 0, 0, 1]])
        return rz

    def translate(self, dx, dy, dz):
        t = np.array([[1, 0, 0, dx],
                      [0, 1, 0, dy],
                      [0, 0, 1, dz],
                      [0, 0, 0, 1]])
        return t
    
    def generate_timesteps(self):
        return int(math.sqrt(abs(180 / self.accelerate) + 0.25) +  + 0.5)

    def forward_kinematics(self, theta):
        P = []
        P.append(np.eye(4))
        R = self.rotate_z(theta)
        T = self.translate(self.link, 0, 0)
        P.append(P[-1].dot(R).dot(T))
        return P

    def inverse_theta(self, theta):
        new_theta = -1*theta
        return new_theta

    def draw(self, theta):
        LINK_COLOR = (255, 255, 255)
        JOINT_COLOR = (0, 0, 0)
        TIP_COLOR = (0, 0, 255)
        theta = self.inverse_theta(theta)
        P = self.forward_kinematics(theta)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0],self.centre_window[1],0)
        base = origin.dot(origin_to_base)
        F_prev = base.copy()
        for i in range(1, len(P)):
            F_next = base.dot(P[i])
            pygame.draw.line(self.screen, LINK_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), (int(F_next[0,3]), int(F_next[1,3])), 5)
            pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)
            F_prev = F_next.copy()
        pygame.draw.circle(self.screen, TIP_COLOR, (int(F_next[0,3]), int(F_next[1,3])), 8)
        
        # draw history
        # for i in range(50):
        #     pygame.draw.circle(self.screen, JOINT_COLOR, (int(F_prev[0,3]), int(F_prev[1,3])), 10)


    def update_pos_target(self):
        self.theta_target = self.theta_target + (20 - abs(self.timesteps)) * self.accelerate
        if self.timesteps == -20 or self.timesteps == 20: 
            self.accelerate = -self.accelerate
        if self.accelerate > 0:
            self.timesteps -= 1
        if self.accelerate < 0:
            self.timesteps += 1
        self.target_pos = self.generate_target_pos(self.theta_target)
        if len(self.history) > 50:
            self.history.pop(0)
        self.history.append(self.target_pos)

    def draw_target(self):
        TARGET_COLOR = (255,0,0)
        HISTORY_TARGET_COLOR = (94,194,191)
        origin = np.eye(4)
        origin_to_base = self.translate(self.centre_window[0], self.centre_window[1], 0)
        base = origin.dot(origin_to_base)
        base_to_target = self.translate(self.target_pos[0], -self.target_pos[1], 0)
        target = base.dot(base_to_target)
        pygame.draw.circle(self.screen, TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 12)
        history_copy = self.history.copy()
        # print(history_copy)
        for i in range(len(history_copy)):
            streak = history_copy.pop()
            base_to_target = self.translate(streak[0], -streak[1], 0)
            target = base.dot(base_to_target)
            pygame.draw.circle(self.screen, HISTORY_TARGET_COLOR, (int(target[0,3]),int(target[1,3])), 2)

    def generate_random_angle(self, min_theta, max_theta):
        theta = random.uniform(min_theta, max_theta)
        return theta

    def generate_random_accelerate(self):
        accelerate = random.uniform(self.min_accelerate, self.max_accelerate)
        return accelerate

    def generate_target_pos(self, theta):
        P = self.forward_kinematics(theta)
        pos = np.array([P[-1][0,3], P[-1][1,3]])
        return pos
    

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def normalize_angle(angle):
        return math.atan2(math.sin(angle), math.cos(angle))

    def step(self, action):
        self.update_pos_target()
        # if self.action[action] == "INC_J":
        #     self.theta += self.rate
        # elif self.action[action] == "DEC_J":
        #     self.theta -= self.rate
        self.theta = action[0]
        self.theta = np.clip(self.theta, self.min_theta, self.max_theta)
        self.theta = self.normalize_angle(self.theta)
        # Calc reward
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]
        distance_error = euclidean(self.target_pos, tip_pos)

        reward = 0
        if distance_error >= self.current_error:
            reward = -1
        epsilon = 10
        if (distance_error > -epsilon and distance_error < epsilon):
            reward = 1

        self.current_error = distance_error
        self.current_score += reward

        if self.current_score == -30 or self.current_score == 30:
            done = True
        else:
            done = False

        # if self.theta_target > self.max_theta or self.theta_target < self.min_theta:
        #     done = True
        observation = np.hstack((self.target_pos, self.theta))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos
        }
        return observation, reward, done, info
    
    def reset(self):
        self.timesteps = 0
        self.accelerate = self.generate_random_accelerate()
        self.theta_target= self.generate_random_angle(self.min_theta_target, self.max_theta_target)
        self.target_pos = self.generate_target_pos(self.theta_target)
        self.current_score = 0
        self.history = [self.target_pos]
        observation = np.hstack((self.target_pos, self.theta))
        return observation

    def render(self, mode='human'):
        SCREEN_COLOR = (50, 168, 52)
        if self.viewer == None:
            pygame.init()
            pygame.display.set_caption("RobotCamera-Env")
            self.screen = pygame.display.set_mode(self.window_size)
            self.clock = pygame.time.Clock()
        self.screen.fill(SCREEN_COLOR)
        self.draw_target()
        self.draw(self.theta)
        # print(self.theta)
        self.clock.tick(60)
        pygame.display.flip()

    def close(self):
        if self.viewer != None:
            pygame.quit()

class RobotCameraEnvV1(RobotCameraEnvV0):
    def __init__(self):
        super(RobotCameraEnvV1, self).__init__()

        self.min_action = 0
        self.max_action = 1
        self.min_theta = np.radians(30)
        self.max_theta = np.radians(150)
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=self.min_theta, high=self.max_theta, dtype=np.float32)
        # self.observation_space = spaces.Box(np.finfo(np.float32).min, np.finfo(np.float32).max, shape=(2,), dtype=np.float32)

    def step(self, action):
        self.update_pos_target()
        theta = np.interp(action, (self.min_action, self.max_action), (self.min_theta, self.max_theta))
        self.theta = theta
        self.theta = self.normalize_angle(self.theta)
        # Calc reward
        P = self.forward_kinematics(self.theta)
        tip_pos = [P[-1][0,3], P[-1][1,3]]
        distance_error = euclidean(self.target_pos, tip_pos)

        # Sharp reward
        reward = 0
        done = False
        epsilon = 5
        if (distance_error > -epsilon and distance_error < epsilon):
            reward += 1

        # if distance_error >= self.current_error:
        #     reward = -1

        self.current_error = distance_error
        self.current_score += reward

        if self.current_score == 30 :
            done = True

        observation = np.hstack((self.target_pos, self.theta))
        info = {
            'distance_error': distance_error,
            'target_position': self.target_pos,
            'current_position': tip_pos
        }
        return observation, reward, done, info

