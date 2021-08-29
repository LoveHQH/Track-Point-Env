import gym
import math 
import random
import pygame
import numpy as np
from gym import utils
from gym import error, spaces
from gym.utils import seeding
import matplotlib
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import time

class TrackPointEnvV0(gym.Env):
    metadata = {'render.modes': ['point']}

    def __init__(self):
        self.ctx = None
        self.cnvs = (960, 720)
        self.set_window_size(self.cnvs)
        # PID
        self.proportional_gain = 0.4 # KP
        self.integral_gain = 0.01   # KI
        self.derivative_gain = 0.0003  # KD
        self.current_time = time.time()
        self.last_time = self.current_time
        # org
        self.setpoint = None
        # Mouse
        self.mouse_position = (0., 0.)
        # Target 
        self.min_velocity = 2.
        self.max_velocity = 5.
        self.target_position = None
        self.target_velocity = None
        self.target_offset = (0., 0.)
        self.count = 1

        # Ball 
        self.ball_position = (480., 320.)
        self.ball_velocity = None 
        self.error = None
        self.last_error = None
        self.integral = None
        self.num_past_errors = None
        self.past_errors = None
        self.last_distance_error = None

        # for train
        self.low_state = np.array([0, 0])
        self.high_state = np.array([960,720])
        self.end = 50
        self.n_count = 100
        self.done = False
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state,
                                            dtype=np.float32)
        self.action_space = spaces.Box(-1, +1, (3,), dtype=np.float32)
        self.action_bound = (3)

        self.seed()
        self.again = False
        self.viewer = None

    def set_window_size(self, window_size):
        self.window_size = window_size
        self.centre_window = [window_size[0]//2, window_size[1]//2]

    #draw ball
    def draw(self):
        POINT_COLOR = (255, 0, 0)
        pygame.draw.circle(self.screen, POINT_COLOR, (self.ball_position[0], self.ball_position[1]), 10)
        
    
    def draw_target(self,target_position):
        TARGET_COLOR = (0,255,0)
        pygame.draw.circle(self.screen, TARGET_COLOR, (target_position[0], target_position[1]), 5)

    def random_target(self):
        # v
        self.target_velocity = random.randint(self.min_velocity, self.max_velocity)
        # pos
        x = float(random.randint(120, self.cnvs[0] - 120))
        y = float(random.randint(120, self.cnvs[1] - 120))
        self.target_position = np.array((x, y))
        # os
        offsetx = random.randint(-10, 10) 
        offsety = random.randint(-10, 10)
        self.target_offset = np.array((offsetx,offsety))

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def update_target_position(self):
        self.count += 1
        if self.count > 100: 
            self.count = 1
            self.target_offset = -self.target_offset
        if self.count%5 == 0: 
            self.target_position = self.target_position + self.target_offset * float(self.count / 100) * self.target_velocity

    # def get_dist(arr1, arr2):
    def check_pos(self, point = []):
        return 0<=point[0]<=self.cnvs[0] and 0<=point[1]<=self.cnvs[1]


    def step(self, action):
        self.done = False
        self.update_target_position()
        distance_error = self.control(action)
        reward = 0
        
        reward = 1 if abs(distance_error) <= 10 else -1

        if self.count >300 :
            # print('count')
            self.done = True

        if abs(distance_error) - abs(self.last_distance_error) > 20:
            reward = -5
            self.done = True
        
        if not self.check_pos(self.ball_position):
            self.ball_position[0]  = np.clip(self.ball_position[0],0,self.cnvs[0])
            self.ball_position[1]  = np.clip(self.ball_position[0],0,self.cnvs[1])
            self.done = True

        self.last_distance_error = distance_error
        return np.array([self.ball_position]), reward, self.done, self.last_error

    def new(self):
        self.random_target()
        self.setpoint = np.copy(self.target_position)
        return

    def reset(self):
        middle = np.divide(self.cnvs,2)
        # self.mouse_position = middle[0]+1, middle[1]
        self.ball_position = middle
        self.ball_velocity = -0.01, 0.01
        self.error = 0., 0.
        self.last_error = 0., 0.
        self.last_distance_error = 0.
        self.integral = 0., 0.
        self.count = 1.
        #
        self.target_position = np.copy(self.setpoint)
        self.ball_position = np.copy(self.setpoint)
        #
        if self.cnvs[0] > self.cnvs[1]: self.num_past_errors = self.cnvs[0];
        else : self.num_past_errors = self.cnvs[1];
        #
        self.past_errors = [];
        for i in range(self.num_past_errors): self.past_errors.append((0., 0.));
        return self.ball_position

    def control(self, action = [0, 0 ,0]):
        setpoint = None
        if self.check_pos(self.target_position):
            setpoint = self.target_position
        else: setpoint = np.divide(self.cnvs,2)

        Kp = action[0]
        Ki = action[1]
        Kd = action[2]

        self.error = setpoint - self.ball_position
        self.current_time = time.time()
        delta_time = self.current_time - self.last_time

        self.integral = self.integral + self.error * 15
        derivative = (self.error - self.last_error) / 15
			
        output = Kp * self.error
        # print("P: ",output)
        output += Ki * self.integral
        # print("I: ",output)
        output += Kd * derivative
        # print("D: ",output)
        distance_error = euclidean(self.error, self.last_error)
        self.last_error = np.copy(self.error)
        self.last_time = np.copy(self.current_time)
        self.ball_velocity = output
        self.past_errors.append(self.error)
        if len(self.past_errors) > self.num_past_errors: self.past_errors.pop(0);    
        self.ball_position += output    
        return distance_error

    def render(self):
        SCREEN_COLOR = (255, 255, 255)
        if self.viewer == None :
            pygame.init()
            pygame.display.set_caption("Track Point-Env")
            # for event in pygame.event.get():
            #     if event.type == pygame.MOUSEMOTION:
            #         self.mouse_position = pygame.mouse.get_pos()
            self.screen = pygame.display.set_mode(self.window_size)
            # self.clock = pygame.time.Clock()
        if self.again:
            pygame.display.update()
            self.again = False
        self.screen.fill(SCREEN_COLOR)
        # self.control()
        self.draw()
        self.draw_target(self.target_position)
        # self.plot(True)
        pygame.display.flip()
    

    def plot(self, plot_path = '', arr_v = [],done = False):
        # is_ipython = 'inline' in matplotlib.get_backend()
        # if is_ipython:
        #     from IPython import display
        pid = str(arr_v[0])
        plt.ion()

        plt.figure(2)
        plt.clf()
        # plt.plot(self.feedbacks)
        plt.plot(self.past_errors)
        plt.xlabel('time (s)')
        plt.ylabel('pid error (PV)')
        plt.title(f'PID Control {arr_v[0]}')
        if done:
            plt.savefig(plot_path + f"/figure_{arr_v[1]}_{arr_v[2]}.png")

        plt.ioff()


    def close(self):
        if self.viewer != None:
            pygame.quit()



