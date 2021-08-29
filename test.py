from os import close
import gym 
import numpy as np
import time
env = gym.make('gym_robot_camera:track-point-v0')
env.new()
env.reset()
reset = 50
while reset>0:
    env.reset()
    done = False
    rew = 0
    while not done:
        env.render()
        observation, reward, done, info = env.step([0.4, 0.2, 0.03 ])
        time.sleep(0.05)
        # print(done)
