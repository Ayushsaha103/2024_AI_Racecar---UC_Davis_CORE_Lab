import os
from math import sin, cos, pi, sqrt
from random import randrange

# import pygame
# from pygame.locals import *

import numpy as np
import gym
from gym import spaces

# import Constants
from Constants import *
from math_helpers import *
from Agent import Drone
import time

from Simulator import Simulator
from Track import Map, wrap
import math
from plot import plotTrajectory, saveTrajectory, SaveGameTrajectory, transform_point_coor
from visualizer import Data_Visualizer

os.system('cls' if os.name == 'nt' else 'clear') # Cleaning library loading information texts
print("Fetching Libraries.. Please Wait..")

# WIDTH, HEIGHT = Constants.WIDTH, Constants.HEIGHT
# TIME_LIMIT = Constants.TIME_LIMIT
# BACKGROUND = Constants.BACKGROUND
# spriter = Constants.spriter #Image displayer
    
class droneEnv(gym.Env):
    def __init__(self):
        super(droneEnv, self).__init__()

        # GAME CONFIGURE
        self.startTime = time.time()
        self.reward = 0
        self.trainstep = 0
        self.uPred = np.zeros([1,2])

        self.numLaps = 3
        self.map = Map(0.5)
        self.sim = Simulator(self.map)
        self.visualizer = Data_Visualizer()

        # GYM CONFIGURE
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(9,), dtype=np.float16)
        
        
        self.info = {}
        self.game_cnt = -1
        report(self)

    def reset(self):

        # PER-GAME DATA VISUALIZATION
        if self.game_cnt > 0 and self.trainstep > 0:
            self.visualizer.visualize(self.s_total, self.v_avg / self.trainstep, (time.time() - self.game_start_time) )
        
        # SAVE GAME TRAJECTORY
        if self.game_cnt > 0 and self.save_game_trajectory:
            SaveGameTrajectory(self.map, self.xcl, self.xcl_glob, self.u_cl)
        
        # DECIDE WHEN TO SAVE GAME TRAJECTORY
        skip = 10
        self.save_game_trajectory = False
        if self.game_cnt > 0 and self.game_cnt % skip == 0:
            self.save_game_trajectory = True

        # always reset the position data vars
        self.xcl = []
        self.xcl_glob = []
        self.u_cl = []

        self.reward = 0
        self.prev_s = -100

        self.x0 = np.array([0.5, 0, 0, 0, 0, 0])
        self.xS = [self.x0.copy(), self.x0.copy()]
        self.prev_XYpsi = [0,0,0]
        
        # self.reward_total = 0
        self.game_start_time = time.time()
        self.s_total = 0
        self.v_avg = 0

        # increment game_cnt, train_step (within game)
        self.game_cnt += 1
        self.trainstep = 0

        return self.get_obs()
    

            
    def get_obs(self) -> np.ndarray:
        vx, vy = self.xS[0][0], self.xS[0][1]
        v, phi_v = vector_magnitude_and_angle(vx, vy)
        ephi, phi = self.xS[0][3], self.xS[1][3]
        phi_rd = phi - ephi
        ephi_v = phi_v - phi_rd
        s = self.xS[0][4]
        curvature = self.map.curvature(s)
        # import pdb; pdb.set_trace()
        return np.array(
            self.uPred[0].tolist() + [v, ephi_v, curvature] + self.xS[0][2:].tolist()
            # [curvature] + self.xS[0][:].tolist()
            # self.xS[0][:].tolist()
        ).astype(np.float16)

    def step(self, actions):
        
        self.render()
        self.reward = 0.0

        vt = 0.9        # self.vt
        x0 = self.xS[0]
        # self.uPred[0, 0] = - 0.6 * x0[5] - 0.9 * x0[3] + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        self.uPred[0, 1] = 1.5 * (vt - x0[0]) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])
        # print(self.uPred)
        #~~~~~~~~~~~~~~~~~~~~
        self.uPred[0,0] = delta = (actions[1] * np.radians(45)) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])
        # self.uPred[0,1] = throttle = 0.7+(actions[0])# + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        # self.uPred[0,1] = max(self.uPred[0,1], 0.4)
        # # print(actions)
        # # print(self.uPred)

        for _ in range(1):
            try:
                # update position
                self.xS[0], self.xS[1] = self.sim.dynModel(self.xS[0], self.xS[1], self.uPred[0,:].copy())

                # game termination
                ey = self.xS[0][5]
                s = self.xS[0][4]
                epsi = self.xS[0][3]
                v = math.sqrt(self.xS[0][0]**2 + self.xS[0][1]**2)
                # ephi = self.xS[0][3]
                
                # TERMINATION CONDITIONS
                if abs(ey) >= self.map.halfWidth: # or abs(ephi) >= np.radians(45):
                    print("COLLISION " + str(self.game_cnt))
                    done = True
                    return self.get_obs(), -15, done, self.info
                    # action = int(action)
                elif s < self.prev_s:
                    print("WRONG BEHAVIOR " + str(self.game_cnt))
                    done = True
                    return self.get_obs(), -15, done, self.info
                    # action = int(action)
                
                # AUTO-RESET after successful 2-lap completion
                if s > 20*self.numLaps:
                    print("SUCCESSFUL FINISH ON GAME " + str(self.game_cnt) + ", AV. LAPTIME: " + str(round((time.time() - self.game_start_time) / self.numLaps, 2)))
                    done = True
                    return self.get_obs(), 0, done, self.info

                # ADD DATA TO xcl, xcl_glob, ucl (conditionally)
                if self.save_game_trajectory:
                    self.xcl.append(self.xS[0])
                    self.xcl_glob.append(self.xS[1])
                    self.u_cl.append(self.uPred[0,:].copy())
            except:
                # import pdb; pdb.set_trace()
                print("BEHAVIOR ISSUE " + str(self.game_cnt))
                done = True
                return self.get_obs(), -15, done, self.info
            # print([self.trainstep, self.uPred])
            self.trainstep += 1

            # rewards = [0.01, throttle*0.3, s*0.01, vel*0.05]
            # rewards = [0.01, s*0.04, -abs(epsi)]
            # rewards = [0.01, s*5, vel*0.3]
            # rewards = [0.01, -abs(epsi)]
            rewards = [0.01, -abs(epsi), -abs(ey)]

            # vt = 0.8        # self.vt
            # x0 = self.xS[0]
            # uPred0_pid = -0.6 * x0[5] - 0.9 * x0[3]     # + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
            # uPred1_pid = 1.5 * (vt - x0[0])     #  + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])
            # rewards = [-abs(self.uPred[0][0] - uPred0_pid),
            #            -abs(self.uPred[0][1] - uPred1_pid)]
            
            self.reward += np.sum(rewards)
            self.prev_s = s

            self.s_total = max(self.s_total, s)
            # self.reward_total += self.reward
            self.v_avg += v
        

        return self.get_obs(), self.reward, False, self.info

    def render(self):
        pass

    def close(self):
        pass
