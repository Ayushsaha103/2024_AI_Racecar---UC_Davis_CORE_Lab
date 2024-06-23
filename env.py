import os
from math import sin, cos, pi, sqrt
from random import randrange

import pygame
from pygame.locals import *

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
from plot import plotTrajectory, transform_point_coor

os.system('cls' if os.name == 'nt' else 'clear') # Cleaning library loading information texts
print("Fetching Libraries.. Please Wait..")

# WIDTH, HEIGHT = Constants.WIDTH, Constants.HEIGHT
# TIME_LIMIT = Constants.TIME_LIMIT
# BACKGROUND = Constants.BACKGROUND
# spriter = Constants.spriter #Image displayer
    
class droneEnv(gym.Env):
    def __init__(self):
        super(droneEnv, self).__init__()
        
        # # VIDEO SETTINGS
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.FPS = FPS
        self.FramePerSec = pygame.time.Clock()

        # # Agent and Target SETTINGS
        self.Agent       = Drone()
        self.car_image = spriter("Car")

        # GAME CONFIGURE
        self.reward = 0
        self.trainstep = 0
        self.uPred = np.zeros([1,2])

        self.map = Map(0.7)
        self.sim = Simulator(self.map)
        
        # GYM CONFIGURE
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float16)
        
        
        self.info = {}
        self.game_cnt = -1
        report(self)

    def reset(self):
        
        # self.Agent.reset()
        self.reward = 0
        self.prev_s = -100

        self.x0 = np.array([0.5, 0, 0, 0, 0, 0])
        self.xS = [self.x0.copy(), self.x0.copy()]
        self.prev_XYpsi = [0,0,0]

        # PLOT TRAJECTORY
        if self.trainstep > 1:
            pass
            # # import pdb; pdb.set_trace()
            # xF = [np.array(self.xcl[-1]) - np.array([0, 0, 0, 0, self.map.TrackLength, 0]), np.array(self.xcl_glob[-1])]
            # self.xcl.pop()
            # self.xcl_glob.pop()

            # self.xcl, self.xcl_glob = np.array(self.xcl), np.array(self.xcl_glob)
            # self.u_cl = np.array(self.u_cl),

            # plotTrajectory(self.map, self.xcl, self.xcl_glob, xF, 'PID')
            # print("Plotted trajectory!")
            
        # always reset the position data vars
        self.xcl = []
        self.xcl_glob = []
        self.u_cl = []
        self.game_cnt += 1
        self.trainstep = 0

        # ADDED for debugging
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        # self.FPS = FPS

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
            self.uPred[0].tolist() + [v, ephi_v] + self.xS[0][2:].tolist()
        ).astype(np.float16)

    def step(self, actions):
        
        self.render()
        self.reward = 0.0

        vt = 0.8        # self.vt
        x0 = self.xS[0]
        # self.uPred[0, 0] = - 0.6 * x0[5] - 0.9 * x0[3] + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        # self.uPred[0, 1] = 1.5 * (vt - x0[0]) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])

        #~~~~~~~~~~~~~~~~~~~~
        # # print(actions)
        self.uPred[0,0] = delta = (actions[1] * np.radians(45)) + np.max([-0.2, np.min([np.random.randn() * 0.10, 0.2])])
        self.uPred[0,1] = throttle = 0.5+(actions[0])# + np.max([-0.9, np.min([np.random.randn() * 0.25, 0.9])])
        # self.uPred[0,1] = max(self.uPred[0,1], 0.4)
        # # print(self.uPred)

        for _ in range(1):
            try:
                # update position
                self.xS[0], self.xS[1] = self.sim.dynModel(self.xS[0], self.xS[1], self.uPred[0,:].copy())

                # game termination
                ey = self.xS[0][5]
                s = self.xS[0][4]
                epsi = self.xS[0][3]
                vel = math.sqrt(self.xS[0][0]**2 + self.xS[0][1]**2)
                # ephi = self.xS[0][3]
                
                if abs(ey) >= self.map.halfWidth or s < self.prev_s: # or abs(ephi) >= np.radians(45):
                    print("COLLISION " + str(self.game_cnt))
                    done = True
                    return self.get_obs(), -15, done, self.info
                    # action = int(action)
                
                # self.xcl.append(self.xS[0])
                # self.xcl_glob.append(self.xS[1])
                # self.u_cl.append(self.uPred[0,:].copy())
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
            rewards = [0.01, -abs(epsi)]
            self.reward += np.sum(rewards)
            self.prev_s = s
            # pass
            # rewards for: agent-to-target distance, surviving thus far
            # successful collision: regenerate target, reward agent
            # termination: time limit constraint, outa bounds
        

        return self.get_obs(), self.reward, False, self.info

    def render(self):
        self.screen.fill(BLACK)

        # transformed coordinates
        zoom = 67
        x_glob = self.xS[1]
        psi = x_glob[3]
        X = x_glob[4]
        Y = x_glob[5]
        if self.trainstep == 0:
            self.x_trans = WIDTH / 2 - X
            self.y_trans = HEIGHT / 2 - Y

        # draw road
        if self.trainstep == 0 and self.game_cnt == 0:
            Points = int(np.floor(10 * (self.map.PointAndTangent[-1, 3] + self.map.PointAndTangent[-1, 4])))
            Points1 = np.zeros((Points, 2))
            Points2 = np.zeros((Points, 2))
            Points0 = np.zeros((Points, 2))
            self.rd_left = []
            self.rd_right = []
            self.rd_mid = []
            skip = 7
            for i in range(0, int(Points), skip):
                a = Points1[i, :] = self.map.getGlobalPosition(i * 0.1, self.map.halfWidth)
                b = Points2[i, :] = self.map.getGlobalPosition(i * 0.1, -self.map.halfWidth)
                c = Points0[i, :] = self.map.getGlobalPosition(i * 0.1, 0)
                a = transform_point_coor(a, self.x_trans, self.y_trans, zoom)
                b = transform_point_coor(b, self.x_trans, self.y_trans, zoom)
                c = transform_point_coor(c, self.x_trans, self.y_trans, zoom)
                # pygame.draw.circle(self.screen, WHITE, transform_point_coor(a, self.x_trans, self.y_trans, zoom), 4)
                # pygame.draw.circle(self.screen, WHITE, transform_point_coor(b, self.x_trans, self.y_trans, zoom), 4)
                # pygame.draw.circle(self.screen, WHITE, transform_point_coor(c, self.x_trans, self.y_trans, zoom), 4)
                self.rd_left.append(b); self.rd_right.append(a); self.rd_mid.append(c)
        pygame.draw.lines(self.screen, WHITE, False, self.rd_left, 3)
        pygame.draw.lines(self.screen, WHITE, False, self.rd_right, 3)
        pygame.draw.lines(self.screen, YELLOW, False, self.rd_mid, 3)

        # # erase old car position
        # X_, Y_, psi_ = self.prev_XYpsi
        # player_sprite = self.car_image[1]
        # player_copy = pygame.transform.rotate(
        #     player_sprite, -math.degrees(psi_ + (np.pi / 2))
        # )
        # self.screen.blit(
        #     player_copy,
        #     transform_point_coor((X_,Y_), self.x_trans, self.y_trans, zoom)
        # )

        # draw car
        player_sprite = self.car_image[0]
        player_copy = pygame.transform.rotate(
            player_sprite, -math.degrees(psi + (np.pi / 2))
        )
        self.screen.blit(
            player_copy,
            transform_point_coor((X,Y), self.x_trans, self.y_trans, zoom)
        )
        self.prev_XYpsi = [X,Y,psi]

        
        ## Update the display
        self.screen.blit( pygame.transform.flip(self.screen, False, True), (0, 0) )  # mirror screen vertical
        pygame.display.flip()  # finally update display
        # pygame.display.update()
        # self.FramePerSec.tick(self.FPS)


    def close(self):
        pass
