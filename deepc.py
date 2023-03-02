import numpy as np
import scipy as sp
from scipy import sparse
import matplotlib.pyplot as plt
from scipy.linalg import hankel
import osqp
import time
import cvxpy as cp
import gym
import mujoco_py

env = gym.make('CartPole-v1')
env.reset()
done = False
for i in range(100):
    time.sleep(0.1)
    env.render()
    env.step(env.action_space.sample()) # take a random action
env.close()

