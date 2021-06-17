import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd


fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
ax.set_xlim([0,30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])

dataset = pd.read_csv("/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv")

dataset_xyz= dataset.iloc[0:180000,0:3]

dataset_target = dataset.iloc[0:180000,3:6]


for i in range(180000):
    pose = dataset_xyz.iloc[i,0:3]
    pose_x = pose.real[0]
    pose_y = pose.real[1]
    pose_z = pose.real[2]
    ax.scatter3D(pose_x, pose_y, pose_z, color='green', s=10)

for i in range(180000):
    target = dataset_target.iloc[i,0:3]
    target_x = target.real[0]
    target_y = target.real[1]
    target_z = target.real[2]
    ax.scatter3D(target_x, target_y, target_z, color='red', s=10)


plt.show()