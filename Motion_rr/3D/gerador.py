import numpy as np
from numpy.linalg import norm
from math import *
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from random import random, randint
from scipy.spatial import ConvexHull
from matplotlib import path
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tools import init_fonts
from path_shortening import shorten_path


def isCollisionFreeVertex(obstacles, point):
    x, y, z = point
    for obstacle in obstacles:
        dx, dy, dz = obstacle.dimensions
        x0, y0, z0 = obstacle.pose
        if abs(x-x0) <= dx/2 and abs(y-y0) <= dy/2 and abs(z-z0) <= dz/2:
            return 0
    return 1


def isCollisionFreeEdge(obstacles, closest_vert, p):
    closest_vert = np.array(closest_vert)
    p = np.array(p)
    collFree = True
    l = norm(closest_vert - p)
    map_resolution = 0.01
    M = int(l / map_resolution)
    if M <= 2:
        M = 20
    t = np.linspace(0, 1, M)
    for i in range(1, M-1):
        point = (1-t[i])*closest_vert + t[i]*p  # calculate configuration
        collFree = isCollisionFreeVertex(obstacles, point)
        if collFree == False:
            return False

    return collFree


class Node3D:
    def __init__(self):
        self.p = [0, 0, 0]
        self.i = 0
        self.iPrev = 0


def closestNode3D(rrt, p):
    distance = []
    for node in rrt:
        distance.append(sqrt((p[0] - node.p[0])**2 +
                        (p[1] - node.p[1])**2 + (p[2] - node.p[2])**2))
    distance = np.array(distance)

    dmin = min(distance)
    ind_min = distance.tolist().index(dmin)
    closest_node = rrt[ind_min]

    return closest_node


def plot_point3D(p, color='blue'):
    ax.scatter3D(p[0], p[1], p[2], color=color)


cont = 0
# Add Obstacles


class Parallelepiped:
    def __init__(self):
        self.dimensions = [0, 0, 0]
        self.pose = [0, 0, 0]
        self.verts = self.vertixes()

    def vertixes(self):
        dx = self.dimensions[0]
        dy = self.dimensions[1]
        dz = self.dimensions[2]
        C = np.array(self.pose)

        Z = np.array([[-dx/2, -dy/2, -dz/2],
                      [dx/2, -dy/2, -dz/2],
                      [dx/2, dy/2, -dz/2],
                      [-dx/2, dy/2, -dz/2],
                      [-dx/2, -dy/2, dz/2],
                      [dx/2, -dy/2, dz/2],
                      [dx/2, dy/2, dz/2],
                      [-dx/2, dy/2, dz/2]])
        Z += C

        # list of sides' polygons of figure
        verts = [[Z[0], Z[1], Z[2], Z[3]],
                 [Z[4], Z[5], Z[6], Z[7]],
                 [Z[0], Z[1], Z[5], Z[4]],
                 [Z[2], Z[3], Z[7], Z[6]],
                 [Z[1], Z[2], Z[6], Z[5]],
                 [Z[4], Z[7], Z[3], Z[0]]]

        return verts

    def draw(self, ax,):
        ax.add_collection3d(Poly3DCollection(
            self.vertixes(), facecolors='k', linewidths=1, edgecolors='k', alpha=0.5))

### Obstacles ###


def add_obstacle(obstacles, pose, dim):
    obstacle = Parallelepiped()
    obstacle.dimensions = dim
    obstacle.pose = pose
    obstacles.append(obstacle)
    return obstacles


def choque(onde_estou, obstacles):

    #### aqui verifica se a direção escolhida esta ocupada ###
    grid = 1
    cont = 0
    choqui = [0, 0, 0, 0, 0, 0]
    direction = [[0, 0, grid], [0, 0, -grid], [0, grid, 0],
                 [0, -grid, 0], [grid, 0, 0], [-grid, 0, 0]]
    for i in direction:
        bateu = isCollisionFreeVertex(obstacles, onde_estou+i)
        #bateu = isCollisionFreeEdge(obstacles,onde_estou+i,onde_estou)

        # BAteu é zero quando tem choque !!
        if bateu == 0:
            choqui[cont] = 1
            cont = cont+1
        else:
            choqui[cont] = 0
            cont = cont+1
    return choqui


def acao(sensores, alvo, pose, pose_antigo):

    grid = 1
    action = [0, 0, 0, 0, 0, 0]
    norma = []
    minimo = 0
    direction = [[0, 0, grid], [0, 0, -grid], [0, grid, 0],
                 [0, -grid, 0], [grid, 0, 0], [-grid, 0, 0]]
    for i in range(len(sensores)):
        # caso sensores[i] for 0 quer dizer que é uma posição a ser avaliada caso não so coloco uma norma alta
        if sensores[i] == 0:
            if np.array_equal(pose+direction[i], pose_antigo):
                norma.append(99999999)
            ### avalia direção baseado na norma , a menor norma vence ###

            else:
                norma.append(np.linalg.norm(alvo - (pose + direction[i])))

        else:
            norma.append(999999)

    minimo = min(norma)

    for i in range(len(norma)):
        if minimo == norma[i]:
            action[i] = 1
        else:
            action[i] = 0
    while sum(action) > 1:
        for i in range(len(action)):
            if action[i] == 1:
                rnd = randint(0, 1)
                if rnd > 0.5:
                    action[i] = 0
                else:
                    action[i] = 1

    return action


def direcao(onde_estou, acao):
    novo_start = onde_estou
    grid = 1
    direction = [[0, 0, grid], [0, 0, -grid], [0, grid, 0],
                 [0, -grid, 0], [grid, 0, 0], [-grid, 0, 0]]
    for i in range(len(acao)):
        if acao[i] == 1:
            novo_start = onde_estou + direction[i]
    return novo_start


def map():

    for x in range(-8, 8, 1):

        for y in range(-8, 8, 1):

            for z in range(-8, 8, 1):
                vetor = np.array([x, y, z])
                
                bateu = isCollisionFreeVertex(obstacles,vetor)
                if bateu: 
                    apa[x, y, z] = 999999999
                else:
                    mapa[x, y, z] = np.linalg.norm(goal-vetor)


def a_star(onde_estou, sensores):
    peso = map()
    grid = 1
    norma = []

#    for x in range(-8, 8, 1):
#
#        for y in range(-8, 8, 1):
#
#            for z in range(-8, 8, 1):
#
#              # mapa[x,y,z] = np.linalg.norm(goal-)


obstacles_poses = [[0, 0, 2]]  # ,[0,2,2]]
obstacles_dims = [[12, 1, 8]]

obstacles = []
for pose, dim in zip(obstacles_poses, obstacles_dims):
    obstacles = add_obstacle(obstacles, pose, dim)

init_fonts()
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
ax.set_xlim([-8, 8])
ax.set_ylim([-8, 8])
ax.set_zlim([-8, 8])

for obstacle in obstacles:
    obstacle.draw(ax)


# RRT Initialization


start = np.array([-2, -4, 0])
ax.scatter3D(start[0], start[1], start[2], color='green', s=100)

start_passado = []

# print("start_passado");print(start_passado)

plt.pause(0.1)

goal = np.array([5, 5, 5])
ax.scatter3D(goal[0], goal[1], goal[2], color='red', s=100)

plt.pause(0.1)


while True:

    while np.linalg.norm(goal-start) > 0.001:

        ### detectar alvos nas possiveis direções ###

        sensores = choque(start, obstacles)

        if np.array_equal(start_passado, start):
            a_estrela(start)
            break

        start_passado = start

        ### melhor saida possivel baseada nos sensores ###

        action = acao(sensores=sensores, alvo=goal,
                      pose=start, pose_antigo=start_passado)

        # Define a direção a ser seguida

        start = direcao(start, action)

        ax.plot([start[0], start_passado[0]], [start[1], start_passado[1]], [
                start[2], start_passado[2]], color='red', linewidth=2)
        plt.pause(0.01)
        #ax1.plot([start[0],new_start[0]],[start[1],new_start[1]],[start[2],new_start[2]],color ='red',linewidth=2)

        print("start -->")
        print(start)

        print("sensores -->")
        print(sensores)

        print("action -->")
        print(action)

    plt.show()
