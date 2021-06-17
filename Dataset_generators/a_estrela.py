# -*- coding: utf-8 -*-
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
import pandas as pd
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection


def entrada(vetor, alvo, g_passado):
    g = g_passado + 1
    h = (vetor[0] - alvo[0])**2 + (vetor[1] -
                                   alvo[1])**2 + ((vetor[2] - alvo[2]))**2
    f = g + h
    return h, g, f


def sensors(pose):
    s_0 = 0
    s_1 = 0
    s_2 = 0
    s_3 = 0
    s_4 = 0
    s_5 = 0

    pose = np.array(pose)

    visao = [[0, 0, 1], [0, 0, -1], [0, 1, 0],
             [0, -1, 0], [1, 0, 0], [-1, 0, 0]]

    for index, sensor in enumerate(visao):
        if isCollisionFreeVertex(obstacles, pose+sensor) == 0:
            if index == 0:
                s_0 = 1
            if index == 1:
                s_1 = 1
            if index == 2:
                s_2 = 1
            if index == 3:
                s_3 = 1
            if index == 4:
                s_4 = 1
            if index == 5:
                s_5 = 1
    return s_0, s_1, s_2, s_3, s_4, s_5


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

        poly = Poly3DCollection(self.vertixes(), linewidths=1)

        poly.set_alpha(0.5)
        poly.set_facecolor('k')

        # poly.set_edgecolor('k')
        ax.add_collection3d(poly)


def isCollisionFreeVertex(obstacles, point):
    x, y, z = point
    for obstacle in obstacles:
        dx, dy, dz = obstacle.dimensions
        x0, y0, z0 = obstacle.pose
        if abs(x-x0) <= dx/2 and abs(y-y0) <= dy/2 and abs(z-z0) <= dz/2:
            return 0
    return 1


def add_obstacle(obstacles, pose, dim):
    obstacle = Parallelepiped()
    obstacle.dimensions = dim
    obstacle.pose = pose
    obstacles.append(obstacle)
    return obstacles


class Node():
    """A node class for A* Pathfinding"""

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position


def score(vector, goal, start):
    g = np.linalg.norm(vector - start)
    h = np.linalg.norm(goal - vector)
    f = g+h
    return g, h, f


def action(primeiro, segundo):
    ac0 = 0
    ac1 = 0
    ac2 = 0
    ac3 = 0
    ac4 = 0
    ac5 = 0

    a = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    vec = [segundo[0]-primeiro[0], segundo[1] -
           primeiro[1], segundo[2]-primeiro[2]]
    for index, value in enumerate(a):
        if vec[0]-value[0] == 0 and vec[1]-value[1] == 0 and vec[2]-value[2] == 0:
            if index == 0:
                ac0 = 1
            if index == 1:
                ac1 = 1
            if index == 2:
                ac2 = 1
            if index == 3:
                ac3 = 1
            if index == 4:
                ac4 = 1
            if index == 5:
                ac5 = 1

    return ac0, ac1, ac2, ac3, ac4, ac5


def banco_de_dados(h,
                   g,
                   f,
                   alvo_x,
                   alvo_y,
                   alvo_z,
                   choque0,
                   choque1,
                   choque2,
                   choque3,
                   choque4,
                   choque5,
                   acao0,
                   acao1,
                   acao2,
                   acao3,
                   acao4,
                   acao5):

    local = {'h': h,
             'g': g,
             'f': f,
             'alvo_x': alvo_x,
             'alvo_y': alvo_y,
             'alvo_z': alvo_z,
             'choque0': choque0,
             'choque1': choque1,
             'choque2': choque2,
             'choque3': choque3,
             'choque4': choque4,
             'choque5': choque5,
             'acao0': acao0,
             'acao1': acao1,
             'acao2': acao2,
             'acao3': acao3,
             'acao4': acao4,
             'acao5': acao5
             }

    df = pd.DataFrame(local, columns=['h',
                                      'g',
                                      'f',
                                      'alvo_x',
                                      'alvo_y',
                                      'alvo_z',
                                      'choque0',
                                      'choque1',
                                      'choque2',
                                      'choque3',
                                      'choque4',
                                      'choque5',
                                      'acao0',
                                      'acao1',
                                      'acao2',
                                      'acao3',
                                      'acao4',
                                      'acao5'])

    print(df)
    print(df.dtypes)

    df.to_csv(
        '/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/a_estrela_com_obstaculos_novo.csv', index=False)

    return 0


############### variáveis do dataset###########################
h = []
f = []
g = []
pose_x = []
pose_y = []
pose_z = []
alvo_x = []
alvo_y = []
alvo_z = []
sensores_0 = []
sensores_1 = []
sensores_2 = []
sensores_3 = []
sensores_4 = []
sensores_5 = []
acao_0 = []
acao_1 = []
acao_2 = []
acao_3 = []
acao_4 = []
acao_5 = []


################## mapa e obstaculos ########################

#obstacles_poses =[[0,2,-2],[0,-8,2],[0,-8,2],[0,-6,-3],[0,-8,2],[0,0,1],[0,2,6],[0,-4,6]]
#obstacles_dims  =[[16,1,3],[16,1,3],[16,1,3],[16,1,3],[16,1,3],[16,1,3],[16,1,3],[16,1,3]]
obstacles_poses = []
obstacles_dims = []

numero_de_obstaculos = 30

for i in range(numero_de_obstaculos):
    vec = [randint(-5, 5), randint(-5, 5), randint(-5, 5)]
    obstacles_poses.append(vec)

for value, i in enumerate(obstacles_poses):
    obstacles_poses.pop(value)
    for f in range(len(obstacles_poses)):
        if np.array_equal(np.array(i), np.array(obstacles_poses[f])):
            continue
        # else:
    obstacles_poses.append(i)

obstacles_poses = [[0, 2, -2], [0, -6, -5]]  # ,[0,2,2]]
obstacles_dims = [[12, 1, 12], [12, 1, 1]]


for i in range(len(obstacles_poses)):
    obstacles_dims.append([1, 1, 1])


obstacles = []
for pose, dim in zip(obstacles_poses, obstacles_dims):
    obstacles = add_obstacle(obstacles, pose, dim)

fig = plt.figure(figsize=(15, 15))
#ax = fig.add_subplot(1, 2, 1, projection='3d')
ax = plt.axes(projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
#ax.set_xlim([-2.5, 2.5])
#ax.set_ylim([-2.5, 2.5])
#ax.set_zlim([0.0, 3.0])
ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_zlim([-6, 6])


for obstacle in obstacles:
    obstacle.draw(ax)

#########################################################################


for i in range(1):

    print(i)

    comeco = (randint(-6, 6), randint(-6, 6), randint(-6, 6))
    alvo = (randint(-6, 6), randint(-6, 6), comeco[2])

    ## garantir que comeco e alvo nao estaram dentro do obstaculo ##

    while isCollisionFreeVertex(obstacles, comeco) and isCollisionFreeVertex(obstacles, alvo) == False:
        #comeco[0] = randint(-8,8)
        #alvo[1] = randint(-8,8)
        print("ESTOU EM UM LOOP")
        comeco = (randint(-6, 6), randint(-6, 6), randint(-6, 6))
        alvo = (randint(-6, 6), randint(-6, 6), comeco[2])
    comeco = (0, -2, 0)
    alvo = (0, 4, 0)

    lista_aberta = []

    lista_fechada = []

    ## defino o nó pela classe node ##
    primeiro_no = Node(None, comeco)
    # o primeiro nó tem g h e f iguais a 0
    primeiro_no.g = primeiro_no.h = primeiro_no.f = 0

    alvo_no = Node(None, alvo)

    alvo_no.g = alvo_no.h = alvo_no.f = 0

    ### o primeiro nó a ser analisado é o comeco ##

    lista_aberta.append(primeiro_no)

    # enquanto open list nao estiver vazio, essa parte vai rodar full
    while len(lista_aberta) > 0:

        print('lista aberta -->')
        print(lista_aberta[len(lista_aberta)-1].position)

        current_node = lista_aberta[0]

        current_index = 0

        ### verifica se tem alguem no open list que tenha valor de f menor  ##
        for index, item in enumerate(lista_aberta):

            if item.f < current_node.f:

                current_node = item

                current_index = index

        lista_aberta.pop(current_index)

        lista_fechada.append(current_node)

        if current_node == alvo_no:
            caminho = []
            current = current_node
            while current is not None:
                #ax.plot([current.position[0], caminho[i+1,0]], [caminho[i,1], caminho[i+1,1]], [caminho[i,2], caminho[i+1,2]], color = 'orange', linewidth=1, zorder=15)
                caminho.append(current.position)
                current = current.parent
                #print('lista aberta -->');print(len(lista_aberta))
                # print('caminho');print(caminho)
            break

        children = []

        # Adjacent squares
        for new_position in [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]:

            # Get node position
            node_position = (current_node.position[0] + new_position[0], current_node.position[1] +
                             new_position[1], current_node.position[2] + new_position[2])

            # Make sure within range
           # if node_position[0] > 8 or node_position[0] < 0 or node_position[1] > 8 or node_position[1] < 0 or node_position[2] > 8 or node_position[2] < 0 :
            if node_position[0] > 6 or node_position[1] > 6 or node_position[2] > 6 or node_position[0] < -6 or node_position[1] < - 6 or node_position[2] < -6:
                continue

            # Make sure walkable terrain
            if isCollisionFreeVertex(obstacles, node_position) == False:
                continue
            # if maze[node_position[0]][node_position[1]] != 0:

                # Create new node
            new_node = Node(current_node, node_position)

            # Append
            children.append(new_node)

            # Loop through children
        for child in children:
            # Child is on the closed list
            for closed_child in lista_fechada:
                if child == closed_child:
                    continue

                  # Create the f, g, and h values
                child.g = current_node.g + 1
                #child.h = (child.position[0] - alvo_no.position[0])**2 + (child.position[1] - alvo_no.position[1])**2 + ((child.position[2] - alvo_no.position[2]))**2
                child.h = (child.position[0] - alvo_no.position[0])**2 + (child.position[1] - alvo_no.position[1])**2 + ((child.position[2] - alvo_no.position[2]))**2
                child.f = child.g + child.h

                # Child is already in the open list
            for open_node in lista_aberta:
                if child == open_node and child.g > open_node.g:
                    continue

            lista_aberta.append(child)

    # print(caminho)

    start = np.array([comeco[0], comeco[1], comeco[2]])
    ax.scatter3D(start[0], start[1], start[2], color='green', s=100)

    goal = np.array([alvo[0], alvo[1], alvo[2]])
    ax.scatter3D(goal[0], goal[1], goal[2], color='red', s=100)

    novo_caminho = caminho[::-1]
    for i in range(len(novo_caminho)-1):
        primeiro = novo_caminho[i]
        segundo = novo_caminho[i+1]
        ax.plot([primeiro[0], segundo[0]], [primeiro[1], segundo[1]], [
                primeiro[2], segundo[2]], color='orange', linewidth=1, zorder=15)
        plt.pause(0.01)

    for i in range(len(novo_caminho)-1):
        ac0, ac1, ac2, ac3, ac4, ac5 = action(
            novo_caminho[i], novo_caminho[i+1])
        acao_0.append(ac0)
        acao_1.append(ac1)
        acao_2.append(ac2)
        acao_3.append(ac3)
        acao_4.append(ac4)
        acao_5.append(ac5)

    novo_caminho.pop(len(novo_caminho)-1)

    g_antigo = 0

   # for index,value in enumerate(novo_caminho):
   #   #if index==0:
   #   #  h.append(0)
   #   #  g.append(0)
   #   #  f.append(0)
   #   #
   #   #else:
   #     heu,ge,fe = entrada(value,alvo,g_antigo)
   #     h.append(heu)
   #     g.append(ge)
   #     f.append(fe)
   #     g_antigo = ge
    for index, value in enumerate(novo_caminho):
        s0, s1, s2, s3, s4, s5 = sensors(value)
        alvo_x.append(alvo[0])
        alvo_y.append(alvo[1])
        alvo_z.append(alvo[2])
        pose_x.append(value[0])
        pose_y.append(value[1])
        pose_z.append(value[2])
        sensores_0.append(s0)
        sensores_1.append(s1)
        sensores_2.append(s2)
        sensores_3.append(s3)
        sensores_4.append(s4)
        sensores_5.append(s5)
    # novo_caminho.pop(len(novo_caminho)-1)

    print("leia aqui")

banco_de_dados(pose_x,
               pose_y,
               pose_z,
               alvo_x,
               alvo_y,
               alvo_z,
               sensores_0,
               sensores_1,
               sensores_2,
               sensores_3,
               sensores_4,
               sensores_5,
               acao_0,
               acao_1,
               acao_2,
               acao_3,
               acao_4,
               acao_5
               )
plt.show()
