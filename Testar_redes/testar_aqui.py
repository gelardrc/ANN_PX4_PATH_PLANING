# -*- coding: utf-8 -*-
from keras.models import load_model
import numpy as np
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from matplotlib import path
from random import random
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon


def isCollisionFreeVertex(obstacles, point):
    x,y,z = point
    for obstacle in obstacles:
	    dx, dy, dz = obstacle.dimensions
	    x0, y0, z0 = obstacle.pose
	    if abs(x-x0)<=dx/2 and abs(y-y0)<=dy/2 and abs(z-z0)<=dz/2:
	        return 0
    return 1
def sensors(pose):
  s_0 = 0  
  s_1 = 0
  s_2 = 0
  s_3 = 0
  s_4 = 0
  s_5 = 0
  
  pose = np.array(pose) 
  
  visao = [[0, 0, 1], [0, 0, -1], [0, 1, 0],[0, -1, 0], [1, 0, 0], [-1, 0, 0]]
  
  for index,sensor in enumerate(visao):
    if isCollisionFreeVertex(obstacles,pose+sensor) == 0 : 
      if index == 0:  s_0 = 1
      if index == 1:  s_1 = 1
      if index == 2:  s_2 = 1
      if index == 3:  s_3 = 1
      if index == 4:  s_4 = 1
      if index == 5:  s_5 = 1
  return s_0,s_1,s_2,s_3,s_4,s_5
def action(out, start, repitido):

    if repitido:
        # pego o valor maximo e zero ele, para que a segunda opçaoo seja a valida
        max = np.max(out)
        for index, value in enumerate(out[0]):
            if value == max:
                out[0, index] = 0

    out = safety(out, sensores)

    max = np.max(out)

    acao = [0, 0, 0, 0, 0, 0]

    for index, value in enumerate(out[0]):

        if max == value:
            direcao = index
            acao[index] = 1

    # print(acao)

    if direcao == 0:
        start = start + [0, 0, 1]
    if direcao == 1:
        start = start + [0, 0, -1]
    if direcao == 2:
        start = start + [0, 1, 0]
    if direcao == 3:
        start = start + [0, -1, 0]
    if direcao == 4:
        start = start + [1, 0, 0]
    if direcao == 5:
        start = start + [-1, 0, 0]

    return acao, start
def safety(out, sensores):
    # garante que nao terá uma ordem de choque
    for index, value in enumerate(out[0]):
        if sensores[index] == 1:
            out[0, index] = 0

    return out
def entrada(estou,alvo,sensor):
    
    entrada = np.zeros((1,12))
    entrada[0,0]=estou[0]
    entrada[0,1]=estou[1]
    entrada[0,2]=estou[2]
    entrada[0,3]=alvo[0]
    entrada[0,4]=alvo[1]
    entrada[0,5]=alvo[2]
    entrada[0,6]=sensor[0]
    entrada[0,7]=sensor[1]
    entrada[0,8]=sensor[2]
    entrada[0,9]=sensor[3]
    entrada[0,10]=sensor[4]
    entrada[0,11]=sensor[5]
    
    return entrada
class Parallelepiped:
    def __init__(self):
        self.dimensions = [0,0,0]
        self.pose = [0,0,0]
        self.verts = self.vertixes()
        
    def vertixes(self):
        dx = self.dimensions[0]
        dy = self.dimensions[1]
        dz = self.dimensions[2]
        C = np.array(self.pose)

        Z = np.array([[-dx/2, -dy/2, -dz/2],
                      [dx/2, -dy/2, -dz/2 ],
                      [dx/2, dy/2, -dz/2],
                      [-dx/2, dy/2, -dz/2],
                      [-dx/2, -dy/2, dz/2],
                      [dx/2, -dy/2, dz/2 ],
                      [dx/2, dy/2, dz/2],
                      [-dx/2, dy/2, dz/2]])
        Z += C

        # list of sides' polygons of figure
        verts = [ [Z[0], Z[1], Z[2], Z[3]],
                  [Z[4], Z[5], Z[6], Z[7]], 
                  [Z[0], Z[1], Z[5], Z[4]], 
                  [Z[2], Z[3], Z[7], Z[6]], 
                  [Z[1], Z[2], Z[6], Z[5]],
                  [Z[4], Z[7], Z[3], Z[0]] ]

        return verts

    def draw(self, ax,):
        poly = Poly3DCollection(self.vertixes(), linewidths=4)
      
        poly.set_alpha(0.05)
        poly.set_facecolor('k') 
      
        poly.set_edgecolor('b')
        ax.add_collection3d(poly)
def add_obstacle(obstacles, pose, dim):
	obstacle = Parallelepiped()
	obstacle.dimensions = dim
	obstacle.pose = pose
	obstacles.append(obstacle)
	return obstacles
def moving_target(target):

  new_target = [target[0]+randint(-1,1),target[1]+randint(-1,1),target[2]+randint(-1,1)]

  while  isCollisionFreeVertex(obstacles,new_target)==0 or new_target[0]>20 or new_target[1]>20 or new_target[2]>20 or new_target[0]<0  or new_target[1]<0  or new_target[2]<0  or list(new_target) in openlist :
        new_target = [target[0]+randint(-1,1),target[1]+randint(-1,1),target[2]+randint(-1,1)]
        print('estou aqui')

    # Atualiza no desenho o plotter
     
    #ax.scatter3D(target[0], target[1], target[2], color='red', s=100)


    
  return new_target
def map():
    
    # Mundo facil

    obstacles_poses = [[15,15,15],[15,5,2],[5,25,25],[25,25,25]]
    obstacles_dims = [[10,10,10],[30,10,4],[10,10,10],[10,10,10]]


    ### mapa medio ###
    #obstacles_poses =[[15,15,6],[15,15,22.5],[15,4,4],[15,4,22.5],[15,23,4],[15,23,22.5],[15,10,27.5],[15,10,10]]#,[15,8,0],[15,7,20]]#,[0,2,2]]
    #obstacles_dims  =[[30,2,12],[30,2,15],[30,2,8],[30,2,15],[30,2,8],[30,2,15],[30,1,5],[30,1,20]]#,[30,10,8],[30,3,20]]

    ### mundo dificil #######

    #obstacles_poses =[[15,15,6],[15,15,22],[15,4,4],[15,4,22],[15,23,4],[15,23,22],[15,10,27],[15,10,10],[0,15,15],[15,0,15],[15,15,30],[15,15,0],[15,30,15],[30,15,15]]
    #obstacles_dims  =[[30,2,12],[30,2,15],[30,2,8],[30,2,15],[30,2,8],[30,2,15],[30,1,5],[30,1,20],[1,30,30],[30,1,30],[30,30,1],[30,30,1],[30,0.5,30],[1,30,30]]

    #for i in range(3): obstacles_poses.append([-6,randint(-6,6),randint(-6,6)])
    #for i in range(len(obstacles_poses)) : obstacles_dims.append([12,1,8])

    obstacles = []
    for pose, dim in zip(obstacles_poses, obstacles_dims):
    	obstacles = add_obstacle(obstacles, pose, dim)

    
    
    return obstacles


fig = plt.figure(figsize=(15,15))
ax = plt.axes(projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])

obstacles = map()

for obstacle in obstacles: obstacle.draw(ax)

REDE =  '/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dijstrk_sem_obj.h5'

rede = load_model(REDE)
openlist = []
start = np.array([15,5,6]);
goal = np.array([15,25,15]);

while np.linalg.norm(goal-start) > 0.001:

            s0,s1,s2,s3,s4,s5 = sensors(start)

            sensores = np.array([s0,s1,s2,s3,s4,s5])
            input = entrada(start,goal,sensores)
            out = rede.predict(input)

            acao, start_futuro = action(out, start, repitido=False)
            start_futuro_list = list(start_futuro)

            while start_futuro_list in openlist: 
                acao,start_futuro = action(out,start,repitido=True)
                start_futuro_list = list(start_futuro)
            
            ax.plot([start[0],start_futuro[0]],[start[1],start_futuro[1]],[start[2],start_futuro[2]],color = 'r',linewidth =2 )
            plt.pause(0.01)

            start = start_futuro
                     #print(start)
            openlist.append(list(start))

plt.show()

