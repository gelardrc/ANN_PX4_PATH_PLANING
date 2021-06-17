import dijkstra3d
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from random import random,randint




#fig = plt.figure(figsize=(15, 15))
##ax = fig.add_subplot(1, 2, 1, projection='3d')
#ax = plt.axes(projection='3d')
#ax.set_xlabel('X, [m]')
#ax.set_ylabel('Y, [m]')
#ax.set_zlabel('Z, [m]')
#ax.set_xlim([0, 3])
#ax.set_ylim([0, 3])
#ax.set_zlim([0, 3])

## 1 - fundo , 2 -  topo , 3 frente ,4 esquerda , 5 direita ,6 direita 








def desenha_objetos(pose,tamanho,ax):


    x = [[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,0,0],[1,1,1,1],[0,0,1,1],[1,1,0,0]]

    y = [[0,1,1,0],[0,1,1,0],[0,0,0,0],[0,0,1,1],[1,1,0,0],[0,0,0,0],[1,1,1,1]]

    z = [[0,0,0,0],[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]]

    


    for x1,y1,z1 in zip(x,y,z):

        x1 = np.array(x1)

        x1 = pose[0] + x1

        x1 = x1*tamanho[0]

        y1 = np.array(y1)

        y1 = pose[1] + y1

        y1 = y1*tamanho[1]

        z1 = np.array(z1)

        z1 = pose[2] + z1

        z1 = z1*tamanho[2]


        nadegas = list(zip(x1,y1,z1))

        poly = Poly3DCollection([nadegas], linewidths=1)
        poly.set_alpha(0.5)
        poly.set_facecolor('k')
        # poly.set_edgecolor('k')
        ax.add_collection3d(poly)
    