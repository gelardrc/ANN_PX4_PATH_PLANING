import dijkstra3d
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from random import random,randint
import pandas as pd




def action(primeiro, segundo):
    ac = 0
    
    primeiro = [primeiro[0],primeiro[1],primeiro[2]]
    
    segundo = [segundo[0],segundo[1],segundo[2]]
    
    a = [[0, 0, 1], [0, 0, -1], [0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0]]
    
    vec = [0,0,0]
    
    #vec = [segundo[0]-primeiro[0],segundo[1]-primeiro[1], segundo[2]-primeiro[2]]
    
    vec = np.subtract(segundo,primeiro,dtype=np.int32)
    
    for index, value in enumerate(a):
         if vec[0]-value[0] == 0 and vec[1]-value[1] == 0 and vec[2]-value[2] == 0:
            if index == 0:
                ac = 1
            if index == 1:
                ac = 2
            if index == 2:
                ac = 3
            if index == 3:
                ac = 4
            if index == 4:
                ac = 5
            if index == 5:
                ac = 6

    return ac


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
        '/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/dirsjtk_3d.csv', index=False)

    return 0


def sensors(pose):
    s_0 = 0
   
pose = np.array(pose)

    visao = [[0, 0, 1], [0, 0, -1], [0, 1, 0],
             [0, -1, 0], [1, 0, 0], [-1, 0, 0]]

    for index, sensor in enumerate(visao):
        view = pose+sensor
        if view[0] > 30 or view[1] > 30 or view[2]>30:
            if index == 0:
                s_0 = 1

            if index == 1:
                s_0 = 2

            if index == 2:
                s_0 = 3

            if index == 3:
                s_0 = 4

            if index == 4:
                s_0 = 5

            if index == 5:
                s_0 = 6
            
            continue
            
        if field[view[0],view[1],view[2]] > 1:
           if index == 0:
                s_0 = 1

            if index == 1:
                s_0 = 2

            if index == 2:
                s_0 = 3

            if index == 3:
                s_0 = 4

            if index == 4:
                s_0 = 5

            if index == 5:
                s_0 = 6
    
    return s_0


def isCollisionFreeVertex(ponto,tamanho, point):
    x, y, z = point
    for pose,dimension in zip(ponto,tamanho):
        dx,dy,dz = dimension
        x0,y0,z0 = pose 
        if abs(x-x0) <= dx and abs(y-y0) <= dy and abs(z-z0) <= dz:
            return 0
    return 1
    

def obj(tamanho,ponto):
    pose = ponto
    colecao = []
    for x in range(tamanho[0]):
        for y in range(tamanho[1]):
            for z in range(tamanho[2]):
                ponto = [pose[0]+x,pose[1]+y,pose[2]+z]
                #ax.scatter3D(ponto[0], ponto[1], ponto[2], color='red')
                colecao.append(ponto)
                if ponto[0] > 30 or ponto[1] >30 or ponto[2] >30:
                    continue  
                field[ponto[0],ponto[1],ponto[2]] = 9999999999
    ponto = pose
    posee = [0,0,0]
    
    desenha_objetos(ponto,tamanho)
   
def desenha_objetos(pose,tamanho):


    x = [[0,0,1,1],[0,0,1,1],[0,0,1,1],[0,0,0,0],[1,1,1,1],[0,0,1,1],[1,1,0,0]]

    y = [[0,1,1,0],[0,1,1,0],[0,0,0,0],[0,0,1,1],[1,1,0,0],[0,0,0,0],[1,1,1,1]]

    z = [[0,0,0,0],[1,1,1,1],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0],[0,1,1,0]]

    


    for x1,y1,z1 in zip(x,y,z):

        x1 = np.array(x1)

        x1 = x1*tamanho[0]
        
        
        x1 = pose[0] + x1

        

        y1 = np.array(y1)

        y1 = y1*tamanho[1]
        
        
        y1 = pose[1] + y1

        

        z1 = np.array(z1)

        z1 = z1*tamanho[2]
        
        
        z1 = pose[2] + z1

        


        nadegas = list(zip(x1,y1,z1))

        poly = Poly3DCollection([nadegas], linewidths=1)
        poly.set_alpha(0.25)
        poly.set_facecolor('k')
        # poly.set_edgecolor('k')
        ax.add_collection3d(poly)
    
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

fig = plt.figure(figsize=(15, 15))
#ax = fig.add_subplot(1, 2, 1, projection='3d')
ax = plt.axes(projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
ax.set_xlim([0, 30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])


#field = np.ones((512, 512, 512), dtype=np.int32)

field = np.ones((31, 31, 31))






#path = dijkstra3d.dijkstra(field, source, target, connectivity=6) 

#print(path)


ponto = [[-1,5,-1],[-1,-1,-1],[-1,10,2],[-1,15,-1],[-1,19,15],[-1,19,10],[-1,19,12],[-1,-1,4],[-1,1,8],[-1,-1,12],[-1,1,16],[-1,-1,20],[-1,15,21],[-1,15,25],[-1,21,23],[-1,20,0],[-1,24,2],[-1,28,0]]
tamanho = [[35,2,30],[35,6,2],[35,2,30],[35,2,20],[35,12,2],[35,10,2],[35,2,3],[35,5,2],[35,5,2],[35,5,2],[35,5,2],[35,5,2],[35,15,2],[35,12,2],[35,4,2],[35,2,4],[35,2,8],[35,2,4]]


for i in zip(tamanho,ponto):
    obj(i[0],i[1])


#source = (3,3,3)
#    
#target = (15, 24, 14)

for vv in range(2000):
    
    source = (randint(0,30),randint(0,30),randint(0,30))
    s1 = np.array(source)
    target = (randint(0,30),randint(0,30),randint(0,30))
    s2= np.array(source)
    while field[s1[0],s1[1],s1[2]] > 1 or  field[s2[0],s2[1],s2[2]] > 1:
    
        source = (randint(0,30),randint(0,30),randint(0,30))
        s1 = np.array(source)
        target = (randint(0,30),randint(0,30),randint(0,30))
        s2= np.array(source)
    
    path = dijkstra3d.dijkstra(field, source, target, connectivity=6) 


    # mostrar o caminho no grafico
    for value,i in enumerate(path):
        if value == 0 :

            i_antigo = i

        else:
            #ax.plot([i_antigo[0], i[0]], [i_antigo[1], i[1]],[i_antigo[2], i[2]], color='red', linewidth=2, zorder=15)

            i_antigo = i

    path = list(path)
    for i in range(len(path)-1):

        ac0 = action(path[i], path[i+1])
        acao.append(ac0)
       

    ## Retira o alvo do path , pq nao precisa calcular os sensores dele 

    #path = list(path)
    path.pop(len(path)-1)

    for index, value in enumerate(path):

        s0= sensors(value)

        alvo_x.append(target[0])
        alvo_y.append(target[1])
        alvo_z.append(target[2])
        pose_x.append(value[0])
        pose_y.append(value[1])
        pose_z.append(value[2])
        
        sensores.append(s0)
        
    # novo_caminho.pop(len(novo_caminho)-1)
    print(vv)





plt.show()



banco_de_dados(pose_x,
               pose_y,
               pose_z,
               alvo_x,
               alvo_y,
               alvo_z,
               sensores,
               acao
              )



