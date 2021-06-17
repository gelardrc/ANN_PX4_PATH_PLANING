import numpy as np
from numpy.linalg import norm
from math import *
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from random import random,randint
from scipy.spatial import ConvexHull
from matplotlib import path
import time
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
from tools import init_fonts
from path_shortening import shorten_path
from keras.models import load_model


def isCollisionFreeVertex(obstacles, point):
    x,y,z = point
    for obstacle in obstacles:
	    dx, dy, dz = obstacle.dimensions
	    x0, y0, z0 = obstacle.pose
	    if abs(x-x0)<=dx/2 and abs(y-y0)<=dy/2 and abs(z-z0)<=dz/2:
	        return 0
    return 1

def isCollisionFreeEdge(obstacles, closest_vert, p):
    closest_vert = np.array(closest_vert); p = np.array(p)
    collFree = True
    l = norm(closest_vert - p)
    map_resolution = 0.01; M = int(l / map_resolution)
    if M <= 2: M = 20
    t = np.linspace(0,1,M)
    for i in range(1,M-1):
        point = (1-t[i])*closest_vert + t[i]*p # calculate configuration
        collFree = isCollisionFreeVertex(obstacles, point) 
        if collFree == False: return False

    return collFree

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

class Node3D:
    def __init__(self):
        self.p     = [0, 0, 0]
        self.i     = 0
        self.iPrev = 0


def closestNode3D(rrt, p):
    distance = []
    for node in rrt:
        distance.append( sqrt((p[0] - node.p[0])**2 + (p[1] - node.p[1])**2 + (p[2] - node.p[2])**2) )
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

      poly = Poly3DCollection(self.vertixes(), linewidths=1)
      
      poly.set_alpha(0.5)
      poly.set_facecolor('k') 
      
      #poly.set_edgecolor('k')
      ax.add_collection3d(poly)
        
### Obstacles ###
def add_obstacle(obstacles, pose, dim):
	obstacle = Parallelepiped()
	obstacle.dimensions = dim
	obstacle.pose = pose
	obstacles.append(obstacle)
	return obstacles

######## minha parte ############
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

def choque(onde_estou,obstacles):
    
    #### aqui verifica se a direção escolhida esta ocupada ###
    grid = 1
    cont = 0
    choqui = [0,0,0,0,0,0]
    direction = [[0,0,grid],[0,0,-grid],[0,grid,0],[0,-grid,0],[grid,0,0],[-grid,0,0]]
    for i in direction:
        bateu = isCollisionFreeVertex(obstacles,onde_estou+i)
        #bateu = isCollisionFreeEdge(obstacles,onde_estou+i,onde_estou)
        
        # BAteu é zero quando tem choque !!
        if bateu == 0:
            choqui[cont] = 1
            cont = cont+1
        else:
            choqui[cont]=0
            cont=cont+1
    return choqui

def direcao(saida,comeco):
    
    vencedor = 0
    #maximo = np.max(saida)
    #acao = 0 
    #grid = 1
    #soma = [[0,0,grid],[0,0,-grid],[0,grid,0],[0,-grid,0],[grid,0,0],[-grid,0,0]]
#
    #for index,value in enumerate(saida): 
    #    if value[index] == maximo : 
    #        acao = index
    #        break
    #comeco = np.sum([comeco,soma[acao]],axis=0)    
    #return comeco
    
    
    
    for i in range(6):
        if saida[0,i]>vencedor:
            vencedor = saida[0,i]
            acao = i
    if acao ==0: comeco = np.sum([comeco,[0,0,1]],axis=0)
    if acao ==1: comeco = np.sum([comeco,[0,0,-1]],axis=0)
    if acao ==2: comeco = np.sum([comeco,[0,+1,0]],axis=0)
    if acao ==3: comeco = np.sum([comeco,[0,-1,0]],axis=0)
    if acao ==4: comeco = np.sum([comeco,[1,0,0]],axis=0)
    if acao ==5: comeco = np.sum([comeco,[-1,0,0]],axis=0)
    
    return comeco

def movimento(obstaculos):
    
    global fim
    
    for g in obstaculos:
        if fim:
            g.pose[0] = g.pose[0]+0.5
            if g.pose[0]>6:
                fim = False
        else:
            g.pose[0]= g.pose[0]-0.5
            if g.pose[0]<-6:
                fim = True
    
    for obstacle in obstaculos: obstacle.draw(ax,ax1)
    #for obstacle in obstaculos: obstacle.draw(ax1)

def dynamic(ax):
    
    global obstacles_poses 
    
    global obstacles_dims  
    
    global obstacles
    
    quantidade = 10
    
    poose = []
    
    dims = []
    
    #lista = [[0,4,0],[0,2,0]]
    
    lista = [[0,1,1],[0,2,3]]
    
    #dims = [[12,1,12],[12,1,11]]
    
    for i in lista: poose.append(i) #poose.append([0,randint(-6,6),5])
       
    for i in range(len(poose)) : dims.append([12,1,1])

    for um, dois in zip(poose, dims):
	        
        obstacles = add_obstacle(obstacles, um, dois)

    #for obstacle in obstacles: obstacle.draw(ax) #obstacle.draw(ax1)
    
    #for obstacle in obstacles: obstacle.draw(ax) #obstacle.draw(ax1)
    
    #obstacles[len(obstacles)-1].draw(ax)
    for i in range(len(poose)):
        obstacles[len(obstacles)-i-1].draw(ax)
    

kidnapped_rtt = False
kidnapped_neural = False
neural = False
rtt_logic = True   
dinamico =  False

cont = 0
dist = 0
sol=True
fim=True

if neural:
    model = load_model('/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk.h5')

#################################

#obstacles_poses = [ [-0.8, 0., 1.5], [ 0., 1., 1.5], [ 0.,-1., 1.5] ]
#obstacles_dims  = [ [1.4, 1.0, 0.3], [3.0, 1.0, 0.3], [3.0, 1.0, 0.3] ]

#obstacles_poses =[[15,7,12]]#,[0,2,2]]
#obstacles_dims  =[[30,2,26]]

obstacles_poses =[[15,15,6],[15,15,22.5],[15,4,4],[15,4,25.5],[15,23,4],[15,23,25.5],[15,10,27.5],[15,10,10]]#,[15,8,0],[15,7,20]]#,[0,2,2]]
obstacles_dims  =[[30,2,12],[30,2,15],[30,2,8],[30,2,20],[30,2,8],[30,2,20],[30,1,5],[30,1,20]]#,[30,10,8],[30,3,20]]




#for i in range(3): obstacles_poses.append([-6,randint(-6,6),randint(-6,6)])
#for i in range(len(obstacles_poses)) : obstacles_dims.append([12,1,8])

obstacles = []
for pose, dim in zip(obstacles_poses, obstacles_dims):
	obstacles = add_obstacle(obstacles, pose, dim)


##################

init_fonts()
fig = plt.figure(figsize=(15,15))
ax = fig.add_subplot(1, 2, 1, projection='3d')
ax1 = fig.add_subplot(1, 2, 2, projection='3d')
ax.set_xlabel('X, [m]')
ax.set_ylabel('Y, [m]')
ax.set_zlabel('Z, [m]')
#ax.set_xlim([-2.5, 2.5])
#ax.set_ylim([-2.5, 2.5])
#ax.set_zlim([0.0, 3.0])
ax.set_xlim([0,30])
ax.set_ylim([0, 30])
ax.set_zlim([0, 30])

ax1.set_xlabel('X, [m]')
ax1.set_ylabel('Y, [m]')
ax1.set_zlabel('Z, [m]')
#ax.set_xlim([-2.5, 2.5])
#ax.set_ylim([-2.5, 2.5])
#ax.set_zlim([0.0, 3.0])
ax1.set_xlim([0,30])
ax1.set_ylim([0, 30])
ax1.set_zlim([0, 30])

for obstacle in obstacles: obstacle.draw(ax);obstacle.draw(ax1)

# parameters

animate = 1

# RRT Initialization
maxiters  = 500
nearGoal = False # This will be set to true if goal has been reached
minDistGoal = 0.05 # Convergence criterion: success when the tree reaches within 0.25 in distance from the goal.
d = 1 # [m], Extension parameter: this controls how far the RRT extends in each step.

# Start and goal positions
sequestro_rtt = np.array([10,13,5])
start = np.array([0, 0, 0]);ax.scatter3D(start[0], start[1], start[2], color='green', s=100);ax1.scatter3D(start[0], start[1], start[2], color='green', s=100)
goal =  np.array([15, 25, 15]);ax.scatter3D(goal[0], goal[1], goal[2], color='red', s=100);ax1.scatter3D(goal[0], goal[1], goal[2], color='red', s=100)
# Initialize RRT. The RRT will be represented as a 2 x N list of points.
# So each column represents a vertex of the tree.
starti = start;
rrt = []
start_node = Node3D()
start_node.p = start
start_node.i = 0
start_node.iPrev = 0
rrt.append(start_node)
#6,6,-7,7,4,-3

# RRT algorithm
start_time = time.time()
iters = 0
tempo = time.time()
tempo_novo = 0
primeiro = True
passos = 0

while neural or rtt_logic:
    

    if neural :
        
        if dinamico==True and cont == 10:
            # Acrescenta um elemento dinamico ao mapa 
            
            
            # dynamic(ax) --> nao existe necessidade, uma vez que ja foi contruido
            # o objeto dinamido em rtt, basta apenas desenha-lo em ax1
            
            
            ### range(quatidade de objetos adicionados)
            din_pose = [10,18,12]
            din_dim = [2,2,2]
            
            obstacles = add_obstacle(obstacles, din_pose, din_dim)
            obstacles[len(obstacles)-1].draw(ax1)   
            #for i in range(2):
                
            
            dinamico = False
        
        if kidnapped_neural == True and cont == 10:
            ## Coloca o robo em um ponto aleátorio
            #start = np.array([randint(-4,4),randint(-4,4),randint(-4,4)])
            start = sequestro_rtt
            ax1.scatter3D(start[0], start[1], start[2], color='green', s=100)
            ## Anula o sequestro
            kidnapped_neural = False
        
        if np.linalg.norm(goal-start) > 0.001:
            
            if primeiro:
                start = np.array([0,0,0])
                primeiro = False
            
            #sensors = choque(start,obstacles)

            s0,s1,s2,s3,s4,s5 = sensors(start)
            
            sensores = np.array([s0,s1,s2,s3,s4,s5])

            input = entrada(start,goal,sensores)

            out = model.predict(input)

            new_start = direcao(out,start)

            #print('start -->'); print(new_start)

            ax1.plot([start[0],new_start[0]],[start[1],new_start[1]],[start[2],new_start[2]],color ='red',linewidth=2)

            start = new_start

            passos+=0.5
            plt.pause(0.01)
        else:
            neural = False
            final_rede = time.time()
            print("Rede neural encontro a solução em -->");print(final_rede - start_time)
            print("Distância percorrida até o alvo -->");print(passos)
        cont+=1    
    
    
    din_time = time.time()
    din_time_novo = 0

    if rtt_logic:
        
        
        if dinamico==True and cont == 10:
            # Acrescenta um elemento dinamico ao mapa 
            
            dynamic(ax)
            
            
            dinamico = False



        if kidnapped_rtt == True and cont == 10:

            start = np.array([randint(0,30),randint(0,30),randint(0,30)])
            sequestro_rtt = start
            rrt = []
            start_node = Node3D()
            start_node.p = start
            start_node.i = 0
            start_node.iPrev = 0
            rrt.append(start_node)

            rnd = random()

            if rnd < 0.10:
                p = goal
            else:
                p = np.array([random()*5-2.5, random()*5-2.5, random()*3]) # Should be a 3 x 1 vector

            closest_node = closestNode3D(rrt, p)
            new_node = Node3D()
            new_node.p = closest_node.p + d * (p - closest_node.p)
            new_node.i = len(rrt)
            new_node.iPrev = closest_node.i
            ## Desenha  
            ax.scatter3D(start[0], start[1], start[2], color='green', s=100)
            kidnapped = False

        # Sample point
        rnd = random()
        # With probability 0.05, sample the goal. This promotes movement to the goal.
        if rnd < 0.10:
            p = goal
        else:
            p = np.array([random()*12-6, random()*12-6, random()*6]) # Should be a 3 x 1 vector

        # Check if sample is collision free
        collFree = isCollisionFreeVertex(obstacles, p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue

        # If it is collision free, find closest point in existing tree. 
        closest_node = closestNode3D(rrt, p)




        # Extend tree towards xy from closest_vert. Use the extension parameter
        # d defined above as your step size. In other words, the Euclidean
        # distance between new_vert and closest_vert should be d.
        new_node = Node3D()
        new_node.p = closest_node.p + d * (p - closest_node.p)
        new_node.i = len(rrt)
        new_node.iPrev = closest_node.i

        if animate:
            ax.plot([closest_node.p[0], new_node.p[0]], [closest_node.p[1], new_node.p[1]], [closest_node.p[2], new_node.p[2]],color = 'b', zorder=5)
          #  ax1.plot([closest_node.p[0], new_node.p[0]], [closest_node.p[1], new_node.p[1]], [closest_node.p[2], new_node.p[2]],color = 'b', zorder=5)
            plt.pause(0.01)

        # Check if new vertice is in collision
        collFree = isCollisionFreeEdge(obstacles, closest_node.p, new_node.p)
        # If it's not collision free, continue with loop
        if not collFree:
            iters += 1
            continue
        
        # If it is collision free, add it to tree    
        rrt.append(new_node)

        # Check if we have reached the goal
        if norm(np.array(goal) - np.array(new_node.p)) < minDistGoal:
            # Add last, goal node
            goal_node = Node3D()
            goal_node.p = goal
            goal_node.i = len(rrt)
            goal_node.iPrev = new_node.i
            if isCollisionFreeEdge(obstacles, new_node.p, goal_node.p):
                rrt.append(goal_node)
                P = [goal_node.p]
            else: P = []
            end_time = time.time()
            nearGoal = True
            rtt_logic = False
            neural = True
            dinamico =True
            cont = 0
            model = load_model('/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk.h5')
            print ('Reached the goal after %.2f seconds:' % (end_time - start_time))

        iters += 1
        din_time_novo = time.time()
        cont +=1 
    
 

if rtt_logic :

    print ('Number of iterations passed: %d / %d' %(iters, maxiters))
    print ('RRT length: ', len(rrt))

    # Path construction from RRT:
    print ('Constructing the path...')
    i = len(rrt) - 1
    while True:
        i = rrt[i].iPrev
        P.append(rrt[i].p)
        if i == 0:
            print ('Reached RRT start node')
            break
    P = np.array(P)
    # drawing a path from RRT
    for i in range(P.shape[0]-1):
        ax.plot([P[i,0], P[i+1,0]], [P[i,1], P[i+1,1]], [P[i,2], P[i+1,2]], color = 'g', linewidth=1, zorder=10)
        dist = dist + np.linalg.norm(P[i+1]-P[i])
        print("norma ->");print(dist)
    # shortened path
    print ('Shortening the path...')
    P = shorten_path(P, obstacles, smoothiters=100)
    for i in range(P.shape[0]-1):
        ax.plot([P[i,0], P[i+1,0]], [P[i,1], P[i+1,1]], [P[i,2], P[i+1,2]], color = 'orange', linewidth=1, zorder=15)


plt.show()  

