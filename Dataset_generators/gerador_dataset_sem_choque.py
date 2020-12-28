# Teste de esfera
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
import csv

def csv(px,py,pz,tx,ty,tz,ox,oy,oz):

    local = {'px' : px,
            'py' : py,
            'pz' : pz,
            'tx' : tx,
            'ty' : ty,
            'tz' : tz,
            'ox' : ox,
            'oy' : oy,
            'oz' : oz
                }

    print(len(px),len(py),len(pz),len(tx),len(ty),len(tz),len(ox),len(oy),len(oz))
    df = pd.DataFrame(local, columns = ['px','py','pz','tx','ty','tz','ox','oy','oz'])

    #df = pd.DataFrame(data=local,    # values
    #            index=['Pose',  'Target','Output'],    # 1st column as index
    #            columns=['Pose',  'Target','Output'])  # 1st row as the column names

    
    print (df.dtypes)
    print (df)

    df.to_csv('dataset_completo_50.csv', index=False)
    
    return 0 

def waypoint(pontos):
    global tempo
    
    global way_point
    
    global pontos_antigos 

    if pontos == pontos_antigos:
        return
    
    way_point.append(pontos)
    print(way_point)
    pontos_antigos = pontos
    
    
def root(a,b,c):
    delta = (b**2 - 4*a*c)
    if delta < 0:
        z = False
        return z 
    s1 = (-b + (delta)**1/2)/2*a
    s2 = (-b - (delta)**1/2)/2*a
    raizes  = [s1,s2]
    z = random.choice(raizes)
    return z


def fitness(pai,target):
    if pai[2] == False :
        return False
    distancia = np.linalg.norm(np.array(target)-np.array(pai))
    global distancia_velha
    global best_pai
    if distancia < distancia_velha:
        best_pai = pai
       #waypoint(best_pai)
        distancia_velha = distancia
        return best_pai
    #distancia_velha = distancia
    return best_pai


def GA(ponto_inicial,passo):
    global escolha
    for i in range(3000):
        x = ponto_inicial[0]+random.triangular(-passo,passo)
        y = ponto_inicial[1]+random.triangular(-passo,passo)
        c = (x-ponto_inicial[0])**2 +(y-ponto_inicial[1])**2 + ponto_inicial[2]**2 -passo**2
        a = 1
        b = -2 * ponto_inicial[2]
        z = root(a,b,c) 
        if z == False:
            continue
        escolha = fitness([x,y,z],target)
    
    return escolha


def mutacao(coco):
    
    pega_alguem = random.choice(coco)
    pega_alguem2 = pega_alguem + random.triangular(-0.1,0.1)
    for i in range(3):
        if coco[i] == pega_alguem:
            coco[i]= pega_alguem2
            num = coco
            print(num)
            break
    celula_mutante = fitness(num,[3,3,3])
    
    return celula_mutante

pontos_antigos = [999999,999999999,9999999999]
random.seed()
tempo = 0
ponto_inicial = [0,0,0]
t=0
way_point = []
distancia_velha = 99999999999
ax = plt.axes(projection='3d')
pose = []
targets = []
output = []
[px,py,pz,tx,ty,tz,ox,oy,oz]=[[],[],[],[],[],[],[],[],[]]
melhor_pai = [0,0,0]
target = [5,5,5]
lista = []
teta = 0
#lista = [[5,5,5],[10,12,13],[20,20,10],[22,11,33],[15,15,15],[13,13,13],[15,16,8]]
for i in range(50):
    lista.append([random.triangular(0,30),random.triangular(0,30),random.triangular(0,30)])

for target in lista:
    print(teta)
    teta = teta + 1
    soma = 0
    melhor_pai = [0,0,0]
    ponto_inicial = [random.triangular(0,10),random.triangular(1,10),random.triangular(1,10)]
    distancia_velha = 99999999999
    while np.linalg.norm(np.array(target)-np.array(melhor_pai)) > 0.06 :
        
    
        melhor_pai = GA(ponto_inicial,1)
        plt.plot([melhor_pai[0],ponto_inicial[0]],[melhor_pai[1],ponto_inicial[1]],[melhor_pai[2],ponto_inicial[2]])
        #px.append(ponto_inicial[0]) 
        #output.append(melhor_pai) 
        #targets.append(target)
        
        px.append(ponto_inicial[0])
        py.append(ponto_inicial[1])
        pz.append(ponto_inicial[2])
        tx.append(melhor_pai[0])
        ty.append(melhor_pai[1])
        tz.append(melhor_pai[2])
        ox.append(target[0])
        oy.append(target[1])
        oz.append(target[2])

        plt.pause(0.1)
        if melhor_pai == ponto_inicial : 
            soma = soma + 1
            if soma == 20:
                break 
        ponto_inicial = melhor_pai
 

plt.plot(target[0] ,target[1] ,target[2] ,'X')
csv(px = px,
    py = py,
    pz = pz,
    tx = tx,
    ty = ty,
    tz = tz,
    ox = ox,
    oy = oy,    
    oz = oz
)

plt.show()




