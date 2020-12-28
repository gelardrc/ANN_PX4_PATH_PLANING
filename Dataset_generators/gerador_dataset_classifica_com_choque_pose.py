# Nova rede_neural baseado na reunião 

# Drone vai ter 6 opções de movimentação em um grid de 1
#
#   Frente   - 0 0 0 0 0 0   x 
#   Tras     - 0 1 0 0 0 0  -x
#   Esquerda - 0 0 1 0 0 0   z
#   Direita  - 0 0 0 1 0 0  -z
#   Cima     - 0 0 0 0 1 0   y
#   Baixo    - 0 0 0 0 0 1  -y 
#
# Devera ser realizado um teste pra saber qual das ações acima sera a melhor escolha 
# para resolver a função de custo
# 
#   L = K ( fator_de_distânciamento ) * distância_objeto / distancia_alvo 
#
#


########## Libs ############
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
import pandas as pd
############################

#### Variavéis globais ####
first = True
distancia_antiga = 0
posicao = [0,0,0]
sensor = [0,0,0]
acao = []
mapa = 50
melhor = None
choque = []
[choque0_x,choque0_y,choque0_z,choque1_x,choque1_y,choque1_z,choque2_x,choque2_y,choque2_z,choque3_x,choque3_y,choque3_z,choque4_x,choque4_y,choque4_z,choque5_x,choque5_y,choque5_z] = [[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]]
[px,py,pz,alvox,alvoy,alvoz,acao0,acao1,acao2,acao3,acao4,acao5]=[[],[],[],[],[],[],[],[],[],[],[],[]]
###########################

# Definir os quadrados que estaram ocupados

def objetos():
    obj = []
        
    for i in range(9000):
        
        obj.append([random.randint(50),random.randint(50),random.randint(50)])
        
        #draw(obj[i],'y^')
        
    return obj

def mapa():

    ax = plt.axes(projection='3d')
  
    ax.set_xlim3d(0,50)
    ax.set_ylim3d(0,50)
    ax.set_zlim3d(0,50)    

    return 0 

def visao(posicao,objetos,original):
    
    grid = 1 
    
    lista = [-grid,grid]
    
    for t in range(3):
        sensor = original
        for i in lista:
            sensor[t] = posicao[t] + i
            print (sensor)
            teste = False
            for l in range(len(objetos)):
                teste = np.array_equal(sensor,objetos[l])
            if teste == False:
                possivel = avalia(sensor)            
            posicao = original
    return possivel

def draw(desenho,cor):
    plt.plot([desenho[0]],[desenho[1]],[desenho[2]],cor)

    return 0

def avalia(best,target):
    ## Avaliação sera dada a principio como o ponto mais proximo do target
    global first
    global distancia_antiga
    global melhor
    if first == True:
        distancia_antiga = 999
        first = False

    distancia = np.linalg.norm(target-best)
    if distancia < distancia_antiga:
         melhor = best
         distancia_antiga = distancia 
    
    return melhor

def lusitano(grid,onde_estou,objetos,target):
    global lista
    teste = False
    global k
    k = [] 
    global choque
    lista = [[0,0,grid],[0,0,-grid],[0,grid,0],[0,-grid,0],[grid,0,0],[-grid,0,0]]
    for i in lista: 
        for l in range(len(objetos)):
            teste = np.array_equal(np.array(onde_estou)+np.array(i),objetos[l])
            # Verificar isso mais tarde
            if teste == True:
                k.extend(objetos[l])
        if teste == False:
            vencedor = avalia(np.array(onde_estou)+np.array(i),target = target)
            k.extend([0,0,0]) 
            
    return vencedor

def action(winner,pose):
    mover = [0,0 ,0, 0, 0 ,0]
    
    direcao = np.array(winner) - np.array(pose)    
    
    l = 0

    for i in lista:
        teste = np.array_equal(np.array(direcao),np.array(i))
        if teste == True:
            mover[l] = 1
            luz = mover  
        l = l+1
    return luz

def banco_de_dados(px,
                py,
                pz,
                alvox,
                alvoy,
                alvoz,
                choque0_x,
                choque0_y,
                choque0_z,
                choque1_x,
                choque1_y,
                choque1_z,
                choque2_x,
                choque2_y,
                choque2_z,
                choque3_x,
                choque3_y,
                choque3_z,
                choque4_x,
                choque4_y,
                choque4_z,
                choque5_x,
                choque5_y,
                choque5_z,
                acao0,
                acao1,
                acao2,
                acao3,
                acao4,
                acao5):
    
    local = {'px' :px,
            'py' :py,
            'pz' :pz,
            'alvox' :alvox,
            'alvoy' :alvoy,
            'alvoz' :alvoz,
            'choque0_x':choque0_x,
            'choque0_y':choque0_y,
            'choque0_z':choque0_z,
            'choque1_x':choque1_x,
            'choque1_y':choque1_y,
            'choque1_z':choque1_z,
            'choque2_x':choque2_x,
            'choque2_y':choque2_y,
            'choque2_z':choque2_z,
            'choque3_x':choque3_x,
            'choque3_y':choque3_y,
            'choque3_z':choque3_z,
            'choque4_x':choque4_x,
            'choque4_y':choque4_y,
            'choque4_z':choque4_z,
            'choque5_x':choque5_x,
            'choque5_y':choque5_y,
            'choque5_z':choque5_z,
            'acao0' :acao0,
            'acao1' :acao1,
            'acao2' :acao2,
            'acao3' :acao3,
            'acao4' :acao4,
            'acao5' :acao5
        }

    df = pd.DataFrame(local, columns = ['px',
                                        'py',
                                        'pz',
                                        'alvox',
                                        'alvoy',
                                        'alvoz',
                                        'choque0_x',
                                        'choque0_y',
                                        'choque0_z',
                                        'choque1_x',
                                        'choque1_y',
                                        'choque1_z',
                                        'choque2_x',
                                        'choque2_y',
                                        'choque2_z',
                                        'choque3_x',
                                        'choque3_y',
                                        'choque3_z',
                                        'choque4_x',
                                        'choque4_y',
                                        'choque4_z',
                                        'choque5_x',
                                        'choque5_y',
                                        'choque5_z',
                                        'acao0',
                                        'acao1',
                                        'acao2',
                                        'acao3',
                                        'acao4',
                                        'acao5'])

    print(df)
    print(df.dtypes)

    df.to_csv('dataset_classificador_com_choque_30000.csv', index=False)
    
    return 0 


## MAIN ##

mapa()

alvo = []
#for i in range(5)

vision = []

luz=[]

pose = []

obj = objetos()

for i in range(30000):

    print('faltam --> %d ', 30000-i )

    choque = []

    first = True

    alvo = [random.randint(50),random.randint(50),random.randint(50)]

    alvox.append(alvo[0])
    alvoy.append(alvo[1])
    alvoz.append(alvo[2])
    
    drone  = [random.randint(50),random.randint(50),random.randint(50)]
        
    #draw(alvo,'ro')

    #draw(drone,'bs')

    pose = lusitano(grid = 1,onde_estou = drone,objetos = obj,target = alvo)  
    
    px.append(drone[0])
    py.append(drone[1])
    pz.append(drone[2])

    acao = action(winner = melhor, pose = drone)
    
    acao0.append(acao[0])
    acao1.append(acao[1])
    acao2.append(acao[2])
    acao3.append(acao[3])
    acao4.append(acao[4])
    acao5.append(acao[5])

    choque0_x.append(k[0])
    print(k) 
    choque0_y.append(k[1]) 
    choque0_z.append(k[2]) 
    choque1_x.append(k[3]) 
    choque1_y.append(k[4]) 
    choque1_z.append(k[5]) 
    choque2_x.append(k[6]) 
    choque2_y.append(k[7]) 
    choque2_z.append(k[8]) 
    choque3_x.append(k[9]) 
    choque3_y.append(k[10])
    choque3_z.append(k[11])
    choque4_x.append(k[12])
    choque4_y.append(k[13])
    choque4_z.append(k[14])
    choque5_x.append(k[15])
    choque5_y.append(k[16])
    choque5_z.append(k[17])
     

banco_de_dados(px,
                py,
                pz,
                alvox,
                alvoy,
                alvoz,
                choque0_x,
                choque0_y,
                choque0_z,
                choque1_x,
                choque1_y,
                choque1_z,
                choque2_x,
                choque2_y,
                choque2_z,
                choque3_x,
                choque3_y,
                choque3_z,
                choque4_x,
                choque4_y,
                choque4_z,
                choque5_x,
                choque5_y,
                choque5_z,
                acao0,
                acao1,
                acao2,
                acao3,
                acao4,
                acao5)

#plt.show()    