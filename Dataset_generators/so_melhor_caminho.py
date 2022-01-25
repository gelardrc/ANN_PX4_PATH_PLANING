import numpy as np 
import random as rnd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd



def constroi_grafico():

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    return ax

def mundo():

    ## Define tamanho do mundo ##

    x = np.zeros([1,60])
    y = np.zeros([1,60])
    z = np.zeros([1,60])
    
    return x,y,z
    ###############################

def objs(x,y,z,numero_de_obj):
    
    #numero_de_obj = 10
    
    X = []
    Y = []
    Z = []
    obj = []
    for i in range(numero_de_obj):
        um = rnd.randint(0,59) 
        dois = rnd.randint(0,59)
        tres = rnd.randint(0,59)
        x[0,um] = 1
        y[0,dois] = 1
        z[0,tres] = 1
        obj.append([um,dois,tres])
        X.append(um)
        Y.append(dois)
        Z.append(tres)
    
    return x,y,z,X,Y,Z,obj

def sensor(pose,field,alvo):
    global saidacsv
    direcao  = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    ex = [0,0,0,0,0,0]
    soma = [ ]
    melhor = [ ]
    menor = 999
    valor = 999 
    
    for i in direcao:
        soma.append([pose[0]+i[0],pose[1]+i[1],pose[2]+i[2]])

    for i in soma: 
        if i in field: 
            soma.remove(i) 
            #print("peguei")
    
    for index,i in enumerate(soma):
        
        melhor.append(np.linalg.norm(np.array(alvo)-np.array(i)))  
        if np.linalg.norm(np.array(alvo)-np.array(i))<menor:
            menor = np.linalg.norm(np.array(alvo)-np.array(i))
            valor = index
    
    x = np.array(soma[valor])-np.array(pose)
    for index,value in enumerate(direcao):
        if (x == value).all():
            valor2= index

    ex[valor2] = 1

    saidacsv.append(ex)

    return soma[valor]

def em_volta(pose,obj):
    direcao  = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
    soma = []
    em_volta = [0,0,0,0,0,0]
    for i in direcao:
        soma.append([i[0]+pose[0],i[1]+pose[1],i[2]+pose[2]])
    for i,value in enumerate(soma):
        if value in obj: em_volta[i] = 1
    return em_volta

def banco_de_dados(pose,alvo,sensores,acao):
    
    px = []
    py = []
    pz = []

    for i in pose:
      px.append(i[0])
      py.append(i[1])
      pz.append(i[2])      
    
    alvox = []
    alvoy = []
    alvoz = []

    for i in alvo:
      alvox.append(i[0])
      alvoy.append(i[1])
      alvoz.append(i[2])     
    
    choque0=[]
    choque1=[]
    choque2=[]
    choque3=[]
    choque4=[]
    choque5=[]
    
   
    for i in sensores:

        choque0.append(i[0])
        choque1.append(i[1])
        choque2.append(i[2])
        choque3.append(i[3])
        choque4.append(i[4])
        choque5.append(i[5])    

    acao0=[]
    acao1=[]
    acao2=[]
    acao3=[]
    acao4=[]
    acao5=[]

    for i in acao:

        acao0.append(i[0])
        acao1.append(i[1])
        acao2.append(i[2])
        acao3.append(i[3])
        acao4.append(i[4])
        acao5.append(i[5])

    
    local = {'px' :px,
            'py' :py,
            'pz' :pz,
            'alvox' :alvox,
            'alvoy' :alvoy,
            'alvoz' :alvoz,
            'choque0' : choque0,
            'choque1' : choque1,
            'choque2' : choque2,
            'choque3' : choque3,
            'choque4' : choque4,
            'choque5' : choque5,
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

    #dataset = pd.read_csv("/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/simples_revisor.csv")    
    #new_data = dataset.to_pd
    
    #print("1",dataset.shape)

    #dataset.append(df)
    
    #print("2",dataset.shape)

    print(df)
    print(df.dtypes)

    df.to_csv('/home/gelo/codes/ANN_PX4_PATH_PLANING/DATASETS/simples_revisor2.csv', index=False)
    
    


## Só faz o basico do gráfico ##

ax  = constroi_grafico()

## Estipula o mundo ###

x,y,z = mundo()

## Constroi os objetos ###

x,y,z,X,Y,Z,obj = objs(x = x,y = y,z = z,numero_de_obj = 8000)


#### Variaveis para plotagem, provavelmente serão apagadas depois ####

list_x = x[0,:]
list_y = y[0,:]
list_z = z[0,:]


### Cria a variável field ###

field = zip(x[0,:],y[0,:],z[0,:])

### Inicializa os vetores path, sensores, comeco e stop ##

resposta = []
path_csv = []
sensores = []
alvo_csv = []
comeco = [ ]
stop  = []
saidacsv = []
direcao  = [[0,0,1],[0,0,-1],[0,1,0],[0,-1,0],[1,0,0],[-1,0,0]]
saida = [0,0,0,0,0,0]

for i in range(30):
    ## Inicializa ##
    value_comeco = [rnd.randint(0,59),rnd.randint(0,59),rnd.randint(0,59)]
    value_alvo = [rnd.randint(0,59),rnd.randint(0,59),rnd.randint(0,59)]
    
    ## Esses dois whiles são pra  garantir que nao escolhemos um ponto que esta ocupado por um objeto ##
    while value_alvo in field : valeu_alvo = [rnd.randint(0,59),rnd.randint(0,59),rnd.randint(0,59)]
    while value_comeco in field : value_comeco = [rnd.randint(0,59),rnd.randint(0,59),rnd.randint(0,59)]
    
    ### Se passou desses dois while é sinal que os pontos escolhidos estao livre e podem sem adicionados em comeco e stop ###
    comeco.append(value_comeco)
    stop.append(value_alvo)

cont = 0

for start,alvo in zip(comeco,stop):
    path = []
    print(cont)
    while np.linalg.norm(np.array(start)-np.array(alvo))>0.01:
        
        ## Adiciona o ponto no path    
        path.append(start)
        
        ## Variavel auxiliar para salvar o banco ##
        path_csv.append(start)
        
        ## Alvo csv e so uma variável para salvar depois no csv ##
        alvo_csv.append(alvo)

        ## sensor retorna a proxima posição que será tomada ##
        caminho = sensor(start,obj,alvo)
        
        ## Atualiza start ##
        start = caminho

    
    cont += 1 
    
    for index,value in enumerate(path) : 
        sensores.append(em_volta(value,obj))
        #if (index+1)<len(path): 
        #    resposta  = np.array(path[index+1]) - np.array(path[index])
        #    for i,v in enumerate(direcao): 
        #        if np.linalg.norm(np.array(v)-np.array(resposta)) == 0: saida[i] = 1
        #        else : saida[i] = 0
        #    saidacsv.append([saida])
        #else: 
        #    resposta = np.array(alvo) - np.array(path[index])
        #    for i,v in enumerate(direcao): 
        #        if np.linalg.norm(np.array(v)-np.array(resposta)) == 0: saida[i] = 1
        #        else : saida[i] = 0
        #    saidacsv.append([saida])
                
       


banco_de_dados(path_csv,alvo_csv,sensores,saidacsv)

x_l = []
y_l = []
z_l = []


#print(path)

for i in path_csv:
    x_l.append(i[0])
    y_l.append(i[1])        
    z_l.append(i[2])


ax.plot3D(x_l,y_l,z_l,color = 'blue')
#ax.scatter3D(X,Y,Z)



plt.show()


#print(x)