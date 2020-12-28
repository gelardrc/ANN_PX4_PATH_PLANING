import asyncio
from mavsdk import System
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


########## VARIŹVEIS GLOBAIS ###########

pontos_antigos = [999999,999999999,9999999999]
random.seed()
tempo = 0
ponto_inicial = [0,0,0]
t=0
way_point = []
distancia_velha = 99999999999
melhor_pai = [0,0,0]
best_pai = [0,0,0]
target = [-22.842171,-43.312731,10] # casa 
#target = [-22.853839,-43.313535,10] # shopping
soma = 0
primeira_vez = True
ax = plt.axes(projection='3d')


########################################

######### GENECTIC ALGORITHM ###########
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

def obstaculo(caminho):
    
    x0 = -22.841218
    y0 = -43.311702
    z0 =  10
    
    circle  = (caminho[0] - x0 )**2 + (caminho[1] - y0 )**2 + (caminho[2] - z0 )**2  
    
    if circle <= (0.001)**2 :
        return True
    return False 
     

def fitness(pai,target):
    if pai[2] == False :
        return False
    distancia = np.linalg.norm(np.array(target)-np.array(pai))
    global distancia_velha
    global best_pai
    #dentro = obstaculo(best_pai)
    if distancia < distancia_velha : #and dentro == False :
        best_pai = pai
       #waypoint(best_pai)
        distancia_velha = distancia
        return best_pai
    #distancia_velha = distancia
    return best_pai


def GA(ponto_inicial,passo):
    global escolha
    for i in range(40000):
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
########################################

############# MAIN FUNCTION ############
async def run():
    
    drone = System()
    await drone.connect(system_address="udp://:14540")

    #Obter localização
    
    position_inicial = await print_position(drone)
    print ("posição ok  --> " ,position_inicial)

    # fazer ga
    
    global target,melhor_pai,ponto_inicial,soma,primeira_vez
    ponto_inicial[0] = position_inicial[0]
    ponto_inicial[1] = position_inicial[1]
    ponto_inicial[2] = 10
    while np.linalg.norm(np.array(target)-np.array(melhor_pai)) > 0 :#29.999996 
        melhor_pai = GA(ponto_inicial,0.001)
      
      
        #plt.plot([melhor_pai[0],ponto_inicial[0]],[melhor_pai[1],ponto_inicial[1]],[melhor_pai[2],ponto_inicial[2]])
        #plt.pause(0.1)
      
        if melhor_pai == ponto_inicial : 
            soma = soma + 1
            if soma == 30:
                print("O melhor que consegui vlw irmão ")
                break 
        
        # Envia missão
        if primeira_vez == True:
            print("Waiting for drone to have a global position estimate...")
            async for health in drone.telemetry.health():
                if health.is_global_position_ok:
                    print("Global position estimate ok")
                    break

            print("-- Arming")
            await drone.action.arm()

            print("-- Taking off")
            await drone.action.takeoff()

            await asyncio.sleep(1)
            primeira_vez = False
        print("Enviando missão ----")
        print(np.linalg.norm(np.array(target)-np.array(melhor_pai)))
        await drone.action.goto_location(melhor_pai[0],melhor_pai[1],melhor_pai[2], float('nan'))

        # Atualiza o ponto
        ponto_inicial = melhor_pai
    
    print("-- Taking off")
    await drone.action.land()
   # await drone.action.disarm()
    print("pousei em -- >",melhor_pai) 
    
    

########################################

########### DRONE ######################    
async def print_position(drone):
    async for position in drone.telemetry.position():
        #print(f"eita nos : {position.latitude_deg}")
        return [position.latitude_deg,position.longitude_deg,position.absolute_altitude_m]
########################################

if __name__ == "__main__":
    # Start the main function
    asyncio.ensure_future(run())

    # Runs the event loop until the program is canceled with e.g. CTRL-C
    asyncio.get_event_loop().run_forever()
