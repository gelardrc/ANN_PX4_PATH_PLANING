class quadrado:
    def __init__(self):
        ## heuristica ##
        self.h = 0
        self.g = 0
        self.f = 0
        self.parent = [0,0]
        self.ponto =[0,0]
    
    def adicionar(self):
        
        vec.append(self)
    def __eq__(self,other):
        return self.ponto == other.ponto
        
        


vec = []
movimento = [[1,0],[-1,0],[0,1],[0,-1]]

comeco = quadrado()
comeco.ponto = [1,1]
alvo = quadrado()
alvo.ponto = [2,4]

open_list = []

primeiro = True

#open_list.append(comeco)

#for index,aberto in enumerate(open_list):

for x in range(5):
    for y in range(4):
        open_list.append([x,y])
    

for index,value in enumerate(open_list):
    
    for mov in movimento :
        
        node = quadrado()
        node.ponto = [value[0] + mov[0],value[1]+mov[1]]
        if node.ponto == value:
            continue
        node.parent = value    
        node.g = (node.ponto[0] - comeco.ponto[0])**2 + (node.ponto[1] - comeco.ponto[1])**2
        node.h =  (node.ponto[0] - alvo.ponto[0])**2 + (node.ponto[1] - alvo.ponto[1])**2 #+ ((node.ponto[0] - alvo.ponto[0))**2
        node.f = node.g+node.f
        
        if node == alvo:
            node.g=0
            node.h=0
            node.f=0
        
        
        if node.ponto[0]>4 and node.ponto[1]>3:
            break
    
        if node.ponto[0]<0 or node.ponto[1]<0:
            continue
            
        node.adicionar()

pais = []
dist = 9999999999

for value in vec : 
    if value == alvo:
        pais.append(value)
        for i in pais:
            if i.f < dist
                
                dist = i.f
            
        
        
        

        
        
        
            








#while len(open_list)>0:
#    
#    for index,aberto in enumerate(open_list): 
#        
#        for mov in movimento:
#            
#            if primeiro:
#                soma = [aberto.ponto[0]+mov[0],aberto.ponto[1]+mov[1]]
#                vetor = quadrado()
#                vetor.ponto = soma
#                vetor.h = (soma[0] - aberto.ponto[0])**2 + (soma[1] - aberto.ponto[1])**2
#                vetor.parent = aberto.ponto   
#                vetor.adicionar()
#                open_list.append(vetor)
#            else:
#                soma = [aberto.ponto[0]+mov[0],aberto.ponto[1]+mov[1]]
#                if soma[0] == aberto.parent[0] and soma[1]==aberto.parent[1]: continue
#                vetor = quadrado()
#                vetor.ponto = soma
#                vetor.h = (soma[0] - aberto.ponto[0])**2 + (soma[1] - aberto.ponto[1])**2
#                vetor.parent = aberto.ponto
#                if soma[0]>2 and soma[1]>2: continue
#                if soma[0]<-2 and soma[1]<-2: continue
#                vetor.adicionar()
#                open_list.append(vetor)
#                print(len(open_list))
#
#        open_list.pop(index)
#
#
#        primeiro = False
#   
#print(vec)
#