import numpy as np
import matplotlib.pyplot as plt


score_original_primeiro = [[1.5109407901763916, 0.4535999894142151], 
         [2.646571397781372, 0.39061665534973145], 
         [1.5729022026062012, 0.44084998965263367], 
         [1.663025140762329, 0.3280999958515167], 
         [1.8154751062393188, 0.40246665477752686], 
         [1.4361085891723633, 0.4594166576862335], 
         [1.4679596424102783, 0.4447999894618988], 
         [1.912434458732605, 0.17260000109672546], 
         [1.8540436029434204, 0.4322499930858612], 
         [1.5679147243499756, 0.445499986410141], 
         [1.8165839910507202, 0.17315000295639038], 
         [2.1842033863067627, 0.41003334522247314], 
         [1.9831740856170654, 0.4261666536331177], 
         [1.5798989534378052, 0.44165000319480896], 
         [2.3628950119018555, 0.4211333394050598], 
         [1.5432887077331543, 0.45225000381469727], 
         [1.6373761892318726, 0.42891666293144226], 
         [1.4657652378082275, 0.44973334670066833], 
         [1.4364070892333984, 0.4481000006198883], 
         [3.35660457611084, 0.38393333554267883], 
         [1.723507285118103, 0.4435666799545288], 
         [2.9960579872131348, 0.39559999108314514], 
         [9.49194622039795, 0.20293332636356354], 
         [2.2748539447784424, 0.39401665329933167], 
         [1.792739987373352, 0.1725333333015442], 
         [4.737788677215576, 0.3792666792869568], 
         [1.8335613012313843, 0.4237000048160553], 
         [1.47469961643219, 0.429833322763443], 
         [1.4912437200546265, 0.43665000796318054], 
         [1.4917047023773193, 0.4517333209514618]]


score_melhores_primeiro = [[1.4361085891723633, 0.4594166576862335]]


score_original_segundo = [[2.0934996604919434, 0.39176666736602783], [1.8157597780227661, 0.40966665744781494], [1.9674463272094727, 0.41040000319480896], [1.4508565664291382, 0.42498332262039185], [2.0650479793548584, 0.24393333494663239], [1.4677188396453857, 0.4572666585445404], [1.7434747219085693, 0.4354499876499176], [1.9655520915985107, 0.41324999928474426], [2.0597658157348633, 0.4105333387851715], [2.8365023136138916, 0.32368332147598267], [1.5172231197357178, 0.43221667408943176], [3.097520351409912, 0.3882833421230316], [2.653599262237549, 0.4032333195209503], [2.364124059677124, 0.3628166615962982], [1.718234658241272, 0.4483500123023987], [1.521957516670227, 0.40996667742729187], [3.5555672645568848, 0.39773333072662354], [1.7076324224472046, 0.43211665749549866], [1.461383581161499, 0.4500333368778229], [1.611161231994629, 0.40933331847190857], [2.8693013191223145, 0.3992166519165039], [5.446191310882568, 0.3738499879837036], [2.5275557041168213, 0.382016658782959], [1.5010013580322266, 0.4265500009059906], [1.6991972923278809, 0.41343334317207336], [1.4766604900360107, 0.40353333950042725], [1.554783582687378, 0.4424000084400177], [3.5566892623901367, 0.39329999685287476], [1.449701189994812, 0.4508666694164276], [1.7588149309158325, 0.25521665811538696]]

score_melhores_segundo = [ 
                         [1.4677188396453857, 0.4572666585445404], 
                         [1.461383581161499, 0.4500333368778229], 
                         [1.449701189994812, 0.4508666694164276] 
                         ]



layers = [['Nºlayers:', 1, 'Nºneurons:', [256], 'Adagrad', 'Batch size :', 300], ['Nºlayers:', 4, 'Nºneurons:', [155, 21, 86, 233], 'RMSprop', 'Batch size :', 550], ['Nºlayers:', 6, 'Nºneurons:', [96, 60, 236, 74, 256, 68], 'Adagrad', 'Batch size :', 200], ['Nºlayers:', 3, 'Nºneurons:', [199, 204, 146], 'Ftrl', 'Batch size :', 500], ['Nºlayers:', 3, 'Nºneurons:', [27, 16, 83], 'RMSprop', 'Batch size :', 800], ['Nºlayers:', 3, 'Nºneurons:', [81, 212, 29], 'Ftrl', 'Batch size :', 500], ['Nºlayers:', 1, 'Nºneurons:', [212], 'Ftrl', 'Batch size :', 400], ['Nºlayers:', 4, 'Nºneurons:', [68, 236, 70, 205], 'Ftrl', 'Batch size :', 200], ['Nºlayers:', 2, 'Nºneurons:', [221, 53], 'Nadam', 'Batch size :', 800], ['Nºlayers:', 6, 'Nºneurons:', [201, 202, 247, 102, 56, 209], 'Adagrad', 'Batch size :', 500], ['Nºlayers:', 4, 'Nºneurons:', [217, 147, 36, 81], 'Ftrl', 'Batch size :', 400], ['Nºlayers:', 1, 'Nºneurons:', [183], 'Adam', 'Batch size :', 150], ['Nºlayers:', 1, 'Nºneurons:', [201], 'Nadam', 'Batch size :', 350], ['Nºlayers:', 6, 'Nºneurons:', [118, 225, 198, 192, 112, 78], 'Adadelta', 'Batch size :', 150], ['Nºlayers:', 3, 'Nºneurons:', [50, 225, 8], 'Adam', 'Batch size :', 750], ['Nºlayers:', 4, 'Nºneurons:', [124, 156, 117, 82], 'Adagrad', 'Batch size :', 550], ['Nºlayers:', 5, 'Nºneurons:', [150, 152, 124, 241, 41], 'Adagrad', 'Batch size :', 550], ['Nºlayers:', 1, 'Nºneurons:', [242], 'Ftrl', 'Batch size :', 850], ['Nºlayers:', 1, 'Nºneurons:', [176], 'Adadelta', 'Batch size :', 850], ['Nºlayers:', 6, 'Nºneurons:', [226, 183, 198, 89, 55, 6], 'Adam', 'Batch size :', 950], ['Nºlayers:', 1, 'Nºneurons:', [162], 'Adam', 'Batch size :', 850], ['Nºlayers:', 6, 'Nºneurons:', [65, 49, 233, 134, 248, 37], 'Adamax', 'Batch size :', 600], ['Nºlayers:', 2, 'Nºneurons:', [24, 195], 'Adadelta', 'Batch size :', 450], ['Nºlayers:', 1, 'Nºneurons:', [184], 'RMSprop', 'Batch size :', 100], ['Nºlayers:', 4, 'Nºneurons:', [193, 7, 225, 4], 'Ftrl', 'Batch size :', 400], ['Nºlayers:', 6, 'Nºneurons:', [127, 109, 117, 127, 65, 152], 'Nadam', 'Batch size :', 300], ['Nºlayers:', 3, 'Nºneurons:', [227, 216, 98], 'SGD', 'Batch size :', 250], ['Nºlayers:', 2, 'Nºneurons:', [54, 252], 'Adagrad', 'Batch size :', 550], ['Nºlayers:', 4, 'Nºneurons:', [134, 83, 197, 128], 'Adadelta', 'Batch size :', 350], ['Nºlayers:', 2, 'Nºneurons:', [136, 143], 'Adagrad', 'Batch size :', 400]]

layers2 = [['Nºlayers:', 2, 'Nºneurons:', [146, 158], 'SGD', 'Batch size :', 100], ['Nºlayers:', 3, 'Nºneurons:', [209, 143, 41], 'SGD', 'Batch size :', 700], ['Nºlayers:', 2, 'Nºneurons:', [149, 13], 'Adamax', 'Batch size :', 150], ['Nºlayers:', 2, 'Nºneurons:', [211, 47], 'Adadelta', 'Batch size :', 600], ['Nºlayers:', 3, 'Nºneurons:', [184, 214, 22], 'Adadelta', 'Batch size :', 350], ['Nºlayers:', 1, 'Nºneurons:', [166], 'Adagrad', 'Batch size :', 600], ['Nºlayers:', 2, 'Nºneurons:', [160, 91], 'Adam', 'Batch size :', 750], ['Nºlayers:', 1, 'Nºneurons:', [220], 'RMSprop', 'Batch size :', 550], ['Nºlayers:', 4, 'Nºneurons:', [109, 54, 53, 76], 'SGD', 'Batch size :', 150], ['Nºlayers:', 5, 'Nºneurons:', [239, 14, 136, 3, 130], 'Nadam', 'Batch size :', 300], ['Nºlayers:', 6, 'Nºneurons:', [90, 29, 79, 110, 241, 61], 'Adagrad', 'Batch size :', 550], ['Nºlayers:', 5, 'Nºneurons:', [62, 136, 77, 118, 3], 'Nadam', 'Batch size :', 650], ['Nºlayers:', 3, 'Nºneurons:', [58, 224, 30], 'Adam', 'Batch size :', 100], ['Nºlayers:', 5, 'Nºneurons:', [245, 163, 135, 219, 225], 'SGD', 'Batch size :', 450], ['Nºlayers:', 2, 'Nºneurons:', [100, 237], 'Adamax', 'Batch size :', 650], ['Nºlayers:', 2, 'Nºneurons:', [62, 73], 'Adagrad', 'Batch size :', 700], ['Nºlayers:', 5, 'Nºneurons:', [213, 21, 217, 127, 198], 'RMSprop', 'Batch size :', 450], ['Nºlayers:', 1, 'Nºneurons:', [115], 'RMSprop', 'Batch size :', 900], ['Nºlayers:', 1, 'Nºneurons:', [218], 'Adagrad', 'Batch size :', 500], ['Nºlayers:', 1, 'Nºneurons:', [182], 'SGD', 'Batch size :', 900], ['Nºlayers:', 4, 'Nºneurons:', [128, 197, 16, 71], 'Nadam', 'Batch size :', 550], ['Nºlayers:', 5, 'Nºneurons:', [155, 256, 53, 230, 80], 'Nadam', 'Batch size :', 350], ['Nºlayers:', 6, 'Nºneurons:', [229, 230, 251, 26, 204, 21], 'RMSprop', 'Batch size :', 100], ['Nºlayers:', 4, 'Nºneurons:', [41, 135, 65, 210], 'Adadelta', 'Batch size :', 250], ['Nºlayers:', 6, 'Nºneurons:', [38, 67, 59, 50, 190, 61], 'Adagrad', 'Batch size :', 100], ['Nºlayers:', 2, 'Nºneurons:', [143, 60], 'Adadelta', 'Batch size :', 650], ['Nºlayers:', 4, 'Nºneurons:', [53, 85, 139, 60], 'Adagrad', 'Batch size :', 450], ['Nºlayers:', 5, 'Nºneurons:', [205, 38, 234, 216, 188], 'RMSprop', 'Batch size :', 750], ['Nºlayers:', 1, 'Nºneurons:', [226], 'Ftrl', 'Batch size :', 600], ['Nºlayers:', 6, 'Nºneurons:', [8, 96, 148, 51, 109, 131], 'Adadelta', 'Batch size :', 350]]

save_index_primeiro = []
save_index_segundo =[]

for index,value in enumerate(score_original_primeiro):
    um = score_melhores_primeiro[0] 
    
    if value[0] == um[0] and value[1] == um[1]:
        save_index_primeiro .append(index)
    

for index,value in enumerate(score_original_segundo):
    um = score_melhores_segundo[0] 
    dois = score_melhores_segundo[1]
    tres = score_melhores_segundo[2]
    
    if value[0] == um[0] and value[1] == um[1]:
        save_index_segundo.append(index)
    if value[0] == dois[0] and value[1] == dois[1]:
        save_index_segundo.append(index)
    if value[0] == tres[0] and value[1] == tres[1]:
        save_index_segundo.append(index)
    
 # [5, 6, 17, 18] --> melhores do primeiro       
#print('primeiro -->',save_index_primeiro)
#print('segundo -->',save_index_segundo)

primeiro = [5]
segundo = [5,18,28]


#for i in score_original_primeiro:
#    loss.append(i[0])
#    accuracia.append(i[1])
score = []
labels = []
loss =[]
accuracia = []
score.append(score_original_primeiro[5])

labels.append(layers[5])
for i in segundo:

    score.append(score_original_segundo[i])
    labels.append(layers2[i])

for i in score:
    loss.append(i[0])
    accuracia.append(i[1])


#accuracia = [1,2,3,4,5,6,7,8]
#loss      = [0,1,2,3,4,5,6,7]
barWidth = 0.4

# Set position of bar on X axis
r1 = np.arange(len(loss))
#r2 = [x + barWidth for x in r1]
#r3 = [x + barWidth for x in r2]
 
# Make the plot
plt.bar(r1, accuracia, color='#2d7f5e', width=barWidth, edgecolor='white', label='Accuracy')
#plt.bar(r1, loss, color='#557f2d', width=barWidth, edgecolor='white', label='Loss')
#plt.bar(r3, tempos_50, color='#2d7f5e', width=barWidth, edgecolor='white', label='50x50x50 world')
 
# Add xticks on the middle of the group bars
plt.xlabel('Neural Networks Models', fontweight='bold')
plt.ylabel('Accuracy',fontweight='bold')
#plt.ylim(0,1)
#plt.title('',fontweight='bold')
#plt.xticks([r for r in range(len(accuracia))], ['SGD', 'RMSprop', 'Adam', 'Adadelta','Adagrad', 'Adamax', 'Nadam', 'Ftrl'])

plt.xticks([r for r in range(len(accuracia))], labels)
 
# Create legend & Show graphic
plt.legend()
plt.show()






