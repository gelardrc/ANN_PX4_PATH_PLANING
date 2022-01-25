from matplotlib import pyplot as plt
from keras.models import load_model
import numpy as np
 
def rectified(x):
	return max(0.0, x)
 

x = []
y = []

for i in range(-50,50):
	x.append(i)
for t in x:
	y.append(rectified(t))

fig= plt.figure(dpi=200)
ax = fig.add_subplot(1,1,1)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_title('ReLu function [-50,50]')
ax.plot(x,y)

model = load_model('/home/gelo/codes/ANN_PX4_PATH_PLANING/Redes_salvas/dritk_qualificacao.h5')

# array([[ 0.,  0.,  0., 15., 25.,  0.,  0.,  0.,  0.,  0.,  0.,  0.]])
# array([0, 0, 0, 0, 0, 3, 0, 1, 0, 1, 0, 0])
input = np.array([[1,1,1,3,1,1,0,1,0,1,0,0]])
out = model.predict(input)

print(out)
# [[3.2365380e-05 9.9950004e-01 5.4626089e-05 4.0887647e-11 5.6507615e-11 4.1298656e-04]]
plt.show()