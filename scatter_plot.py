import matplotlib as plt
import numpy as np

bald=np.load("celebA/bald.json.npy")
black=np.load("celebA/black.json.npy")
brown=np.load("celebA/brown.json.npy")
blond=np.load("celebA/blond.json.npy")
gray=np.load("celebA/gray.json.npy")

begin=bald.shape[0]

x=np.concatenate( (bald,black,brown,blond,gray) , axis=0)
y=np.ones(x.shape[0])

for i in range(5):
	y[ i*begin:begin*(i+1) ] =i

colors=['#ffff00' , '#00ff00' , '#ff0000' ,'#000000' , '#0000ff'  ]

color_labels=[]
for i in range(x.shape[0]):
	color_labels.append(colors[y[i]])

plt.scatter( x[:,0] , x[:,1] , c=color_labels  )
plt.savefig( "Cluster_Analysis.png")

