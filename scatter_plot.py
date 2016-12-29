import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
num_show=200
bald=np.load("celebA/bald.json.npy")[:num_show,:]
black=np.load("celebA/black.json.npy")[:num_show,:]
brown=np.load("celebA/brown.json.npy")[:num_show,:]
blond=np.load("celebA/blond.json.npy")[:num_show,:]
gray=np.load("celebA/gray.json.npy")[:num_show,:]

begin=bald.shape[0]

attributes = ['bald','black','brown','blond','gray']
x=np.concatenate( (bald,black,brown,blond,gray) , axis=0)
y=np.ones(x.shape[0]).astype(int)

for i in range(5):
	y[ i*begin:begin*(i+1) ] =i

colors=['#ffff00' , '#00ff00' , '#ff0000' ,'#000000' , '#0000ff'  ]

color_labels=[]
for i in range(x.shape[0]):
	color_labels.append(colors[y[i]])

legend = []
for i in range(len(attributes)):
	legend.append(mpatches.Patch(color=colors[i], label=attributes[i]))  #Legend is for the size of clusters containing only paramSet
plt.legend(handles=legend)
plt.scatter( x[:,0] , x[:,1] , c=color_labels  )
plt.savefig( "Cluster_Analysis.png")

