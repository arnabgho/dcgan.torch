from sklearn.datasets import fetch_mldata
from skimage import io
mnist = fetch_mldata('MNIST original')
import random
import numpy as np
import scipy.misc
img=np.zeros(28*28*3).reshape(28,28,3)
base="/mnt/raid/arnab/stacked-mnist/"

for i in xrange(0,25600):
    for j in xrange(0,3):
        id=random.randint(0,mnist.data.shape[0]+1)
        img[:,:,j]=mnist.data[id].reshape(28,28)
    scipy.misc.imsave(base+str(i)+'.png', img)
