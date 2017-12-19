from sklearn.datasets import fetch_mldata
from skimage import io
mnist = fetch_mldata('MNIST original')
import random
import numpy as np
import scipy.misc
img=np.zeros(56*56*3).reshape(56,56,3)
base="/mnt/raid/arnab/compositional-mnist/"

for i in xrange(0,25600):
    for j in xrange(0,3):
        id=random.randint(0,mnist.data.shape[0]+1)
        id=id%mnist.data.shape[0]
        x=j%2
        y=j/2
        img[0+x*28:0+x*28+28,0+y*28:0+y*28+28,0]=mnist.data[id].reshape(28,28)
        img[0+x*28:0+x*28+28,0+y*28:0+y*28+28,1]=mnist.data[id].reshape(28,28)
        img[0+x*28:0+x*28+28,0+y*28:0+y*28+28,2]=mnist.data[id].reshape(28,28)
    scipy.misc.imsave(base+str(i)+'.png', img)
