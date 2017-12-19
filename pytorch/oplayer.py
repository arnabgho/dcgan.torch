import caffe
import numpy as np


class OuterProductLayer(caffe.Layer):
    """
    Compute the Euclidean Loss in the same manner as the C++ EuclideanLossLayer
    to demonstrate the class interface for developing layers in Python.
    """

    def setup(self, bottom, top):
        # check input pair
        if len(bottom) != 2:
            raise Exception("Need two inputs to compute outer product.")

    def reshape(self, bottom, top):
        # check input dimensions match
        #if bottom[0].count != bottom[1].count:
        #    raise Exception("Inputs must have the same dimension.")
        # difference is shape of inputs
        self.batch_size=bottom[0].data.shape[0]
        self.diff_0 = np.zeros_like(bottom[0].data, dtype=np.float32)
        self.diff_1 = np.zeros_like(bottom[1].data, dtype=np.float32)
        # loss output is scalar
        top[0].reshape(self.batch_size,bottom[0].data.shape[1],bottom[1].data.shape[1])

    def forward(self, bottom, top):
        for i in range(self.batch_size):
            for j in range(bottom[1].data.shape[1]):
                top[0][i,:,j]=bottom[0][i]*bottom[1][i,j]

    def backward(self, top, propagate_down, bottom):
        for i in range(self.batch_size):
            for j in range(bottom[0].shape[1]):
                for k in range(bottom[1].shape[1]):
                    self.diff_0[i,j]+=bottom[1][i,k]*top[0].diff[i,j,k]

         for i in range(self.batch_size):
            for j in range(bottom[1].shape[1]):
                for k in range(bottom[0].shape[0]):
                    self.diff_1[i,j]+=bottom[0][i,k]*top[0].diff[i,k,j]


        bottom[0].diff[...] = self.diff_0
        bottom[1].diff[...] = self.diff_1


