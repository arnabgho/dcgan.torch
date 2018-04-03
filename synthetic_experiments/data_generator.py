import os
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
sns.set(color_codes=True)

class MoG1DDataset(Dataset):
    "MoG1D Dataset"
    def __init__(self,spec_file='specs1D.txt',num_samples=1000000,load_file='data1D.npy',load=False):
        self.spec_file=spec_file
        self.num_samples=num_samples
        self.mode_info=self.read_spec_file(self.spec_file)
        if load==False:
            self.samples=self.generate_samples(load_file)
        else:
            self.samples=np.load(load_file)
        self.plot_samples()


    def plot_samples(self,filename='samples.png'):
        fig,axs= plt.subplots()
        sns.distplot(self.samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-5),int(max(self.mode_info['modes'])+5),0.1) , kde=False)
        plt.savefig(filename)
        plt.clf()

    def plot_generated_samples(self,generated_samples,filename='generated_samples.png'):
        fig,axs= plt.subplots()
        sns.distplot(self.samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-5),int(max(self.mode_info['modes'])+5),0.1) , kde=False)
        sns.distplot(generated_samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-5),int(max(self.mode_info['modes'])+5),0.1) , kde=False)
        plt.savefig(filename)
        plt.clf()

    def read_spec_file(self,spec_file):
        lines=[]
        modes=[]
        stds=[]
        densities=[]
        with open(spec_file) as f:
            for line in f:
                lines.append(line)

        total_density=0
        for line in lines:
            if len(line)==0:
                continue
            line=line.rstrip()
            mode=float(line.split(' ')[0])
            std=float(line.split(' ')[1])
            density=float(line.split(' ')[2])
            total_density=density+total_density
            modes.append(mode)
            stds.append(std)
            densities.append(density)

        for (i,density) in enumerate(densities):
            densities[i]=density/total_density
        print(densities)
        return {'modes':modes, 'stds':stds , 'densities':densities}

    def generate_samples(self,filename):
        samples=np.zeros(self.num_samples)
        mode_selections=np.random.choice(range(len(self.mode_info['modes'])), size = self.num_samples, p = self.mode_info['densities'])

        for i in range(self.num_samples):
            samples[i]=np.random.normal(self.mode_info['modes'][ mode_selections[i]], self.mode_info[ 'densities'][ mode_selections[i] ] )

        np.save(filename,samples)
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        return self.samples[idx]
