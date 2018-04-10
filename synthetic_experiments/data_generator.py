import os
import torch
import torch.nn as nn
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pylab as pl
from torch.autograd import Variable
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
        sns.distplot(self.samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-20),int(max(self.mode_info['modes'])+20),0.1) , kde=False)
        plt.savefig(filename)
        plt.clf()

    def plot_generated_samples(self,generated_samples,filename='generated_samples.png'):
        fig,axs= plt.subplots()
        sns.distplot(self.samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-20),int(max(self.mode_info['modes'])+20),0.1) , kde=False)
        sns.distplot(generated_samples,ax=axs, bins=pl.frange(int(min(self.mode_info['modes'])-20),int(max(self.mode_info['modes'])+20),0.1) , kde=False)
        plt.savefig(filename)
        plt.clf()

    def estimate_prob_discriminator( self ,netD ,batch_size, bin_range  ):
        num_bins=bin_range.shape[0]
        output_D=np.zeros( num_bins)
        input=torch.FloatTensor(batch_size,1)
        input= input.cuda()
        bin_range_rounded=np.zeros( int( num_bins/batch_size + 1   )*batch_size  )
        bin_range_rounded[0:num_bins]=bin_range
        total_gen=0
        while(total_gen<=num_bins):
            input.copy_(torch.Tensor(bin_range_rounded[total_gen:total_gen+batch_size]))
            inputv=Variable(input)
            output=netD(inputv)
            output_np=output.data.cpu().numpy().reshape(batch_size)

            if total_gen+batch_size<num_bins:
                output_D[ total_gen:total_gen+batch_size ]=output_np
            else:
                output_D[ total_gen:num_bins ] =output_np[0:num_bins-total_gen]
            total_gen+=batch_size

        return output_D

    def generate_samples_D(self,original_samples,bin_range,output_D):
        hist, bin_edges = np.histogram(original_samples, bins = bin_range )
        maxim = max(hist)
        total_gen_bin = output_D
        total_generations = 0
        for i in range(output_D.shape[0]):
            total_gen_bin[i] = int(output_D[i] * maxim)
            total_generations+= int(total_gen_bin[i])

        final_generations=np.zeros(total_generations)

        index=0
        for i in range(total_gen_bin.shape[0]):
            for j in range(int(total_gen_bin[i])):
                final_generations[ index ] = bin_range[i]
                index+=1

        return final_generations

    def plot_generated_samples_discriminator(self,generated_samples,netD,batch_size,filename='generated_samples.png'):
        fig,axs= plt.subplots()
        bin_range=pl.frange(int(min(self.mode_info['modes'])-20),int(max(self.mode_info['modes'])+20),0.1)
        output_D=self.estimate_prob_discriminator(netD,batch_size,bin_range)
        samples_D=self.generate_samples_D(self.samples,bin_range,output_D)
        sns.distplot(self.samples,ax=axs, bins=bin_range , kde=False)
        sns.distplot(generated_samples,ax=axs, bins=bin_range , kde=False)
        sns.distplot(samples_D,ax=axs,bins=bin_range, kde=False)
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
        return {'modes':modes, 'stds':stds , 'densities':densities}

    def generate_samples(self,filename):
        samples=np.zeros(self.num_samples)
        mode_selections=np.random.choice(range(len(self.mode_info['modes'])), size = self.num_samples, p = self.mode_info['densities'])

        for i in range(self.num_samples):
            samples[i]=np.random.normal(self.mode_info['modes'][ mode_selections[i]], self.mode_info[ 'stds'][ mode_selections[i] ] )

        np.save(filename,samples)
        return samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self,idx):
        return self.samples[idx]
