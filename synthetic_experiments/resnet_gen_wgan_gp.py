from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.autograd as autograd
import visdom
from common_net import *
import math
import sys
from data_generator import *
vis = visdom.Visdom()
vis.env = 'wgan_gp_resnet'
parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=4096, help='input batch size')
parser.add_argument('--out_dim', type=int, default=1, help='the output dimension')
parser.add_argument('--nz', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=4)
parser.add_argument('--ndf', type=int, default=4)
parser.add_argument('--ngres', type=int, default=4)
parser.add_argument('--ndres', type=int, default=4)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--num_samples', type=int, default=1000000, help='number of samples in the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate, default=0')
parser.add_argument('--lambda_gp', type=float, default=0.1, help='lambda for WGAN-GP, default=0.1')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./wgan_resnet_gen/', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int,default=7, help='manual seed')

opt = parser.parse_args()
print(opt)
torch.manual_seed(7)
try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

dataset=MoG1DDataset()
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,shuffle=True, num_workers=int(opt.workers))
ngpu=int(opt.ngpu)
class _netG(nn.Module):
    def __init__(self,ngpu):
        super(_netG,self).__init__()
        self.ngpu=ngpu
        main_block=[]

        #Input is z going to series of rsidual blocks

        main_block+=[nn.Linear(opt.nz,opt.ngf) ]

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+=[BATCHResBlock(opt.ngf,opt.dropout)]

        # Final layer to map to 1D

        main_block+=[nn.Linear(opt.ngf,opt.out_dim)]

        self.main=nn.Sequential(*main_block)


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)

if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))

class _netD(nn.Module):
    def __init__(self,ngpu):
        super(_netD,self).__init__()
        self.ngpu=ngpu
        main_block=[]

        #Input is 1D going to series of residual blocks

        main_block+=[nn.Linear(opt.out_dim,opt.ngf) ]
        main_block+=[nn.ReLU()]
        # Sets of residual blocks start

        for i in range(opt.ndres):
            main_block+=[BATCHResBlock(opt.ngf,opt.dropout)]

        # Final layer to map to sigmoid output

        main_block+=[nn.Linear(opt.ngf,1)]
	#main_block+=[nn.Sigmoid()]

        self.main=nn.Sequential(*main_block)


    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netD = _netD(ngpu)

if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))


with open(opt.outf+'/architecture.txt' , 'w'  ) as f:
    print(netD,file=f)
    print(netG,file=f)

criterion=nn.BCELoss()

def calc_gradient_penalty(netD, real_data, fake_data):
    alpha = torch.rand(opt.batchSize, 1)
    alpha = alpha.expand(real_data.size())
    alpha = alpha.cuda() if opt.cuda else alpha

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = autograd.Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda() if opt.cuda else torch.ones(
                                  disc_interpolates.size()),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda_gp
    return gradient_penalty

input=torch.FloatTensor(opt.batchSize,opt.out_dim)
noise = torch.FloatTensor(opt.batchSize, opt.nz)
fixed_noise = torch.FloatTensor(opt.batchSize, opt.nz).normal_(0, 1)
generated_samples=np.zeros(opt.num_samples)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1
if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input = input.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    one, mone = one.cuda() , mone.cuda()
fixed_noise = Variable(fixed_noise)

#setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

for epoch in range(opt.niter):
    for i,data in enumerate(dataloader,0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu = data.float()
        batch_size = real_cpu.size(0)
        if batch_size!=opt.batchSize:
            continue
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        real_cpu=real_cpu.resize_as_(input)
        input.resize_as_(real_cpu).copy_(real_cpu)
        realv = Variable(input)

        D_real = netD(realv)
	D_real = D_real.mean()
	D_real.backward(mone)

        # train with fake
        noise.resize_(batch_size, opt.nz).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        D_fake=netD(fake.detach())
	D_fake=D_fake.mean()
	D_fake.backward(one)

        # gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, realv.data, fake.data)
        gradient_penalty.backward()

        D_cost = D_fake - D_real + gradient_penalty
        Wasserstein_D = D_real - D_fake

        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        G = netD(fake)
        G = G.mean()
        G.backward(mone)
        G_cost = -G
        optimizerG.step()

	print('[%d/%d][%d/%d] D_cost: %.4f G_cost: %.4f Wasserstein_D: %.4f '
              % (epoch, opt.niter, i, len(dataloader),
                 D_cost.cpu().data.numpy(), G_cost.cpu().data.numpy(), Wasserstein_D.cpu().data.numpy()))

    num_sampled=0
    while(num_sampled<opt.num_samples):
        noise.resize_(opt.batchSize, opt.nz).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        fake_cpu_np=fake.data.cpu().numpy().reshape(opt.batchSize)
        if num_sampled+opt.batchSize<opt.num_samples:
            generated_samples[num_sampled:num_sampled+opt.batchSize ] = fake_cpu_np
        else:
            generated_samples[num_sampled:opt.num_samples] = fake_cpu_np[ 0:opt.num_samples-num_sampled ]
        num_sampled+=opt.batchSize
    dataset.plot_generated_samples(generated_samples,filename=opt.outf+'generated_samples_'+str(epoch)+'.png')

