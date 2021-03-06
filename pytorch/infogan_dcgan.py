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
import visdom
import numpy as np
vis = visdom.Visdom()
vis.env = 'infogan_dcgan'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ndiscrete', type=int, default=10, help='size of the latent z vector')
parser.add_argument('--ndiscrete_chosen', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--ncontinuous', type=int, default=2, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda_continuous', type=float, default=1, help='lambda_continuous default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netQ', default='', help="path to netQ (to continue training)")
parser.add_argument('--netFE', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./infogan-dcgan', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

opt = parser.parse_args()
print(opt)

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

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


class log_gaussian:
    def __call__(self, x, mu, var):
        logli = -0.5*(var.mul(2*np.pi)+1e-6).log() - (x-mu).pow(2).div(var.mul(2.0)+1e-6)
        return logli.sum(1).mean().mul(-1)

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def _noise_sample(dis_c,con_c,noise_c):
    discrete_np=np.zeros((opt.batchSize,opt.ndiscrete))
    for i in range(opt.batchSize):
        activated=np.random.choice(opt.ndiscrete,opt.ndiscrete_chosen,replace=False)
        discrete_np[i][activated]=1

    dis_c.data.copy_(torch.Tensor(discrete_np))
    con_c.data.uniform_(-1.0,1.0)
    noise_c.data.uniform_(-1.0,1.0)
    z=torch.cat([noise_c,dis_c,con_c],1).view(-1,opt.nz+opt.ndiscrete + opt.ncontinuous,1,1)

    return z , discrete_np

class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     opt.nz + opt.ndiscrete+opt.ncontinuous  , ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)




class _netFE(nn.Module):
    def __init__(self, ngpu):
        super(_netFE, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output  #output.view(-1, 1).squeeze(1)
netFE = _netFE(ngpu)
netFE.apply(weights_init)
if opt.netFE != '':
    netFE.load_state_dict(torch.load(opt.netD))
print(netFE)

class _netQ(nn.Module):
    def __init__(self,ngpu):
        super(_netQ,self).__init__()
        self.ngpu=ngpu
        self.continuous_branch=nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            )
        self.discrete_branch=nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, opt.ndiscrete, 4, 1, 0, bias=False)
            )

        self.mu_branch = nn.Sequential( nn.Conv2d(ndf * 8, opt.ncontinuous, 4, 1, 0, bias=False) )
        self.std_branch = nn.Sequential( nn.Conv2d(ndf * 8, opt.ncontinuous, 4, 1, 0, bias=False) )
    def forward(self,input):
        output_continuous = self.continuous_branch(input)
        output_continuous_mu = self.mu_branch( output_continuous )
        output_continuous_std = self.std_branch( output_continuous  )
        output_discrete = self.discrete_branch(input)

        return [output_discrete.view(-1,opt.ndiscrete),output_continuous_mu.view(-1,opt.ncontinuous), output_continuous_std.view(-1,opt.ncontinuous) ]  #output.view(-1, 1).squeeze(1)

netQ = _netQ(ngpu)
netQ.apply(weights_init)
if opt.netQ != '':
    netQ.load_state_dict(torch.load(opt.netD))
print(netQ)


class _netD(nn.Module):
    def __init__(self,ngpu):
        super(_netD,self).__init__()
        self.ngpu=ngpu
        self.main = nn.Sequential(
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )
    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()
criterion_discrete= nn.MultiLabelSoftMarginLoss()
criterion_continuous = log_gaussian()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, opt.nz+opt.ndiscrete+opt.ncontinuous, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz + opt.ndiscrete + opt.ncontinuous , 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    netFE.cuda()
    netQ.cuda()
    criterion.cuda()
    #criterion_Q.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam( [ { 'params':netFE.parameters()  } , { 'params' : netD.parameters() } ] , lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam( [ { 'params':netG.parameters()  } , { 'params' : netQ.parameters() } ], lr=opt.lr, betas=(opt.beta1, 0.999))

dis_c= torch.FloatTensor(opt.batchSize, opt.ndiscrete).cuda()
con_c = torch.FloatTensor(opt.batchSize, opt.ncontinuous).cuda()
noise_c = torch.FloatTensor(opt.batchSize, opt.nz).cuda()
dis_c = Variable(dis_c)
con_c = Variable(con_c)
noise_c = Variable(noise_c)

fix_dis_c= torch.FloatTensor(100, opt.ndiscrete).cuda()
fix_con_c = torch.FloatTensor(100, opt.ncontinuous).cuda()
fix_noise_c = torch.FloatTensor(100, opt.nz).cuda()
fix_dis_c = Variable(fix_dis_c)
fix_con_c = Variable(fix_con_c)
fix_noise_c = Variable(fix_noise_c)

# fixed random variables
c = np.linspace(-1, 1, 10).reshape(1, -1)
c = np.repeat(c, 10, 0).reshape(-1, 1)

c1 = np.hstack([c, np.zeros_like(c)])
c2 = np.hstack([np.zeros_like(c), c])

####################
#idx = np.arange(10).repeat(10)
#one_hot = np.zeros((100, 10))
#one_hot[range(100), idx] = 1
###################

one_hot = np.zeros((100, 10))
for i in range(10):
    idx=np.random.choice(opt.ndiscrete,opt.ndiscrete_chosen,replace=False)
    for j in range(10):
        one_hot[i*10+j][idx] = 1



fix_noise_c_fix = torch.Tensor(100, opt.nz).uniform_(-1, 1)


rec_win=None
real_win=None
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        optimizerD.zero_grad() #netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if batch_size!=opt.batchSize:
            continue
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(netFE(inputv))
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fake
        noise, idx = _noise_sample(dis_c,con_c,noise_c)
        noisev=noise.resize(batch_size, opt.nz + opt.ndiscrete + opt.ncontinuous , 1, 1)
        #noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        fe_output=netFE( fake.detach())
        output = netD(fe_output )
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z))) & Q Network
        ###########################
        optimizerG.zero_grad() #netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        fe_output= netFE(fake)
        output = netD(fe_output)
        errG_GAN = criterion(output, labelv)


        q_logits,q_mu,q_var = netQ(fe_output)
        errG_discrete = criterion_discrete(q_logits,dis_c)
        errG_continuous = criterion_continuous(con_c , q_mu , q_var) * opt.lambda_continuous

        errG = errG_GAN + errG_continuous + errG_discrete
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        rec_win = vis.image(fake.data[0].cpu()*0.5+0.5,win = rec_win)
        real_win = vis.image(data[0][0]*0.5+0.5,win = real_win)

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            fix_noise_c.data.copy_(fix_noise_c_fix)
            fix_dis_c.data.copy_(torch.Tensor(one_hot))
            fix_con_c.data.copy_(torch.from_numpy(c1))
            z = torch.cat([fix_noise_c, fix_dis_c, fix_con_c], 1).view(-1, opt.nz + opt.ndiscrete + opt.ncontinuous, 1, 1)
            x_save = netG(z)
            vutils.save_image(x_save.data, '%s/c1.png' % opt.outf, nrow=10)

            fix_con_c.data.copy_(torch.from_numpy(c2))
            z = torch.cat([fix_noise_c, fix_dis_c, fix_con_c], 1).view(-1, opt.nz + opt.ndiscrete + opt.ncontinuous , 1, 1)
            x_save = netG(z)
            vutils.save_image(x_save.data, '%s/c2.png' % opt.outf , nrow=10)
    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
