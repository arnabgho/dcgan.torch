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

vis = visdom.Visdom()
vis.env = 'wgan_gp_resnet'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--lambda_gp', type=float, default=10, help='lambda for WGAN-GP default=10')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./wgan_resnet_gen', help='folder to output images and model checkpoints')
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


#class _netG(nn.Module):
#    def __init__(self, ngpu):
#        super(_netG, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            # input is Z, going into a convolution
#            nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#            nn.BatchNorm2d(ngf * 8),
#            nn.ReLU(True),
#            # state size. (ngf*8) x 4 x 4
#            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 4),
#            nn.ReLU(True),
#            # state size. (ngf*4) x 8 x 8
#            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf * 2),
#            nn.ReLU(True),
#            # state size. (ngf*2) x 16 x 16
#            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ngf),
#            nn.ReLU(True),
#            # state size. (ngf) x 32 x 32
#            nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#            nn.Tanh()
#            # state size. (nc) x 64 x 64
#        )
#
#    def forward(self, input):
#        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#        else:
#            output = self.main(input)
#        return output


class _netG(nn.Module):
    def __init__(self,ngpu):
        super(_netG,self).__init__()
        self.ngpu=ngpu
        main_block=[]
        # input is Z, going into a convolution
        main_block += [nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False)]
        main_block += [ nn.BatchNorm2d(ngf * 8) ]
        main_block += [nn.ReLU(True)]

        # Residual Layers
        tch=ngf*8
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]

        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]

        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]



        # state size. (ngf*8) x 4 x 4
        main_block += [nn.ConvTranspose2d( ngf*8 , ngf * 4, 4, 2, 1, bias=False)]
        main_block += [ nn.BatchNorm2d(ngf * 4) ]
        main_block += [nn.ReLU(True)]
        # state size. (ngf*4) x 8 x 8
        tch=ngf*4
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]


        main_block += [nn.ConvTranspose2d( ngf*4 , ngf * 2, 4, 2, 1, bias=False)]
        main_block += [ nn.BatchNorm2d(ngf * 2) ]
        main_block += [nn.ReLU(True)]
        # state size. (ngf*2) x 16 x 16

        tch=ngf*2
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]


        main_block += [nn.ConvTranspose2d( ngf*2 , ngf , 4, 2, 1, bias=False)]
        main_block += [ nn.BatchNorm2d(ngf ) ]
        main_block += [nn.ReLU(True)]
        # state size. (ngf) x 32 x 32

        tch=ngf
        #main_block += [BATCHResBlock(tch, tch)]

        main_block += [ nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False)  ]
        main_block += [ nn.Tanh() ]
        # state size. (nc) x 64 x 64
        self.main= nn.Sequential( *main_block )

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

orig_stdout=sys.stdout
#f = open(opt.outf+'/architecture.txt' , 'w'  )
#sys.stdout=f
#print(netG)
#sys.stdout = orig_stdout



class _netD(nn.Module):
    def __init__(self,ngpu):
        super(_netD,self).__init__()
        self.ngpu=ngpu
        main_block  =[]
        # input is (nc) x 64 x 64
        main_block += [ nn.Conv2d( nc, ndf , 4, 2, 1, bias=False  )  ]
        main_block += [nn.LeakyReLU(0.2, inplace=True)]

        tch=ndf
        #main_block += [BATCHResBlock(tch, tch)]

        # state size (ndf) x 32 x 32
        main_block += [ nn.Conv2d( ndf , ndf*2 , 4, 2, 1, bias=False  )  ]
        main_block += [ nn.BatchNorm2d(ndf * 2) ]
        main_block += [ nn.LeakyReLU(0.2, inplace=True) ]
        # state size. (ndf*2) x 16 x 16

        tch=ndf*2

        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]

        main_block += [ nn.Conv2d( ndf*2 , ndf*4 , 4, 2, 1, bias=False  )  ]
        main_block += [ nn.BatchNorm2d(ndf * 4) ]
        main_block += [ nn.LeakyReLU(0.2, inplace=True) ]
        # state size. (ndf*4) x 8 x 8
        tch=ndf*4
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]


        main_block += [ nn.Conv2d( ndf*4 , ndf*8 , 4, 2, 1, bias=False  )  ]
        main_block += [ nn.BatchNorm2d(ndf * 8) ]
        main_block += [ nn.LeakyReLU(0.2, inplace=True) ]
        # state size. (ndf*8) x 4 x 4
        tch=ndf*8
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]
        main_block += [BATCHResBlock(tch, tch)]

        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]

        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]

        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]
        #main_block += [BATCHResBlock(tch, tch)]


        main_block += [ nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False) ]
        #main_block += [ nn.Sigmoid() ]

        self.main = nn.Sequential( *main_block  )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


#class _netD(nn.Module):
#    def __init__(self, ngpu):
#        super(_netD, self).__init__()
#        self.ngpu = ngpu
#        self.main = nn.Sequential(
#            # input is (nc) x 64 x 64
#            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf) x 32 x 32
#            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 2),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*2) x 16 x 16
#            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 4),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*4) x 8 x 8
#            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#            nn.BatchNorm2d(ndf * 8),
#            nn.LeakyReLU(0.2, inplace=True),
#            # state size. (ndf*8) x 4 x 4
#            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#            nn.Sigmoid()
#        )
#
#    def forward(self, input):
#        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#        else:
#            output = self.main(input)
#
#        return output.view(-1, 1).squeeze(1)

#class _netD(nn.Module):
#    def __init__(self, ngpu):
#        super(_netD, self).__init__()
#        self.ngpu = ngpu
#        input_dim_a = 3
#        ch = 64
#        n_enc_front_blk  = 3
#        n_enc_latter_blk = 1  #2
#        n_enc_res_blk    = 3
#        n_enc_shared_blk = 1
#        encA=[]
#        encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=2, padding=1)]
#        tch=ch
#        for i in range(0,n_enc_front_blk):
#            encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
#            tch *= 2
#        for i in range(0,n_enc_latter_blk):
#            encA += [ReLUINSConv2d(tch, tch , kernel_size=3, stride=2, padding=1)]  #[ReLUINSConv2d(tch, tch , kernel_size=3, stride=2, padding=1)]
#        encA += [nn.Conv2d(tch, tch , 3, 2, 1,bias=False)]
#        encA += [INSResBlock(tch, tch)]
#        encA += [INSResBlock(tch, tch)]
#        encA += [INSResBlock(tch, tch)]
#
#        encA += [INSResBlock(tch, tch)]
#        encA += [INSResBlock(tch, tch)]
#        encA += [INSResBlock(tch, tch)]
#
#
#        encA+=[nn.Sigmoid()]
#        self.fch=tch
#        #for i in range(0, n_enc_res_blk):
#        #    encA += [INSResBlock(tch, tch)]
#
#        #for i in range(0, n_enc_shared_blk):
#        #    encA += [INSResBlock(tch, tch)]
#
#        self.linear=nn.Sequential(
#        nn.Conv2d(self.fch,1,1,1,0,bias=False),
#        nn.Sigmoid()
#        )
#        #encA += [GaussianNoiseLayer()]
#        self.main=nn.Sequential(*encA)
#
#    def forward(self, input):
#        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#        #    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
#        #else:
#        output = self.main(input)      #self.main(input)
#        output = self.linear(output)               #output.view(-1, self.fch))
#        return  output.view(-1, 1).squeeze(1) # output

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

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

with open(opt.outf+'/architecture.txt' , 'w'  ) as f:
    print(netD,file=f)
    print(netG,file=f)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0
one = torch.FloatTensor([1])
mone = one * -1

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()
    one, mone = one.cuda() , mone.cuda()
fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

rec_win=None
real_win=None

for epoch in range(opt.niter):
    for i,data in enumerate(dataloader,0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu,_ = data
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
        noise.resize_(batch_size, opt.nz,1,1).normal_(0, 1)
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
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
