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
from common_net import *
import math
import copy
vis = visdom.Visdom()
vis.env = 'tails_madgan_unit'
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--nresidual', type=int, default=3, help='number of residual layers in discriminator')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--ngen', type=int, default=3)
parser.add_argument('--niter', type=int, default=50, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./tails_madgan_unit', help='folder to output images and model checkpoints')
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
ngen = opt.ngen

# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if 'weight' in m.__dict__.keys():
            m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class _netG(nn.Module):
    def __init__(self,ngpu):
        super(_netG,self).__init__()
        self.ngpu=ngpu
        self.main=nn.Sequential(
                # input is Z, going into a convolution
                nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
                nn.BatchNorm2d(ngf * 8),
                nn.ReLU(True)
                )
    def forward(self,input):
        if isinstance(input.data,torch.cuda.FloatTensor) and self.ngpu>1:
            output=nn.parallel.data_parallel(self.main,input,range(self.ngpu))
        else:
            output = self.main(input)
        return output

class _netG_tail(nn.Module):
    def __init__(self, ngpu):
        super(_netG_tail, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
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

netG_tail = _netG_tail(ngpu)
netG_tail.apply(weights_init)
netG = []
net = _netG(ngpu)
net.apply(weights_init)
state = net.state_dict()
netG.append(net)
for i in range(ngen-1):
    state_clone = copy.deepcopy(state)
    net=_netG(ngpu)
    net.load_state_dict(state_clone)
    netG.append(net)

#if opt.netG != '':
#    netG.load_state_dict(torch.load(opt.netG))

print(netG[0])


class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        input_dim_a = 3
        ch = ndf
        n_enc_front_blk  = 3
        n_enc_latter_blk = 1  #2
        n_enc_res_blk    = 3
        n_enc_shared_blk = 1
        encA=[]
        encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=2, padding=1)]
        tch=ch
        for i in range(0,n_enc_front_blk):
            encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        for i in range(0,n_enc_latter_blk):
            encA += [ReLUINSConv2d(tch, tch , kernel_size=3, stride=2, padding=1)]  #[ReLUINSConv2d(tch, tch , kernel_size=3, stride=2, padding=1)]
        encA += [nn.Conv2d(tch, tch , 3, 2, 1,bias=False)]

        for i in range(opt.nresidual):
            encA += [INSResBlock(tch, tch)]
        #encA += [INSResBlock(tch, tch)]
        #encA += [INSResBlock(tch, tch)]
        #encA+=[nn.Softmax()]
        self.fch=tch
        #for i in range(0, n_enc_res_blk):
        #    encA += [INSResBlock(tch, tch)]

        #for i in range(0, n_enc_shared_blk):
        #    encA += [INSResBlock(tch, tch)]

        self.linear=nn.Sequential(
        nn.Conv2d(self.fch,ngen+1,1,1,0,bias=False),
        #nn.Sigmoid()
        )
        #encA += [GaussianNoiseLayer()]
        self.main=nn.Sequential(*encA)

    def forward(self, input):
        #if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
        #    output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        #else:
        output = self.main(input)      #self.main(input)
        output = self.linear(output)               #output.view(-1, self.fch))
        return output.view(-1,ngen+1)  # output.view(-1, 1).squeeze(1) # output


netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.CrossEntropyLoss()  #nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.LongTensor(opt.batchSize)   #torch.FloatTensor(opt.batchSize)
real_label = ngen
fake_labels = np.arange(ngen)

if opt.cuda:
    netD.cuda()
    netG_tail.cuda()
    for k in range(ngen):
        netG[k].cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG_tail= optim.Adam(netG_tail.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG=[]
for i in range(ngen):
    optimizerG.append(optim.Adam(netG[i].parameters(), lr=opt.lr, betas=(opt.beta1, 0.999)))

fake_win=[]
for i in range(ngen):
    fake_win.append(None)
real_win=None
for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        netD.zero_grad()
        real_cpu, _ = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        label.resize_(batch_size).fill_(real_label)
        inputv = Variable(input)
        labelv = Variable(label)

        output = netD(inputv)
        errD_real = criterion(output, labelv)
        errD_real.backward()
        D_x = output.data.mean()

        # train with fakes
        fakes=[]
        D_G_z1=0
        errD = errD_real
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        for k in range(ngen):
            noisev = Variable(noise)
            head_output=netG[k](noisev)
            fake=netG_tail(head_output)
            labelv = Variable(label.fill_(fake_labels[k].item()))
            output = netD(fake.detach())
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 += output.data.mean()
            fakes.append(fake)
            errD=errD+errD_fake
        optimizerD.step()
        D_G_z1/=ngen
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        D_G_z2=0
        errG=0
        errGs=[]
        for k in range(ngen):
            errGs.append(None)
        netG_tail.zero_grad()
        for k in range(ngen):
            netG[k].zero_grad()
            netD.zero_grad()
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            output = netD(fakes[k])
            errG += criterion(output, labelv)
            errGs[k]=criterion(output,labelv)
            errGs[k].backward( retain_graph=True )
            D_G_z2 += output.data.mean()
            optimizerG[k].step()
        D_G_z2/=ngen
        optimizerG_tail.step()
        for k in range(ngen):
            fake_win[k]= vis.image(fakes[k].data[0].cpu()*0.5+0.5,win = fake_win[k])
        real_win = vis.image(data[0][0]*0.5+0.5,win = real_win)

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            for k in range(ngen):
                head_output=netG[k](fixed_noise)
                fake = netG_tail( head_output  )
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d_gen_%03d.png' % (opt.outf, epoch,k),
                        normalize=True)

    # do checkpointing
    #torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    #torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
