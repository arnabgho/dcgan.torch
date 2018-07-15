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
vis = visdom.Visdom()
vis.env = 'dcgan'

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=128, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=128, help='size of the latent z vector')
parser.add_argument('--nc', type=int, default=3, help='num channels')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--spectral_G', type=bool, default=False)
parser.add_argument('--spectral_D', type=bool, default=False)
parser.add_argument('--outf', default='./resnet_dcgan', help='folder to output images and model checkpoints')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout value')
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


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
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
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            Reshape( opt.batchSize,opt.nz ),
            nn.Linear( opt.nz , 4*4*8*opt.ngf),
            Reshape( opt.batchSize,8*opt.ngf,4,4 ),
            # state size. (8*ngf) x 4 x 4
            UpConvResBlock(8*opt.ngf,8*opt.ngf,use_sn=opt.spectral_G),
            # state size. (8*ngf) x 8 x 8
            UpConvResBlock(8*opt.ngf,4*opt.ngf,use_sn=opt.spectral_G),
            # state size. (4*ngf) x 16 x 16
            UpConvResBlock(4*opt.ngf,4*opt.ngf,use_sn=opt.spectral_G),
            # state size. (4*ngf) x 32 x 32
            UpConvResBlock(4*opt.ngf,2*opt.ngf,use_sn=opt.spectral_G),
            #UpConvResBlock(4*opt.ngf,opt.ngf,use_sn=opt.spectral_G),
            # state size. (2*ngf) x 64 x 64
            UpConvResBlock(2*opt.ngf,opt.ngf,use_sn=opt.spectral_G),
            # state size. (ngf) x 128 x 128

            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(inplace=True),

            nn.Conv2d(    ngf,  nc, kernel_size=3,stride=1, padding=1, bias=False),
            nn.Tanh()
            #nn.Sigmoid()
            # state size. (nc) x 128 x 128
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


netG = _netG(ngpu)
#netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))
print(netG)

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
#

class _netD(nn.Module):
    def __init__(self, ngpu):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 128 x 128
            DownConvResBlock(opt.nc,opt.ndf,use_sn=opt.spectral_D),
            # input is (ndf) x 64 x 64
            DownConvResBlock(opt.ndf,2*opt.ndf,use_sn=opt.spectral_D) ,
            #DownConvResBlock(opt.nc,2*opt.ndf,use_sn=opt.spectral_D) ,
            # state size. (2*ndf) x 32 x 32
            DownConvResBlock(2*opt.ndf,4*opt.ndf,use_sn=opt.spectral_D) ,
            # state size. (ndf*4) x 16 x 16
            DownConvResBlock(4*opt.ndf,4*opt.ndf,use_sn=opt.spectral_D) ,
            # state size. (ndf*4) x 8 x 8
            DownConvResBlock(4*opt.ndf,8*opt.ndf,use_sn=opt.spectral_D) ,
            # state size. (ndf*8) x 4 x 4
            DownConvResBlock(8*opt.ndf,8*opt.ndf,use_sn=opt.spectral_D) ,
            # state size. (ndf*8) x 2 x 2
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,stride=2),
            Reshape( opt.batchSize , 8*opt.ndf ),
            nn.Linear(8*opt.ndf,1),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1).squeeze(1)


netD = _netD(ngpu)
#netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)
real_label = 1
fake_label = 0

if opt.cuda:
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

rec_win=None
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

        # train with fake
        noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
        noisev = Variable(noise)
        fake = netG(noisev)
        labelv = Variable(label.fill_(fake_label))
        output = netD(fake.detach())
        errD_fake = criterion(output, labelv)
        errD_fake.backward()
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, labelv)
        errG.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()

        rec_win = vis.image(fake.data[0].cpu()*0.5+0.5,win = rec_win)
        real_win = vis.image(data[0][0]*0.5+0.5,win = real_win)

        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
              % (epoch, opt.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
        if i % 100 == 0:
            vutils.save_image(real_cpu,
                    '%s/real_samples.png' % opt.outf,
                    normalize=True)
            fake = netG(fixed_noise)
            vutils.save_image(fake.data,
                    '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                    normalize=True)

    # do checkpointing
    torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
    torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
