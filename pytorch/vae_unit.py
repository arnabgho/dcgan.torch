from __future__ import print_function
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
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
from common_net import *
import visdom

vis = visdom.Visdom()
vis.env = 'vae_unit'

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=216, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=512, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00002, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int,default=1, help='manual seed')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
opt = parser.parse_args()
args = parser.parse_args()
args.cuda = opt.cuda and torch.cuda.is_available()

try:
    os.makedirs(opt.outf)
except OSError:
    pass

torch.manual_seed(opt.manualSeed)
if args.cuda:
    torch.cuda.manual_seed(opt.manualSeed)
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
train_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
test_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
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



kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
#train_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=True, download=True,
#                   transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)
#test_loader = torch.utils.data.DataLoader(
#    datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
#    batch_size=args.batch_size, shuffle=True, **kwargs)


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        input_dim_a = 3
        ch = 64
        n_enc_front_blk  = 3
        n_enc_res_blk    = 3
        n_enc_shared_blk = 1
        n_gen_shared_blk = 1
        n_gen_res_blk    = 3
        n_gen_front_blk  = 6
        encA = []

        encA += [LeakyReLUConv2d(input_dim_a, ch, kernel_size=7, stride=2, padding=1)]
        tch=ch
        for i in range(0,n_enc_front_blk):
            encA += [ReLUINSConv2d(tch, tch * 2, kernel_size=3, stride=2, padding=1)]
            tch *= 2
        fch=tch
        for i in range(0, n_enc_res_blk):
            encA += [INSResBlock(tch, tch)]

        for i in range(0, n_enc_shared_blk):
            encA += [INSResBlock(tch, tch)]

        encA += [GaussianNoiseLayer()]

        self.encoder=nn.Sequential(*encA)

        decA = []
        for i in range(0, n_gen_shared_blk):
            decA += [INSResBlock(tch, tch)]
        # Decoders
        for i in range(0, n_gen_res_blk):
            decA += [INSResBlock(tch, tch)]
        for i in range(0, n_gen_front_blk):
            decA += [ReLUINSConvTranspose2d(tch, tch//2, kernel_size=3, stride=2, padding=1, output_padding=1)]
            tch = tch//2
        decA += [nn.ConvTranspose2d(tch, input_dim_a, kernel_size=1, stride=1, padding=0)]
        decA += [nn.Tanh()]

        self.decoder=nn.Sequential(*decA)

        #self.decoder = nn.Sequential(
        #    # input is Z, going into a convolution
        #    nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
        #    nn.BatchNorm2d(ngf * 8),
        #    nn.ReLU(True),
        #    # state size. (ngf*8) x 4 x 4
        #    nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 4),
        #    nn.ReLU(True),
        #    # state size. (ngf*4) x 8 x 8
        #    nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf * 2),
        #    nn.ReLU(True),
        #    # state size. (ngf*2) x 16 x 16
        #    nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ngf),
        #    nn.ReLU(True),
        #    # state size. (ngf) x 32 x 32
        #    nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
        #    nn.Tanh()
        #    # state size. (nc) x 64 x 64
        #)

        #self.encoder = nn.Sequential(
        #    # input is (nc) x 64 x 64
        #    nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. (ndf) x 32 x 32
        #    nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ndf * 2),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. (ndf*2) x 16 x 16
        #    nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ndf * 4),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. (ndf*4) x 8 x 8
        #    nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        #    nn.BatchNorm2d(ndf * 8),
        #    nn.LeakyReLU(0.2, inplace=True),
        #    # state size. (ndf*8) x 4 x 4
        #    nn.Conv2d(ndf * 8, nz, 4, 1, 0, bias=False),
        #    nn.Sigmoid()
        #)

        #self.fc1 = nn.Linear(784, 400)
        self.fc21 =   nn.Conv2d(fch, nz, 4, 1, 0, bias=False)     #nn.Linear(nz, nz)
        self.fc22 =   nn.Conv2d(fch, nz, 4, 1, 0, bias=False)   #nn.Linear(nz, nz)
        #self.fc3 = nn.Linear(20, 400)
        #self.fc4 = nn.Linear(400, 784)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x):
        h1 = self.encoder(x)
        return  self.fc21(h1),self.fc22(h1)          #self.fc21(h1.view(h1.size(0),-1)), self.fc22(h1.view(h1.size(0),-1))

    def reparameterize(self, mu, logvar):
        if self.training:
          std = logvar.mul(0.5).exp_()
          eps = Variable(std.data.new(std.size()).normal_())
          return eps.mul(std).add_(mu)
        else:
          return mu

    def decode(self, z):
        return self.decoder(z.view(z.size(0),z.size(1),1,1))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

MSECriterion=nn.MSELoss()
model = VAE()
if args.cuda:
    model.cuda()
    MSECriterion.cuda()


def loss_function(recon_x, x, mu, logvar):
    #BCE = F.binary_cross_entropy(recon_x, x.view(-1, 784))
    #MSE=MSECriterion(recon_x,x)
    L1=torch.nn.L1Loss()
    MSE=L1(recon_x,x)
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # Normalise by same number of elements as in reconstruction
    KLD /= args.batchSize * opt.imageSize * opt.imageSize
    #KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    #KLD = torch.sum(KLD_element).mul_(-0.5)

    return MSE + KLD


optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(opt.beta1,0.999))

data_win=None
rec_win=None
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = Variable(data)
        if args.cuda:
            data = data.cuda()
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        data_win=vis.image(data.data[0].cpu()*0.5+0.5) #,win=data_win)
        rec_win=vis.image(recon_batch.data[0].cpu()*0.5+0.5) # ,win=rec_win)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.data[0]
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.data[0] / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))


def test(epoch):
    model.eval()
    test_loss = 0
    for i, (data, _) in enumerate(test_loader):
        if args.cuda:
            data = data.cuda()
        data = Variable(data, volatile=True)
        recon_batch, mu, logvar = model(data)
        test_loss += loss_function(recon_batch, data, mu, logvar).data[0]
        if i == 0:
          n = min(data.size(0), 8)
          comparison = torch.cat([data[:n],
                                  recon_batch.view(args.batchSize, 3, opt.imageSize, opt.imageSize)[:n]])
          save_image(comparison.data.cpu(),
                     opt.outf + '/reconstruction_' + str(epoch) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


for epoch in range(1, opt.niter + 1):
    train(epoch)
    test(epoch)
    sample = Variable(torch.randn(opt.batchSize, nz))
    if args.cuda:
       sample = sample.cuda()
    sample = model.decode(sample).cpu()
    save_image(sample.data.view(opt.batchSize, 3, opt.imageSize, opt.imageSize),
               opt.outf + '/sample_' + str(epoch) + '.png')
