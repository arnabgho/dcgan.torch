import argparse
import os
import numpy as np
import math

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from common_net import *

parser = argparse.ArgumentParser()
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout value')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--latent_dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--n_classes', type=int, default=10, help='number of classes for dataset')
parser.add_argument('--img_size', type=int, default=28, help='size of each image dimension')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--nc', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=400, help='interval between image sampling')
parser.add_argument('--ngf', type=int, default=16)
parser.add_argument('--ngf_gate', type=int, default=32)
parser.add_argument('--ndf', type=int, default=16)
parser.add_argument('--ndf_gate', type=int, default=32)
parser.add_argument('--ngres', type=int, default=32)
parser.add_argument('--ngres_gate', type=int, default=32)
parser.add_argument('--ndres', type=int, default=32)
parser.add_argument('--ndres_gate', type=int, default=32)
parser.add_argument('--spectral_G', type=bool, default=False)
parser.add_argument('--spectral_D', type=bool, default=True)
parser.add_argument('--outf', default='./conv_gated_resnet_resnet_gen/', help='folder to output images and model checkpoints')
parser.add_argument('--dataset', default='MNIST', help='folder to output images and model checkpoints')
opt = parser.parse_args()
print(opt)
try:
    os.makedirs(opt.outf)
except OSError:
    pass
opt.nsalient=opt.n_classes
opt.nnoise = opt.latent_dim
opt.batchSize = opt.batch_size
img_shape = (opt.channels, opt.img_size, opt.img_size)

cuda = True if torch.cuda.is_available() else False


class block(nn.Module):
    def __init__(self,in_feat,out_feat,normalize=True):
        super(block,self).__init__()

        layers = [  nn.Linear(in_feat, out_feat)]
        if normalize:
            layers.append(nn.BatchNorm1d(out_feat, 0.8))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.model=nn.Sequential(*layers)

    def forward(self,x):
        output = self.model(x)
        return output


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.label_emb = nn.Embedding(opt.n_classes, opt.n_classes)



        self.model = nn.Sequential(
            block(opt.latent_dim+opt.n_classes, 128, normalize=False),
            block(128, 256),
            block(256, 512),
            block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # Concatenate label embedding and image to produce input
        gen_input = torch.cat((self.label_emb(labels), noise), -1)
        img = self.model(gen_input)
        img = img.view(img.size(0), *img_shape)
        return img

class GatedResnetConvResnetG(nn.Module):
    def __init__(self,opt):
        super(GatedResnetConvResnetG, self).__init__()
        self.opt=opt

        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        self.main_initial = nn.Sequential(
          nn.ConvTranspose2d(opt.nnoise, 4*opt.ngf, 1, 1, bias=False),
          nn.BatchNorm2d(4*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(4*opt.ngf, 2*opt.ngf, 7, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, 2*opt.ngf, 4, 2, 1, bias=False),
          nn.BatchNorm2d(2*opt.ngf),
          nn.ReLU(True),
          nn.ConvTranspose2d(2*opt.ngf, opt.ngf, 4, 2, 1, bias=False),
          #nn.Sigmoid()
        )

        main_block=[]
        #Input is z going to series of rsidual blocks

        # Sets of residual blocks start

        for i in range(opt.ngres):
            main_block+= [GatedConvResBlock(opt.ngf,opt.ngf,dropout=opt.dropout,use_sn=opt.spectral_G)] #[BATCHResBlock(opt.ngf,opt.dropout)]


        # Final layer to map to 1 channel

        main_block+=[nn.Conv2d(opt.ngf,opt.nc,kernel_size=3,stride=1,padding=1)]
        main_block+=[nn.Tanh()]
        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ nn.Linear(opt.nsalient ,opt.ngf_gate)]
        #gate_block+=[ nn.BatchNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        for i in range(opt.ngres_gate):
            gate_block+=[ResBlock(opt.ngf_gate,opt.dropout)]
        gate_block+=[ nn.Linear(opt.ngf_gate,opt.ngres) ]
        gate_block+= [ nn.Sigmoid()]# [nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate=nn.Sequential(*gate_block)

    def forward(self, noise , labels):
        input_gate = self.label_embedding(labels)
        input_main = noise
        input_main = input_main.resize(self.opt.batchSize,self.opt.nnoise,1,1)

        output_gate = self.gate(input_gate)
        output = self.main_initial(input_main)
        for i in range(self.opt.ngres):
            alpha = output_gate[:,i]
            alpha = alpha.resize(self.opt.batchSize,1,1,1)
            output=self.main[i](output,alpha)

        output=self.main[self.opt.ngres](output)
        output=self.main[self.opt.ngres+1](output)
        return output


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.label_embedding = nn.Embedding(opt.n_classes, opt.n_classes)

        self.model = nn.Sequential(
            nn.Linear(opt.n_classes + int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.Dropout(0.4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Concatenate label embedding and image to produce input
        d_in = torch.cat((img.view(img.size(0), -1), self.label_embedding(labels)), -1)
        validity = self.model(d_in)
        return validity

class GatedResnetConvResnetD(nn.Module):
    def __init__(self,opt):
        super(GatedResnetConvResnetD, self).__init__()
        self.opt=opt

        self.label_embedding = nn.Embedding(opt.n_classes, opt.nsalient)
        if opt.spectral_D:
            self.main_latter = nn.Sequential(
            spectral_norm(nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1)),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(opt.ndf, 2*opt.ndf, 4, 2, 1, bias=False)),
            nn.BatchNorm2d(2*opt.ndf),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(2*opt.ndf, 2*opt.ndf, 7, bias=False)),
            nn.BatchNorm2d(2*opt.ndf),
            nn.LeakyReLU(0.1, inplace=True),
            spectral_norm(nn.Conv2d(2*opt.ndf, 1, 1)),
            nn.Sigmoid()
            )
        else:
            self.main_latter = nn.Sequential(
            nn.Conv2d(opt.ndf, opt.ndf, 4, 2, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(opt.ndf, 2*opt.ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(2*opt.ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2*opt.ndf, 2*opt.ndf, 7, bias=False),
            nn.BatchNorm2d(2*opt.ndf),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(2*opt.ndf, 1, 1),
            nn.Sigmoid()
            )


        main_block=[]
        #Input is z going to series of rsidual blocks
        # First layer to map to ndf channel
        if opt.spectral_D:
            main_block+=[spectral_norm(nn.Conv2d(opt.nc,opt.ndf,kernel_size=3,stride=1,padding=1))]
        else:
            main_block+=[nn.Conv2d(opt.nc,opt.ndf,kernel_size=3,stride=1,padding=1)]
        # Sets of residual blocks start

        for i in range(opt.ndres):
            main_block+= [GatedConvResBlock(opt.ndf,opt.ndf,dropout=opt.dropout,use_sn=opt.spectral_D)] #[BATCHResBlock(opt.ngf,opt.dropout)]


        self.main=nn.Sequential(*main_block)

        gate_block =[]
        gate_block+=[ nn.Linear(opt.nsalient ,opt.ndf_gate)]
        #gate_block+=[ nn.BatchNorm1d(opt.ngf_gate) ]
        gate_block+=[ nn.ReLU()]
        for i in range(opt.ndres_gate):
            gate_block+=[ResBlock(opt.ndf_gate,opt.dropout)]
        gate_block+=[ nn.Linear(opt.ndf_gate,opt.ndres) ]
        gate_block+= [nn.Sigmoid()] #[nn.Softmax()]  #[ nn.Sigmoid()]

        self.gate=nn.Sequential(*gate_block)

    def forward(self, img, labels):
        input_gate = self.label_embedding(labels)
        input_main = img

        output_gate = self.gate(input_gate)
        output = self.main[0](img)
        for i in xrange(1,1+self.opt.ndres):
            alpha = output_gate[:,i-1]
            alpha = alpha.resize(self.opt.batchSize,1,1,1)
            output=self.main[i](output,alpha)

        output = self.main_latter(output)
        return output
# Loss functions
adversarial_loss = torch.nn.BCELoss()  #torch.nn.MSELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()

# Initialize generator and discriminator
#generator = Generator()
#discriminator = Discriminator()
generator = GatedResnetConvResnetG(opt)
discriminator = GatedResnetConvResnetD(opt)

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()

# Configure data loader
if opt.dataset=="CIFAR":
    dataset = datasets.CIFAR10('./cifar_dataset', transform=transforms.Compose([ transforms.CenterCrop(opt.img_size) ,  transforms.ToTensor()]), download=True)
elif opt.dataset=="MNIST":
    dataset= datasets.MNIST('./mnist', train=True, download=True,
                   transform=transforms.Compose([
                        transforms.Resize(opt.img_size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                   ]))
dataloader = torch.utils.data.DataLoader( dataset,batch_size=opt.batch_size, shuffle=True)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=10*opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


def sample_image(n_row, batches_done):
    """Saves a grid of generated digits ranging from 0 to n_classes"""
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (n_row**2, opt.latent_dim))))
    # Get labels ranging from 0 to n_classes for n rows
    labels = np.array([num for _ in range(n_row) for num in range(n_row)])
    labels = Variable(LongTensor(labels))
    gen_imgs = generator(z, labels)
    save_image(gen_imgs.data,'%s/%d.png' % (opt.outf,batches_done), nrow=n_row, normalize=True)

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # Adversarial ground truths
        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(FloatTensor))
        labels = Variable(labels.type(LongTensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))

        # Generate a batch of images
        gen_imgs = generator(z, gen_labels)

        # Loss measures generator's ability to fool the discriminator
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss for real images
        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        # Loss for fake images
        validity_fake = discriminator(gen_imgs.detach(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # Total discriminator loss
        d_loss = (d_real_loss + d_fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [D_real: %f] [D_fake: %f]" % (epoch, opt.n_epochs, i, len(dataloader),
                                                            d_loss.item(), g_loss.item() , d_fake_loss.item() , d_real_loss.item() ))

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

            # do checkpointing
            torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, batches_done))
            torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, batches_done))
