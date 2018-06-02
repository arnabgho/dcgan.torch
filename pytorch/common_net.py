"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import cv2
#from init import *
import numpy as np
# custom weights initialization called on netG and netD

def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class GaussianSmoother(nn.Module):
  def __init__(self, kernel_size=5):
    super(GaussianSmoother, self).__init__()
    self.sigma = 0.3*((kernel_size-1)*0.5-1)+0.8
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel2d = np.dot(kernel.reshape(kernel_size,1),kernel.reshape(1,kernel_size))
    data = torch.Tensor(3, 1, kernel_size, kernel_size)
    self.pad = (kernel_size-1)/2
    for i in range(0,3):
      data[i,0,:,:] = torch.from_numpy(kernel2d)
    self.blur_kernel = Variable(data, requires_grad=False)

  def forward(self, x):
    out = nn.functional.pad(x, [self.pad, self.pad, self.pad, self.pad], mode ='replicate')
    out = nn.functional.conv2d(out, self.blur_kernel, groups=3)
    return out

  def cuda(self, gpu):
    self.blur_kernel = self.blur_kernel.cuda(gpu)

class INSResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(INSResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class BATCHResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(BATCHResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    model += [nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    model += [nn.BatchNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class ResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(ResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    #model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    #model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class upDeconvBATCHResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(upDeconvBATCHResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, inplanes, stride)]
    model += [nn.BatchNorm2d(inplanes)]
    model += [nn.ReLU(inplace=True)]
    model += [ nn.ConvTranspose2d( inplanes , inplanes , 4, 2, 1, bias=False)]
    model += [nn.BatchNorm2d(inplanes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
  def forward(self, x):
    residual = self.upsample(x)
    out = self.model(x)
    out += residual
    out = self.final(out)
    return out

class upDeconvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(upDeconvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, inplanes, stride)]
    #model += [nn.BatchNorm2d(inplanes)]
    model += [nn.ReLU(inplace=True)]
    model += [nn.ConvTranspose2d( inplanes , inplanes , 4, 2, 1, bias=False)]
    #model += [nn.BatchNorm2d(inplanes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
    self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
  def forward(self, x):
    residual = self.upsample(x)
    out = self.model(x)
    out += residual
    out = self.final(out)
    return out

class downConvBATCHResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(downConvBATCHResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, inplanes, stride)]
    model += [nn.BatchNorm2d(inplanes)]
    model += [nn.ReLU(inplace=True)]
    model += [nn.Conv2d( inplanes , inplanes , 4, 2, 1, bias=False  )]
    model += [nn.BatchNorm2d(inplanes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
    self.downsample = nn.AvgPool2d((2, 2), stride=(2, 2))
  def forward(self, x):
    residual = self.downsample(x)
    out = self.model(x)
    out += residual
    out = self.final(out)
    return out

class downConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(downConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, inplanes, stride)]
    #model += [nn.BatchNorm2d(inplanes)]
    model += [nn.ReLU(inplace=True)]
    model += [nn.Conv2d( inplanes , inplanes , 4, 2, 1, bias=False  )]
    #model += [nn.BatchNorm2d(inplanes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
    self.downsample = nn.AvgPool2d((2, 2), stride=(2, 2))
  def forward(self, x):
    residual = self.downsample(x)
    out = self.model(x)
    out += residual
    out = self.final(out)
    return out


class UpResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(UpResBlock, self).__init__()

    self.model1=ResBlock(inplanes,inplanes,stride,dropout)
    self.model2=ResBlock(inplanes,inplanes,stride,dropout)
    self.model3=ResBlock(inplanes,inplanes,stride,dropout)
    self.model4=ResBlock(inplanes,inplanes,stride,dropout)

    self.model1.apply(gaussian_weights_init)
    self.model2.apply(gaussian_weights_init)
    self.model3.apply(gaussian_weights_init)
    self.model4.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
  def forward(self, x):
    out1=self.model1(x)
    out2=self.model2(x)
    out3=self.model3(x)
    out4=self.model4(x)

    out12 = torch.cat([out1,out2],2)
    out34 = torch.cat([out3,out4],2)
    out = torch.cat([ out12, out34 ],3)
    out = self.final(out)
    return out

class UpBATCHResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(UpBATCHResBlock, self).__init__()

    self.model1=ResBlock(inplanes,inplanes,stride,dropout)
    self.model2=ResBlock(inplanes,inplanes,stride,dropout)
    self.model3=ResBlock(inplanes,inplanes,stride,dropout)
    self.model4=ResBlock(inplanes,inplanes,stride,dropout)

    self.model1.apply(gaussian_weights_init)
    self.model2.apply(gaussian_weights_init)
    self.model3.apply(gaussian_weights_init)
    self.model4.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
  def forward(self, x):
    out1=self.model1(x)
    out2=self.model2(x)
    out3=self.model3(x)
    out4=self.model4(x)

    out12 = torch.cat([out1,out2],2)
    out34 = torch.cat([out3,out4],2)
    out = torch.cat([ out12, out34 ],3)
    out = self.final(out)
    return out

class UpRANDBATCHResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(UpRANDBATCHResBlock, self).__init__()

    self.model1=ResBlock(inplanes,inplanes,stride,dropout)
    self.model2=ResBlock(inplanes,inplanes,stride,dropout)
    self.model3=ResBlock(inplanes,inplanes,stride,dropout)
    self.model4=ResBlock(inplanes,inplanes,stride,dropout)

    self.model1.apply(gaussian_weights_init)
    self.model2.apply(gaussian_weights_init)
    self.model3.apply(gaussian_weights_init)
    self.model4.apply(gaussian_weights_init)
    self.final = self.conv3x3(inplanes,planes,stride)
  def forward(self, x):
    out1=self.model1(x)
    out2=self.model2(x)
    out3=self.model3(x)
    out4=self.model4(x)

    choice=np.random.randint(2)
    if choice==0:
        out12 = torch.cat([out1,out2],2)
        out34 = torch.cat([out3,out4],2)
    elif choice==1:
        out12 = torch.cat([out3,out4],2)
        out34 = torch.cat([out1,out2],2)

    out = torch.cat([ out12, out34 ],3)
    out = self.final(out)
    return out

class MAX_SELECTResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1):
    return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0):
    super(MAX_SELECTResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride)]
    #model += [nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes)]
    #model += [nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out = torch.max(out,residual)
    return out
class LeakyReLUConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
    super(LeakyReLUConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=True)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUBNConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUBNConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)]
    model += [nn.BatchNorm2d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUBNConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0, output_padding=0):
    super(LeakyReLUBNConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, output_padding=output_padding, bias=False)]
    model += [nn.BatchNorm2d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class ReLUINSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(ReLUINSConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class ReLUINSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding, output_padding):
    super(ReLUINSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding,
                         output_padding=output_padding, bias=True)]
    model += [nn.InstanceNorm2d(n_out, affine=False)]
    model += [nn.ReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class GaussianNoiseLayer(nn.Module):
  def __init__(self,):
    super(GaussianNoiseLayer, self).__init__()

  def forward(self, x):
    if self.training == False:
      return x
    noise = Variable(torch.randn(x.size()).cuda(x.data.cuda().get_device()))
    return x + noise

class GaussianVAE2D(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(GaussianVAE2D, self).__init__()
    self.en_mu = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
    self.en_sigma = nn.Conv2d(n_in, n_out, kernel_size, stride, padding)
    self.softplus = nn.Softplus()
    self.reset_parameters()

  def reset_parameters(self):
    self.en_mu.weight.data.normal_(0, 0.002)
    self.en_mu.bias.data.normal_(0, 0.002)
    self.en_sigma.weight.data.normal_(0, 0.002)
    self.en_sigma.bias.data.normal_(0, 0.002)

  def forward(self, x):
    mu = self.en_mu(x)
    sd = self.softplus(self.en_sigma(x))
    return mu, sd

  def sample(self, x):
    mu = self.en_mu(x)
    sd = self.softplus(self.en_sigma(x))
    noise = Variable(torch.randn(mu.size(0), mu.size(1), mu.size(2), mu.size(3))).cuda(x.data.cuda().get_device())
    return mu + sd.mul(noise), mu, sd

class Bias2d(nn.Module):
  def __init__(self, channels):
    super(Bias2d, self).__init__()
    self.bias = nn.Parameter(torch.Tensor(channels))
    self.reset_parameters()

  def reset_parameters(self):
    self.bias.data.normal_(0, 0.002)

  def forward(self, x):
    n, c, h, w = x.size()
    return x + self.bias.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand(n, c, h, w)

class LeakyReLUBNNSConv2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUBNNSConv2d, self).__init__()
    model = []
    model += [nn.Conv2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.BatchNorm2d(n_out, affine=False)]
    model += [Bias2d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)

class LeakyReLUBNNSConvTranspose2d(nn.Module):
  def __init__(self, n_in, n_out, kernel_size, stride, padding=0):
    super(LeakyReLUBNNSConvTranspose2d, self).__init__()
    model = []
    model += [nn.ConvTranspose2d(n_in, n_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=True)]
    model += [nn.BatchNorm2d(n_out, affine=False)]
    model += [Bias2d(n_out)]
    model += [nn.LeakyReLU(inplace=True)]
    self.model = nn.Sequential(*model)
    self.model.apply(gaussian_weights_init)

  def forward(self, x):
    return self.model(x)
