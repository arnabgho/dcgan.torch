import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
def gaussian_weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class BATCHResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(BATCHResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out


class INSResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(INSResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.InstanceNorm1d(num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.InstanceNorm1d(num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class ResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(ResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class GatedResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(GatedResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x,alpha):
        residual=x
        out=alpha*self.model(x)
        out+=residual
        return out

class GatedConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)

  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
    super(GatedConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]  #[nn.InstanceNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]  #[nn.InstanceNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    #self.model.apply(gaussian_weights_init)

  def forward(self, x,alpha):
    residual = x
    out = alpha*self.model(x)
    out += residual
    return out

class ConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
    super(ConvResBlock, self).__init__()
    model = []
    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)
    #self.model.apply(gaussian_weights_init)

  def forward(self, x):
    residual = x
    out = self.model(x)
    out += residual
    return out

class UpConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
    super(UpConvResBlock, self).__init__()
    model = []
    model += [nn.Upsample( scale_factor=2,mode='nearest')]
    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block += [nn.Upsample(scale_factor=2,mode='nearest')]
    residual_block += [self.conv3x3(inplanes,planes,stride,use_sn)]
    #self.model.apply(gaussian_weights_init)
    self.residual_block=nn.Sequential(*residual_block)

  def forward(self, x):
    residual = self.residual_block(x)
    out = self.model(x)
    out += residual
    return out

class DownConvResBlock(nn.Module):
  def conv3x3(self, inplanes, out_planes, stride=1,use_sn=True):
    if use_sn:
        return spectral_norm(nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1))
    else:
        return nn.Conv2d(inplanes, out_planes, kernel_size=3, stride=stride, padding=1)


  def __init__(self, inplanes, planes, stride=1, dropout=0.0,use_sn=False):
    super(DownConvResBlock, self).__init__()
    model = []
    model += [nn.AvgPool2d(2, stride=2)]
    model += [self.conv3x3(inplanes, planes, stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    model += [nn.ReLU(inplace=True)]
    model += [self.conv3x3(planes, planes,stride,use_sn)]
    model += [nn.BatchNorm2d(planes)]
    if dropout > 0:
      model += [nn.Dropout(p=dropout)]
    self.model = nn.Sequential(*model)

    residual_block = []
    residual_block += [self.conv3x3(inplanes,planes,stride,use_sn)]
    residual_block += [nn.AvgPool2d(2, stride=2)]
    #self.model.apply(gaussian_weights_init)
    self.residual_block=nn.Sequential(*residual_block)

  def forward(self, x):
    residual = self.residual_block(x)
    out = self.model(x)
    out += residual
    return out



class MAX_SELECTResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(MAX_SELECTResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out=torch.max(out,residual)
        return out

class MAX_PARALLELResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(MAX_PARALLELResBlock, self).__init__()

        model_1 = []
        model_1 += [nn.Linear(num_neurons,num_neurons)]
        model_1 += [nn.ReLU(inplace=True)]
        model_1 += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model_1 += [nn.Dropout(p=dropout)]
        self.model_1 = nn.Sequential(*model_1)

        model_2 = []
        model_2 += [nn.Linear(num_neurons,num_neurons)]
        model_2 += [nn.ReLU(inplace=True)]
        model_2 += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model_2 += [nn.Dropout(p=dropout)]
        self.model_2 = nn.Sequential(*model_2)


    def forward(self,x):
        residual=x
        out_1=self.model_1(x)
        out_2=self.model_2(x)
        out_max=torch.max(out_1,out_2)
        out = residual + out_max
        return out


class RELUResBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(RELUResBlock, self).__init__()

        model = []
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU()]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

class LinearRELUBlock(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(LinearRELUBlock,self).__init__()
        model=[]
        model += [nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU()]
        if dropout>0:
            model+= [nn.Dropout(p=dropout)]

        self.model = nn.Sequential(*model)

    def forward(self,x):
       out=self.model(x)
       return out
