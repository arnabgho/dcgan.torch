import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)

class ResBlock1D(nn.Module):
    def __init__(self,num_neurons,dropout=0.0):
        super(ResBlock1D, self).__init__()

        model = []
        model += [ nn.Conv1d( num_neurons , num_neurons  , kernel_size=3, stride=1,padding=1)  ]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU(inplace=True)]
        model += [ nn.Conv1d( num_neurons , num_neurons  , kernel_size=3, stride=1,padding=1)  ]
        model += [nn.BatchNorm1d(num_neurons)] # Just testing might be removed
        model += [nn.ReLU()]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out

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

def residual_op(out,residual,op='add'):
    if op == 'add':
        return out + residual
    elif op=='sub':
        return out - residual
    elif op == 'max':
        return torch.max(out,residual)
    elif op == 'min':
        return torch.min(out,residual)


class ResBlock(nn.Module):
    def linear(self,num_neurons,use_sn=True):
        if use_sn:
            return spectral_norm(nn.Linear(num_neurons,num_neurons))
        else:
            return nn.Linear(num_neurons,num_neurons)
    def __init__(self,num_neurons,dropout=0.0,use_sn=True):
        super(ResBlock, self).__init__()

        model = []
        model += [self.linear(num_neurons,use_sn)] #[nn.Linear(num_neurons,num_neurons)]
        model += [nn.ReLU(inplace=True)]
        model += [self.linear(num_neurons,use_sn)]  #nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x,alpha=1.0,beta=0.0):
        if type(alpha)!=float:
            alpha=alpha.expand_as(x)
        if type(beta)!=float:
            beta=beta.expand_as(x)
        residual=x + beta
        out=self.model(x)
        #out+=residual
        out = alpha*out
        out = residual_op(out,residual)
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
