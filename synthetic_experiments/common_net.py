import torch
import torch.nn as nn

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
        model += [nn.ReLU(inplace=True)]
        model += [nn.Linear(num_neurons,num_neurons)]
        if dropout > 0:
            model += [nn.Dropout(p=dropout)]
        self.model = nn.Sequential(*model)

    def forward(self,x):
        residual=x
        out=self.model(x)
        out+=residual
        return out
