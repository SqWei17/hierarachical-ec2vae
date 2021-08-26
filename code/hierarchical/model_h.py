import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal

__all__ = ['Decoder8to4', 'Decoder4to2']

class Decoder8to4(nn.Module):
    def __init__(self,hidden_dims,
                 z1p_dim,z1r_dim,
                 z2p_dim,z2r_dim,
                 n_step,
                 k=1000):
        super(Decoder8to4, self).__init__()
        self.grucell_0 = nn.GRUCell(z1p_dim+z2p_dim,
                                    hidden_dims)
        self.grucell_1 = nn.GRUCell(
            z1r_dim+z2r_dim, hidden_dims)
        self.linear_init_0 = nn.Linear(z1p_dim, hidden_dims)
        self.linear_init_1 = nn.Linear(z1r_dim, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, z2p_dim)
        self.linear_out_1 = nn.Linear(hidden_dims, z2r_dim)
        self.linear_out_0_ = nn.Linear(hidden_dims, z2p_dim)
        self.linear_out_1_ = nn.Linear(hidden_dims, z2r_dim)
        self.n_step = n_step
        self.hidden_dims = hidden_dims
        self.z1p_dim = z1p_dim
        self.z1r_dim = z1r_dim
        self.z2p_dim = z2p_dim
        self.z2r_dim = z2r_dim
        self.eps = 1
        self.samplep = None
        self.sampler = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])


    def final_decoder84p(self, z):
        out = torch.zeros((z.size(0), self.z2p_dim))
        out[:, -1] = 1.
        x, hx = [], [None]
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out,  z], 1)
            hx = self.grucell_0(out, hx)
            out = self.linear_out_0(hx)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = out
                    #out = self.samplep[:, i, :]
                    #out = out.squeeze(1)
                else:
                    out = out
            else:
                out = out
        return torch.stack(x, 1)

    def final_decoder84r(self, z):
        out = torch.zeros((z.size(0), self.z2r_dim))
        out[:, -1] = 1.
        x, hx = [], [None]
        t = torch.tanh(self.linear_init_1(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out,  z], 1)
            hx = self.grucell_1(out, hx)
            out = self.linear_out_1(hx)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = out
                   # out = self.sampler[:, i, :]
                   # out = out.squeeze(1)
                else:
                    out = out
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
            else:
                out = out
        return torch.stack(x, 1)



    def forward(self, z_8p,z_8r,z_4p,z_4r):
        if self.training:
            self.samplep = z_4p
            self.sampler = z_4r
            self.iteration += 1
        z4p = self.final_decoder84p(z_8p)
        z4r=   self.final_decoder84r(z_8r)
        output = (z4p,z4r)
        return output


class Decoder4to2(nn.Module):
    def __init__(self,hidden_dims,
                 z1p_dim,z1r_dim,
                 z2p_dim,z2r_dim,
                 n_step,
                 k=1000):
        super(Decoder4to2, self).__init__()
        self.grucell_0 = nn.GRUCell(z1p_dim+z2p_dim,
                                    hidden_dims)
        self.grucell_1 = nn.GRUCell(
            z1r_dim+z2r_dim, hidden_dims)
        self.linear_init_0 = nn.Linear(z1p_dim, hidden_dims)
        self.linear_init_1 = nn.Linear(z1r_dim, hidden_dims)
        self.linear_out_0 = nn.Linear(hidden_dims, z2p_dim)
        self.linear_out_1 = nn.Linear(hidden_dims, z2r_dim)
        self.linear_out_0_ = nn.Linear(hidden_dims, z2p_dim)
        self.linear_out_1_ = nn.Linear(hidden_dims, z2r_dim)
        self.n_step = n_step
        self.hidden_dims = hidden_dims
        self.z1p_dim = z1p_dim
        self.z1r_dim = z1r_dim
        self.z2p_dim = z2p_dim
        self.z2r_dim = z2r_dim
        self.eps = 1
        self.samplep = None
        self.sampler = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])


    def final_decoder42p(self, z):
        out = torch.zeros((z.size(0), self.z2p_dim))
        out[:, -1] = 1.
        x, hx = [], [None]
        t = torch.tanh(self.linear_init_0(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out,  z], 1)
            hx = self.grucell_0(out, hx)
            out = self.linear_out_0(hx)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out  = out
                    #out = self.samplep[:, i, :]
                    #out = out.squeeze(1)
                else:
                    out = out
            else:
                out = out
        return torch.stack(x, 1)

    def final_decoder42r(self, z):
        out = torch.zeros((z.size(0), self.z2r_dim))
        out[:, -1] = 1.
        x, hx = [], [None]
        t = torch.tanh(self.linear_init_1(z))
        hx = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out,  z], 1)
            hx = self.grucell_1(out, hx)
            out = self.linear_out_1(hx)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = out
                    #out = self.sampler[:, i, :]
                    #out = out.squeeze(1)
                else:
                    out = out
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
            else:
                out = out
        return torch.stack(x, 1)



    def forward(self, z_4p,z_4r,z_2p,z_2r):
        if self.training:
            self.samplep = z_2p
            self.sampler = z_2r
            self.iteration += 1
        z2p = self.final_decoder42p(z_4p)
        z2r =  self.final_decoder42r(z_4r)
        output = (z2p,z2r)
        return output
