import math
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

class Swish(nn.Module):
    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)

def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class ST_Joint_Att(nn.Module):
    def __init__(self, channel, reduct_ratio, bias, **kwargs):
        super(ST_Joint_Att, self).__init__()

        inner_channel = channel // reduct_ratio

        self.fcn = nn.Sequential(
            nn.Conv2d(channel, inner_channel, kernel_size=1, bias=bias),
            nn.BatchNorm2d(inner_channel),
            nn.Hardswish(),
        )

        self.conv_t = nn.Conv2d(inner_channel, channel, kernel_size=1)
        self.conv_v = nn.Conv2d(inner_channel, channel, kernel_size=1)

        self.func = nn.Sequential(
            nn.BatchNorm2d(inner_channel),
            Swish(),
        )
        self.swish = Swish(inplace=True)

    def forward(self, x):
        N, C, T, V = x.size()
        res = x
        x_t = x.mean(3, keepdims=True)
        x_v = x.mean(2, keepdims=True).transpose(2, 3)
        x_att = self.fcn(torch.cat([x_t, x_v], dim=2))
        x_t, x_v = torch.split(x_att, [T, V], dim=2)
        x_t_att = self.conv_t(x_t).sigmoid()
        x_v_att = self.conv_v(x_v.transpose(2, 3)).sigmoid()
        x_att = x_t_att * x_v_att
        return x_att * res

class APGC(nn.Module):
    def __init__(self, in_channels, out_channels, vertex_nums, pseudo_num, A, num_subset=8, rel_reduction=2):
        super(APGC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.pseudo_num = pseudo_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset
        mid_in_channels = in_channels // num_subset
        mid_out_channels = out_channels // num_subset
        self.mid_in_channels = mid_in_channels
        self.mid_out_channels = mid_out_channels
        self.hidden_channels = mid_in_channels // rel_reduction
        self.to_V = nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset)

        self.to_W = nn.Sequential(
            nn.Conv1d(in_channels, num_subset * self.hidden_channels, kernel_size=1, groups=num_subset),
            nn.LeakyReLU(),
            nn.Conv1d(num_subset * self.hidden_channels, num_subset, kernel_size=1),
            nn.Tanh()
        )
        pseudo_joint = torch.zeros(self.pseudo_num, in_channels)
        pseudo_joint[0, :] = 0.01
        pseudo_joint[1, :] = 0.0
        pseudo_joint[2, :] = -0.01
        # pseudo_joint[0, :] = 0.01 * random.randint(-10, 10)
        # pseudo_joint[1, :] = 0.01 * random.randint(-10, 10)
        # pseudo_joint[2, :] = 0.01 * random.randint(-10, 10)

        self.pseudo_joint = nn.Parameter(pseudo_joint)
        self.alpha = nn.Parameter(torch.ones(1))
        self.sigma_param = nn.Parameter(torch.ones(1))
        self.softmax = nn.Softmax(dim=-1)
        self.raw_weight = nn.Parameter(torch.tensor(0.5))
        self.sigmoid = nn.Sigmoid()

        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1, groups=num_subset)
        self.Ad = A
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=False)
        self.edge_importance = nn.Parameter(torch.ones(A.shape))

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        conv_init(self.to_V)
        conv_init(self.to_W[0])
        conv_init(self.to_W[2])
        conv_init(self.conv_d)

        self.conv1 = nn.Conv2d(self.in_channels, mid_out_channels, kernel_size=1, groups=num_subset)
        self.conv3 = nn.Conv2d(self.in_channels, self.out_channels, kernel_size=1, groups=num_subset)
        self.conv4 = nn.Conv2d(mid_out_channels, self.out_channels, kernel_size=1, groups=num_subset)
        self.tanh = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)

    def pseudo_norm(self, H, W):
        w = torch.diag_embed(W)
        norm_w = torch.norm(H, 1, dim=2, keepdim=True) + 1e-8
        w_ = w / norm_w
        H_w = H @ w
        norm_v = torch.norm(H_w, 1, dim=3, keepdim=True) + 1e-8
        h_ = H_w / norm_v
        A = h_ @ w_ @ H.transpose(3, 2)
        return A

    def a_norm(self, A):
        d_r = torch.norm(A, 1, dim=2, keepdim=True) + 1e-8
        return A / d_r

    def forward(self, x):
        N, C, T, V = x.size()
        h_x = self.pseudo_joint
        h_x = (h_x.T).unsqueeze(1)
        x = torch.cat([x, h_x.repeat(N, 1, T, 1)], dim=-1)
        V += self.pseudo_num
        A = self.PA.cuda(x.get_device())
        A = self.edge_importance * A
        A = self.a_norm(A)

        t_x = x.mean(2)
        v_x = self.to_V(t_x)
        dis_v_x = v_x.view(N, self.num_subset, self.hidden_channels, V)
        dis_v_x = dis_v_x.permute(0, 1, 3, 2).contiguous()
        distance_x = torch.cdist(dis_v_x, dis_v_x)   # 1, 8, 25, 25
        r = self.sigmoid(distance_x)

        median_dist = torch.median(distance_x.view(N, self.num_subset, V, V), dim=-1)[0]    # 32, 8, 25   32:batch_size
        mean_dist = torch.mean(distance_x.view(N, self.num_subset, V, V), dim=-1)
        dist_t = (median_dist + mean_dist) * 0.5
        rho = 0.7
        threshold = dist_t.unsqueeze(-1) * r * rho
        mask = distance_x < threshold
        exp_neg_dist = torch.exp(-distance_x) * mask
        sum_exp_neg_dist = exp_neg_dist.sum(dim=-1, keepdim=True)
        sum_exp_neg_dist = torch.where(sum_exp_neg_dist == 0, torch.ones_like(sum_exp_neg_dist), sum_exp_neg_dist)
        topk_v = exp_neg_dist / sum_exp_neg_dist

        H = topk_v
        W = self.to_W(t_x)
        H = self.pseudo_norm(H, W)
        alpha = self.alpha
        alpha = self.relu(alpha)
        A = A + alpha * H

        x1, x3 = self.conv1(x).mean(-2), self.conv3(x)
        x1 = self.tanh(x1.unsqueeze(-1) - x1.unsqueeze(-2))
        x1 = self.conv4(x1)
        x1 = torch.einsum('ncuv,nctv->nctu', x1, x3)
        x4 = self.tanh(x1.mean(-3).unsqueeze(-1) - x1.mean(-3).unsqueeze(-2))
        x3 = x3.permute(0, 2, 1, 3)
        x5 = torch.einsum('btmn,btcn->bctm', x4, x3)

        d_x = x1 + x5
        d_x = d_x.view(N, self.num_subset, self.mid_out_channels, T, V)

        y = torch.einsum('nkuv,nkctv->nkctu', A, d_x).contiguous()
        y = y.view(N, self.out_channels, T, V)
        x = x[..., :self.vertex_nums]
        y = y[..., :self.vertex_nums]
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)

        return y, self.pseudo_joint

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[2, 4, 6, 8],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = (len(dilations)) * 4
        branch_channels = out_channels // self.num_branches
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size] * len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_size, dilations):
            self.branches.append(nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    branch_channels,
                    kernel_size=1,
                    padding=0),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
                TemporalConv(
                    branch_channels,
                    branch_channels,
                    kernel_size=ks,
                    stride=stride,
                    dilation=dilation),
            ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)
        self.conv_up = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, kernel_size=1, padding=0, stride=(stride, 1)),
            nn.BatchNorm2d(out_channels),
            Swish(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.a = 0.5
        self.b = 0.5

    def forward(self, x):
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = self.conv_up(torch.cat(branch_outs, dim=1))
        out = (out * self.sigmoid(out) + out*self.a + res*self.b)
        return out

class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x

class Zero_Layer(nn.Module):
    def __init__(self):
        super(Zero_Layer, self).__init__()

    def forward(self, x):
        return 0

class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, pseudo_joints, A, stride=1, residual=True, kernel_size=5,
                 dilations=[1, 2]):
        super(TCN_GCN_unit, self).__init__()
        self.gcn = APGC(in_channels, out_channels, num_point, pseudo_joints, A)
        self.tcn = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                            dilations=dilations,
                                            residual=False)
        self.relu = Swish(inplace=True)

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x):
        res = self.residual(x)
        y, h_x = self.gcn(x)
        y = self.tcn(y)
        y = self.relu(y + res)
        return y, h_x

class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=9,
                 pseudo_joints=0,
                 drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(pseudo_joints, **graph_args)

        A = self.graph.A
        self.num_class = num_class
        self.num_point = num_point
        self.embedding_channels = 64

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.to_joint_embedding = nn.Conv2d(in_channels, self.embedding_channels, 1, bias=True)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        self.l1 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, pseudo_joints, A)
        self.l2 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, pseudo_joints, A)
        self.l3 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, pseudo_joints, A)
        self.fc = nn.Linear(self.embedding_channels, self.num_class)
        self.att = ST_Joint_Att(self.embedding_channels, 8, True)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        self.drop1 = nn.Dropout(0.2)
        self.drop2 = nn.Dropout(0.4)
        self.drop3 = nn.Dropout(0.4)

    def forward(self, x):
        N, C, T, V, M = x.size()
        x = x.view(N * M, C, T, V).contiguous()
        x = self.to_joint_embedding(x).view(N, -1, T, V, M).permute(0, 4, 2, 3, 1).contiguous()
        x += self.pos_embedding[:, :self.num_point]
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N, M * V * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, self.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, self.embedding_channels, T, V)
        res1 = x
        x1, h_x1 = self.l1(x)
        x2, h_x2 = self.l2(x + x1)
        x3, h_x3 = self.l3(x + x2)
        x = x3 + res1
        x = self.drop1(x)
        x= self.att(x)
        x = self.drop2(x)
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop3(x)

        return self.fc(x), [h_x1, h_x2, h_x3]
