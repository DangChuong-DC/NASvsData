import torch
import torch.nn as nn
import copy

from modules.basic_operations import *
from modules.genotypes import SUPER_PRIMITIVES, CHILD_PRIMITIVES


class MixedEdge(nn.Module):

    def __init__(self, C, stride, is_bequeath):
        super(MixedEdge, self).__init__()
        self.C = C
        self.stride = stride
        self.is_bequeath = is_bequeath

        self._opers = nn.ModuleList()
        if is_bequeath:
            for name in SUPER_PRIMITIVES:
                op = CANDIDATES[name](C, stride, False)
                self._opers.append(op)
        else:
            for name in CHILD_PRIMITIVES:
                op = CANDIDATES[name](C, stride, True)
                self._opers.append(op)

    def forward(self, x, ops_binaries):
        output = []
        has_residual = True if ops_binaries[0] == 1 else False
        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                m_oi = self._opers[idx](x, has_residual)
                output.append(m_oi*bin)
            else:
                m_oi = self._opers[idx](x, has_residual)
                output.append(m_oi.detach()*bin)
        output = torch.sum(torch.stack(output, dim=0), dim=0)
        return output

    def set_edge_ops(self, ops_binaries):
        assert ops_binaries.size(-1) == 2, 'Wrong input for alphas'
        opers = []
        if self.is_bequeath:
            has_residual = True if ops_binaries[0] == 1 else False
        else:
            has_residual = None

        for idx, bin in enumerate(ops_binaries):
            if bin == 1:
                opers.append(self._opers[idx])
        if len(opers) == 0:
            opers.append(CANDIDATES['zero'](self.C, self.stride, not self.is_bequeath))
        return Edge(opers, has_residual)


class Edge(nn.Module):

    def __init__(self, opers, has_residual):
        super(Edge, self).__init__()
        self._opers = nn.ModuleList(opers)
        self.has_residual = has_residual

    def forward(self, x, *arg):
        output = []
        for op in self._opers:
            output.append(op(x, self.has_residual))
        output = torch.sum(torch.stack(output, dim=0), dim=0)
        return output


class Cell(nn.Module):

    def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev, is_bequeath):
        super(Cell, self).__init__()
        self.reduction = reduction
        self.steps = steps
        self.multiplier = multiplier
        self.is_bequeath = is_bequeath

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)

        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        self._edges = nn.ModuleList()
        self._compile(C, reduction, is_bequeath)

    def _compile(self, C, reduction, is_bequeath):
        for i in range(self.steps):
            for j in range(2 + i):
                stride = 2 if reduction and j < 2 else 1
                op = MixedEdge(C, stride, is_bequeath)
                self._edges.append(op)
        self.ops_alphas = None

    def forward(self, s0, s1):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        offset = 0
        for i in range(self.steps):
            s = sum(self._edges[offset + j](h, self.ops_alphas[offset + j]) for j, h in enumerate(states))
            offset += len(states)
            states.append(s)
        return torch.cat(states[-self.multiplier:], dim=1)

    def generate_rand_alphas(self, drop_prob=0.5):
        k = sum(1 for i in range(self.steps) for n in range(2 + i))
        num_op = len(SUPER_PRIMITIVES) if self.is_bequeath else len(CHILD_PRIMITIVES)
        keep_prob = 1. - drop_prob
        ops_alps = torch.rand(k, num_op).bernoulli_(keep_prob)
        self.ops_alphas = ops_alps
        return self.ops_alphas

    def set_edge_fixed(self, ops_matrix):
        for idx, edg in enumerate(self._edges):
            self._edges[idx] = edg.set_edge_ops(ops_matrix[idx])
