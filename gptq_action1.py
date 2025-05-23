import math
import time

import torch
import torch.nn as nn
import transformers

from quant_testChao import *
# from quant_Chao import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


class GPTQ:

    def __init__(self, layer):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        # inp = inp.float()
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())

    def fasterquant(
        self, blocksize=1, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, name=None, layer_idx=None, R_aver_container=None, ele_sum_container=None, R_record_container=None, col_idx=None, alpha=None
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        # if not self.quantizer.ready():
        #     self.quantizer.find_params(W, weight=True)
        W = W.flatten(1)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            invperm = torch.argsort(perm)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        
        H_ori = H.clone()
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        device = self.quantizer.scale.device
        Nrow = W.shape[0]
        Ncol = W.shape[1]

        W_mean = W.mean(dim=1, keepdim=True)
        W_mean = W_mean.to(device)
        
        tmp = torch.zeros(W.shape[0], device=W.device)
        Wmax_vec = torch.maximum(torch.max(W, dim=1).values, tmp)  # clamp max to ≥ 0
        Wmin_vec = torch.minimum(torch.min(W, dim=1).values, tmp)  # clamp min to ≤ 0
        Wmax_vec = Wmax_vec.to(device)
        Wmin_vec = Wmin_vec.to(device)

        tmp = (Wmin_vec == 0) & (Wmax_vec == 0)
        Wmin_vec[tmp] = -1
        Wmax_vec[tmp] = +1
        
        # d_vec = diag(Hinv)
        d_vec = torch.diag(Hinv)  # shape [Ncol]
        d_vec = d_vec.to(device)
        
        ratio = torch.tensor(0.00007, device=device)
        # ratio = ratio * alpha[layer_idx, col_idx]
        # C = torch.sum(Wmax_vec - Wmin_vec) / Nrow
        C = torch.sum(Wmax_vec - Wmin_vec)/1000
        ratio = ratio / C

        L = torch.trace(W @ H_ori @ W.T) * ratio / Ncol
        L = L.to(device)
        
        # R_vec is a row vector, each value is the R that allocated for a col
        d_vec = d_vec.view(-1) # set to shape [1,N]
        R_vec = torch.round(-0.5 * torch.log2(12 * L * d_vec**2))
        R_vec = torch.clamp_min(R_vec, 0.0)
        maxq = (2 ** R_vec - 1).to(device)

        ele_sum = ele_sum_container['value']
        R_aver = R_aver_container['value']
        R_mean = R_vec.mean().item()
        print(f"R_mean = {R_mean:.4f}")
        ele = Nrow * Ncol / 1000
        R_aver = (R_aver * ele_sum + R_mean * ele) / (ele_sum + ele)
        R_aver_container['value'] = R_aver
        ele_sum_container['value'] = ele_sum + ele
        
        R_record_container['value'][layer_idx, col_idx] = R_mean

        eps = torch.tensor(1e-8, device=device)
        diff = Wmax_vec - Wmin_vec
        diff[diff == 0] += eps  # avoid division by zero
        scale = diff.unsqueeze(1) / (maxq + 1e-8)
        scale = scale.to(device)
        
        zero = torch.round(- Wmin_vec.unsqueeze(1) / scale)
        zero = zero.to(device)

        # for i1 in range(0, self.columns, blocksize):
        for idx_i1, i1 in enumerate(range(0, self.columns, blocksize)):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]
                
                scale_vec = scale [:, idx_i1]
                zero_vec = zero [:, idx_i1]
                maxq_scalar = maxq[idx_i1]  # 取出标量值
                maxq_vec = torch.full_like(scale_vec, maxq_scalar)  # 和 scale_vec 形状一致的列向量

                q = quantize_vec(
                    w.unsqueeze(1), scale_vec.unsqueeze(1), zero_vec.unsqueeze(1), maxq_vec.unsqueeze(1), W_mean, layer_idx=layer_idx, name = name, 
                ).flatten()
                
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if actorder:
            Q = Q[:, invperm]

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
