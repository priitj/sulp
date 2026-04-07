#
# Copyright (c) 2026 Priit Järv
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this source file (the " Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify,
# merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice (including the next
# paragraph) shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import torch
from torch.func import functional_call, vmap, grad

def grad_add(g1, g2):
    return [p1 + p2 for p1, p2 in zip(g1, g2)]

def sum_grad_slice(g, s_idx, e_idx):
    # minor optimization, saves ~15% if most users have 1 sample
    # instead of this optimization, it may be better to
    # vectorize operations over layers, where gradient has the same shape
    #if s_idx + 1 == e_idx:
    #    return [p[s_idx, ...] for p in g]
    return [p[s_idx:e_idx, ...].sum(dim=0) for p in g]

class GradAccumulator:
    """Manage clipped gradient state
       Used for implementing user level differential privacy

       - Implements per-layer clipping
       - Sensitivity is to the sum of group gradients
       - GradAccumulator.apply() sets the gradient to mean (over samples)
            ==> noise should be scaled to    (z * max_grad_norm) / q * N
                where z is noise scale for (or estimated by) the privacy
                accountant, q is sample rate and N dataset size
    """
    def __init__(self, named_params, max_grad_norm, qN):
        self.params = []
        self.param_names = []
        for name, p in named_params:
            if p.requires_grad:
                self.params.append(p)
                self.param_names.append(name)
        if not self.params:
            raise ValueError("nothing requires grad?")
        self.max_grad_norm = torch.tensor(max_grad_norm)
        self.qN = torch.tensor(qN)
        self._reset_sum()
        self._reset_group()
        self.norm_stats = dict((name, []) for name in self.param_names)
        self.norm_stats["flat"] = []

    def _reset_sum(self):
        self.grad_sum = [torch.zeros(p.shape, dtype=torch.float32,
                                                            device=p.device)
                         for p in self.params]

    def _reset_group(self):
        self.last_group_sum = None
        self.last_group_id = None
        self.num_groups = 0

    def _norm_clip(self, grad_sum):
        # flat clipping
        layer_norms = []
        for i, g in enumerate(grad_sum):
            norm = torch.linalg.norm(g)
            #self.norm_stats[self.param_names[i]].append(norm.item())
            layer_norms.append(norm)
        grad_norm = torch.linalg.norm(torch.tensor(layer_norms))
        #self.norm_stats["flat"].append(grad_norm.item())
        if grad_norm < self.max_grad_norm:
            return grad_sum

        clipped_grad = []
        ratio = self.max_grad_norm / grad_norm
        for g in grad_sum:
            clipped_grad.append(g * ratio)
        return clipped_grad

    def accumulate(self, per_sample_grad_dict, group_ids):
        per_sample_grad = [per_sample_grad_dict[name]
                for name in self.param_names]
        # assert per_sample_grad.shape[0] == group_ids.shape[0]

        # this is _currently_ very similar to group offset computation
        # in GroupPoissonSampler. The computation is repeated, because groups
        # are permuted, so the offsets inside the sampler can not be reused
        # here. Also, having separate code here saves a few tensor operations
        # (which may or may not mean anything) because in the sampler it is
        # convenient to compute group size, here we want the group endpoint.

        batch_size = per_sample_grad[0].shape[0]
        group_start = torch.ones(batch_size, dtype=torch.int32)
        group_start[1:] = ((group_ids[1:] - group_ids[:-1]) != 0).long()
        group_offset = group_start.nonzero().ravel()
        group_end_offset = torch.roll(group_offset, shifts=-1)
        group_end_offset[-1] = batch_size

        # likely the slow part:
        # outer loop over groups
        #     inner loop over params
        #         innermost stuff (vectorized on GPU)
        for s_idx, e_idx in zip(group_offset.tolist(),
                                group_end_offset.tolist()):
            group_id = group_ids[s_idx].item()
            if self.last_group_id is None or self.last_group_id != group_id:
                # commit last group to summed gradient if needed
                if self.last_group_sum is not None:
                    self.grad_sum = grad_add(self.grad_sum,
                                         self._norm_clip(self.last_group_sum))

                # fresh group
                self.num_groups += 1
                group_sum = sum_grad_slice(per_sample_grad, s_idx, e_idx)
            else:
                # leftovers from last batch
                group_sum = grad_add(sum_grad_slice(per_sample_grad,
                                                    s_idx, e_idx),
                                     self.last_group_sum)

            self.last_group_sum = group_sum
            self.last_group_id = group_id

    def apply(self):
        # accumulate last group
        if self.last_group_sum is not None:
            self.grad_sum = grad_add(self.grad_sum,
                                     self._norm_clip(self.last_group_sum))
        self._reset_group()
        # XXX: leave the no_grad context manager to the user?
        with torch.no_grad():
            # self.params only contains those with p.requires_grad set
            for p, g in zip(self.params, self.grad_sum):
                p.grad = g / self.qN
        self._reset_sum()

def detach_params(model):
    params = {k: v.detach() for k, v in model.named_parameters()}
    buffers = {k: v.detach() for k, v in model.named_buffers()}
    return params, buffers

def make_gradient_func(model, loss_fn):
    def compute_loss(params, buffers, sample, target):
        batch = sample.unsqueeze(0)
        targets = target.unsqueeze(0)

        predictions = functional_call(model, (params, buffers), (batch,))
        loss = loss_fn(predictions, targets)
        return loss
    return vmap(grad(compute_loss), in_dims=(None, None, 0, 0),
            randomness="different")

def add_noise(params, z, max_grad_norm, qN):
    sigma = (z * max_grad_norm) / qN
    for p in params:
        p.grad += torch.normal(0.0, sigma, size=p.shape, device=p.device)

