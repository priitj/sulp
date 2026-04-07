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

class GroupPoissonSampler(torch.utils.data.Sampler):
    """For implementing user level differential privacy
       "group" refers to a group of samples, e.g. all samples of one user

       The sampler will:
       - sample from groups with probability sample_rate
       - permute the groups
       - permute the records inside groups
       - return record indices to dataset
    """
    def __init__(self, group_ids, sample_rate, seed=42):
        self.dataset_size = len(group_ids)
        sorted_ids, self.sort_map = torch.sort(torch.as_tensor(group_ids,
            dtype=torch.int32))
        self.q = sample_rate
        self.rng = torch.Generator().manual_seed(seed)

        # compute things that don't change between steps
        # e.g. need to be done once per dataloader initialization
        group_start = torch.ones(self.dataset_size, dtype=torch.int32)
        group_start[1:] = ((sorted_ids[1:] - sorted_ids[:-1]) > 0).long()
        self.N = group_start.sum().item()
        self.group_offset = group_start.nonzero().ravel()
        self.group_size = torch.zeros(self.N, dtype=torch.int32)
        self.group_size[:-1] = self.group_offset[1:] - self.group_offset[:-1]
        self.group_size[-1] = self.dataset_size - self.group_offset[-1].item()

    def __iter__(self):
        # the Poisson sampling part
        group_mask = torch.rand(self.N, generator=self.rng) < self.q

        # shuffle is unnecessary in ULDP but is useful for testing
        sample_size = group_mask.sum().item()
        shuffled_groups = group_mask.nonzero()[torch.randperm(sample_size,
                                                generator=self.rng)]

        # this lazy generator loop runs once (in parallel) with the typical
        # training loop, so the impact on training speed is negligible
        for group_idx in shuffled_groups.ravel().tolist():
            shuffle_order = torch.randperm(self.group_size[group_idx].item(),
                                                generator=self.rng)
            shuffled_samples = shuffle_order + self.group_offset[group_idx]
            yield from self.sort_map[shuffled_samples.ravel()].tolist()

