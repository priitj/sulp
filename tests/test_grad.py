import pytest
from sulp import grad
import torch

TEST_TENSORS = {
    # scalar
    "A": torch.tensor(3),
    "B": torch.tensor(-2),
    "C": torch.tensor(1),    # C=A+B
    # one dimension
    "D": torch.tensor([-5, -7,  1, -4, -8]),
    "E": torch.tensor([-2, -4,  1,  0,  4]),
    "F": torch.tensor([-7, -11,   2,  -4,  -4]), # F=D+E
    # two dimensions
    "G": torch.tensor([[3,  4,  5,  3],
        [6,  7,  0, -1],
        [4,  8,  0.9999,  7],
        [0.5019,  0.2306,  0.1023,  0.4355],
        [1.9268, -0.0433,  0.0869,  0.6877]]),
    "H": torch.tensor([[1,  -2,   4,  -8],
        [-4,   8, -10,  -4],
        [7,  -7,  -5, -10],
        [0.5459, -0.1659,  0.0675, -0.5560],
        [0.1831,  0.3647, -0.04860027,  0.6436]]),
    "J": torch.tensor([[4,   2,   9,  -5],
        [2,  15, -10,  -5],
        [11,   1,  -4.0001,  -3],
        [1.0478,  0.0647,  0.1698, -0.1205],
        [2.1099,  0.3214,  0.03829973,  1.3313]]),   # J=G+H
    "Y": torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0])
}

TEST_SLICE_SUM = {
    # 0:1, 1:2, 2:3, 0:3, 1:4, 0:5
    "D": [-5, -7, 1, -11, -10, -23],
    "F": [-7, -11, 2, -16, -13, -24],
    "J": [[4,   2,   9,  -5],
        [2,  15, -10,  -5],
        [11,   1,  -4.0001,  -3],
        [17,  18, -5.0001, -13],
        [14.0478,  16.0647, -13.8303,  -8.1205],
        [20.1577,  18.3861,  -4.79200027, -11.7892]]
}

TEST_GRAD = {
    "H": {"w": torch.tensor([[-0.6, 1.2, -2.4, 4.8],
            [11.2, -22.4, 28, 11.2],
            [-58.8, 58.8, 42, 84],
            [0.894468068, -0.271830468, 0.1106001, -0.91101712],
            [0.488551082, 0.973099834, -0.1296762, 1.717266288]]),
        "b": torch.tensor([-0.6, -2.8, -8.4, 1.63852, 2.66822])},
    "J": {"w": torch.tensor([[28, 14, 63, -35],
            [0.8, 6, -4, -2],
            [19.79934, 1.79994, -7.199939994, -5.39982],
            [2.348035976, 0.144987524, 0.380508216, -0.27003086],
            [7.676997402, 1.169433132, 0.139355859, 4.844014712]]),
        "b": torch.tensor([7, 0.4, 1.79994, 2.24092, 3.63856])}
}

TEST_NORMS = {
    "ADG": 21.064184188842773,
    "BEH": 23.371400833129883,
    "CFJ": 29.008188247680664
}

TEST_USERIDS = torch.tensor([[1, 1, 1, 1, 1],  # 0:5 slice
    [0, 1, 1, 1, 2], # 0:1, 1:4, 4:5 slices
    [7, 7, 7, 9, 9]]) # 0:3, 3:5 slices

# simple tensor computations

def test_grad_add():
    g1 = [TEST_TENSORS[v] for v in ["A", "D", "G"]]
    g2 = [TEST_TENSORS[v] for v in ["B", "E", "H"]]
    gsum = [TEST_TENSORS[v] for v in ["C", "F", "J"]]
    for p1, p2 in zip(grad.grad_add(g1, g2), gsum):
        assert p1 == pytest.approx(p2)

def test_sum_grad_slice():
    test_cases = ["D", "F", "J"]
    test_slices = [(0, 1), (1, 2), (2, 3), (0, 3), (1, 4), (0, 5)]
    g = [TEST_TENSORS[v] for v in test_cases]
    for i, (s_idx, e_idx) in enumerate(test_slices):
        gsum = [torch.tensor(TEST_SLICE_SUM[v][i]) for v in test_cases]
        for p1, p2 in zip(grad.sum_grad_slice(g, s_idx, e_idx), gsum):
            assert p1 == pytest.approx(p2)

def test_add_noise():
    means = [1, 2, -7]
    sizes = [300, (2, 40), (5, 4, 30)]
    params = [torch.randn(s, dtype=torch.float32) for s in sizes]
    for z, C, qN in [(0.5, 10.0, 100), (1.0, 10.0, 100), (1.0, 1.0, 500)]:
        for p, m in zip(params, means):
            p.grad = torch.full(p.shape, m, dtype=torch.float32)
        grad.add_noise(params, z, C, qN)
        sigma = (z * C) / qN
        for p, m in zip(params, means):
            std, mean = torch.std_mean(p.grad)
            assert std == pytest.approx(sigma, rel=0.2)
            assert mean == pytest.approx(m, rel=0.1)

# model functions

class NN1(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([0.1, 0.2, 0.3, 0.4],
                dtype=torch.float32))
        self.b = torch.nn.Parameter(torch.tensor(2.0, dtype=torch.float32))
        self.mean = torch.nn.parameter.Buffer(torch.tensor(0.25))

    def forward(self, x):
        self.mean = x.sum() / 4.0
        return (x * self.w).sum(dim=1) + self.b

def test_detach_params():
    model = NN1()
    assert model.w.requires_grad == True
    assert model.b.requires_grad == True
    params, buffers = grad.detach_params(model)
    for i, v in enumerate([0.1, 0.2, 0.3, 0.4]):
        assert params["w"][i] == pytest.approx(v)
        assert params["w"].requires_grad == False
    assert params["b"] == pytest.approx(2)
    assert params["b"].requires_grad == False
    assert buffers["mean"] == pytest.approx(0.25)

def test_make_gradient_func():
    # test if the gradient func computes the correct gradient
    model = NN1()
    loss_fn = torch.nn.MSELoss()
    params, buffers = grad.detach_params(model)
    gf = grad.make_gradient_func(model, loss_fn)
    for test_case in ["H", "J"]:
        sample_grads = gf(params, buffers,
                TEST_TENSORS[test_case].float(), TEST_TENSORS["Y"].float())
        assert TEST_GRAD[test_case]["w"] == pytest.approx(sample_grads["w"])
        assert TEST_GRAD[test_case]["b"] == pytest.approx(sample_grads["b"])

# class methods

def test_ga_reset_sum():
    model = NN1()
    ga = grad.GradAccumulator(model.named_parameters(), 1.0, 100)
    shapes = [p.shape for _, p in model.named_parameters()]
    ga.grad_sum = []
    ga._reset_sum()
    for t, s in zip(ga.grad_sum, shapes):
        assert t == pytest.approx(torch.zeros(s))

def test_ga_reset_group():
    model = NN1()
    ga = grad.GradAccumulator(model.named_parameters(), 1.0, 100)
    ga.last_group_sum = [TEST_TENSORS["J"]]
    ga.last_group_id = 79
    ga.num_groups = 1000
    ga._reset_group()
    assert ga.last_group_sum is None
    assert ga.last_group_id is None
    assert ga.num_groups == 0

def test_ga_norm_clip():
    model = NN1()
    grads = {
        "ADG": [TEST_TENSORS[v].float() for v in ["A", "D", "G"]],
        "BEH": [TEST_TENSORS[v].float() for v in ["B", "E", "H"]],
        "CFJ": [TEST_TENSORS[v].float() for v in ["C", "F", "J"]]
    }
    for max_norm in [1.0, 22.0, 30.0]:
        ga = grad.GradAccumulator(model.named_parameters(), max_norm, 100)
        for k in ["ADG", "BEH", "CFJ"]:
            expected = min(max_norm, TEST_NORMS[k])
            clipped = ga._norm_clip(grads[k])
            vector_repr = torch.cat([t.ravel() for t in clipped])
            assert expected == pytest.approx(vector_repr.norm().item())

def test_ga_accumulate():
    model = NN1()
    grad_dict = {
        "w": TEST_TENSORS["J"].float(),
        "b": TEST_TENSORS["D"].float()
    }

    # case 1: nothing should be accumulated [1 1 1 1 1]
    # no clipping happens, to isolate clip test and this test
    ga = grad.GradAccumulator(model.named_parameters(), 30.0, 100)
    ga.accumulate(grad_dict, TEST_USERIDS[0].long())
    # param order is "w", "b"
    for t, s in zip(ga.grad_sum, [[4], []]):
        assert t == pytest.approx(torch.zeros(s))
    for t1, t2 in zip(ga.last_group_sum, [TEST_SLICE_SUM["J"][5],
                                        TEST_SLICE_SUM["D"][5]]):
        assert t1 == pytest.approx(t2)
    assert ga.last_group_id == 1

    # case 2: 2 groups accumulated  [0 1 1 1 2]
    ga = grad.GradAccumulator(model.named_parameters(), 30.0, 100)
    ga.accumulate(grad_dict, TEST_USERIDS[1].long())
    expected = [torch.tensor(sums[0]).float() + torch.tensor(sums[4]).float()
                for sums in [TEST_SLICE_SUM["J"], TEST_SLICE_SUM["D"]]]
    for t1, t2 in zip(ga.grad_sum, expected):
        assert t1 == pytest.approx(t2)
    assert ga.last_group_id == 2

    # case 3: previous batch leftover + 1 group ... 7]  [7 7 7 9 9]
    ga = grad.GradAccumulator(model.named_parameters(), 32.0, 100)
    ga.last_group_id = 7
    ga.last_group_sum = [torch.ones(4), torch.tensor(1.0)]
    ga.accumulate(grad_dict, TEST_USERIDS[2].long())
    expected = [torch.tensor(TEST_SLICE_SUM["J"][3]).float() + 1.0,
            torch.tensor(TEST_SLICE_SUM["D"][3]).float() + 1.0]
    for t1, t2 in zip(ga.grad_sum, expected):
        assert t1 == pytest.approx(t2)
    assert ga.last_group_id == 9

def test_ga_apply():
    model = NN1()
    ga = grad.GradAccumulator(model.named_parameters(), 32.0, 100)
    # param order "w", "b"
    ga.grad_sum = [TEST_TENSORS["G"][4].float(),
            TEST_TENSORS["A"].float()]
    ga.last_group_sum = [TEST_TENSORS["H"][4].float(),
            TEST_TENSORS["B"].float()]
    expected = {"b": TEST_TENSORS["C"] / 100.0,
            "w": TEST_TENSORS["J"][4] / 100.0}
    ga.apply()
    for name, p in model.named_parameters():
        assert p.grad == pytest.approx(expected[name])

