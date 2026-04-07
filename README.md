# Simple User-Level Privacy

Implements "user level" differential privacy for training PyTorch models with DP-SGD.

By convention in the differential privacy literature, a "user" is not somebody who trains or uses the machine learning model, but a person whose data is in the training set. An example would be a patient in a clinical dataset. Therefore, user level differential privacy (ULDP) protects the privacy of data subjects.

User level privacy extends the privacy protection from one training example to all training examples that were contributed by one person. Without ULDP, even if inferences cannot be made about individual training examples, inferences about people who are represented by many examples may still be possible.

`sulp` is minimalistic and research oriented. It is currently not efficient enough to scale to LLMs.

## Installation

```
pip install --upgrade git+https://github.com/priitj/sulp.git
```

`sulp` is designed to work with PyTorch and installing it will pull the `torch` package as a dependency. Because there are several good libraries that include DP accountants, `sulp` does not have its own accounting functionality. You can use the accountants from [Opacus](https://opacus.ai/api/accounting/accounting.html) instead.

## Documentation

Please see the [tutorial notebook](examples/tutorial.ipynb).

More examples and API documentation coming soon.

## Other Resources

In case you don't need user level privacy, [Opacus](https://github.com/meta-pytorch/opacus) is a mature production quality DP library for PyTorch models.

If you want to train LLMs with user level privacy, you may be interested in these papers: https://arxiv.org/abs/2407.07737, https://arxiv.org/abs/2406.14322.

`sulp` does not address these issues with DP in production use: [randomness sources](https://dl.acm.org/doi/pdf/10.1145/3411497.3420211), [floating point precision](https://dl.acm.org/doi/pdf/10.1145/3548606.3560708).

## Citation

Coming soon.

