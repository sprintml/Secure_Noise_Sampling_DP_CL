# converted from tensorflow:
# https://github.com/google-research/federated/blob/master/distributed_dp/discrete_gaussian_utils.py

import torch
import torch.distributions as dist


def _sample_discrete_laplace(t, shape):
    geometric_probs = 1.0 - torch.exp(-1.0 / t.float())
    geo1 = dist.Geometric(probs=geometric_probs).sample(shape)
    geo2 = dist.Geometric(probs=geometric_probs).sample(shape)
    return (geo1 - geo2).long()


def _sample_bernoulli(p):
    return dist.Bernoulli(probs=p).sample().long()


def _check_input_args(scale, shape, dtype):
    if dtype not in (torch.int32, torch.int64):
        raise ValueError(
            f'Only torch.int32 and torch.int64 are supported. Found dtype `{dtype}`.')

    if scale < 0:
        raise ValueError('Scale must be non-negative.')
    if not isinstance(scale, int):
        raise ValueError('Scale must be an integer.')

    return scale, shape, dtype


def _sample_discrete_gaussian_helper(scale, shape, dtype):
    scale = torch.tensor(scale, dtype=torch.int64)
    sq_scale = scale ** 2

    dlap_scale = scale
    oversample_factor = 1.5

    min_n = 1000
    target_n = torch.prod(torch.tensor(shape)).int()
    oversample_n = (oversample_factor * target_n.float()).int()
    draw_n = max(min_n, oversample_n)

    accepted_n = 0
    result = torch.empty(0, dtype=torch.int64)

    while accepted_n < target_n:
        samples = _sample_discrete_laplace(dlap_scale, shape=(draw_n,))
        z_numer = ((samples.abs() - scale) ** 2)
        z_denom = 2 * sq_scale
        bern_probs = torch.exp(-z_numer / z_denom)
        accept = _sample_bernoulli(bern_probs)
        accepted_samples = samples[accept == 1]
        accepted_n += accepted_samples.numel()
        result = torch.cat((result, accepted_samples))
        draw_n = max(min_n, ((target_n - accepted_n).float() * oversample_factor).int())

    return result[:target_n].reshape(shape).to(dtype)

def sample_discrete_gaussian(scale, shape, dtype=torch.int32, device='cpu'):
  """Draws (possibly inexact) samples from the discrete Gaussian distribution.

  We relax some integer constraints to use vectorized implementations of
  Bernoulli and discrete Laplace sampling. Integer operations are done in
  tf.int64 as TF does not have direct support for fractions.

  Args:
    scale: The scale of the discrete Gaussian distribution.
    shape: The shape of the output tensor.
    dtype: The type of the output.

  Returns:
    A tensor of the specified shape filled with random values.
  """
  scale, shape, dtype = _check_input_args(scale, shape, dtype)
  if scale == 0.:
      return torch.zeros(shape, dtype=dtype).to(device)
  else:
      return _sample_discrete_gaussian_helper(scale, shape, dtype).to(device)

