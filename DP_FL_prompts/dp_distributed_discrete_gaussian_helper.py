# Copyright 2021, Google LLC. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""DP Accounting with composition for Gaussian and DDGaussian."""

import math

import numpy as np
from scipy import optimize
from scipy import special


RDP_ORDERS = tuple(range(2, 129)) + (256,)
DIV_EPSILON = 1e-22

# https://github.com/tensorflow/privacy/blob/085b7ddfecc8443910e3a374e6d3c80bdc4ae618/tensorflow_privacy/privacy/analysis/rdp_accountant.py

# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""RDP analysis of the Sampled Gaussian Mechanism.
Functionality for computing Renyi differential privacy (RDP) of an additive
Sampled Gaussian Mechanism (SGM). Its public interface consists of two methods:
  compute_rdp(q, noise_multiplier, T, orders) computes RDP for SGM iterated
                                   T times.
  get_privacy_spent(orders, rdp, target_eps, target_delta) computes delta
                                   (or eps) given RDP at multiple orders and
                                   a target value for eps (or delta).
Example use:
Suppose that we have run an SGM applied to a function with l2-sensitivity 1.
Its parameters are given as a list of tuples (q1, sigma1, T1), ...,
(qk, sigma_k, Tk), and we wish to compute eps for a given delta.
The example code would be:
  max_order = 32
  orders = range(2, max_order + 1)
  rdp = np.zeros_like(orders, dtype=float)
  for q, sigma, T in parameters:
   rdp += rdp_accountant.compute_rdp(q, sigma, T, orders)
  eps, _, opt_order = rdp_accountant.get_privacy_spent(rdp, target_delta=delta)
"""

import math
import sys

import numpy as np
from scipy import special

########################
# LOG-SPACE ARITHMETIC #
########################


def _log_add(logx, logy):
  """Add two numbers in the log space."""
  a, b = min(logx, logy), max(logx, logy)
  if a == -np.inf:  # adding 0
    return b
  # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
  return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_sub(logx, logy):
  """Subtract two numbers in the log space. Answer must be non-negative."""
  if logx < logy:
    raise ValueError("The result of subtraction must be non-negative.")
  if logy == -np.inf:  # subtracting 0
    return logx
  if logx == logy:
    return -np.inf  # 0 is represented as -np.inf in the log space.

  try:
    # Use exp(x) - exp(y) = (exp(x - y) - 1) * exp(y).
    return math.log(math.expm1(logx - logy)) + logy  # expm1(x) = exp(x) - 1
  except OverflowError:
    return logx


def _log_sub_sign(logx, logy):
  """Returns log(exp(logx)-exp(logy)) and its sign."""
  if logx > logy:
    s = True
    mag = logx + np.log(1 - np.exp(logy - logx))
  elif logx < logy:
    s = False
    mag = logy + np.log(1 - np.exp(logx - logy))
  else:
    s = True
    mag = -np.inf

  return s, mag


def _log_print(logx):
  """Pretty print."""
  if logx < math.log(sys.float_info.max):
    return "{}".format(math.exp(logx))
  else:
    return "exp({})".format(logx)


def _log_comb(n, k):
  return (special.gammaln(n + 1) - special.gammaln(k + 1) -
          special.gammaln(n - k + 1))


def _compute_log_a_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha. 0 < q < 1."""
  assert isinstance(alpha, int)

  # Initialize with 0 in the log space.
  log_a = -np.inf

  for i in range(alpha + 1):
    log_coef_i = (
        _log_comb(alpha, i) + i * math.log(q) + (alpha - i) * math.log(1 - q))

    s = log_coef_i + (i * i - i) / (2 * (sigma**2))
    log_a = _log_add(log_a, s)

  return float(log_a)


def _compute_log_a_frac(q, sigma, alpha):
  """Compute log(A_alpha) for fractional alpha. 0 < q < 1."""
  # The two parts of A_alpha, integrals over (-inf,z0] and [z0, +inf), are
  # initialized to 0 in the log space:
  log_a0, log_a1 = -np.inf, -np.inf
  i = 0

  z0 = sigma**2 * math.log(1 / q - 1) + .5

  while True:  # do ... until loop
    coef = special.binom(alpha, i)
    log_coef = math.log(abs(coef))
    j = alpha - i

    log_t0 = log_coef + i * math.log(q) + j * math.log(1 - q)
    log_t1 = log_coef + j * math.log(q) + i * math.log(1 - q)

    log_e0 = math.log(.5) + _log_erfc((i - z0) / (math.sqrt(2) * sigma))
    log_e1 = math.log(.5) + _log_erfc((z0 - j) / (math.sqrt(2) * sigma))

    log_s0 = log_t0 + (i * i - i) / (2 * (sigma**2)) + log_e0
    log_s1 = log_t1 + (j * j - j) / (2 * (sigma**2)) + log_e1

    if coef > 0:
      log_a0 = _log_add(log_a0, log_s0)
      log_a1 = _log_add(log_a1, log_s1)
    else:
      log_a0 = _log_sub(log_a0, log_s0)
      log_a1 = _log_sub(log_a1, log_s1)

    i += 1
    if max(log_s0, log_s1) < -30:
      break

  return _log_add(log_a0, log_a1)


def _compute_log_a(q, sigma, alpha):
  """Compute log(A_alpha) for any positive finite alpha."""
  if float(alpha).is_integer():
    return _compute_log_a_int(q, sigma, int(alpha))
  else:
    return _compute_log_a_frac(q, sigma, alpha)


def _log_erfc(x):
  """Compute log(erfc(x)) with high accuracy for large x."""
  try:
    return math.log(2) + special.log_ndtr(-x * 2**.5)
  except NameError:
    # If log_ndtr is not available, approximate as follows:
    r = special.erfc(x)
    if r == 0.0:
      # Using the Laurent series at infinity for the tail of the erfc function:
      #     erfc(x) ~ exp(-x^2-.5/x^2+.625/x^4)/(x*pi^.5)
      # To verify in Mathematica:
      #     Series[Log[Erfc[x]] + Log[x] + Log[Pi]/2 + x^2, {x, Infinity, 6}]
      return (-math.log(math.pi) / 2 - math.log(x) - x**2 - .5 * x**-2 +
              .625 * x**-4 - 37. / 24. * x**-6 + 353. / 64. * x**-8)
    else:
      return math.log(r)


def _compute_delta(orders, rdp, eps):
  """Compute delta given a list of RDP values and target epsilon.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    eps: The target epsilon.
  Returns:
    Pair of (delta, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if eps < 0:
    raise ValueError("Value of privacy loss bound epsilon must be >=0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   delta = min( np.exp((rdp_vec - eps) * (orders_vec - 1)) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4):
  logdeltas = []  # work in log space to avoid overflows
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")
    # For small alpha, we are better of with bound via KL divergence:
    # delta <= sqrt(1-exp(-KL)).
    # Take a min of the two bounds.
    logdelta = 0.5 * math.log1p(-math.exp(-r))
    if a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value for alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      rdp_bound = (a - 1) * (r - eps + math.log1p(-1 / a)) - math.log(a)
      logdelta = min(logdelta, rdp_bound)

    logdeltas.append(logdelta)

  idx_opt = np.argmin(logdeltas)
  return min(math.exp(logdeltas[idx_opt]), 1.), orders_vec[idx_opt]


def _compute_eps(orders, rdp, delta):
  """Compute epsilon given a list of RDP values and target delta.
  Args:
    orders: An array (or a scalar) of orders.
    rdp: A list (or a scalar) of RDP guarantees.
    delta: The target delta.
  Returns:
    Pair of (eps, optimal_order).
  Raises:
    ValueError: If input is malformed.
  """
  orders_vec = np.atleast_1d(orders)
  rdp_vec = np.atleast_1d(rdp)

  if delta <= 0:
    raise ValueError("Privacy failure probability bound delta must be >0.")
  if len(orders_vec) != len(rdp_vec):
    raise ValueError("Input lists must have the same length.")

  # Basic bound (see https://arxiv.org/abs/1702.07476 Proposition 3 in v3):
  #   eps = min( rdp_vec - math.log(delta) / (orders_vec - 1) )

  # Improved bound from https://arxiv.org/abs/2004.00010 Proposition 12 (in v4).
  # Also appears in https://arxiv.org/abs/2001.05990 Equation 20 (in v1).
  eps_vec = []
  for (a, r) in zip(orders_vec, rdp_vec):
    if a < 1:
      raise ValueError("Renyi divergence order must be >=1.")
    if r < 0:
      raise ValueError("Renyi divergence must be >=0.")

    if delta**2 + math.expm1(-r) >= 0:
      # In this case, we can simply bound via KL divergence:
      # delta <= sqrt(1-exp(-KL)).
      eps = 0  # No need to try further computation if we have eps = 0.
    elif a > 1.01:
      # This bound is not numerically stable as alpha->1.
      # Thus we have a min value of alpha.
      # The bound is also not useful for small alpha, so doesn't matter.
      eps = r + math.log1p(-1 / a) - math.log(delta * a) / (a - 1)
    else:
      # In this case we can't do anything. E.g., asking for delta = 0.
      eps = np.inf
    eps_vec.append(eps)

  idx_opt = np.argmin(eps_vec)
  return max(0, eps_vec[idx_opt]), orders_vec[idx_opt]


def _stable_inplace_diff_in_log(vec, signs, n=-1):
  """Replaces the first n-1 dims of vec with the log of abs difference operator.
  Args:
    vec: numpy array of floats with size larger than 'n'
    signs: Optional numpy array of bools with the same size as vec in case one
      needs to compute partial differences vec and signs jointly describe a
      vector of real numbers' sign and abs in log scale.
    n: Optonal upper bound on number of differences to compute. If negative, all
      differences are computed.
  Returns:
    The first n-1 dimension of vec and signs will store the log-abs and sign of
    the difference.
  Raises:
    ValueError: If input is malformed.
  """

  assert vec.shape == signs.shape
  if n < 0:
    n = np.max(vec.shape) - 1
  else:
    assert np.max(vec.shape) >= n + 1
  for j in range(0, n, 1):
    if signs[j] == signs[j + 1]:  # When the signs are the same
      # if the signs are both positive, then we can just use the standard one
      signs[j], vec[j] = _log_sub_sign(vec[j + 1], vec[j])
      # otherwise, we do that but toggle the sign
      if not signs[j + 1]:
        signs[j] = ~signs[j]
    else:  # When the signs are different.
      vec[j] = _log_add(vec[j], vec[j + 1])
      signs[j] = signs[j + 1]


def _get_forward_diffs(fun, n):
  """Computes up to nth order forward difference evaluated at 0.
  See Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf
  Args:
    fun: Function to compute forward differences of.
    n: Number of differences to compute.
  Returns:
    Pair (deltas, signs_deltas) of the log deltas and their signs.
  """
  func_vec = np.zeros(n + 3)
  signs_func_vec = np.ones(n + 3, dtype=bool)

  # ith coordinate of deltas stores log(abs(ith order discrete derivative))
  deltas = np.zeros(n + 2)
  signs_deltas = np.zeros(n + 2, dtype=bool)
  for i in range(1, n + 3, 1):
    func_vec[i] = fun(1.0 * (i - 1))
  for i in range(0, n + 2, 1):
    # Diff in log scale
    _stable_inplace_diff_in_log(func_vec, signs_func_vec, n=n + 2 - i)
    deltas[i] = func_vec[0]
    signs_deltas[i] = signs_func_vec[0]
  return deltas, signs_deltas


def _compute_rdp(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling rate.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """
  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  return _compute_log_a(q, sigma, alpha) / (alpha - 1)


def compute_rdp(q, noise_multiplier, steps, orders):
  """Computes RDP of the Sampled Gaussian Mechanism.
  Args:
    q: The sampling rate.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp(q, noise_multiplier, orders)
  else:
    rdp = np.array(
        [_compute_rdp(q, noise_multiplier, order) for order in orders])

  return rdp * steps


def compute_rdp_sample_without_replacement(q, noise_multiplier, steps, orders):
  """Compute RDP of Gaussian Mechanism using sampling without replacement.
  This function applies to the following schemes:
  1. Sampling w/o replacement: Sample a uniformly random subset of size m = q*n.
  2. ``Replace one data point'' version of differential privacy, i.e., n is
     considered public information.
  Reference: Theorem 27 of https://arxiv.org/pdf/1808.00087.pdf (A strengthened
  version applies subsampled-Gaussian mechanism)
  - Wang, Balle, Kasiviswanathan. "Subsampled Renyi Differential Privacy and
  Analytical Moments Accountant." AISTATS'2019.
  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    noise_multiplier: The ratio of the standard deviation of the Gaussian noise
      to the l2-sensitivity of the function to which it is added.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders, can be np.inf.
  """
  if np.isscalar(orders):
    rdp = _compute_rdp_sample_without_replacement_scalar(
        q, noise_multiplier, orders)
  else:
    rdp = np.array([
        _compute_rdp_sample_without_replacement_scalar(q, noise_multiplier,
                                                       order)
        for order in orders
    ])

  return rdp * steps


def _compute_rdp_sample_without_replacement_scalar(q, sigma, alpha):
  """Compute RDP of the Sampled Gaussian mechanism at order alpha.
  Args:
    q: The sampling proportion =  m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """

  assert (q <= 1) and (q >= 0) and (alpha >= 1)

  if q == 0:
    return 0

  if q == 1.:
    return alpha / (2 * sigma**2)

  if np.isinf(alpha):
    return np.inf

  if float(alpha).is_integer():
    return _compute_rdp_sample_without_replacement_int(q, sigma, alpha) / (
        alpha - 1)
  else:
    # When alpha not an integer, we apply Corollary 10 of [WBK19] to interpolate
    # the CGF and obtain an upper bound
    alpha_f = math.floor(alpha)
    alpha_c = math.ceil(alpha)

    x = _compute_rdp_sample_without_replacement_int(q, sigma, alpha_f)
    y = _compute_rdp_sample_without_replacement_int(q, sigma, alpha_c)
    t = alpha - alpha_f
    return ((1 - t) * x + t * y) / (alpha - 1)


def _compute_rdp_sample_without_replacement_int(q, sigma, alpha):
  """Compute log(A_alpha) for integer alpha, subsampling without replacement.
  When alpha is smaller than max_alpha, compute the bound Theorem 27 exactly,
    otherwise compute the bound with Stirling approximation.
  Args:
    q: The sampling proportion = m / n.  Assume m is an integer <= n.
    sigma: The std of the additive Gaussian noise.
    alpha: The order at which RDP is computed.
  Returns:
    RDP at alpha, can be np.inf.
  """

  max_alpha = 256
  assert isinstance(alpha, int)

  if np.isinf(alpha):
    return np.inf
  elif alpha == 1:
    return 0

  def cgf(x):
    # Return rdp(x+1)*x, the rdp of Gaussian mechanism is alpha/(2*sigma**2)
    return x * 1.0 * (x + 1) / (2.0 * sigma**2)

  def func(x):
    # Return the rdp of Gaussian mechanism
    return 1.0 * x / (2.0 * sigma**2)

  # Initialize with 1 in the log space.
  log_a = 0
  # Calculates the log term when alpha = 2
  log_f2m1 = func(2.0) + np.log(1 - np.exp(-func(2.0)))
  if alpha <= max_alpha:
    # We need forward differences of exp(cgf)
    # The following line is the numerically stable way of implementing it.
    # The output is in polar form with logarithmic magnitude
    deltas, _ = _get_forward_diffs(cgf, alpha)
    # Compute the bound exactly requires book keeping of O(alpha**2)

    for i in range(2, alpha + 1):
      if i == 2:
        s = 2 * np.log(q) + _log_comb(alpha, 2) + np.minimum(
            np.log(4) + log_f2m1,
            func(2.0) + np.log(2))
      elif i > 2:
        delta_lo = deltas[int(2 * np.floor(i / 2.0)) - 1]
        delta_hi = deltas[int(2 * np.ceil(i / 2.0)) - 1]
        s = np.log(4) + 0.5 * (delta_lo + delta_hi)
        s = np.minimum(s, np.log(2) + cgf(i - 1))
        s += i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)
    return float(log_a)
  else:
    # Compute the bound with stirling approximation. Everything is O(x) now.
    for i in range(2, alpha + 1):
      if i == 2:
        s = 2 * np.log(q) + _log_comb(alpha, 2) + np.minimum(
            np.log(4) + log_f2m1,
            func(2.0) + np.log(2))
      else:
        s = np.log(2) + cgf(i - 1) + i * np.log(q) + _log_comb(alpha, i)
      log_a = _log_add(log_a, s)

    return log_a


def compute_heterogeneous_rdp(sampling_probabilities, noise_multipliers,
                              steps_list, orders):
  """Computes RDP of Heteregoneous Applications of Sampled Gaussian Mechanisms.
  Args:
    sampling_probabilities: A list containing the sampling rates.
    noise_multipliers: A list containing the noise multipliers: the ratio of the
      standard deviation of the Gaussian noise to the l2-sensitivity of the
      function to which it is added.
    steps_list: A list containing the number of steps at each
      `sampling_probability` and `noise_multiplier`.
    orders: An array (or a scalar) of RDP orders.
  Returns:
    The RDPs at all orders. Can be `np.inf`.
  """
  assert len(sampling_probabilities) == len(noise_multipliers)

  rdp = 0
  for q, noise_multiplier, steps in zip(sampling_probabilities,
                                        noise_multipliers, steps_list):
    rdp += compute_rdp(q, noise_multiplier, steps, orders)

  return rdp


def get_privacy_spent(orders, rdp, target_eps=None, target_delta=None):
  """Computes delta (or eps) for given eps (or delta) from RDP values.
  Args:
    orders: An array (or a scalar) of RDP orders.
    rdp: An array of RDP values. Must be of the same length as the orders list.
    target_eps: If not `None`, the epsilon for which we compute the
      corresponding delta.
    target_delta: If not `None`, the delta for which we compute the
      corresponding epsilon. Exactly one of `target_eps` and `target_delta` must
      be `None`.
  Returns:
    A tuple of epsilon, delta, and the optimal order.
  Raises:
    ValueError: If target_eps and target_delta are messed up.
  """
  if target_eps is None and target_delta is None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (Both are).")

  if target_eps is not None and target_delta is not None:
    raise ValueError(
        "Exactly one out of eps and delta must be None. (None is).")

  if target_eps is not None:
    delta, opt_order = _compute_delta(orders, rdp, target_eps)
    return target_eps, delta, opt_order
  else:
    eps, opt_order = _compute_eps(orders, rdp, target_delta)
    return eps, target_delta, opt_order


####################
###### Shared ######
####################


def log_comb(n, k):
  gammaln = special.gammaln
  return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def _compute_rdp_subsampled(alpha, gamma, eps, upper_bound=True):
  """Computes RDP with subsampling.

  Reference: http://proceedings.mlr.press/v97/zhu19c/zhu19c.pdf.

  Args:
    alpha: The RDP order.
    gamma: The subsampling probability.
    eps: The RDP function taking alpha as input.
    upper_bound: A bool indicating whether to use Theorem 5 of the referenced
      paper above (if set to True) or Theorem 6 (if set to False).

  Returns:
    The RDP with subsampling.
  """
  if isinstance(alpha, float):
    assert alpha.is_integer()
    alpha = int(alpha)
  assert alpha > 1
  assert 0 < gamma <= 1

  if upper_bound:
    a = [0, eps(2)]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1),
         special.comb(alpha, 2) * (gamma**2) * (1 - gamma)**(alpha - 2)]

    for l in range(3, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(3)

  else:
    a = [0]
    b = [((1 - gamma)**(alpha - 1)) * (alpha * gamma - gamma + 1)]

    for l in range(2, alpha + 1):
      a.append((l - 1) * eps(l) + log_comb(alpha, l) +
               (alpha - l) * np.log(1 - gamma) + l * np.log(gamma))
      b.append(1)

  return special.logsumexp(a=a, b=b) / (alpha - 1)


def rounded_l2_norm_bound(l2_norm_bound, beta, dim):
  """Computes the L2 norm bound after stochastic rounding to integers.

  Note that this function is *agnostic* to the actual vector whose coordinates
  are to be rounded, and it does *not* consider the effect of scaling (i.e.
  we assume the input norm is already scaled before rounding).

  See Theorem 1 of https://arxiv.org/pdf/2102.06387.pdf.

  Args:
    l2_norm_bound: The L2 norm (bound) of the vector whose coordinates are to be
      stochastically rounded to the integer grid.
    beta: A float constant in [0, 1). See the initializer docstring of the
      aggregator.
    dim: The dimension of the vector to be rounded.

  Returns:
    The inflated L2 norm bound after stochastic rounding (conditionally
    according to beta).
  """
  assert int(dim) == dim and dim > 0, f'Invalid dimension: {dim}'
  assert 0 <= beta < 1, 'beta must be in the range [0, 1)'
  assert l2_norm_bound > 0, 'Input l2_norm_bound should be positive.'

  bound_1 = l2_norm_bound + np.sqrt(dim)
  if beta == 0:
    return bound_1

  squared_bound_2 = np.square(l2_norm_bound) + 0.25 * dim
  squared_bound_2 += (
      np.sqrt(2.0 * np.log(1.0 / beta)) * (l2_norm_bound + 0.5 * np.sqrt(dim)))
  bound_2 = np.sqrt(squared_bound_2)
  return min(bound_1, bound_2)


def rounded_l1_norm_bound(l2_norm_bound, dim):
  # In general we have L1 <= sqrt(d) * L2. In the scaled and rounded domain
  # where coordinates are integers we also have L1 <= L2^2.
  return l2_norm_bound * min(np.sqrt(dim), l2_norm_bound)


def heuristic_scale_factor(local_stddev,
                           l2_clip,
                           bits,
                           num_clients,
                           dim,
                           k_stddevs,
                           rho=1.0):
  """Selects a scaling factor by assuming subgaussian aggregates.

  Selects scale_factor = 1 / gamma such that k stddevs of the noisy, quantized,
  aggregated client values are bounded within the bit-width. The aggregate at
  the server is assumed to follow a subgaussian distribution. See Section 4.2
  and 4.4 of https://arxiv.org/pdf/2102.06387.pdf for more details.

  Specifically, the implementation is solving for gamma using the following
  expression:

    2^b = 2k * sqrt(rho / dim * (cn)^2 + (gamma^2 / 4 + sigma^2) * n) / gamma.

  Args:
    local_stddev: The local noise standard deviation.
    l2_clip: The initial L2 clip norm. See the __init__ docstring.
    bits: The bit-width. See the __init__ docstring.
    num_clients: The expected number of clients. See the __init__ docstring.
    dim: The dimension of the client vector that includes any necessary padding.
    k_stddevs: The number of standard deviations of the noisy and quantized
      aggregate values to bound within the bit-width.
    rho: (Optional) The subgaussian flatness parameter of the random orthogonal
      transform as part of the DDP procedure. See Section 4.2 of the above paper
      for more details.

  Returns:
    The selected scaling factor in tf.float64.
  """
  c = l2_clip
  n = num_clients
  sigma = local_stddev

  if 2.0**(2.0 * bits) < n * k_stddevs**2:
    raise ValueError(f'The selected bit-width ({bits}) is too small for the '
                     f'given parameters (num_clients = {n}, k_stddevs = '
                     f'{k_stddevs}). You may decrease `num_clients`, '
                     f'increase `bits`, or decrease `k_stddevs`.')

  numer = np.sqrt(2.0**(2.0 * bits) - n * k_stddevs**2)
  denom = 2.0 * k_stddevs * np.sqrt(rho / dim * c**2 * n**2 + n * sigma**2)
  scale_factor = numer / denom
  return scale_factor


#####################################
######## Gaussian Accounting ########
#####################################


def analytic_gauss_stddev(epsilon, delta, norm_bound, tol=1.e-12):
  """Compute the stddev for the Gaussian mechanism with the given DP params.

  Calibrate a Gaussian perturbation for differential privacy using the
  analytic Gaussian mechanism of [Balle and Wang, ICML'18].

  Reference: http://proceedings.mlr.press/v80/balle18a/balle18a.pdf.

  Arguments:
    epsilon: Target epsilon (epsilon > 0).
    delta: Target delta (0 < delta < 1).
    norm_bound: Upper bound on L2 global sensitivity (norm_bound >= 0).
    tol: Error tolerance for binary search (tol > 0).

  Returns:
    sigma: Standard deviation of Gaussian noise needed to achieve
      (epsilon,delta)-DP under the given norm_bound.
  """

  exp = math.exp
  sqrt = math.sqrt

  def phi(t):
    return 0.5 * (1.0 + special.erf(float(t) / sqrt(2.0)))

  def case_one(eps, s):
    return phi(sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  def case_two(eps, s):
    return phi(-sqrt(eps * s)) - exp(eps) * phi(-sqrt(eps * (s + 2.0)))

  def doubling_trick(predicate_stop, s_inf, s_sup):
    while not predicate_stop(s_sup):
      s_inf = s_sup
      s_sup = 2.0 * s_inf
    return s_inf, s_sup

  def binary_search(predicate_stop, predicate_left, s_inf, s_sup):
    s_mid = s_inf + (s_sup - s_inf) / 2.0
    while not predicate_stop(s_mid):
      if predicate_left(s_mid):
        s_sup = s_mid
      else:
        s_inf = s_mid
      s_mid = s_inf + (s_sup - s_inf) / 2.0
    return s_mid

  delta_thr = case_one(epsilon, 0.0)

  if delta == delta_thr:
    alpha = 1.0

  else:
    if delta > delta_thr:
      predicate_stop_dt = lambda s: case_one(epsilon, s) >= delta
      function_s_to_delta = lambda s: case_one(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) > delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) - sqrt(s / 2.0)

    else:
      predicate_stop_dt = lambda s: case_two(epsilon, s) <= delta
      function_s_to_delta = lambda s: case_two(epsilon, s)
      predicate_left_bs = lambda s: function_s_to_delta(s) < delta
      function_s_to_alpha = lambda s: sqrt(1.0 + s / 2.0) + sqrt(s / 2.0)

    predicate_stop_bs = lambda s: abs(function_s_to_delta(s) - delta) <= tol

    s_inf, s_sup = doubling_trick(predicate_stop_dt, 0.0, 1.0)
    s_final = binary_search(predicate_stop_bs, predicate_left_bs, s_inf, s_sup)
    alpha = function_s_to_alpha(s_final)

  sigma = alpha * norm_bound / sqrt(2.0 * epsilon)
  return sigma


def get_eps_gaussian(q, noise_multiplier, steps, target_delta, orders):
  """Compute eps for the Gaussian mechanism given the DP params."""
  rdp = compute_rdp(
      q=q, noise_multiplier=noise_multiplier, steps=steps, orders=orders)
  eps, _, _ = get_privacy_spent(orders, rdp, target_delta=target_delta)
  return eps


def get_gauss_noise_multiplier(target_eps,
                               target_delta,
                               target_sampling_rate,
                               steps,
                               orders=RDP_ORDERS):
  """Compute the Gaussian noise multiplier given the DP params."""

  def get_eps_for_noise_multiplier(z):
    eps = get_eps_gaussian(
        q=target_sampling_rate,
        noise_multiplier=z,
        steps=steps,
        target_delta=target_delta,
        orders=orders)
    return eps

  def opt_fn(z):
    return get_eps_for_noise_multiplier(z) - target_eps

  min_nm, max_nm = 0.001, 1000
  result, r = optimize.brentq(opt_fn, min_nm, max_nm, full_output=True)
  if r.converged:
    return result
  else:
    return -1


##################################################
######## (Distributed) Discrete Gaussian  ########
##################################################


def compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders):
  """Compute RDP of the Sampled (Distributed) Discrete Gaussian Mechanism.

  See Proposition 14 / Eq. 17 (Page 16) of the main paper.

  Args:
    q: The sampling rate.
    l1_scale: The l1 scale of the discrete Gaussian mechanism (i.e.,
      l1_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    l2_scale: The l2 scale of the discrete Gaussian mechanism (i.e.,
      l2_sensitivity/stddev). For distributed version, stddev is the noise
      stddev after summing all the noise shares.
    tau: The inflation parameter due to adding multiple discrete Gaussians. Set
      to zero when analyzing the the discrete Gaussian mechanism. For the
      distributed discrete Gaussian mechanisn, see Theorem 1.
    dim: The dimension of the vector query.
    steps: The number of steps.
    orders: An array (or a scalar) of RDP orders, must all be greater than 1. If
      provided orders are not integers, they are rounded down to the nearest
      integer.

  Returns:
    The RDPs at all orders, can be np.inf.
  """
  orders = [int(order) for order in orders]

  def eps(order):
    assert order > 1, 'alpha must be greater than 1.'
    # Proposition 14 of https://arxiv.org/pdf/2102.06387.pdf.
    term_1 = (order / 2.0) * l2_scale**2 + tau * dim
    term_2 = (order / 2.0) * (l2_scale**2 + 2 * l1_scale * tau + tau**2 * dim)
    term_3 = (order / 2.0) * (l2_scale + np.sqrt(dim) * tau)**2
    return min(term_1, term_2, term_3)

  if q == 1:
    rdp = np.array([eps(order) for order in orders])
  else:
    rdp = np.array([
        min(_compute_rdp_subsampled(order, q, eps), eps(order))
        for order in orders
    ])

  return rdp * steps


def ddgauss_epsilon(gamma,
                    local_stddev,
                    num_clients,
                    l2_sens,
                    beta,
                    dim,
                    q,
                    steps,
                    delta,
                    l1_sens=None,
                    rounding=True,
                    orders=RDP_ORDERS):
  """Computes epsilon of (distributed) discrete Gaussian via RDP."""
  scale = 1.0 / (gamma + DIV_EPSILON)
  l1_sens = l1_sens or (l2_sens * np.sqrt(dim))
  if rounding:
    l2_sens = rounded_l2_norm_bound(l2_sens * scale, beta, dim) / scale
    l1_sens = rounded_l1_norm_bound(l2_sens * scale, dim) / scale

  tau = 0
  for k in range(1, num_clients):
    tau += math.exp(-2 * (math.pi * local_stddev * scale)**2 * (k / (k + 1)))
  tau *= 10

  l1_scale = l1_sens / (np.sqrt(num_clients) * local_stddev)
  l2_scale = l2_sens / (np.sqrt(num_clients) * local_stddev)
  rdp = compute_rdp_dgaussian(q, l1_scale, l2_scale, tau, dim, steps, orders)
  eps, _, order = get_privacy_spent(orders, rdp, target_delta=delta)
  return eps, order


def ddgauss_local_stddev(q,
                         epsilon,
                         l2_clip_norm,
                         gamma,
                         beta,
                         steps,
                         num_clients,
                         dim,
                         delta,
                         orders=RDP_ORDERS):
  """Selects the local stddev for the distributed discrete Gaussian."""

  def stddev_opt_fn(stddev):
    stddev += DIV_EPSILON
    cur_epsilon, _ = ddgauss_epsilon(
        gamma,
        stddev,
        num_clients,
        l2_clip_norm,
        beta,
        dim,
        q,
        steps,
        delta,
        orders=orders)
    return (epsilon - cur_epsilon)**2

  stddev_result = optimize.minimize_scalar(stddev_opt_fn)
  if stddev_result.success:
    return stddev_result.x
  else:
    return -1


def ddgauss_params(q,
                   epsilon,
                   l2_clip_norm,
                   bits,
                   num_datapoints_total_per_round,
                   num_clients_to_sample_noise_per_round,
                   dim,
                   delta,
                   beta,
                   steps,
                   k=4,
                   rho=1,
                   sqrtn_norm_growth=False,
                   orders=RDP_ORDERS):
  """Selects gamma and local noise standard deviation from the given parameters.

  Args:
    q: The sampling factor.
    epsilon: The target overall epsilon.
    l2_clip_norm: The l2 clipping norm for the client vectors.
    bits: The number of bits per coordinate for the aggregated noised vector.
    num_datapoints_total_per_round: The number of data points per aggregate minibatch per step.
    num_clients_to_sample_noise_per_round: The number of clients per step.
    dim: The dimension of the vector query.
    delta: The target delta.
    beta: The constant in [0, 1) controlling conditional randomized rounding.
      See Proposition 22 of the paper.
    steps: The total number of steps.
    k: The number of standard deviations of the signal to bound (see Thm. 34 /
      Eq. 61 of the paper).
    rho: The flatness parameter of the random rotation (see Lemma 29).
    sqrtn_norm_growth: A bool indicating whether the norm of the sum of the
      vectors grow at a rate of `sqrt(n)` (i.e. norm(sum_i x_i) <= sqrt(n) * c).
      If `False`, we use the upper bound `norm(sum_i x_i) <= n * c`. See also
      Eq. 61 of the paper.
    orders: The RDP orders.

  Returns:
    The selected gamma and the local noise standard deviation.
  """
  n_factor = num_datapoints_total_per_round ** (1 if sqrtn_norm_growth else 2)

  def stddev(x):
    return ddgauss_local_stddev(q, epsilon, l2_clip_norm, x, beta, steps,
                                num_clients_to_sample_noise_per_round, dim, delta, orders)

  def mod_min(x):
    return k * math.sqrt(rho / dim * l2_clip_norm ** 2 * n_factor +
                         (x**2 / 4.0 * num_datapoints_total_per_round + stddev(x)**2*num_clients_to_sample_noise_per_round))

  def gamma_opt_fn(x):
    return (math.pow(2, bits) - 2 * mod_min(x) / (x + DIV_EPSILON))**2

  gamma_result = optimize.minimize_scalar(gamma_opt_fn)
  if not gamma_result.success:
    raise ValueError('Cannot compute gamma.')

  gamma = gamma_result.x
  # Select the local_stddev that gave the best gamma.
  local_stddev = ddgauss_local_stddev(q, epsilon, l2_clip_norm, gamma, beta,
                                      steps, num_clients_to_sample_noise_per_round, dim, delta, orders)
  return gamma, local_stddev




if __name__ == '__main__':
  gamma_2, local_stddev_2 = ddgauss_params(900/67200,
                     8,
                     0.01,
                     16,
                     900,
                     1,
                     2**14,
                     1/67200,
                     math.exp(-0.5),
                     1554,
                     k=4,
                     rho=1,
                     sqrtn_norm_growth=False,
                     orders=RDP_ORDERS)