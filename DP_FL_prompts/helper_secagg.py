import numpy as np
import torch
import copy
import warnings
from functools import reduce
from operator import mul
import torch.nn.functional as F
import numpy as np
import torch

DEFAULT_BETA = np.exp(-0.5)


def flatten_and_concat_grad(grad, template):
    flattened = []
    for param_name in sorted(template.keys()):
        param = grad[param_name]
        flattened.append(param.reshape([param.size(0), -1])) #reshape to batch_size, -1
    return torch.cat(flattened, dim=-1)

def unflatten_and_concat_grad(flattened, template):
    grad = {}
    ind = 0
    for param_name in sorted(template.keys()):
        grad[param_name] = flattened[ind: template[param_name][0]+ind].reshape(template[param_name][1]).to(template[param_name][2])
        ind += template[param_name][0]
    return grad


def pad_zeros(x):
    """Pads a vector with shape (d,) with zeros to the next power of two."""
    assert x.dim() == 1
    dim = len(x)
    pad_dim = int(np.math.pow(2, np.ceil(np.log2(dim))))
    return F.pad(x, (0, max(0, pad_dim - dim)))


def sample_rademacher(shape, dtype, seed_pair):
  """Sample uniform random +1/-1 values with specified shape/dtype/seed_pair."""
  rs = np.random.RandomState(seed_pair.cpu().tolist())
  rand_uniform = torch.from_numpy(rs.random(shape)).to(seed_pair.device)
  return torch.sign(rand_uniform - 0.5).to(dtype)


def randomized_hadamard_transform(x, seed_pair, repeat=1):
    def apply_transform(repeat_index, x):
        cur_seed = seed_pair + repeat_index
        signs = sample_rademacher(x.shape, dtype=x.dtype, seed_pair=cur_seed).to(x.device)
        rademacher_x = signs * x
        encoded_x = torch.squeeze(fast_walsh_hadamard_transform(rademacher_x.unsqueeze(0)), 0)
        return encoded_x

    assert x.dtype == torch.float32
    padded_x = pad_zeros(x)
    i = 0
    result_x = padded_x
    while i < repeat:
        result_x = apply_transform(i, result_x)
        i += 1
    return result_x


def inverse_randomized_hadamard_transform(x, original_dim, seed_pair, repeat=1):
    def inverse_transform(repeat_index, x):
        cur_seed = seed_pair + repeat_index
        unrotated_x = fast_walsh_hadamard_transform(x.unsqueeze(0))
        unrotated_x = torch.squeeze(unrotated_x, 0)
        signs = sample_rademacher(unrotated_x.shape, dtype=x.dtype, seed_pair=cur_seed).to(x.device)
        decoded_x = signs * unrotated_x
        return decoded_x

    assert x.dtype == torch.float32
    i = repeat - 1
    result_x = x
    while i >= 0:
        result_x = inverse_transform(i, result_x)
        i -= 1

    return result_x[:original_dim]


def fast_walsh_hadamard_transform(x):
    # x = torch.tensor(x)
    if len(x.shape) != 2:
        raise ValueError('Number of dimensions of x must be 2. Shape of x: {}'.format(x.shape))

    original_x_shape = x.shape
    dim = x.shape[-1]

    if dim is None:
        dim = x.shape[-1]
        log2 = int(torch.round(torch.log2(torch.tensor(dim)).float()).item())
        assert dim == 2 ** log2, 'The dimension of x must be a power of two. Provided dimension is: {}'.format(dim)
    else:
        if not (dim and ((dim & (dim - 1)) == 0)):
            raise ValueError('The dimension of x must be a power of two. Provided dimension is: {}'.format(dim))
        log2 = int(np.ceil(np.log2(dim)))
        if dim == 1:
            return x

    h_core = torch.tensor([[1., 1.], [1., -1.]], dtype=x.dtype).to(x.device)
    permutation = (0, 2, 1)

    def _hadamard_step(x, dim):
        x_shape = x.shape
        x = x.reshape(-1, 2)
        x = torch.matmul(x, h_core)
        x = x.reshape(-1, dim // 2, 2)
        x = x.permute(permutation)
        x = x.reshape(x_shape)
        return x

    def _fwht(x, dim, log2):
        x = x.reshape(-1, 2, dim // 2)
        i = 0
        while i < log2:
            x = _hadamard_step(x, dim)
            i += 1
        return x

    if dim == 1:
        x = x
    else:
        x = _fwht(x, dim, log2)

    x = x.reshape(-1, dim) / torch.sqrt(torch.tensor(dim, dtype=x.dtype).to(x.device))
    return x



def stochastic_rounding(x, conditional, l2_norm_bound=None, beta=DEFAULT_BETA):
    """Randomly rounds the elements of a tensor to integer values (keeps dtype)."""

    def post_rounding_l2_norm_bound(x, l2_norm_bound, beta):
        """Computes the L2 norm bound of a vector after rounding."""
        # beta = beta.to(x.dtype)
        dim = x.numel()
        if l2_norm_bound is None:
            x_norm = torch.norm(x, p=2)
        else:
            x_norm = l2_norm_bound.to(x.dtype)

        bound1 = x_norm + torch.sqrt(torch.tensor(dim, dtype=x.dtype))
        squared_bound2 = x_norm ** 2 + 0.25 * torch.tensor(dim, dtype=x.dtype)
        squared_bound2 += (
            torch.sqrt(2.0 * torch.log(1.0 / beta)) * (x_norm + 0.5 * torch.sqrt(torch.tensor(dim, dtype=x.dtype))))
        bound2 = torch.sqrt(squared_bound2)
        return torch.minimum(bound1, bound2)

    conditional = conditional.bool()
    l2_norm_threshold = post_rounding_l2_norm_bound(x, l2_norm_bound, beta)
    floored_x = torch.floor(x)
    decimal_x = x - floored_x

    i = 0
    def round_fn(repeat, _):
        uniform = torch.rand_like(x, dtype=x.dtype)
        bernoulli = uniform < decimal_x
        rounded_x = floored_x + bernoulli.to(x.dtype)
        rounded_l2_norm = torch.norm(rounded_x, p=2)
        repeat = conditional and (rounded_l2_norm > l2_norm_threshold)
        return repeat, rounded_x

    repeat = True
    while repeat:
        # print(i)
        i += 1
        repeat, x = round_fn(repeat, x)

    return x


def scaled_quantization(x,
                        scale,
                        stochastic,
                        conditional,
                        l2_norm_bound,
                        beta=DEFAULT_BETA):
    """Scales the tensors and rounds to integers."""
    scale = torch.tensor(scale).type(x.dtype).to(x.device)
    l2_norm_bound = torch.tensor(l2_norm_bound).type(x.dtype).to(x.device)
    conditional = torch.tensor(conditional).type(x.dtype).to(x.device)
    beta = torch.tensor(beta).type(x.dtype).to(x.device)

    scaled_x = x * scale
    scaled_bound = l2_norm_bound * scale

    if stochastic:
        quantized_x = stochastic_rounding(scaled_x, conditional, scaled_bound, beta)
    else:
        quantized_x = torch.round(scaled_x)

    return quantized_x




def rounding_to_int(vec, args):

    for param_name in vec:
        vec[param_name] = torch.round(vec[param_name]).type(torch.long)
    return vec


def mod_vector_m(value, clip_range_lower=None, clip_range_upper=None, args=None):
    if clip_range_lower is None or clip_range_upper is None:
        clip_range_upper = 2 ** (args.mod_num_bits-1)
        clip_range_lower = - 2 ** (args.mod_num_bits - 1)
    def mod_clip(v):
        width = clip_range_upper - clip_range_lower
        period = torch.floor(v / width - clip_range_lower / width).to(v.dtype)
        v_mod_clipped = v - period * width
        return v_mod_clipped
    if isinstance(value, dict):
        return {k: mod_clip(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [mod_clip(v) for v in value]
    else:
        return mod_clip(value)


def reverse_mod_vector_m(vec, m=None, args=None):
    if m is None:
        m = 2** args.mod_num_bits
    for param_name in vec:
        vec[param_name] = vec[param_name] - (vec[param_name] > (m/2 -1) ) * m
    return vec


def get_actual_sentivity(args):
    if args.use_discrete:

        return args.per_sample_max_grad_norm
    else:
        return args.per_sample_max_grad_norm

def get_seed_pair_hadamard():
    return torch.randint(low=0, high=2**32, size=(2,))

def postprocessing_client_gradient(grad, template, seed_pair, args):
    # step 0: the individual gradients are already clipped, flatten them into a single vector (B, size_trainable_param_all)
    grad = flatten_and_concat_grad(grad, template)
    seed_pair = seed_pair.to(grad.device)
    # step 1: flattening
    num_trainable_params = grad.shape[-1]
    rotated_grad = [randomized_hadamard_transform(grad[i].view(num_trainable_params), seed_pair=seed_pair) for i in range(len(grad))]


    # step 2: rounding to integers
    quantized_grad = [scaled_quantization(
        rotated_grad[i],
        1/args.discretization_granularity,
        stochastic=args.stochastic_rounding,
        conditional=args.beta > 0,
        l2_norm_bound=args.per_sample_max_grad_norm,
        beta=args.beta).to(torch.int32).unsqueeze(0) for i in range(len(rotated_grad))]
    # prev_grad = copy.deepcopy(grad)

    # step 2b: for each client, sum up all the clipped gradients
    sum_grad = torch.cat(quantized_grad, dim=0).sum(0)

    # step 3: mod m=2**B
    grad = mod_vector_m(sum_grad, args=args)

    return grad


def postprocessing_client_gradient_distributed(grad, template, seed_pair, args):
    # step 0: the individual gradients are already clipped, flatten them into a single vector (B, size_trainable_param_all)
    grad = flatten_and_concat_grad(grad, template)
    seed_pair = seed_pair.to(grad.device)
    # step 1: flattening
    num_trainable_params = grad.shape[-1]
    rotated_grad = [randomized_hadamard_transform(grad[i].view(num_trainable_params), seed_pair=seed_pair) for i in range(len(grad))]


    # step 2: rounding to integers
    quantized_grad = [scaled_quantization(
        rotated_grad[i],
        1/args.discretization_granularity,
        stochastic=args.stochastic_rounding,
        conditional=args.beta > 0,
        l2_norm_bound=args.per_sample_max_grad_norm,
        beta=args.beta).to(torch.int32).unsqueeze(0) for i in range(len(rotated_grad))]
    # prev_grad = copy.deepcopy(grad)

    # step 2b: for each client, sum up all the clipped gradients
    sum_grad = torch.cat(quantized_grad, dim=0).sum(0)

    return sum_grad


def inverse_scaled_quantization(x, scale):
  """Restores the value range of `x` from `scaled_quantization`."""
  return x / torch.tensor(scale).to(x.dtype)


def postprocessing_server_gradient(grad, template, sample_hadamard_seed, args):
    grad = grad.to(torch.float32).to(grad.device)
    # step 1 unscale by gamma
    grad = inverse_scaled_quantization(grad, 1/args.discretization_granularity)
    # step 2
    grad = inverse_randomized_hadamard_transform(grad, original_dim=args.trainable_param_size, seed_pair=sample_hadamard_seed)

    grad = unflatten_and_concat_grad(grad, template)
    return grad


def add_two_model_states(m1, m2, param_names=None):
    if isinstance(m1, dict):
        aggregate_gradient = {}
        if param_names is None:
            param_names = m1.keys()
        for param_name in param_names:
            aggregate_gradient[param_name] = m1[param_name] + m2[param_name]
    else:
        aggregate_gradient = torch.cat([m1, m2], dim=0).sum(0)
    return aggregate_gradient


def secure_aggregate_gradients(list_client_gradient_update, args):
    # receives the gradients in 2**B
    # return the aggregated gradient in 2**B
    aggregate_gradient = {}

    first_client_id = list(list_client_gradient_update.keys())[0]

    if isinstance(list_client_gradient_update[first_client_id], dict):
        for param_name in list_client_gradient_update[first_client_id]:
            aggregate_gradient[param_name] = torch.zeros_like(list_client_gradient_update[first_client_id][param_name])
        for client_id, client_gradient_update in list_client_gradient_update.items():
            for param_name, param in client_gradient_update.items():
                aggregate_gradient[param_name] += client_gradient_update[param_name]
    else:
        aggregate_gradient = torch.cat([v.unsqueeze(0) for k, v in list_client_gradient_update.items()], dim=0).sum(0)
        aggregate_gradient = aggregate_gradient.reshape([1, len(aggregate_gradient)])
        # print(aggregate_gradient.shape)
        # exit()

    if args.use_discrete:
        aggregate_gradient = mod_vector_m(aggregate_gradient, args=args)
    return aggregate_gradient


def get_model_update(trainer, pre_state, trainable_params, client_training_args_ind):
    current_state = copy.deepcopy(trainer.model.state_dict())

    if len(trainable_params) > 0:
        model_update = {}
        for param_name in trainable_params:
            model_update[param_name] = torch.zeros_like(pre_state[param_name])
        for param_name in trainable_params:
            # current_state = init_state - lr * avg_clipped_gradients
            # avg_clipped_gradient = (init_state - current_state) / lr
            # sum_clipped_gradient = avg_clipped_gradient * client_minibatch_size
            model_update[param_name] = (pre_state[param_name] - current_state[param_name]) / client_training_args_ind.learning_rate \
                          * client_training_args_ind.per_device_train_batch_size * client_training_args_ind.gradient_accumulation_steps
        return model_update

    else:
        for k in current_state.keys():
            if "running_mean" in k or "running_var" in k or "num_batches_tracked" in k:
                warnings.warn("batch normalization is used. make sure "
                              "it is not tracking the running stats. otherwise the "
                              "dp guarantee is WRONG!!!")
                continue
            current_state[k] = current_state[k] - pre_state[k]

        return current_state



def get_trainable_param_size(trainer, args):
    param_sizes = [param.numel() for name, param in trainer.model.named_parameters() if param.requires_grad]
    args.trainable_param_size = sum(param_sizes)
    args.padded_trainable_param_size = int(np.math.pow(2, np.ceil(np.log2(args.trainable_param_size))))
    return args.trainable_param_size, args.padded_trainable_param_size


