from typing import Iterator, List, Tuple, Union
import torch
def calc_sample_norms(
    named_params: Iterator[Tuple[str, torch.Tensor]], flat: bool = True
) -> List[torch.Tensor]:
    r"""
    Calculates the norm of the given tensors for each sample.

    This function calculates the overall norm of the given tensors for each sample,
    assuming the each batch's dim is zero.

    Args:
        named_params: An iterator of tuples <name, param> with name being a
            string and param being a tensor of shape ``[B, ...]`` where ``B``
            is the size of the batch and is the 0th dimension.
        flat: A flag, when set to `True` returns a flat norm over all
            layers norms

    Example:
        >>> t1 = torch.rand((2, 5))
        >>> t2 = torch.rand((2, 5))
        >>> calc_sample_norms([("1", t1), ("2", t2)])
            [tensor([1.5117, 1.0618])]

    Returns:
        A list of tensor norms where length of the list is the number of layers
    """
    norms = [param.view(len(param), -1).norm(2, dim=-1) for name, param in named_params.items()]
    # calc norm over all layer norms if flat = True
    if flat:
        norms = [torch.stack(norms, dim=0).norm(2, dim=0)]
    return norms

def calc_clipping_factors(norms: List[torch.Tensor], flat_value=1.0, numerical_stability_constant=1e-6
):
    """
    Calculates the clipping factor based on the given
    norm of gradients for all layers, so that the new
    norm of clipped gradients is at most equal to
    ``self.flat_value``.

    Args:
        norms: List containing a single tensor of dimension (1,)
            with the norm of all gradients.

    Returns:
        Tensor containing the single threshold value to be used
        for all layers.
    """
    # Expects a list of size one.
    if len(norms) != 1:
        raise ValueError(
            "Waring: flat norm selected but "
            f"received norm for {len(norms)} layers"
        )
    per_sample_clip_factor = flat_value / (norms[0] + numerical_stability_constant)
    # We are *clipping* the gradient, so if the factor is ever >1 we set it to 1
    per_sample_clip_factor = per_sample_clip_factor.clamp(max=1.0)
    # return this clipping factor for all layers
    return per_sample_clip_factor