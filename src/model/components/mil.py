from functools import partial
from typing import Optional
from torch import Tensor, nn
import torch
import torch.nn.functional as F


def get_mil_aggregation(mil_aggregation, use_no_finding_and: bool = True, use_no_finding_or: bool = False, **kwargs):
    if mil_aggregation is None:
        return None
    elif mil_aggregation == 'lse':
        pool_fn = partial(lse_pool, **kwargs)
        mask_ignore_value = -torch.inf
    elif mil_aggregation == 'max':
        pool_fn = partial(torch.amax, dim=1)
        mask_ignore_value = -torch.inf
    elif mil_aggregation == 'noisyOR':
        pool_fn = lambda x: 1. - (1. - x).prod(dim=1)
        mask_ignore_value = 0.
    else: 
        raise ValueError(mil_aggregation)

    def _aggregate(regions_class_probs: Tensor, mask=None, no_finding_index: Optional[int] = None):
        return mil_aggregate(pool_fn, mask_ignore_value, regions_class_probs, mask=mask, 
            no_finding_index=no_finding_index, use_no_finding_and=use_no_finding_and, use_no_finding_or=use_no_finding_or)

    return _aggregate

def lse_pool(regions_class_probs: Tensor, lse_r: float=5.0):
    probs_max = regions_class_probs.amax(dim=1, keepdim=True)  # (N, 1, C+1)
    aggregated_probs = regions_class_probs - probs_max  # (N x A x C+1)
    # (N x C+1)
    aggregated_probs = torch.div((lse_r * aggregated_probs).exp().mean(dim=1).log(), lse_r)
    aggregated_probs = aggregated_probs + probs_max[:, 0, :]
    return aggregated_probs

def mil_aggregate(pool_fn, mask_ignore_value, regions_class_probs: Tensor, mask=None, 
    no_finding_index: Optional[int] = None, 
    use_no_finding_and: bool = True, use_no_finding_or: bool = False):
    """
    :param regions_class_probs: (N x A x C)
    :param lse_r:
    :param mask: (N x A)
    :param no_finding_index:
    :return: (N x C)
    """
    N, A, C = regions_class_probs.shape 
    if mask is not None:
        regions_class_probs = torch.masked_fill(regions_class_probs, ~mask[:, :, None], mask_ignore_value)
    if no_finding_index is not None:
        # add AND variant of no-finding class (in contrast to the OR-variant used by default)
        # i.e. min instead of max
        # (N x A)
        inverse_no_finding_probs = 1. - regions_class_probs[:, :, no_finding_index]
        if mask is not None:
            inverse_no_finding_probs = torch.masked_fill(inverse_no_finding_probs, ~mask, mask_ignore_value)
        # (N x A x C+1)
        regions_class_probs = torch.cat([regions_class_probs, inverse_no_finding_probs[:, :, None]], dim=-1)

    aggregated_probs = pool_fn(regions_class_probs)

    if no_finding_index is not None:
        # extract AND variant of no-finding class
        no_finding_and_probs = 1. - aggregated_probs[:, -1].clone()
        # extract OR variant of no-finding class
        no_finding_or_probs = aggregated_probs[:, no_finding_index].clone()

        if use_no_finding_and:
            # place the AND variant of no-finding class at the no-finding position (this is the default using the no-finding label)
            aggregated_probs[:, no_finding_index] = no_finding_and_probs
        else:
            # dummy value for no-finding class (to keep the order of classes)
            aggregated_probs[:, no_finding_index] = 0.

        if use_no_finding_or:
            # place the OR variant of no-finding class at the last position
            aggregated_probs[:, -1] = no_finding_or_probs
        else:
            # remove the OR variant of no-finding class
            aggregated_probs = aggregated_probs[:, :-1]

    return aggregated_probs
   