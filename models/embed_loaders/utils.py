import os
import os.path as osp
import pickle

from torch.distributed import barrier
from mmengine.dist import get_dist_info


def collect_strings(strings_part, tmpdir='data/tmp'):
    """Collect string lists from all processes in CPU mode.

    This function will save the string lists from different processes to
    `tmpdir` and collect them in the rank 0 process.

    Args:
        strings_part (List[str]): String list containing parts to be collected.
        tmpdir (str | None): Temporary directory for collected strings to store.
            If set to None, it will create a random temporary directory for it.
            Defaults to None.

    Returns:
        List[str] or None: The collected strings in rank 0, None in other ranks.

    Example:
        >>> import mmengine.dist as dist
        >>> dist.init_process_group(backend='nccl', ...)
        >>> rank = dist.get_rank()
        >>> world_size = dist.get_world_size()
        >>> strings_part = ["string1", "string2"] if rank == 0 else ["string3", "string4"]
        >>> collected_strings = collect_strings_cpu(strings_part)
        >>> if rank == 0:
        >>>     print(collected_strings)  # ['string1', 'string2', 'string3', 'string4']
    """
    rank, world_size = get_dist_info()
    if world_size == 1:
        return strings_part

    os.makedirs(tmpdir, exist_ok=True)

    with open(osp.join(tmpdir, f'strings_part_{rank}.pkl'), 'wb') as f:
        pickle.dump(strings_part, f)

    barrier()

    if rank == 0:
        collected_strings = []
        for i in range(world_size):
            with open(osp.join(tmpdir, f'strings_part_{i}.pkl'), 'rb') as f:
                collected_strings.extend(pickle.load(f))
            os.remove(osp.join(tmpdir, f'strings_part_{i}.pkl'))
        return collected_strings
    else:
        return None


def broadcast_strings(strings, tmpdir='data/tmp'):
    """Broadcast a string list from the rank 0 process to all other processes.

    This function will save the string list from the rank 0 process to a temporary
    directory and then load this list in all other processes. It ensures that all
    processes in a distributed environment have the same string list.

    Args:
        strings (List[str]): The string list to be broadcasted. This should be
            provided only by the rank 0 process.
        tmpdir (str | None): Temporary directory for storing the broadcasted data.
            If set to None, it will create a random temporary directory for it.
            Defaults to 'data/tmp'.

    Returns:
        List[str]: The string list broadcasted to all processes. In processes other
        than rank 0, this will be the list received from rank 0.

    Example:
        >>> import mmengine.dist as dist
        >>> dist.init_process_group(backend='nccl', ...)
        >>> rank = dist.get_rank()
        >>> strings = ["string1", "string2"] if rank == 0 else []
        >>> broadcasted_strings = broadcast_strings(strings)
        >>> print(broadcasted_strings)  # ['string1', 'string2'] in all processes
    """
    rank, world_size = get_dist_info()

    if world_size == 1:
        return strings

    os.makedirs(tmpdir, exist_ok=True)

    if rank == 0:
        with open(osp.join(tmpdir, 'strings.pkl'), 'wb') as f:
            pickle.dump(strings, f)

    barrier()

    if rank != 0:
        with open(osp.join(tmpdir, 'strings.pkl'), 'rb') as f:
            strings = pickle.load(f)

    barrier()

    if rank == 0:
        os.remove(osp.join(tmpdir, 'strings.pkl'))

    return strings
