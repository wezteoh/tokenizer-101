from typing import Union


def get_stats(
    ids: list[int], counts: Union[dict[tuple, int], None] = None
) -> dict[tuple[int, int], int]:
    """
    Get the countings of every unique byte pairs in the list of bytes
    """
    # cannot set default argument as empty dict,
    # because mutable default argument will be shared across all calls
    if counts is None:
        counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: list[int], id) -> list[int]:
    """
    Merge the pair of bytes in the list of bytes
    """
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and (ids[i], ids[i + 1]) == tuple(pair):
            new_ids.append(id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids
