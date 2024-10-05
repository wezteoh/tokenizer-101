def get_stats(ids: list[int]) -> dict[tuple[int, int], int]:
    """
    Get the countings of every unique byte pairs in the list of bytes
    """
    stats = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        stats[pair] = stats.get(pair, 0) + 1
    return stats


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
