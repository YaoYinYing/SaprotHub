
from typing import List, Tuple, Union


def shorter_range(
    input_list: Union[List[int], Tuple[int]],
    connector: str = '-',
    seperator: str = '+',
) -> str:
    """
    Shorten a list of integers by representing consecutive ranges with hyphens,
    and non-consecutive integers with plus signs.

    Parameters:
    input_list (list): A list of integers to be shortened.
    connector (str): A string for connecting consecutive ranges
    seperator (str): A string for separating non-consecutive ranges

    Returns:
    str: A string expression representing the shortened integer list.

    Example:
    >>> input_list = [395, 396, 397, 398, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409]
    >>> result = shorter_range(input_list)
    >>> print(result)
    "395-409"

    >>> input_list = [395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409]
    >>> result = shorter_range(input_list)
    >>> print(result)
    "395-401+403-409"
    """

    # Filter out non-integer items and sort the list
    input_list = sorted([item for item in input_list if isinstance(item, int)])

    if not input_list:
        return

    range_pairs = []
    start, end = input_list[0], input_list[0]

    for item in input_list[1:]:
        if item == end + 1:
            end = item
        else:
            if start == end:
                range_pairs.append(str(start))
            else:
                range_pairs.append(f"{start}{connector}{end}")
            start, end = item, item

    # Handle the last range or single number
    if start == end:
        range_pairs.append(str(start))
    else:
        range_pairs.append(f"{start}{connector}{end}")

    return seperator.join(range_pairs)


def expand_range(
    shortened_str: str, connector: str = '-', seperator: str = '+'
) -> list[int]:
    """
    Expand a shortened string expression representing a list of integers to the original list.

    Parameters:
    shortened_str (str): A shortened string expression representing a list of integers.
    connector (str): A string for connecting consecutive ranges
    seperator (str): A string for separating non-consecutive ranges

    Returns:
    list: A list of integers corresponding to the original input.

    Example:
    >>> shortened_str = "395-401+403-409"
    >>> result = expand_range(shortened_str)
    >>> print(result)
    [395, 396, 397, 398, 399, 400, 401, 403, 404, 405, 406, 407, 408, 409]
    """
    expanded_list = []

    if shortened_str.isdigit():
        return [int(shortened_str)]

    ranges = shortened_str.split(seperator)

    for rng in ranges:
        if '-' in rng:
            start, end = map(int, rng.split(connector))
            expanded_list.extend(range(start, end + 1))
        else:
            expanded_list.append(int(rng))

    return expanded_list