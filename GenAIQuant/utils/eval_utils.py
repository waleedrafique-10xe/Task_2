from typing import List

# Provides utility functions for evaluation
__all__ = ["flatten_dict", "suppress_repeats"]


def flatten_dict(d: dict, parent_key: tuple = ()):
    """
    Flatten nested dictionary into list of (path, value).
    Each path is a tuple of keys.
    """
    rows = []
    for k, v in d.items():
        new_key = parent_key + (k,)
        if isinstance(v, dict):
            rows.extend(flatten_dict(v, new_key))
        else:
            rows.append((new_key, v))
    return rows


def suppress_repeats(rows: List[list]):
    """
    Given a list of rows like [[a,b,c], [a,b,d], [e,f,g]], suppress repeating items.
    """
    last_seen = [None] * len(rows[0])
    suppressed = []

    for row in rows:
        new_row = []
        for i, val in enumerate(row):
            if i < len(last_seen) - 1:  # suppress only keys, not value
                if val == last_seen[i]:
                    new_row.append("")
                else:
                    new_row.append(val)
                    last_seen[i] = val
            else:
                new_row.append(val)  # always show the final value
        suppressed.append(new_row)

    return suppressed
