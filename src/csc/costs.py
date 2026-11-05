"""Fixed hierarchy costs in model output order."""

from __future__ import annotations

import re

import numpy as np

from ._validation import names


def hierarchy_cost_matrix(paths) -> np.ndarray:
    """Build costs 1 - LCA_depth / D for equal-depth root-to-label paths.

    Paths exclude the shared root at depth zero. Each path is a sequence of
    nonempty node names; its position determines the corresponding score column.
    All modeled labels must have the same depth and distinct paths.
    """
    paths = tuple(paths)
    if not paths or any(isinstance(path, str) for path in paths):
        raise ValueError("paths must be a nonempty sequence of node sequences")
    paths = tuple(tuple(path) for path in paths)
    depth = len(paths[0])
    if depth == 0 or any(len(path) != depth for path in paths):
        raise ValueError("all paths must have the same positive depth")
    if any(not isinstance(node, str) or not node.strip() for path in paths for node in path):
        raise ValueError("path nodes must be nonempty strings")
    if len(set(paths)) != len(paths):
        raise ValueError("modeled labels must have distinct hierarchy paths")

    result = np.zeros((len(paths), len(paths)), dtype=np.float64)
    for i, left in enumerate(paths):
        for j in range(i + 1, len(paths)):
            shared_depth = 0
            for a, b in zip(left, paths[j]):
                if a != b:
                    break
                shared_depth += 1
            result[i, j] = result[j, i] = 1.0 - shared_depth / depth
    return result


def atc_cost_matrix(label_codes, *, level: int = 4) -> np.ndarray:
    """Build a prefix-based ATC cost matrix, truncated at the modeled level.

    Codes must be unique, uppercase, and all at the requested ATC level (1--5).
    This validates code syntax, not membership in a particular ATC release.
    No taxonomy files or network access are required. Source drug-to-ATC mapping
    remains the caller's responsibility.
    """
    patterns = (r"[A-Z]", r"[A-Z][0-9]{2}", r"[A-Z][0-9]{2}[A-Z]",
                r"[A-Z][0-9]{2}[A-Z]{2}", r"[A-Z][0-9]{2}[A-Z]{2}[0-9]{2}")
    if isinstance(level, bool) or not isinstance(level, (int, np.integer)) or not 1 <= level <= 5:
        raise ValueError("level must be an integer from 1 to 5")
    if isinstance(label_codes, str):
        raise ValueError("label_codes must be a sequence of ATC codes")
    codes = tuple(label_codes)
    names(codes, len(codes))
    if not codes or any(re.fullmatch(patterns[level - 1], code) is None for code in codes):
        raise ValueError(f"all label_codes must be uppercase ATC level-{level} codes")
    lengths = (1, 3, 4, 5, 7)[:level]
    return hierarchy_cost_matrix([tuple(code[:length] for length in lengths) for code in codes])
