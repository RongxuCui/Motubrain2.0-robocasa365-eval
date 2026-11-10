"""Shared observation-grid selection for compressed visual history."""


def history_offsets(frame_stride: int) -> tuple[int, ...]:
    """Return four uniformly spaced past offsets, independent of the horizon."""
    if type(frame_stride) is not int or frame_stride <= 0:
        raise ValueError("history frame_stride must be a positive integer")
    return (-4 * frame_stride, -3 * frame_stride, -2 * frame_stride, -frame_stride)


def select_history_indices(
    current_index: int,
    offsets: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return requested and clamped episode indices, including the anchor."""
    requested = (0, *(current_index + offset for offset in offsets))
    return requested, tuple(max(0, index) for index in requested)
