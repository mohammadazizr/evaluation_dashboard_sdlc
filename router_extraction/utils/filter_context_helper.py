import logging
from typing import Optional

def apply_keyword_filters(
    candidates: list[str],
    filters: Optional[list[str]],
    fallback_list: list[str],
    logger: Optional[logging.Logger] = None
) -> list[str]:
    """
    Apply keyword-based filtering with fallback and logging.

    Filters a list of candidates based on extracted keywords. If no filters
    are provided or filtering produces no matches, falls back to the complete list.

    Args:
        candidates: All available items (e.g., activity names)
        filters: Filtered items from keyword extraction (list or None)
        fallback_list: Default list to use if filters empty or produce no matches
        logger: Optional logger for debug output

    Returns:
        Filtered list + ["null"] if filters match, otherwise fallback + ["null"]

    Examples:
        >>> apply_keyword_filters(
        ...     candidates=["ORP Portal", "ORP Workflow", "BOC"],
        ...     filters=["ORP Workflow"],
        ...     fallback_list=["ORP Portal", "ORP Workflow", "BOC"]
        ... )
        ['ORP Workflow', 'null']

        >>> apply_keyword_filters(
        ...     candidates=["ORP Portal", "ORP Workflow"],
        ...     filters=None,
        ...     fallback_list=["ORP Portal", "ORP Workflow"],
        ...     logger=logger
        ... )
        # Logs: "No filters provided, using fallback list (2 items)"
        ['ORP Portal', 'ORP Workflow', 'null']
    """
    # If no filters, use fallback
    if not filters or len(filters) == 0:
        if logger:
            logger.debug(f"[FILTER] No filters provided, using fallback list ({len(fallback_list)} items)")
        return fallback_list + ["null"]

    # Filter candidates: keep only those in filters list
    filtered = [item for item in candidates if item in filters]

    if logger:
        logger.debug(f"[FILTER] Candidates: {len(candidates)} → Filtered: {len(filtered)} matches")
        if filtered:
            logger.debug(f"[FILTER] Included items: {filtered}")
        else:
            logger.warning(f"[FILTER] No matches found! Filters were: {filters}. Using fallback.")

    # If filtering produced no results, use fallback
    if not filtered:
        if logger:
            logger.warning(f"[FILTER] Filtering produced no matches, falling back to full list ({len(fallback_list)} items)")
        return fallback_list + ["null"]

    return filtered + ["null"]
