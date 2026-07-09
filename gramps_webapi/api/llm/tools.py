#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2025      David Straub
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
#

"""Pydantic AI tools for LLM interactions."""

from __future__ import annotations

import json
from datetime import datetime
from functools import wraps
from typing import Any

from pydantic_ai import RunContext

from ..people_families_cache import CachePeopleFamiliesProxy
from ..resources.anniversaries import upcoming_anniversaries
from ..resources.filters import apply_filter
from ..resources.kinship import (
    _ancestors_bfs,
    _descendants_bfs,
    common_ancestors,
    relatives_of,
)
from ..resources.ru_surnames import (
    count_unique_surnames,
    get_family_surname,
    normalize_surname_gender,
)
from ..resources.util import (
    get_event_profile_for_object,
    get_event_summary_from_object,
    get_one_relationship,
    get_person_profile_for_object,
)
from ..search import get_search_indexer, get_semantic_search_indexer
from ..search.indexer import SearchIndexerBase
from ..search.text import obj_strings_from_object
from ..util import get_db_outside_request, get_locale_for_language, get_logger
from .deps import AgentDeps


def _build_date_expression(before: str, after: str) -> str:
    """Build a date string from before/after parameters.

    Args:
        before: Year before which to filter (e.g., "1900")
        after: Year after which to filter (e.g., "1850")

    Returns:
        A date string for Gramps filters:
        - "between 1850 and 1900" for date ranges
        - "after 1850" for only after
        - "before 1900" for only before
    """
    if before and after:
        return f"between {after} and {before}"
    if after:
        return f"after {after}"
    if before:
        return f"before {before}"
    return ""


def _get_relationship_prefix(db_handle, anchor_person, result_person, logger) -> str:
    """Get a relationship string prefix for a result person.

    Args:
        db_handle: Database handle
        anchor_person: The Person object to calculate relationship from
        result_person: The Person object to calculate relationship to
        logger: Logger instance

    Returns:
        A formatted relationship prefix like "[grandfather] " or empty string
    """
    try:
        rel_string, dist_orig, dist_other = get_one_relationship(
            db_handle=db_handle,
            person1=anchor_person,
            person2=result_person,
            depth=10,
        )
        if rel_string and rel_string.lower() not in ["", "self"]:
            return f"[{rel_string}] "
        elif dist_orig == 0 and dist_other == 0:
            return "[self] "
    except Exception as e:  # pylint: disable=broad-except
        logger.warning(
            "Error calculating relationship between %s and %s: %s",
            anchor_person.gramps_id,
            result_person.gramps_id,
            e,
        )
    return ""


def _apply_gramps_filter(
    ctx: RunContext[AgentDeps],
    namespace: str,
    rules: list[dict[str, Any]],
    max_results: int,
    empty_message: str = "No results found matching the filter criteria.",
    show_relation_with: str = "",
    logic: str = "and",
    handles: list | None = None,
) -> str:
    """Apply a Gramps filter and return formatted results.

    This is a common helper for filter tools that handles:
    - Database handle management
    - Filter application
    - Result iteration with privacy checking
    - Context length limiting
    - Truncation messages
    - Error handling
    - Optional relationship calculation

    Args:
        ctx: The Pydantic AI run context with dependencies
        namespace: Gramps object namespace ("Person", "Event", "Family", etc.)
        rules: List of filter rule dictionaries
        max_results: Maximum number of results to return (already validated)
        empty_message: Message to return when no results found
        show_relation_with: Gramps ID of anchor person for relationship calculation (Person namespace only)
        logic: How to combine multiple rules: "and" (default) or "or"
        handles: Optional pre-filtered list of handles to restrict the search space

    Returns:
        Formatted string with matching objects or error message
    """
    logger = get_logger()
    db_handle = None

    try:
        # Use get_db_outside_request to avoid Flask's g caching, since Pydantic AI's
        # run_sync() uses an event loop that can violate SQLite's thread-safety.
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )

        filter_dict: dict[str, Any] = {"rules": rules}
        if len(rules) > 1 or logic == "or":
            filter_dict["function"] = logic

        filter_rules = json.dumps(filter_dict)
        logger.debug("%s filter rules: %s", namespace, filter_rules)

        args = {"rules": filter_rules}
        matching_handles = apply_filter(
            db_handle=db_handle,
            args=args,
            namespace=namespace,
            handles=handles,
        )

        if not matching_handles:
            db_handle.close()
            return empty_message

        total_matches = len(matching_handles)
        matching_handles = matching_handles[:max_results]

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000
        current_length = 0

        # Get the anchor person for relationship calculation if requested
        anchor_person = None
        if show_relation_with and namespace == "Person":
            try:
                anchor_person = db_handle.get_person_from_gramps_id(show_relation_with)
                if not anchor_person:
                    logger.warning(
                        "Anchor person %s not found for relationship calculation",
                        show_relation_with,
                    )
            except Exception as e:  # pylint: disable=broad-except
                logger.warning(
                    "Error fetching anchor person %s: %s", show_relation_with, e
                )

        # Get the appropriate method to fetch objects
        get_method_name = f"get_{namespace.lower()}_from_handle"
        get_method = getattr(db_handle, get_method_name)

        for handle in matching_handles:
            try:
                obj = get_method(handle)

                if not ctx.deps.include_private and obj.private:
                    continue

                obj_dict = obj_strings_from_object(
                    db_handle=db_handle,
                    class_name=namespace,
                    obj=obj,
                    semantic=True,
                )

                if not obj_dict:
                    continue

                # obj_strings_from_object always returns string_all/string_public
                content = (
                    obj_dict["string_all"]
                    if ctx.deps.include_private
                    else obj_dict["string_public"]
                )

                if not content:
                    continue

                # Add relationship prefix if anchor person is set
                if anchor_person and namespace == "Person":
                    rel_prefix = _get_relationship_prefix(
                        db_handle, anchor_person, obj, logger
                    )
                    content = rel_prefix + content

                # Truncate individual items if they're too long
                if len(content) > per_item_max:
                    logger.debug(
                        "Truncating %s content from %d to %d chars",
                        namespace,
                        len(content),
                        per_item_max,
                    )
                    content = _truncate_content(content, per_item_max)

                # Check if adding this item would exceed total limit
                if current_length + len(content) > max_length:
                    logger.debug(
                        "Reached max context length (%d chars), stopping at %d results",
                        max_length,
                        len(context_parts),
                    )
                    break

                context_parts.append(content)
                current_length += len(content) + 2

            except Exception as e:  # pylint: disable=broad-except
                logger.warning("Error processing %s %s: %s", namespace, handle, e)
                continue

        if not context_parts:
            db_handle.close()
            return f"{empty_message} (or all results are private)."

        result = "\n\n".join(context_parts)

        # Add truncation messages
        returned_count = len(context_parts)
        if returned_count < total_matches:
            result += f"\n\n---\nShowing {returned_count} of {total_matches} matching {namespace.lower()}s. Use max_results parameter to see more."
        elif total_matches == max_results:
            result += f"\n\n---\nShowing {returned_count} {namespace.lower()}s (limit reached). There may be more matches."

        logger.debug(
            "Tool filter_%ss returned %d results (%d chars)",
            namespace.lower(),
            returned_count,
            len(result),
        )

        db_handle.close()
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error filtering %ss: %s", namespace.lower(), e)
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass
        return f"Error filtering {namespace.lower()}s: {str(e)}"


def _truncate_content(
    content: str, max_chars: int, head: int = 4000, tail: int = 1000
) -> str:
    """Truncate long content using head+tail strategy.

    Keeps the first `head` chars and last `tail` chars, with an elision marker in
    between showing how many characters were dropped. This is more useful to the
    model than a front-only cut because the tail often contains summary information.
    """
    if len(content) <= max_chars:
        return content
    if head + tail >= len(content):
        # Truncation would not reduce the content; return as-is.
        return content
    elided = len(content) - head - tail
    return content[:head] + f"\n\n...[{elided} chars elided]...\n\n" + content[-tail:]


def log_tool_call(func):
    """Decorator to log tool usage."""
    logger = get_logger()

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.debug("Tool called: %s", func.__name__)
        return func(*args, **kwargs)

    return wrapper


@log_tool_call
def get_current_date(_ctx: RunContext[AgentDeps]) -> str:
    """Returns today's date in ISO format (YYYY-MM-DD)."""
    logger = get_logger()

    result = datetime.now().date().isoformat()
    logger.debug("Tool get_current_date returned: %s", result)
    return result


@log_tool_call
def search_genealogy_database(
    ctx: RunContext[AgentDeps],
    query: str,
    object_type: str = "",
    max_results: int = 20,
    search_type: str = "semantic",
) -> str:
    """Search the family tree database.

    Args:
        query: Keywords or descriptive phrase to match. For semantic search, use
            content-describing phrases ("Catholic farmer Bavaria born 1840"), not
            verbatim questions. For fulltext, use exact names or phrases to match.
        object_type: Restrict to one object type: "Person", "Family", "Event",
            "Place", "Source", "Citation", "Repository", "Note", "Media".
            Leave empty to search all types.
        max_results: Maximum results to return (default: 20, max: 50)
        search_type: "semantic" (default) for concept/biography queries;
            "fulltext" for exact name/phrase matches.

    Returns:
        Formatted genealogical data matching the query.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback(
            "search_genealogy_database", "Searching genealogy database..."
        )

    logger = get_logger()

    # Limit max_results to reasonable bounds
    max_results = min(max(1, max_results), 50)

    object_types = [object_type.lower()] if object_type else None

    try:
        searcher: SearchIndexerBase
        if search_type == "fulltext":
            searcher = get_search_indexer(ctx.deps.tree)
        else:
            searcher = get_semantic_search_indexer(ctx.deps.tree)
        _, hits = searcher.search(
            query=query,
            page=1,
            pagesize=max_results,
            include_private=ctx.deps.include_private,
            include_content=True,
            object_types=object_types,
        )

        if not hits:
            return "No results found in the genealogy database."

        context_parts: list[str] = []
        max_length = ctx.deps.max_context_length
        per_item_max = 10000  # Maximum chars per individual item
        current_length = 0

        for hit in hits:
            content = hit.get("content", "")

            # Truncate individual items if they're too long
            if len(content) > per_item_max:
                logger.debug(
                    "Truncating search result from %d to %d chars",
                    len(content),
                    per_item_max,
                )
                content = _truncate_content(content, per_item_max)

            if current_length + len(content) > max_length:
                logger.debug(
                    "Reached max context length (%d chars), stopping at %d results",
                    max_length,
                    len(context_parts),
                )
                break
            context_parts.append(content)
            current_length += len(content) + 2

        result = "\n\n".join(context_parts)
        logger.debug(
            "Tool search_genealogy_database returned %d results (%d chars) for query: %r",
            len(context_parts),
            len(result),
            query,
        )
        return result

    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error searching genealogy database: %s", e)
        return f"Error searching the database: {str(e)}"


@log_tool_call
def filter_people(
    ctx: RunContext[AgentDeps],
    given_name: str = "",
    surname: str = "",
    birth_year_before: str = "",
    birth_year_after: str = "",
    birth_place: str = "",
    death_year_before: str = "",
    death_year_after: str = "",
    death_place: str = "",
    ancestor_of: str = "",
    ancestor_generations: int = 10,
    descendant_of: str = "",
    descendant_generations: int = 10,
    is_male: bool = False,
    is_female: bool = False,
    probably_alive_on_date: str = "",
    has_common_ancestor_with: str = "",
    degrees_of_separation_from: str = "",
    degrees_of_separation: int = 2,
    combine_filters: str = "and",
    max_results: int = 50,
    show_relation_with: str = "",
) -> str:
    """Filters people in the family tree based on simple criteria.

    IMPORTANT: When filtering by relationships (ancestor_of, descendant_of, degrees_of_separation_from),
    ALWAYS set show_relation_with to the same Gramps ID to get relationship labels in results.
    Without it, you cannot determine specific relationships like "grandfather" vs "father".

    Args:
        given_name: Given/first name to search for (partial match)
        surname: Surname/last name to search for (partial match)
        birth_year_before: Year before which people were born (e.g., "1900"). Use only the year.
        birth_year_after: Year after which people were born (e.g., "1850"). Use only the year.
        birth_place: Place name where person was born (partial match)
        death_year_before: Year before which people died (e.g., "1950"). Use only the year.
        death_year_after: Year after which people died (e.g., "1800"). Use only the year.
        death_place: Place name where person died (partial match)
        ancestor_of: Gramps ID of person to find ancestors of (e.g., "I0044")
        ancestor_generations: Maximum generations to search for ancestors (default: 10)
        descendant_of: Gramps ID of person to find descendants of (e.g., "I0044")
        descendant_generations: Maximum generations to search for descendants (default: 10)
        is_male: Filter to only males (True/False)
        is_female: Filter to only females (True/False)
        probably_alive_on_date: Date to check if person was likely alive (YYYY-MM-DD)
        has_common_ancestor_with: Gramps ID to find people sharing an ancestor (e.g., "I0044")
        degrees_of_separation_from: Gramps ID of person to find relatives connected to (e.g., "I0044")
        degrees_of_separation: Maximum relationship path length (default: 2). Each parent-child
            or spousal connection counts as 1. Examples: sibling=2, grandparent=2, uncle=3,
            first cousin=4, brother-in-law=2
        combine_filters: How to combine multiple filters: "and" (default) or "or"
        max_results: Maximum results to return (default: 50, max: 100)
        show_relation_with: Gramps ID of person to show relationships relative to (e.g., "I0044").
            When set, each result will include the relationship to this anchor person.

    Returns:
        Formatted list of people matching the filter criteria.

    Examples:
        - Find people with surname Smith: surname="Smith"
        - Find people born before 1900: birth_year_before="1900"
        - Find people born between 1850-1900: birth_year_after="1850", birth_year_before="1900"
        - Find who was alive in 1880: probably_alive_on_date="1880-01-01"
        - Find cousins: has_common_ancestor_with="I0044"
        - Find someone's parents (with labels): ancestor_of="I0044", ancestor_generations=1, show_relation_with="I0044"
        - Find someone's grandfathers (with labels): ancestor_of="I0044", ancestor_generations=2, is_male=True, show_relation_with="I0044"
        - Find siblings (with labels): degrees_of_separation_from="I0044", degrees_of_separation=2, show_relation_with="I0044"
        - Find extended family (uncles, aunts): degrees_of_separation_from="I0044", degrees_of_separation=3
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback(
            "filter_people", "Filtering people in family tree..."
        )

    logger = get_logger()

    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if given_name or surname:
        rules.append(
            {
                "name": "HasNameOf",
                "values": [given_name, surname, "", "", "", "", "", "", "", "", ""],
            }
        )

    if birth_year_before or birth_year_after or birth_place:
        date_expr = _build_date_expression(birth_year_before, birth_year_after)
        rules.append({"name": "HasBirth", "values": [date_expr, birth_place, ""]})

    if death_year_before or death_year_after or death_place:
        date_expr = _build_date_expression(death_year_before, death_year_after)
        rules.append({"name": "HasDeath", "values": [date_expr, death_place, ""]})

    if ancestor_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [ancestor_of, str(ancestor_generations + 1)],
            }
        )

    if descendant_of:
        rules.append(
            {
                "name": "IsLessThanNthGenerationDescendantOf",
                "values": [descendant_of, str(descendant_generations + 1)],
            }
        )

    if has_common_ancestor_with:
        rules.append(
            {"name": "HasCommonAncestorWith", "values": [has_common_ancestor_with]}
        )

    if degrees_of_separation_from:
        # Check if DegreesOfSeparation filter is available (from FilterRules addon)
        from ..resources.filters import get_rule_list

        available_rules = [rule.__name__ for rule in get_rule_list("Person")]  # type: ignore
        if "DegreesOfSeparation" in available_rules:
            rules.append(
                {
                    "name": "DegreesOfSeparation",
                    "values": [degrees_of_separation_from, str(degrees_of_separation)],
                }
            )
        else:
            logger.warning(
                "DegreesOfSeparation filter not available. "
                "Install FilterRules addon to use this feature."
            )
            return (
                "DegreesOfSeparation filter is not available. "
                "The FilterRules addon must be installed to use this feature."
            )

    if is_male:
        rules.append({"name": "IsMale", "values": []})

    if is_female:
        rules.append({"name": "IsFemale", "values": []})

    if probably_alive_on_date:
        rules.append({"name": "ProbablyAlive", "values": [probably_alive_on_date]})

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter."
        )

    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Person",
        rules=rules,
        max_results=max_results,
        empty_message="No people found matching the filter criteria.",
        show_relation_with=show_relation_with,
        logic=combine_filters.lower(),
    )


@log_tool_call
def filter_events(
    ctx: RunContext[AgentDeps],
    event_type: str = "",
    date_before: str = "",
    date_after: str = "",
    place: str = "",
    description_contains: str = "",
    participant_id: str = "",
    max_results: int = 50,
) -> str:
    """Filter events in the genealogy database.

    Use this tool to find events matching specific criteria. Events are occurrences in
    people's lives (births, deaths, marriages, etc.) or general historical events.

    Args:
        event_type: Type of event (e.g., "Birth", "Death", "Marriage", "Baptism",
            "Census", "Emigration", "Burial", "Occupation", "Residence")
        date_before: Latest year to include (inclusive). For "between 1892 and 1900", use "1900".
            Use only the year as a string.
        date_after: Earliest year to include (inclusive). For "between 1892 and 1900", use "1892".
            Use only the year as a string.
        place: Location name to search for (e.g., "Boston", "Massachusetts")
        description_contains: Text that should appear in the event description
        participant_id: Gramps ID of a person who participated in the event (e.g., "I0001")
        max_results: Maximum number of results to return (1-100, default 50)

    Returns:
        A formatted string containing matching events with their details, or an error message.

    Examples:
        - "births in 1850": filter_events(event_type="Birth", date_after="1850", date_before="1850")
        - "marriages in Boston": filter_events(event_type="Marriage", place="Boston")
        - "events between 1892 and 1900": filter_events(date_after="1892", date_before="1900")
        - "events after 1850": filter_events(date_after="1850")
        - "events before 1900": filter_events(date_before="1900")
        - "events for person I0044": filter_events(participant_id="I0044")
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("filter_events", "Filtering events...")

    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if event_type or date_before or date_after or place or description_contains:
        date_expr = _build_date_expression(before=date_before, after=date_after)
        rules.append(
            {
                "name": "HasData",
                "values": [
                    event_type or "",
                    date_expr,
                    place or "",
                    description_contains or "",
                ],
            }
        )

    if not rules and not participant_id:
        return (
            "No filter criteria provided. Please specify at least one filter parameter "
            "(event_type, date_before, date_after, place, description_contains, or participant_id)."
        )

    # Resolve participant_id to a set of event handles before filtering.
    # MatchesPersonFilter requires a named filter stored in Gramps and cannot
    # accept inline JSON, so we look up the person's events directly instead.
    participant_handles: list | None = None
    if participant_id:
        logger = get_logger()
        try:
            db_handle = get_db_outside_request(
                tree=ctx.deps.tree,
                view_private=ctx.deps.include_private,
                readonly=True,
                user_id=ctx.deps.user_id,
            )
            try:
                person = db_handle.get_person_from_gramps_id(participant_id)
            finally:
                db_handle.close()
            if person is None:
                return f"No person found with Gramps ID '{participant_id}'."
            participant_handles = [ref.ref for ref in person.get_event_ref_list()]
            if not participant_handles:
                return f"No events found for person '{participant_id}'."
        except Exception as e:  # pylint: disable=broad-except
            logger.error("Error looking up participant %s: %s", participant_id, e)
            return f"Error looking up participant '{participant_id}': {e}"

    if not rules:
        # participant_id only — return all events for that person directly
        rules = [{"name": "AllEvents", "values": []}]

    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Event",
        rules=rules,
        max_results=max_results,
        empty_message="No events found matching the filter criteria.",
        handles=participant_handles,
    )


@log_tool_call
def get_person(ctx: RunContext[AgentDeps], gramps_id: str) -> str:
    """Fetch the full record for a single person by their Gramps ID.

    Use this after finding a Gramps ID in search or filter results to retrieve
    the complete details for that person (names, events, families, notes, etc.).
    This is the most efficient way to look up a known individual.

    Args:
        gramps_id: The Gramps ID of the person (e.g., "I0044")

    Returns:
        Full person record as formatted text, or an error message.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_person", "Fetching person record...")

    logger = get_logger()
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )
        try:
            person = db_handle.get_person_from_gramps_id(gramps_id)
            if person is None:
                return f"No person found with Gramps ID '{gramps_id}'."
            obj_dict = obj_strings_from_object(
                db_handle=db_handle,
                class_name="Person",
                obj=person,
                semantic=True,
            )
        finally:
            db_handle.close()
        if not obj_dict:
            return f"No content available for person '{gramps_id}'."
        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )
        if not content:
            return f"No content available for person '{gramps_id}'."
        return _truncate_content(content, min(ctx.deps.max_context_length // 2, 10000))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching person %s: %s", gramps_id, e)
        return f"Error fetching person '{gramps_id}': {e}"


@log_tool_call
def get_family(ctx: RunContext[AgentDeps], gramps_id: str) -> str:
    """Fetch the full record for a single family by their Gramps ID.

    Use this after finding a family Gramps ID in search or filter results to
    retrieve the complete record: spouses, children, marriage events, notes, etc.

    Args:
        gramps_id: The Gramps ID of the family (e.g., "F0001")

    Returns:
        Full family record as formatted text, or an error message.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_family", "Fetching family record...")

    logger = get_logger()
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )
        try:
            family = db_handle.get_family_from_gramps_id(gramps_id)
            if family is None:
                return f"No family found with Gramps ID '{gramps_id}'."
            obj_dict = obj_strings_from_object(
                db_handle=db_handle,
                class_name="Family",
                obj=family,
                semantic=True,
            )
        finally:
            db_handle.close()
        if not obj_dict:
            return f"No content available for family '{gramps_id}'."
        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )
        if not content:
            return f"No content available for family '{gramps_id}'."
        return _truncate_content(content, min(ctx.deps.max_context_length // 2, 10000))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching family %s: %s", gramps_id, e)
        return f"Error fetching family '{gramps_id}': {e}"


@log_tool_call
def get_event(ctx: RunContext[AgentDeps], gramps_id: str) -> str:
    """Fetch the full record for a single event by its Gramps ID.

    Use this after finding an event Gramps ID in search or filter results to
    retrieve the complete details: type, date, place, description, participants, etc.

    Args:
        gramps_id: The Gramps ID of the event (e.g., "E0123")

    Returns:
        Full event record as formatted text, or an error message.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_event", "Fetching event record...")

    logger = get_logger()
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )
        try:
            event = db_handle.get_event_from_gramps_id(gramps_id)
            if event is None:
                return f"No event found with Gramps ID '{gramps_id}'."
            obj_dict = obj_strings_from_object(
                db_handle=db_handle,
                class_name="Event",
                obj=event,
                semantic=True,
            )
        finally:
            db_handle.close()
        if not obj_dict:
            return f"No content available for event '{gramps_id}'."
        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )
        if not content:
            return f"No content available for event '{gramps_id}'."
        return _truncate_content(content, min(ctx.deps.max_context_length // 2, 10000))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching event %s: %s", gramps_id, e)
        return f"Error fetching event '{gramps_id}': {e}"


@log_tool_call
def get_place(ctx: RunContext[AgentDeps], gramps_id: str) -> str:
    """Fetch the full record for a single place by its Gramps ID.

    Use this after finding a place Gramps ID in search or filter results to
    retrieve the complete details: name, type, coordinates, place hierarchy, etc.

    Args:
        gramps_id: The Gramps ID of the place (e.g., "P0042")

    Returns:
        Full place record as formatted text, or an error message.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_place", "Fetching place record...")

    logger = get_logger()
    try:
        db_handle = get_db_outside_request(
            tree=ctx.deps.tree,
            view_private=ctx.deps.include_private,
            readonly=True,
            user_id=ctx.deps.user_id,
        )
        try:
            place = db_handle.get_place_from_gramps_id(gramps_id)
            if place is None:
                return f"No place found with Gramps ID '{gramps_id}'."
            obj_dict = obj_strings_from_object(
                db_handle=db_handle,
                class_name="Place",
                obj=place,
                semantic=True,
            )
        finally:
            db_handle.close()
        if not obj_dict:
            return f"No content available for place '{gramps_id}'."
        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )
        if not content:
            return f"No content available for place '{gramps_id}'."
        return _truncate_content(content, min(ctx.deps.max_context_length // 2, 10000))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching place %s: %s", gramps_id, e)
        return f"Error fetching place '{gramps_id}': {e}"


@log_tool_call
def filter_families(
    ctx: RunContext[AgentDeps],
    father_given_name: str = "",
    father_surname: str = "",
    mother_given_name: str = "",
    mother_surname: str = "",
    marriage_year_before: str = "",
    marriage_year_after: str = "",
    marriage_place: str = "",
    relationship_type: str = "",
    combine_filters: str = "and",
    max_results: int = 50,
) -> str:
    """Filter families in the genealogy database.

    A "family" in Gramps represents a couple (with or without children) and may
    include marriage events, children, and notes. Use this to find families by
    the names of spouses, marriage date/place, or relationship type.

    Args:
        father_given_name: Given name of the father/first spouse (partial match)
        father_surname: Surname of the father/first spouse (partial match)
        mother_given_name: Given name of the mother/second spouse (partial match)
        mother_surname: Surname of the mother/second spouse (partial match)
        marriage_year_before: Latest year of marriage to include (e.g., "1900")
        marriage_year_after: Earliest year of marriage to include (e.g., "1850")
        marriage_place: Place of marriage (partial match, e.g., "Boston")
        relationship_type: Family relationship type (e.g., "Married", "Unmarried",
            "Civil Union", "Unknown")
        combine_filters: How to combine multiple filters: "and" (default) or "or"
        max_results: Maximum results to return (default: 50, max: 100)

    Returns:
        Formatted list of families matching the filter criteria.

    Examples:
        - Find families where father is named Smith: father_surname="Smith"
        - Find families married in Boston: marriage_place="Boston"
        - Find marriages before 1900: marriage_year_before="1900"
        - Find families with a Smith father and Jones mother:
            father_surname="Smith", mother_surname="Jones"
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("filter_families", "Filtering families...")

    max_results = min(max(1, max_results), 100)

    rules: list[dict[str, Any]] = []

    if father_given_name or father_surname:
        rules.append(
            {
                "name": "FatherHasNameOf",
                "values": [
                    father_given_name,
                    father_surname,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            }
        )

    if mother_given_name or mother_surname:
        rules.append(
            {
                "name": "MotherHasNameOf",
                "values": [
                    mother_given_name,
                    mother_surname,
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                ],
            }
        )

    if marriage_year_before or marriage_year_after or marriage_place:
        date_expr = _build_date_expression(marriage_year_before, marriage_year_after)
        rules.append(
            {
                "name": "HasEvent",
                "values": ["Marriage", date_expr, marriage_place, "", ""],
            }
        )

    if relationship_type:
        rules.append({"name": "HasRelType", "values": [relationship_type]})

    if not rules:
        return (
            "No filter criteria provided. Please specify at least one filter parameter."
        )

    return _apply_gramps_filter(
        ctx=ctx,
        namespace="Family",
        rules=rules,
        max_results=max_results,
        empty_message="No families found matching the filter criteria.",
        logic=combine_filters.lower(),
    )


# ---------------------------------------------------------------------------
# Home person / kinship / anniversaries / statistics tools
# ---------------------------------------------------------------------------


def _tool_locale():
    """Return the server default GrampsLocale for rendering tool output."""
    return get_locale_for_language(None, default=True)


def _open_db(ctx: RunContext[AgentDeps]):
    """Open a thread-safe read-only DB handle for a tool. Caller must close it.

    Uses get_db_outside_request for the same thread-safety reason as the other
    tools (Pydantic AI runs tools in a thread pool, which would violate SQLite's
    thread affinity if the request-cached handle were reused).
    """
    return get_db_outside_request(
        tree=ctx.deps.tree,
        view_private=ctx.deps.include_private,
        readonly=True,
        user_id=ctx.deps.user_id,
    )


def _open_cached_db(ctx: RunContext[AgentDeps]):
    """Open a read-only handle wrapped in the people/families cache.

    Returns ``(proxy, raw_handle)``. The caller MUST close ``raw_handle`` (the
    proxy does not own the underlying connection). Wrapping
    ``get_db_outside_request`` in ``CachePeopleFamiliesProxy`` mirrors what
    ``RelativesResource`` does with the request handle, so the kinship BFS does
    not re-hit the DB for every person/family lookup.
    """
    raw = _open_db(ctx)
    proxy = CachePeopleFamiliesProxy(raw)
    proxy.cache_people()
    proxy.cache_families()
    return proxy, raw


def _fmt_life_dates(profile: dict[str, Any]) -> str:
    """Return a compact ' (b. …; d. …)' suffix from a person profile, or ''."""

    def _date(key: str) -> str:
        value = profile.get(key)
        if isinstance(value, dict):
            return (value.get("date") or "").strip()
        return ""

    parts: list[str] = []
    born = _date("birth")
    died = _date("death")
    if born:
        parts.append(f"b. {born}")
    if died:
        parts.append(f"d. {died}")
    return f" ({'; '.join(parts)})" if parts else ""


def _person_line(
    db_handle,
    person,
    locale,
    relationship: str | None = None,
    kind: str | None = None,
) -> str:
    """Render a single person as a compact markdown list entry with a link."""
    profile = get_person_profile_for_object(db_handle, person, args=[], locale=locale)
    name = profile.get("name_display") or profile.get("gramps_id") or "?"
    gid = profile.get("gramps_id") or ""
    link = f"[{name}](/person/{gid})" if gid else name
    line = link + _fmt_life_dates(profile)
    # Guard against degenerate labels the relationship calculator emits for
    # distant relations it has no word for: it inserts a literal "None" token
    # (e.g. "None матери супруга/супруги"). Drop the whole label in that case —
    # the [in-law] marker and the group header still convey the relation. Clean
    # labels are kept as-is.
    if relationship and "none" not in relationship.strip().lower().split():
        line += f" — {relationship}"
    if kind == "inlaw":
        line += " [in-law]"
    return line


def _marriage_note(db_handle, family, locale) -> str:
    """Return a compact " (m. YYYY; divorced)" annotation for a marriage.

    Lets the agent (and reader) tell current vs former spouses apart when a
    person has several marriages. Empty string if nothing is recorded.
    """
    marriage_date = ""
    divorced = False
    try:
        for event_ref in family.get_event_ref_list():
            event = db_handle.get_event_from_handle(event_ref.ref)
            if event is None:
                continue
            etype = event.get_type()
            if etype.is_marriage():
                date_obj = event.get_date_object()
                if date_obj is not None and not date_obj.is_empty():
                    marriage_date = locale.date_displayer.display(date_obj)
            elif etype.is_divorce():
                divorced = True
    except Exception:  # pylint: disable=broad-except
        return ""
    parts = []
    if marriage_date:
        parts.append(f"m. {marriage_date}")
    if divorced:
        parts.append("divorced")
    return f" ({'; '.join(parts)})" if parts else ""


@log_tool_call
def get_home_person(ctx: RunContext[AgentDeps]) -> str:
    """Identify the user's "home person" — the individual representing the user.

    ALWAYS call this FIRST whenever the user refers to themselves ("I", "me",
    "my", "мой", "меня", "мои") so you know whose relatives, ancestors, or
    descendants they mean. Do not ask the user who they are until this tool has
    returned no home person.

    Returns:
        The home person's full record (name, dates, families), or a note that no
        home person is configured.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_home_person", "Identifying the home person...")

    gid = (ctx.deps.home_person_gramps_id or "").strip()
    if not gid:
        return (
            "No home person is set for this user. Ask the user who they are in "
            "the family tree (their name or Gramps ID) before answering questions "
            "about their own relatives."
        )

    logger = get_logger()
    try:
        db_handle = _open_db(ctx)
        try:
            person = db_handle.get_person_from_gramps_id(gid)
            if person is None:
                return (
                    f"The configured home person (Gramps ID '{gid}') was not found "
                    "in the tree. Ask the user to confirm who they are."
                )
            obj_dict = obj_strings_from_object(
                db_handle=db_handle,
                class_name="Person",
                obj=person,
                semantic=True,
            )
        finally:
            db_handle.close()
        if not obj_dict:
            return f"The home person '{gid}' has no readable record."
        content = (
            obj_dict["string_all"]
            if ctx.deps.include_private
            else obj_dict["string_public"]
        )
        if not content:
            return f"The home person '{gid}' has no readable record."
        header = (
            f"The user's home person (i.e. the user themselves, 'I'/'me'/'my') is "
            f"Gramps ID {gid}. Full record:\n\n"
        )
        return header + _truncate_content(
            content, min(ctx.deps.max_context_length // 2, 10000)
        )
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error fetching home person %s: %s", gid, e)
        return f"Error fetching the home person '{gid}': {e}"


@log_tool_call
def get_relatives(ctx: RunContext[AgentDeps], gramps_id: str = "") -> str:
    """List all relatives of a person, grouped by kinship category.

    This is the best tool for "who are my cousins / uncles / nephews", "list
    X's relatives", or any request for a person's whole kinship network. It
    returns both blood relatives and in-law (marriage) relatives, each already
    labelled with its exact relationship (e.g. "двоюродный брат", "second
    cousin", "aunt"). Prefer this over filter_people for relationship questions.

    Args:
        gramps_id: Gramps ID of the anchor person (e.g. "I0044"). Leave empty to
            use the user's home person (call get_home_person first if unsure who
            that is).

    Returns:
        Relatives grouped by category, each linked, or a note if none/unknown.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_relatives", "Finding relatives...")

    gid = (gramps_id or ctx.deps.home_person_gramps_id or "").strip()
    if not gid:
        return (
            "No person was specified and no home person is set. Ask the user who "
            "they are, or provide a Gramps ID."
        )

    logger = get_logger()
    raw = None
    try:
        db_handle, raw = _open_cached_db(ctx)
        anchor = db_handle.get_person_from_gramps_id(gid)
        if anchor is None:
            return f"No person found with Gramps ID '{gid}'."

        locale = _tool_locale()
        result = relatives_of(db_handle, anchor, locale)

        anchor_line = _person_line(db_handle, anchor, locale)
        out: list[str] = [f"Relatives of {anchor_line}:"]
        max_length = ctx.deps.max_context_length
        current_length = len(out[0])
        truncated = False

        # relatives_of groups by blood-equivalent category and does NOT include
        # the anchor's own spouse (a spouse has no blood path). Surface direct
        # spouse(s) explicitly and first, so the agent can resolve "my wife /
        # husband" reliably and pass their Gramps ID to other tools. Each spouse
        # is annotated with the marriage year and a "divorced" marker so multiple
        # marriages (current vs former) can be told apart.
        spouse_lines: list[str] = []
        for fam_handle in anchor.get_family_handle_list():
            family = db_handle.get_family_from_handle(fam_handle)
            if family is None:
                continue
            spouse_handle = (
                family.get_mother_handle()
                if family.get_father_handle() == anchor.handle
                else family.get_father_handle()
            )
            if not spouse_handle:
                continue
            spouse = db_handle.get_person_from_handle(spouse_handle)
            if spouse is None or (not ctx.deps.include_private and spouse.private):
                continue
            spouse_lines.append(
                "- "
                + _person_line(db_handle, spouse, locale)
                + _marriage_note(db_handle, family, locale)
            )
        if spouse_lines:
            block = f"\n\n### spouse ({len(spouse_lines)})\n" + "\n".join(spouse_lines)
            out.append(block)
            current_length += len(block)

        for group in result["groups"]:
            group_lines: list[str] = []
            for entry in group["people"]:
                person = db_handle.get_person_from_handle(entry["handle"])
                if person is None:
                    continue
                if not ctx.deps.include_private and person.private:
                    continue
                group_lines.append(
                    "- "
                    + _person_line(
                        db_handle,
                        person,
                        locale,
                        relationship=entry.get("relationship"),
                        kind=entry.get("kind"),
                    )
                )
            if not group_lines:
                continue
            block = f"\n\n### {group['category_key']} ({len(group_lines)})\n" + "\n".join(
                group_lines
            )
            if current_length + len(block) > max_length:
                truncated = True
                break
            out.append(block)
            current_length += len(block)

        if len(out) == 1:
            return f"No relatives found for {anchor_line}."
        text = "".join(out)
        if truncated:
            text += "\n\n---\n(Some more distant relatives were omitted to fit the response.)"
        return text
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error finding relatives of %s: %s", gid, e)
        return f"Error finding relatives of '{gid}': {e}"
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:  # pylint: disable=broad-except
                pass


@log_tool_call
def get_relationship(
    ctx: RunContext[AgentDeps], gramps_id_a: str, gramps_id_b: str = ""
) -> str:
    """Explain how two people are related, with their common ancestor(s).

    Use for "how are X and Y related?", "what is the relationship between X and
    Y?", or "who is the common ancestor of X and Y?". Handles both blood and
    in-law (marriage) relationships and shows the ancestral path.

    Args:
        gramps_id_a: Gramps ID of the first person (e.g. "I0044").
        gramps_id_b: Gramps ID of the second person. Leave empty to compare
            against the user's home person.

    Returns:
        The relationship label plus common ancestor(s) and the path between them,
        or a note if they are unrelated / a person is unknown.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_relationship", "Computing relationship...")

    a_id = (gramps_id_a or "").strip()
    b_id = (gramps_id_b or ctx.deps.home_person_gramps_id or "").strip()
    if not a_id:
        return "Specify at least the first person via gramps_id_a."
    if not b_id:
        return (
            "No second person specified and no home person is set. Provide "
            "gramps_id_b, or ask the user who they are."
        )

    logger = get_logger()
    raw = None
    try:
        db_handle, raw = _open_cached_db(ctx)
        person_a = db_handle.get_person_from_gramps_id(a_id)
        if person_a is None:
            return f"No person found with Gramps ID '{a_id}'."
        person_b = db_handle.get_person_from_gramps_id(b_id)
        if person_b is None:
            return f"No person found with Gramps ID '{b_id}'."

        locale = _tool_locale()
        line_a = _person_line(db_handle, person_a, locale)
        line_b = _person_line(db_handle, person_b, locale)

        if person_a.handle == person_b.handle:
            return f"{line_a} and {line_b} are the same person."

        # Localized relationship label (blood + in-law aware). Direction:
        # get_one_relationship(person1=B, person2=A) yields A's relationship to B.
        rel_str = ""
        try:
            rel_str, _da, _db = get_one_relationship(
                db_handle=db_handle,
                person1=person_b,
                person2=person_a,
                depth=15,
                locale=locale,
            )
        except Exception as e:  # pylint: disable=broad-except
            logger.warning("get_one_relationship failed for %s/%s: %s", a_id, b_id, e)

        parts: list[str] = []
        if rel_str and rel_str.strip():
            parts.append(f"{line_a} is **{rel_str}** of {line_b}.")
        else:
            parts.append(
                f"No direct relationship label was found between {line_a} and "
                f"{line_b}."
            )

        # Common ancestor(s) + path (blood relationships only).
        ca = common_ancestors(db_handle, person_a, person_b, locale)

        def _render_people(handle_people) -> str:
            names: list[str] = []
            for p in handle_people:
                if p is None:
                    continue
                if not ctx.deps.include_private and p.private:
                    continue
                names.append(_person_line(db_handle, p, locale))
            return ", ".join(names)

        rendered_any = False
        for entry in ca.get("ancestors", []):
            anc = [
                db_handle.get_person_from_handle(h)
                for h in entry.get("ancestor_handles", [])
            ]
            anc_text = _render_people(anc)
            if not anc_text:
                continue
            rendered_any = True
            parts.append(f"\nCommon ancestor(s): {anc_text}")
            path_a = [
                db_handle.get_person_from_handle(h) for h in entry.get("path_a", [])
            ]
            path_b = [
                db_handle.get_person_from_handle(h) for h in entry.get("path_b", [])
            ]
            pa = _render_people(path_a)
            pb = _render_people(path_b)
            if pa:
                parts.append(f"Path from {line_a}: {pa}")
            if pb:
                parts.append(f"Path from {line_b}: {pb}")

        if not rendered_any and not (rel_str and rel_str.strip()):
            parts.append(
                "They share no common ancestor in the tree and no marriage link "
                "was found — they appear to be unrelated within the recorded data."
            )
        return "\n".join(parts)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error computing relationship %s/%s: %s", a_id, b_id, e)
        return f"Error computing the relationship between '{a_id}' and '{b_id}': {e}"
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:  # pylint: disable=broad-except
                pass


@log_tool_call
def get_anniversaries(
    ctx: RunContext[AgentDeps],
    within_days: int = 31,
    scope: str = "all",
    event_types: str = "",
) -> str:
    """List upcoming birthdays and anniversaries (yearly-recurring events).

    Use for "whose birthday is coming up?", "any anniversaries this month?", or
    "what family dates are in July?". Matches events by their month/day recurrence
    within the next `within_days` days.

    Args:
        within_days: How many days ahead to look (default 31, max 366).
        scope: "all" for the whole tree, or "home" to restrict to the family
            around the user's home person.
        event_types: Optional comma-separated event types to include (e.g.
            "Birth", "Marriage", "Death"). Empty = Birth and Marriage.

    Returns:
        A dated list of upcoming anniversaries with links, or a note if none.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_anniversaries", "Finding upcoming dates...")

    within_days = min(max(1, within_days), 366)
    types = [t.strip() for t in event_types.split(",") if t.strip()] or [
        "Birth",
        "Marriage",
    ]

    anchor_gid = None
    scope_note = ""
    if scope == "home":
        anchor_gid = (ctx.deps.home_person_gramps_id or "").strip() or None
        if anchor_gid is None:
            scope_note = (
                "\n\n(No home person is set, so this covers the whole tree.)"
            )

    logger = get_logger()
    db_handle = None
    try:
        db_handle = _open_db(ctx)
        results = upcoming_anniversaries(
            db_handle,
            within_days=within_days,
            event_types=types,
            anchor_gramps_id=anchor_gid,
        )
        if not results:
            return (
                f"No birthdays or anniversaries ({', '.join(types)}) in the next "
                f"{within_days} days." + scope_note
            )

        locale = _tool_locale()
        lines: list[str] = [
            f"Upcoming anniversaries in the next {within_days} days:"
        ]
        max_length = ctx.deps.max_context_length
        current_length = len(lines[0])
        for item in results:
            event = item["event"]
            if not ctx.deps.include_private and event.private:
                continue
            summary = get_event_summary_from_object(db_handle, event, locale)
            gid = event.gramps_id or ""
            date_label = item["date"].strftime("%d %B")
            link = f" ([details](/event/{gid}))" if gid else ""
            line = f"- **{date_label}** — {summary}{link}"
            if current_length + len(line) > max_length:
                lines.append("- …(more omitted)")
                break
            lines.append(line)
            current_length += len(line)
        return "\n".join(lines) + scope_note
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error listing anniversaries: %s", e)
        return f"Error listing anniversaries: {e}"
    finally:
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass


@log_tool_call
def get_tree_statistics(ctx: RunContext[AgentDeps]) -> str:
    """Return high-level counts for the family tree (people, families, etc.).

    Use for "how big is the tree?", "how many people/families/events are there?",
    or "what are the most common surnames?". For counts matching specific criteria
    (e.g. "how many people born in Kazan"), use filter_people instead and read the
    "Showing N of M" footer.

    Returns:
        Object counts and the most common surnames.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_tree_statistics", "Gathering statistics...")

    logger = get_logger()
    db_handle = None
    try:
        db_handle = _open_db(ctx)
        counts = {
            "people": db_handle.get_number_of_people(),
            "families": db_handle.get_number_of_families(),
            "events": db_handle.get_number_of_events(),
            "places": db_handle.get_number_of_places(),
            "sources": db_handle.get_number_of_sources(),
            "citations": db_handle.get_number_of_citations(),
            "repositories": db_handle.get_number_of_repositories(),
            "media": db_handle.get_number_of_media(),
            "notes": db_handle.get_number_of_notes(),
        }
        surnames = db_handle.get_surname_list() or []

        lines = ["Family tree statistics:"]
        lines.append(
            "- People: {people}\n- Families: {families}\n- Events: {events}\n"
            "- Places: {places}\n- Sources: {sources}\n- Citations: {citations}\n"
            "- Repositories: {repositories}\n- Media: {media}\n- Notes: {notes}".format(
                **counts
            )
        )
        lines.append(f"- Distinct surnames: {count_unique_surnames(surnames)}")

        # Top surnames by frequency (bounded to avoid heavy scans on huge trees).
        # Use the family surname (patronymic excluded) collapsed to its
        # masculine base so gender forms (Соболевский/Соболевская) count as one.
        if counts["people"] and counts["people"] <= 20000:
            freq: dict[str, int] = {}
            for person in db_handle.iter_people():
                try:
                    surname = normalize_surname_gender(
                        get_family_surname(person.primary_name)
                    )
                except Exception:  # pylint: disable=broad-except
                    surname = ""
                if surname:
                    freq[surname] = freq.get(surname, 0) + 1
            top = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
            if top:
                lines.append(
                    "\nMost common surnames:\n"
                    + "\n".join(f"- {name}: {n}" for name, n in top)
                )
        return "\n".join(lines)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error gathering tree statistics: %s", e)
        return f"Error gathering tree statistics: {e}"
    finally:
        if db_handle is not None:
            try:
                db_handle.close()
            except Exception:  # pylint: disable=broad-except
                pass


@log_tool_call
def get_timeline(ctx: RunContext[AgentDeps], gramps_id: str = "") -> str:
    """Return a chronological life timeline for a person.

    Use for "tell me about X", "X's life story", "what happened to X and when".
    Lists the person's own events (birth, education, residence, death, …) and
    their marriages, sorted by date. Clearer than get_person for narrative
    questions because it is ordered in time.

    Args:
        gramps_id: Gramps ID of the person. Leave empty for the home person.

    Returns:
        A dated, chronological list of the person's life events with links.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_timeline", "Building the timeline...")

    gid = (gramps_id or ctx.deps.home_person_gramps_id or "").strip()
    if not gid:
        return (
            "No person was specified and no home person is set. Ask the user who "
            "they are, or provide a Gramps ID."
        )

    logger = get_logger()
    raw = None
    try:
        db_handle, raw = _open_cached_db(ctx)
        person = db_handle.get_person_from_gramps_id(gid)
        if person is None:
            return f"No person found with Gramps ID '{gid}'."

        locale = _tool_locale()
        entries: list[tuple[int, str]] = []

        def _sortval(event) -> int:
            date_obj = event.get_date_object()
            if date_obj is None or date_obj.is_empty():
                return 10**18  # undated events sort last
            return date_obj.get_sort_value()

        def _event_line(event, prefix: str = "") -> str:
            prof = get_event_profile_for_object(db_handle, event, args=[], locale=locale)
            date_str = prof.get("date") or "?"
            etype = prof.get("type") or ""
            place = prof.get("place") or ""
            egid = event.gramps_id or ""
            text = f"- **{date_str}** — {prefix}{etype}"
            if place:
                text += f", {place}"
            if egid:
                text += f" ([event](/event/{egid}))"
            return text

        # The person's own events.
        for event_ref in person.get_event_ref_list():
            event = db_handle.get_event_from_handle(event_ref.ref)
            if event is None:
                continue
            if not ctx.deps.include_private and event.private:
                continue
            entries.append((_sortval(event), _event_line(event)))

        # Marriage / family events, labelled with the spouse.
        for fam_handle in person.get_family_handle_list():
            family = db_handle.get_family_from_handle(fam_handle)
            if family is None:
                continue
            spouse_handle = family.get_mother_handle()
            if spouse_handle == person.handle:
                spouse_handle = family.get_father_handle()
            spouse_note = ""
            if spouse_handle:
                spouse = db_handle.get_person_from_handle(spouse_handle)
                if spouse is not None and (
                    ctx.deps.include_private or not spouse.private
                ):
                    sp = get_person_profile_for_object(
                        db_handle, spouse, args=[], locale=locale
                    )
                    sp_name = sp.get("name_display") or ""
                    sp_gid = sp.get("gramps_id") or ""
                    spouse_note = (
                        f" [{sp_name}](/person/{sp_gid}): "
                        if sp_gid
                        else f"{sp_name}: "
                    )
            for event_ref in family.get_event_ref_list():
                event = db_handle.get_event_from_handle(event_ref.ref)
                if event is None:
                    continue
                if not ctx.deps.include_private and event.private:
                    continue
                entries.append((_sortval(event), _event_line(event, prefix=spouse_note)))

        if not entries:
            person_line = _person_line(db_handle, person, locale)
            return f"No dated events found for {person_line}."

        entries.sort(key=lambda item: item[0])
        person_line = _person_line(db_handle, person, locale)
        body = "\n".join(text for _sv, text in entries)
        result = f"Life timeline of {person_line}:\n{body}"
        return _truncate_content(result, ctx.deps.max_context_length)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error building timeline for %s: %s", gid, e)
        return f"Error building the timeline for '{gid}': {e}"
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:  # pylint: disable=broad-except
                pass


def _generation_label(gen: int, ascending: bool) -> str:
    """Human label for a pedigree generation (ascending=ancestors)."""
    if ascending:
        names = {1: "parents", 2: "grandparents", 3: "great-grandparents"}
        return names.get(gen, f"generation {gen} up (great×{gen - 2}-grandparents)")
    names = {1: "children", 2: "grandchildren", 3: "great-grandchildren"}
    return names.get(gen, f"generation {gen} down (great×{gen - 2}-grandchildren)")


def _render_pedigree(
    ctx: RunContext[AgentDeps],
    gramps_id: str,
    ascending: bool,
    generations: int,
) -> str:
    """Shared implementation for get_ancestors / get_descendants."""
    generations = min(max(1, generations), 20)
    gid = (gramps_id or ctx.deps.home_person_gramps_id or "").strip()
    if not gid:
        return (
            "No person was specified and no home person is set. Ask the user who "
            "they are, or provide a Gramps ID."
        )

    logger = get_logger()
    raw = None
    try:
        db_handle, raw = _open_cached_db(ctx)
        anchor = db_handle.get_person_from_gramps_id(gid)
        if anchor is None:
            return f"No person found with Gramps ID '{gid}'."

        locale = _tool_locale()
        if ascending:
            gen_map = _ancestors_bfs(db_handle, anchor)  # {handle: gen}
            pairs = list(gen_map.items())
        else:
            pairs = _descendants_bfs(db_handle, anchor)  # [(handle, gen)]

        by_gen: dict[int, list[str]] = {}
        max_gen = 0
        for handle, gen in pairs:
            if gen > generations:
                continue
            person = db_handle.get_person_from_handle(handle)
            if person is None:
                continue
            if not ctx.deps.include_private and person.private:
                continue
            by_gen.setdefault(gen, []).append(_person_line(db_handle, person, locale))
            max_gen = max(max_gen, gen)

        anchor_line = _person_line(db_handle, anchor, locale)
        kind = "Ancestors" if ascending else "Descendants"
        if not by_gen:
            return f"No {kind.lower()} recorded for {anchor_line}."

        out = [f"{kind} of {anchor_line} (up to {generations} generations):"]
        max_length = ctx.deps.max_context_length
        current = len(out[0])
        for gen in sorted(by_gen):
            people = sorted(by_gen[gen])
            block = (
                f"\n\n### {_generation_label(gen, ascending)} ({len(people)})\n"
                + "\n".join(f"- {p}" for p in people)
            )
            if current + len(block) > max_length:
                out.append("\n\n---\n(Further generations omitted to fit.)")
                break
            out.append(block)
            current += len(block)

        if ascending and max_gen:
            deepest = by_gen.get(max_gen, [])
            out.append(
                f"\n\nFurthest known ancestor(s), {max_gen} generations back: "
                + "; ".join(deepest)
            )
        return "".join(out)
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Error building pedigree for %s: %s", gid, e)
        return f"Error building the pedigree for '{gid}': {e}"
    finally:
        if raw is not None:
            try:
                raw.close()
            except Exception:  # pylint: disable=broad-except
                pass


@log_tool_call
def get_ancestors(
    ctx: RunContext[AgentDeps], gramps_id: str = "", generations: int = 6
) -> str:
    """List a person's direct ancestors, grouped by generation.

    Use for "who are my ancestors", "show my pedigree / lineage N generations
    back", "who is my furthest known ancestor". Returns only the direct ancestral
    line (parents, grandparents, …), not collateral relatives (for those use
    get_relatives).

    Args:
        gramps_id: Gramps ID of the person. Leave empty for the home person.
        generations: How many generations up to include (default 6, max 20).

    Returns:
        Ancestors grouped by generation, plus the furthest known ancestor(s).
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_ancestors", "Tracing ancestors...")
    return _render_pedigree(ctx, gramps_id, ascending=True, generations=generations)


@log_tool_call
def get_descendants(
    ctx: RunContext[AgentDeps], gramps_id: str = "", generations: int = 6
) -> str:
    """List a person's direct descendants, grouped by generation.

    Use for "who are X's descendants", "show X's descendant lineage". Returns the
    direct descendant line (children, grandchildren, …).

    Args:
        gramps_id: Gramps ID of the person. Leave empty for the home person.
        generations: How many generations down to include (default 6, max 20).

    Returns:
        Descendants grouped by generation.
    """
    if ctx.deps.progress_callback:
        ctx.deps.progress_callback("get_descendants", "Tracing descendants...")
    return _render_pedigree(ctx, gramps_id, ascending=False, generations=generations)
