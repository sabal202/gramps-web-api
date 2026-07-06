#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      Gramps Web contributors
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#

"""Anniversaries ICS resource."""

import json
from datetime import date, datetime, timedelta, timezone
from typing import Any, Optional

from flask import Response
from gramps.gen.lib import Event
from gramps.gen.lib.date import gregorian
from marshmallow import Schema
from webargs import fields, validate

from ...auth import (
    get_permissions,
    get_user_from_access_token,
    is_tree_disabled,
)
from ...auth.const import ACCESS_TOKEN_SCOPE_ANNIVERSARIES_ICS, PERM_VIEW_PRIVATE
from ..blueprint import api_blueprint
from ..util import (
    abort_with_message,
    close_db,
    get_config,
    get_db_outside_request,
    get_tree_id,
)
from . import Resource
from .filters import apply_filter
from .util import get_backlinks, get_event_summary_from_object


def _escape_ics_text(value: str) -> str:
    """Escape text fields according to RFC 5545."""
    return (
        value.replace("\\", "\\\\")
        .replace(";", "\\;")
        .replace(",", "\\,")
        .replace("\r\n", "\\n")
        .replace("\n", "\\n")
    )


# LOCAL DEVIATION from vendored PR #792: fold content lines to <=75 octets
# (RFC 5545 sec 3.1) so multibyte (e.g. Cyrillic) SUMMARY/UID lines are not
# silently dropped by stricter parsers. Splits on character boundaries so a
# UTF-8 sequence is never cut; continuation lines start with a single space.
def _fold_ics_line(line: str) -> str:
    """Fold a single content line to 75-octet segments per RFC 5545."""
    if len(line.encode("utf-8")) <= 75:
        return line
    chunks: list[str] = []
    cur = ""
    cur_len = 0
    for ch in line:
        clen = len(ch.encode("utf-8"))
        limit = 75 if not chunks else 74  # continuation lines carry a leading space
        if cur_len + clen > limit:
            chunks.append(cur)
            cur, cur_len = ch, clen
        else:
            cur += ch
            cur_len += clen
    chunks.append(cur)
    return "\r\n ".join(chunks)


def _event_matches_type(event: Event, allowed_types: set[str]) -> bool:
    """Check if an event type matches one of the requested filters."""
    if not allowed_types:
        return True
    event_values = {
        str(event.get_type()).casefold().strip(),
        event.get_type().xml_str().casefold().strip(),
    }
    return not event_values.isdisjoint(allowed_types)


def _get_anniversary_date_components(event: Event) -> Optional[tuple[int, int, int]]:
    """Get Gregorian (year, month, day) tuple for an event date."""
    if event.date is None or not event.date.is_valid():
        return None
    gdate = gregorian(event.date)
    month = gdate.get_month()
    day = gdate.get_day()
    if month < 1 or day < 1:
        return None
    year = gdate.get_year()
    if year < 1:
        year = 1970
    if year > 9999:
        year = 9999
    return year, month, day


def _is_event_in_anchor_scope(
    db_handle,
    event: Event,
    allowed_people: set[str],
    allowed_families: set[str],
) -> bool:
    """Check if an event is linked to people/families in anchor scope."""
    backlinks = get_backlinks(db_handle, event.handle)
    people = set(backlinks.get("person", []))
    if not people.isdisjoint(allowed_people):
        return True
    families = set(backlinks.get("family", []))
    return not families.isdisjoint(allowed_families)


def _resolve_anchor_people_handles(
    db_handle, anchor_gramps_id: str, generation_depth: int
) -> set[str]:
    """Resolve people handles to include for an anchor+generation filter."""
    anchor = db_handle.get_person_from_gramps_id(anchor_gramps_id)
    if anchor is None:
        abort_with_message(404, "Anchor person not found")
    rules = {
        "function": "or",
        "rules": [
            {
                "name": "IsLessThanNthGenerationAncestorOf",
                "values": [anchor_gramps_id, generation_depth],
            },
            {
                "name": "IsLessThanNthGenerationDescendantOf",
                "values": [anchor_gramps_id, generation_depth],
            },
        ],
    }
    handles = db_handle.get_person_handles(sort_handles=True)
    return set(
        apply_filter(
            db_handle,
            {"rules": json.dumps(rules)},
            "Person",
            handles,
        )
    )


def _resolve_family_handles_for_people(db_handle, people_handles: set[str]) -> set[str]:
    """Resolve families attached to in-scope people handles."""
    family_handles = set()
    for handle in people_handles:
        person = db_handle.get_person_from_handle(handle)
        if person is None:
            continue
        family_handles.update(person.family_list)
        family_handles.update(person.parent_family_list)
    return family_handles


def _build_ics(events: list[Event], db_handle, tree_id: str) -> str:
    """Build ICS calendar content for a list of events."""
    dtstamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    # LOCAL DEVIATION from vendored PR #792: link each anniversary back to its
    # Gramps Web event page (BASE_URL/event/<gramps_id>).
    base_url = (get_config("BASE_URL") or "").rstrip("/")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//Gramps Web//Anniversaries//EN",
        "CALSCALE:GREGORIAN",
        "METHOD:PUBLISH",
        "X-WR-CALNAME:Gramps Anniversaries",
    ]
    for event in events:
        date_components = _get_anniversary_date_components(event)
        if date_components is None:
            continue
        year, month, day = date_components
        dtstart = f"{year:04d}{month:02d}{day:02d}"
        gramps_id = event.gramps_id or ""
        event_url = f"{base_url}/event/{gramps_id}" if base_url and gramps_id else ""
        summary = _escape_ics_text(get_event_summary_from_object(db_handle, event))
        # LOCAL FIX: vendored code wrote a literal "\n" (backslash-n) which
        # _escape_ics_text then double-escaped, so clients showed a literal
        # "\n". Use a real newline so it renders as a line break; append the
        # event link as a third line.
        description_text = (
            f"Gramps ID: {gramps_id}\n"
            f"Type: {event.get_type().xml_str()}"
        )
        if event_url:
            description_text += f"\n{event_url}"
        description = _escape_ics_text(description_text)
        uid = _escape_ics_text(f"{event.handle}@{tree_id}.anniversaries.gramps-web")
        vevent = [
            "BEGIN:VEVENT",
            f"UID:{uid}",
            f"DTSTAMP:{dtstamp}",
            f"DTSTART;VALUE=DATE:{dtstart}",
            "RRULE:FREQ=YEARLY",
            f"SUMMARY:{summary}",
        ]
        if event_url:
            # URL is a URI value (not TEXT) - do not backslash-escape it.
            vevent.append(f"URL:{event_url}")
        vevent.append(f"DESCRIPTION:{description}")
        vevent.append("END:VEVENT")
        lines.extend(vevent)
    lines.append("END:VCALENDAR")
    return "\r\n".join(_fold_ics_line(line) for line in lines) + "\r\n"


def _next_anniversary_date(today: date, month: int, day: int) -> Optional[date]:
    """Return the next date on or after *today* matching (month, day).

    Handles Feb 29 in non-leap years by observing the anniversary on Mar 1.
    """

    def _make(year: int) -> Optional[date]:
        try:
            return date(year, month, day)
        except ValueError:
            if month == 2 and day == 29:
                return date(year, 3, 1)
            return None

    occurrence = _make(today.year)
    if occurrence is None:
        return None
    if occurrence < today:
        occurrence = _make(today.year + 1)
    return occurrence


def upcoming_anniversaries(
    db_handle,
    within_days: int = 31,
    today: Optional[date] = None,
    event_types: Optional[list[str]] = None,
    anchor_gramps_id: Optional[str] = None,
    generation_depth: int = 4,
) -> list[dict[str, Any]]:
    """Return events whose yearly anniversary falls within the next *within_days*.

    Pure helper (no Flask): iterates all events, keeps those with a valid
    recurring (month, day) whose next occurrence is within ``[today,
    today + within_days]``. Optionally restricts to the family scope around an
    anchor person (ancestors/descendants within ``generation_depth``). A missing
    anchor is ignored (whole tree) rather than raising.

    Returns a list sorted by the upcoming occurrence date::

        [{"date": date, "event": Event}, ...]
    """
    if today is None:
        today = datetime.now().date()
    window_end = today + timedelta(days=within_days)
    allowed_types = {
        event_type.casefold().strip()
        for event_type in (event_types or [])
        if event_type and event_type.strip()
    }

    allowed_people: Optional[set[str]] = None
    allowed_families: Optional[set[str]] = None
    if anchor_gramps_id:
        anchor = db_handle.get_person_from_gramps_id(anchor_gramps_id)
        if anchor is not None:
            rules = {
                "function": "or",
                "rules": [
                    {
                        "name": "IsLessThanNthGenerationAncestorOf",
                        "values": [anchor_gramps_id, generation_depth],
                    },
                    {
                        "name": "IsLessThanNthGenerationDescendantOf",
                        "values": [anchor_gramps_id, generation_depth],
                    },
                ],
            }
            handles = db_handle.get_person_handles(sort_handles=True)
            allowed_people = set(
                apply_filter(db_handle, {"rules": json.dumps(rules)}, "Person", handles)
            )
            allowed_people.add(anchor.handle)
            allowed_families = _resolve_family_handles_for_people(
                db_handle, allowed_people
            )

    results: list[dict[str, Any]] = []
    for handle in db_handle.get_event_handles():
        event = db_handle.get_event_from_handle(handle)
        if event is None:
            continue
        if not _event_matches_type(event, allowed_types):
            continue
        components = _get_anniversary_date_components(event)
        if components is None:
            continue
        _year, month, day = components
        occurrence = _next_anniversary_date(today, month, day)
        if occurrence is None or occurrence > window_end:
            continue
        if allowed_people is not None and not _is_event_in_anchor_scope(
            db_handle, event, allowed_people, allowed_families or set()
        ):
            continue
        results.append({"date": occurrence, "event": event})

    results.sort(key=lambda item: (item["date"], item["event"].handle))
    return results


def _event_sort_key(event: Event) -> tuple[int, int, int, str]:
    """Return stable sort key for anniversary events."""
    date_components = _get_anniversary_date_components(event)
    if date_components is None:
        return (12, 31, 9999, event.handle)
    year, month, day = date_components
    return (month, day, year, event.handle)


class AnniversariesIcsQueryArgs(Schema):
    """Query arguments for GET /anniversaries.ics."""

    token = fields.Str(
        required=True,
        validate=validate.Length(min=1),
        metadata={"description": "Persistent access token value."},
    )
    event_types = fields.DelimitedList(
        fields.Str(validate=validate.Length(min=1)),
        metadata={"description": "Comma-delimited event type names to include."},
    )
    anchor_gramps_id = fields.Str(
        metadata={"description": "Anchor person Gramps ID for family-scope filtering."},
    )
    generation_depth = fields.Integer(
        load_default=4,
        validate=validate.Range(min=1, max=9),
        metadata={"description": "Generation depth around the anchor person."},
    )


class AnniversariesIcsResource(Resource):
    """Public anniversaries ICS feed resource."""

    @api_blueprint.arguments(AnniversariesIcsQueryArgs, location="query")
    def get(self, args: dict) -> Response:
        """Return anniversaries in ICS format."""
        user = get_user_from_access_token(
            args["token"], ACCESS_TOKEN_SCOPE_ANNIVERSARIES_ICS
        )
        if user is None:
            abort_with_message(401, "Invalid access token")
        if user.role is None or user.role < 0:
            abort_with_message(403, "User account is disabled")

        tree_id = get_tree_id(str(user.id))
        if is_tree_disabled(tree=tree_id):
            abort_with_message(503, "This tree is temporarily disabled")

        permissions = get_permissions(username=user.name, tree=tree_id)
        view_private = PERM_VIEW_PRIVATE in permissions
        db_handle = get_db_outside_request(
            tree=tree_id,
            view_private=view_private,
            readonly=True,
            user_id=str(user.id),
        )
        try:
            iter_event_handles = db_handle.method("iter_event_handles")
            get_event_from_handle = db_handle.method("get_event_from_handle")
            assert iter_event_handles is not None
            assert get_event_from_handle is not None
            events = []
            for handle in iter_event_handles():
                event = get_event_from_handle(handle)
                if event is not None:
                    events.append(event)

            allowed_types = {
                event_type.casefold().strip()
                for event_type in args.get("event_types", [])
                if event_type and event_type.strip()
            }
            events = [
                event for event in events if _event_matches_type(event, allowed_types)
            ]
            events = [
                event
                for event in events
                if _get_anniversary_date_components(event) is not None
            ]

            if args.get("anchor_gramps_id"):
                allowed_people = _resolve_anchor_people_handles(
                    db_handle, args["anchor_gramps_id"], args["generation_depth"]
                )
                allowed_families = _resolve_family_handles_for_people(
                    db_handle, allowed_people
                )
                events = [
                    event
                    for event in events
                    if _is_event_in_anchor_scope(
                        db_handle, event, allowed_people, allowed_families
                    )
                ]

            events.sort(key=_event_sort_key)
            payload = _build_ics(events=events, db_handle=db_handle, tree_id=tree_id)
        finally:
            close_db(db_handle)

        response = Response(payload, status=200, mimetype="text/calendar")
        response.headers["Content-Disposition"] = "inline; filename=anniversaries.ics"
        return response
