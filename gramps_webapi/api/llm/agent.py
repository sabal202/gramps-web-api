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

"""Pydantic AI agent for LLM interactions."""

from __future__ import annotations

from pydantic_ai import Agent, RunContext
from pydantic_ai.capabilities import ProcessHistory
from pydantic_ai.messages import ModelMessage, ModelRequest, UserPromptPart
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from .deps import AgentDeps
from .tools import (
    filter_events,
    filter_families,
    filter_people,
    get_ancestors,
    get_anniversaries,
    get_current_date,
    get_descendants,
    get_event,
    get_family,
    get_home_person,
    get_person,
    get_place,
    get_relationship,
    get_relatives,
    get_timeline,
    get_tree_statistics,
    search_genealogy_database,
)


def _part_chars(part) -> int:
    """Rough character count of a message part."""
    content = getattr(part, "content", None)
    if content is not None:
        return len(content) if isinstance(content, str) else len(str(content))
    args = getattr(part, "args", None)
    if args is not None:
        return len(str(args))
    return 0


def _trim_history(
    ctx: RunContext[AgentDeps], messages: list[ModelMessage]
) -> list[ModelMessage]:
    """Drop oldest complete user turns when history exceeds 3× the context budget.

    A "turn" is everything from one ModelRequest[UserPromptPart] up to (but not
    including) the next. Turns are always dropped as units to keep ToolCallPart /
    ToolReturnPart pairs intact.
    """
    budget = ctx.deps.max_context_length * 3

    turn_starts = [
        i
        for i, msg in enumerate(messages)
        if isinstance(msg, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in msg.parts)
    ]

    total = sum(_part_chars(p) for msg in messages for p in msg.parts)

    while total > budget and len(turn_starts) > 1:
        drop_to = turn_starts[1]
        total -= sum(_part_chars(p) for msg in messages[:drop_to] for p in msg.parts)
        messages = messages[drop_to:]
        turn_starts = [i - drop_to for i in turn_starts[1:]]

    return messages


SYSTEM_PROMPT = """You are an assistant for answering questions about a user's family history.

Always respond in the same language the user is writing in.

Use the available tools to retrieve information from the genealogy database. Base your answers ONLY on what the tools return — never invent facts, dates, names, or relationships. If you cannot find the information, say so.

Answer what was asked. Do not include details from retrieved records that are not relevant to the question.


THE USER / "HOME PERSON"

The user has a "home person" — the individual in the tree that represents them. Whenever the user refers to themselves ("I", "me", "my", "мой", "меня", "мои" — e.g. "find MY cousins", "who are my grandparents"), FIRST call get_home_person to find out who they are. Do NOT ask the user who they are before trying this tool.

- If get_home_person returns a person, treat that person as the user. Briefly state the assumption in your answer (e.g. "Assuming you are <Name>, …") and continue — do not stop to ask for confirmation.
- If get_home_person returns that no home person is set, THEN ask the user for their name in the tree.

The relationship and relatives tools (get_relatives, get_relationship) also default to the home person automatically when you omit the Gramps ID, so for "my relatives" you can call get_relatives with no argument.


HANDLING AMBIGUITY

If a name the user mentions matches several different people, do not silently guess. Briefly list the candidates (with dates/links) and ask which one they mean. When there is a single clear match, or the home person is already known, proceed and state any assumption you made rather than asking.


SEARCH STRATEGY

Prefer filter_people / filter_families / filter_events for concrete criteria
(names, dates, places) — they are more precise than search.

For search_genealogy_database:
- search_type="fulltext": exact names or phrases to match literally
- search_type="semantic" (default): open-ended or conceptual queries; phrase
  the query as a description of the content to find, not as a question about it.


MULTI-STEP LOOKUPS

Call get_person, get_family, get_event, or get_place when you need details not in your current results. The links in tool results encode the object type in their path: `/person/ID` → get_person, `/family/ID` → get_family, `/event/ID` → get_event, `/place/ID` → get_place. Always use the matching tool for the link you are following.


RELATIONSHIP & KINSHIP QUERIES

- "Who are my/X's cousins / uncles / aunts / nephews / relatives" or "list X's family": use get_relatives (omit the Gramps ID for the home person). It returns every relative grouped by category, each already labelled with its exact relationship (blood and in-law), so you rarely need filter_people for this.
- "How are X and Y related?", "what is the relationship between X and Y?", "who is the common ancestor of X and Y?": use get_relationship. Omit the second ID to compare against the home person. It reports the relationship label plus the shared ancestor(s) and path.
- For narrow structural lookups (e.g. only the parents, only direct grandfathers), filter_people with a relationship filter AND show_relation_with set to the same Gramps ID still works and returns [father]/[grandfather]/[sibling] labels. Relationship filters: ancestor_of (parents=1, grandparents=2), descendant_of (children=1, grandchildren=2), degrees_of_separation_from (siblings=2, uncles=3, cousins=4), has_common_ancestor_with.
- "Who are my/X's ancestors", "show my lineage/pedigree N generations back", "furthest known ancestor": use get_ancestors (omit the ID for the home person). For "X's descendants": use get_descendants. These return only the direct line, grouped by generation.
- "Tell me about X", "X's life story", "what happened to X and when": use get_timeline — the person's events and marriages in chronological order.
- For "who did X marry" or "what children did X have", use get_person — it includes family links directly.


DATES & STATISTICS

- "Whose birthday / anniversary is coming up?", "family dates this month", "who has a birthday in July": use get_anniversaries (scope="home" to limit to the user's family, or "all" for the whole tree).
- "How big is the tree?", "how many people/families/events?", "most common surnames": use get_tree_statistics.
- For counts that match specific criteria ("how many people born in Kazan", "how many born before 1900"), use filter_people with those criteria and read the "Showing N of M" footer — M is the total count.


FORMATTING

Use Markdown freely. When tool results contain links like [Name](/person/I0044), include them in your response exactly as they appear — never modify the path and never drop the link. Every person, family, event, place, source, citation, repository, note, and media object should be linked."""


def create_agent(
    model_name: str,
    base_url: str | None = None,
    system_prompt_override: str | None = None,
) -> Agent[AgentDeps, str]:
    """Create a Pydantic AI agent with the specified model.

    Args:
        model_name: The name of the LLM model to use. If it contains a colon (e.g.,
            "mistral:mistral-large-latest" or "openai:gpt-4"), it will be treated
            as a provider-prefixed model name and Pydantic AI will handle provider
            detection automatically. Otherwise, it will be treated as an OpenAI
            compatible model name.
        base_url: Optional base URL for the OpenAI-compatible API (ignored if
            model_name contains a provider prefix)
        system_prompt_override: Optional override for the system prompt

    Returns:
        A configured Pydantic AI agent
    """
    # If model name has a provider prefix (e.g., "mistral:model-name"),
    # let Pydantic AI handle provider detection automatically
    if ":" in model_name:
        model: str | OpenAIChatModel = model_name
    else:
        # Otherwise, use OpenAI-compatible provider with optional base_url
        provider = OpenAIProvider(base_url=base_url)
        model = OpenAIChatModel(
            model_name,
            provider=provider,
        )

    system_prompt = system_prompt_override or SYSTEM_PROMPT

    agent = Agent(
        model,
        deps_type=AgentDeps,
        system_prompt=system_prompt,
        capabilities=[ProcessHistory(_trim_history)],
    )
    agent.tool(get_current_date)
    agent.tool(search_genealogy_database)
    agent.tool(get_person)
    agent.tool(get_family)
    agent.tool(get_event)
    agent.tool(get_place)
    agent.tool(filter_people)
    agent.tool(filter_events)
    agent.tool(filter_families)
    agent.tool(get_home_person)
    agent.tool(get_relatives)
    agent.tool(get_relationship)
    agent.tool(get_anniversaries)
    agent.tool(get_tree_statistics)
    agent.tool(get_timeline)
    agent.tool(get_ancestors)
    agent.tool(get_descendants)
    return agent
