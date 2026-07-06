#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2024      David Straub
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

"""AI chat endpoint."""

import asyncio
import json

from flask import Response, stream_with_context
from flask_jwt_extended import get_jwt_identity
from marshmallow import Schema, validate
from webargs import fields

from ..util import (
    get_tree_from_jwt_or_fail,
    abort_with_message,
    check_quota_ai,
    get_logger,
    update_usage_ai,
)
from ..blueprint import api_blueprint
from ..tasks import AsyncResult, make_task_response, process_chat, run_task
from . import ProtectedResource
from .schemas import ChatResponseSchema
from ...auth.const import PERM_USE_CHAT, PERM_VIEW_PRIVATE
from ..auth import has_permissions, require_permissions


class ChatMessageSchema(Schema):
    role = fields.Str(
        required=True,
        metadata={
            "description": "Role of the message sender: one of 'human', 'ai', 'system', 'assistant', or 'error'."
        },
    )
    message = fields.Str(
        required=True,
        metadata={"description": "The message content."},
    )


class ChatBodyArgs(Schema):
    """Body arguments for POST /chat/."""

    query = fields.Str(
        required=True,
        metadata={"description": "The chat prompt to answer."},
    )
    history = fields.List(
        fields.Nested(ChatMessageSchema),
        required=False,
        metadata={
            "description": "Optional list of prior conversation messages ({role, message})."
        },
    )
    message_history_raw = fields.Str(
        required=False,
        load_default=None,
        validate=validate.Length(max=1_000_000),
        metadata={
            "description": "Serialized message history from a previous response's "
            "message_history_raw field. Preserves full tool call context across turns. "
            "Takes precedence over history when both are provided."
        },
    )
    home_person_gramps_id = fields.Str(
        required=False,
        load_default=None,
        metadata={
            "description": "Gramps ID of the user's home person (the person "
            "representing the user in the tree). When provided, the assistant "
            "resolves self-references ('I', 'my', 'me') to this person."
        },
    )


class ChatQueryArgs(Schema):
    """Query arguments for POST /chat/."""

    background = fields.Boolean(
        load_default=False,
        metadata={
            "description": "If true, process the chat in the background and return HTTP 202."
        },
    )
    verbose = fields.Boolean(
        load_default=False,
        metadata={
            "description": "If true, include detailed agent metadata (tool calls, token usage) in the response."
        },
    )


class ChatResource(ProtectedResource):
    """AI chat resource."""

    @api_blueprint.response(200, ChatResponseSchema())
    @api_blueprint.arguments(ChatBodyArgs, location="json")
    @api_blueprint.arguments(ChatQueryArgs, location="query")
    def post(self, args_json, args_query):
        """Create a chat response."""
        require_permissions({PERM_USE_CHAT})
        check_quota_ai(requested=1)
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        if args_query["background"]:
            task = run_task(
                process_chat,
                tree=tree,
                user_id=user_id,
                query=args_json["query"],
                include_private=include_private,
                history=args_json.get("history"),
                verbose=args_query["verbose"],
                message_history_raw=args_json.get("message_history_raw"),
                home_person_gramps_id=args_json.get("home_person_gramps_id"),
            )
            if isinstance(task, AsyncResult):
                return make_task_response(task)
            update_usage_ai(new=1)
            return task, 200

        try:
            result = process_chat(
                tree=tree,
                user_id=user_id,
                query=args_json["query"],
                include_private=include_private,
                history=args_json.get("history"),
                verbose=args_query["verbose"],
                message_history_raw=args_json.get("message_history_raw"),
                home_person_gramps_id=args_json.get("home_person_gramps_id"),
            )
        except ValueError:
            abort_with_message(422, "Invalid message format")

        update_usage_ai(new=1)
        return result


class ChatStreamResource(ProtectedResource):
    """Streaming AI chat resource — Server-Sent Events (text/event-stream).

    Runs the agent in the web process (not Celery) via run_stream_events so
    answer text and tool-call events are streamed to the client in real time.
    Emits ``data: {json}\\n\\n`` frames of type delta / tool / done / error.
    """

    @api_blueprint.arguments(ChatBodyArgs, location="json")
    def post(self, args_json) -> Response:
        """Stream a chat response as Server-Sent Events."""
        require_permissions({PERM_USE_CHAT})
        check_quota_ai(requested=1)
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})

        query = args_json["query"]
        history = args_json.get("history")
        message_history_raw = args_json.get("message_history_raw")
        home_person_gramps_id = args_json.get("home_person_gramps_id")

        # Imported lazily so the chat module loads without AI deps installed.
        from ..llm import stream_agent_events

        def generate():
            loop = asyncio.new_event_loop()
            agen = stream_agent_events(
                prompt=query,
                tree=tree,
                include_private=include_private,
                user_id=user_id,
                history=history,
                message_history_raw=message_history_raw,
                home_person_gramps_id=home_person_gramps_id,
                verbose=True,
            )
            completed = False
            try:
                while True:
                    try:
                        event = loop.run_until_complete(agen.__anext__())
                    except StopAsyncIteration:
                        break
                    if event.get("type") == "done":
                        completed = True
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as e:  # pylint: disable=broad-except
                get_logger().error("Chat stream error: %s", e)
                yield (
                    "data: "
                    + json.dumps(
                        {"type": "error", "message": "Unexpected error."},
                        ensure_ascii=False,
                    )
                    + "\n\n"
                )
            finally:
                try:
                    loop.run_until_complete(agen.aclose())
                except Exception:  # pylint: disable=broad-except
                    pass
                loop.close()
                if completed:
                    try:
                        update_usage_ai(new=1)
                    except Exception:  # pylint: disable=broad-except
                        pass

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
