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

from flask import Response, current_app, stream_with_context
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
from ...auth.const import (
    PERM_ADD_OBJ,
    PERM_DEL_OBJ,
    PERM_EDIT_OBJ,
    PERM_USE_CHAT,
    PERM_VIEW_PRIVATE,
)
from ..auth import has_permissions, require_permissions

# keep in sync with gramps_webapi.api.llm.images, which needs the AI dependencies
_MAX_IMAGES = 4
# base64 of 10 MiB, plus room for a data URL header
_MAX_IMAGE_CHARS = 4 * (10 * 1024 * 1024) // 3 + 100


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
        validate=validate.Length(max=20),
        metadata={
            "description": "Gramps ID of the user's home person; anchors the "
            "get_home_person and relationship/relatives tools."
        },
    )
    previous_proposal = fields.Str(
        required=False,
        load_default=None,
        validate=validate.OneOf(["applied", "not_applied"]),
        metadata={
            "description": "'applied' if, since the last prompt, the user "
            "applied changes the assistant proposed in any earlier response. "
            "'not_applied' if nothing was applied since the last prompt and "
            "the assistant's immediately preceding response had a proposal. "
            "Omit it otherwise."
        },
    )
    images = fields.List(
        fields.Str(validate=validate.Length(max=_MAX_IMAGE_CHARS)),
        required=False,
        load_default=None,
        validate=validate.Length(max=_MAX_IMAGES),
        metadata={
            "description": f"Up to {_MAX_IMAGES} JPEG, PNG, WebP or GIF images "
            "attached to the query, base64 encoded, optionally as data URLs. "
            "Only allowed if the server metadata reports chat_vision. Images "
            "are not kept in the returned message_history_raw."
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


def _normalize_images(images: list[str] | None) -> list[str] | None:
    """Validate and downscale attached images, or abort."""
    if not images:
        return None
    if not current_app.config["LLM_VISION"]:
        abort_with_message(422, "The AI model does not accept images.")
    # import here to avoid error if AI dependencies are not installed
    from ..llm.images import InvalidImageError, normalize_image

    try:
        return [normalize_image(image) for image in images]
    except InvalidImageError as e:
        abort_with_message(422, str(e))
    return None  # unreachable, for type checkers


class ChatResource(ProtectedResource):
    """AI chat resource."""

    @api_blueprint.response(200, ChatResponseSchema())
    @api_blueprint.arguments(ChatBodyArgs, location="json")
    @api_blueprint.arguments(ChatQueryArgs, location="query")
    def post(self, args_json, args_query):
        """Create a chat response."""
        require_permissions({PERM_USE_CHAT})
        # each image counts as another message, see process_chat
        check_quota_ai(requested=1 + len(args_json.get("images") or []))
        tree = get_tree_from_jwt_or_fail()
        user_id = get_jwt_identity()
        include_private = has_permissions({PERM_VIEW_PRIVATE})
        # proposals are applied as one transaction, which needs all three
        allow_write_proposals = has_permissions(
            {PERM_ADD_OBJ, PERM_EDIT_OBJ, PERM_DEL_OBJ}
        )
        images = _normalize_images(args_json.get("images"))

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
                allow_write_proposals=allow_write_proposals,
                images=images,
                previous_proposal=args_json.get("previous_proposal"),
            )
            if isinstance(task, AsyncResult):
                return make_task_response(task)
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
                allow_write_proposals=allow_write_proposals,
                images=images,
                previous_proposal=args_json.get("previous_proposal"),
            )
        except ValueError:
            abort_with_message(422, "Invalid message format")

        return result


class ChatStreamResource(ProtectedResource):
    """Streaming AI chat resource — Server-Sent Events (text/event-stream).

    Runs the agent in the web process (not Celery) via stream_agent_events so
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
