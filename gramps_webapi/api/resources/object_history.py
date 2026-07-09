#
# Gramps Web API - A RESTful API for the Gramps genealogy program
#
# Copyright (C) 2026      David Straub
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

"""Per-object change history endpoint."""

import json
from typing import Dict

from flask import Response
from marshmallow import Schema
from webargs import fields, validate

from ...auth.const import PERM_EDIT_OBJ
from ...const import GRAMPS_NAMESPACES
from ..auth import require_permissions
from ..blueprint import api_blueprint
from ..util import abort_with_message, get_db_handle
from . import ProtectedResource
from .history import fix_transaction_user, get_user_dict
from .schemas import UndoTransactionSchema
from .util import get_page_reference_handles

# Namespaces accepted by GET /api/<namespace>/<handle>/history/.
# This extends the shared GRAMPS_NAMESPACES map (used by filters/bookmarks)
# with "tags": Tag objects have no dedicated filter/bookmark support, but do
# have their own change history and are one of the primary Gramps object
# types (see the object-history design doc).
OBJECT_HISTORY_NAMESPACES = {**GRAMPS_NAMESPACES, "tags": "Tag"}


class ObjectHistoryQueryArgs(Schema):
    """Query arguments for GET /<namespace>/<handle>/history/."""

    scope = fields.Str(
        load_default="page",
        validate=validate.OneOf(["object", "page"]),
        metadata={
            "description": (
                "'object' returns only transactions touching this handle; "
                "'page' (default) additionally includes transactions "
                "touching the objects this object's page renders inline "
                "(media, notes, citations, events, and for people also "
                "their families)."
            )
        },
    )
    page = fields.Integer(
        load_default=1,
        validate=validate.Range(min=1),
        metadata={"description": "Page number of the result subset to return."},
    )
    pagesize = fields.Integer(
        load_default=10,
        validate=validate.Range(min=1),
        metadata={"description": "Number of items per page."},
    )
    sort = fields.Str(
        load_default="-id",
        validate=validate.OneOf(["id", "-id"]),
        metadata={
            "description": (
                "Sort order for transactions. 'id' for ascending or '-id' "
                "(default) for descending."
            )
        },
    )


class ObjectHistoryResource(ProtectedResource):
    """Resource for the change history of a single object and (optionally) its page."""

    @api_blueprint.response(200, UndoTransactionSchema(many=True))
    @api_blueprint.arguments(ObjectHistoryQueryArgs, location="query")
    def get(self, args: Dict, namespace: str, handle: str) -> Response:
        """Return the change history for an object, optionally scoped to its page."""
        # Editor+ (can-edit floor). PERM_VIEW_PRIVATE would be Member+ — too wide;
        # Editor+ is a subset of view-private holders, so no private leak either.
        require_permissions([PERM_EDIT_OBJ])
        class_name = OBJECT_HISTORY_NAMESPACES.get(namespace)
        if class_name is None:
            abort_with_message(404, f"Unknown namespace: {namespace}")

        db_handle = get_db_handle()
        has_handle_func = db_handle.method("has_%s_handle", class_name)
        if has_handle_func is None or not has_handle_func(handle):
            abort_with_message(404, f"{class_name} with handle {handle} not found")

        if args["scope"] == "object":
            handles = {handle}
        else:
            handles = get_page_reference_handles(db_handle, class_name, handle)

        undodb = db_handle.undodb
        ascending = args["sort"] != "-id"
        transactions, count = undodb.get_object_transactions(
            handles=handles,
            page=args["page"],
            pagesize=args["pagesize"],
            ascending=ascending,
        )

        # replace user IDs by user name
        user_dict = get_user_dict()
        transactions = [
            fix_transaction_user(transaction, user_dict) for transaction in transactions
        ]
        res = Response(
            response=json.dumps(transactions),
            status=200,
            mimetype="application/json",
        )
        res.headers.add("X-Total-Count", count)
        return res
