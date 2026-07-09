"""DB adapter: build a TreeGraph snapshot from a Gramps DB, then run the pure
algorithms in graph_primitives on it. All analysis functions are parameterized.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from gramps.gen.db.base import DbReadBase
from gramps.gen.errors import HandleError
from gramps.gen.lib import ChildRefType

from . import graph_primitives as gp

# ---------------------------------------------------------------------------
# Small handle-safety helpers (used by lineages()'s surname grouping and by
# integrity()'s dangling-reference scans).
# ---------------------------------------------------------------------------


def _safe_person(db: DbReadBase, handle: str):
    try:
        return db.get_person_from_handle(handle)
    except HandleError:
        return None


def _safe_family(db: DbReadBase, handle: str):
    try:
        return db.get_family_from_handle(handle)
    except HandleError:
        return None


@dataclass
class TreeGraph:
    """In-memory snapshot of the tree as graphs (person handles are the nodes)."""

    undirected: dict[str, set[str]] = field(default_factory=dict)  # connection graph
    parents: dict[str, list[str]] = field(default_factory=dict)  # child -> [parents]
    parents_birth: dict[str, list[str]] = field(
        default_factory=dict
    )  # birth-only


def build_tree_graph(db: DbReadBase) -> TreeGraph:
    """One pass over people + families → TreeGraph.

    Undirected edges: spouse↔spouse and parent↔child (ALL child refs, incl.
    adoptive/step — else an adopted child is a false island).
    Directed edges: child→parent, with a birth-only variant filtered by
    ChildRefType.BIRTH on both the father-ref and mother-ref.
    """
    g = TreeGraph()
    # Seed every person as a node so childless singletons appear as islands.
    for handle in db.get_person_handles():
        g.undirected.setdefault(handle, set())
        g.parents.setdefault(handle, [])
        g.parents_birth.setdefault(handle, [])

    def link(a: str, b: str) -> None:
        if a and b and a != b:
            g.undirected[a].add(b)
            g.undirected[b].add(a)

    for family in db.iter_families():
        father = family.get_father_handle()
        mother = family.get_mother_handle()
        link(father, mother)  # spouse edge
        for cref in family.get_child_ref_list():
            child = cref.ref
            if child is None:
                continue
            for parent, rel in (
                (father, cref.get_father_relation()),
                (mother, cref.get_mother_relation()),
            ):
                if not parent:
                    continue
                link(parent, child)
                g.parents.setdefault(child, []).append(parent)
                if rel == ChildRefType.BIRTH:
                    g.parents_birth.setdefault(child, []).append(parent)
    return g


def connectivity(
    db: DbReadBase, *, include_singletons: bool = True, g: TreeGraph | None = None
) -> dict[str, Any]:
    g = g or build_tree_graph(db)
    comps = gp.connected_components(g.undirected)
    total = sum(len(c) for c in comps)
    main = comps[0] if comps else set()
    islands, orphans, pairs = [], [], []
    for comp in comps[1:]:  # everything not in the largest
        if len(comp) == 1:
            if include_singletons:
                orphans.append(next(iter(comp)))
        elif len(comp) == 2:
            pairs.append(sorted(comp))
        islands.append({"size": len(comp), "handles": sorted(comp)})
    return {
        "person_count": total,
        "component_count": len(comps),
        "main_component_size": len(main),
        "islands": islands,  # size>1 non-main
        "isolated_pairs": pairs,
        "orphans": orphans,  # singletons
    }


def deepest_ancestors(
    db: DbReadBase,
    anchor,
    *,
    generations: int = 0,
    birth_only: bool = False,
    top: int = 10,
    g: TreeGraph | None = None,
) -> dict[str, Any]:
    g = g or build_tree_graph(db)
    pmap = g.parents_birth if birth_only else g.parents
    # BFS up from anchor to collect the ancestor subgraph (with generation).
    gen: dict[str, int] = {}
    q: deque[tuple[str, int]] = deque([(anchor.handle, 0)])
    seen = {anchor.handle}
    while q:
        h, d = q.popleft()
        if generations and d >= generations:
            continue
        for p in pmap.get(h, ()):
            if p not in seen:
                seen.add(p)
                gen[p] = d + 1
                q.append((p, d + 1))
    if not gen:
        return {"anchor": anchor.handle, "max_depth": 0, "by_line": [], "furthest": []}
    max_depth = max(gen.values())
    furthest = sorted(h for h, d in gen.items() if d == max_depth)
    # Lineage roots = the top-most (parentless within the ancestor subgraph).
    sub_parents = {h: [p for p in pmap.get(h, ()) if p in gen] for h in gen}
    line_roots = gp.roots(sub_parents)
    # children-within-subgraph map, to collect each root's OWN line members
    # (going down from the root toward the anchor). Fixes the bug where every
    # line reported the full ancestor set.
    children_sub: dict[str, list[str]] = {}
    for h, ps in sub_parents.items():
        for p in ps:
            children_sub.setdefault(p, []).append(h)
    by_line: list[dict[str, Any]] = []
    for r in line_roots:
        members = {r}
        stack = [r]
        while stack:
            n = stack.pop()
            for c in children_sub.get(n, ()):
                if c not in members:
                    members.add(c)
                    stack.append(c)
        # r is the deepest point of its line, so line depth = gen[r].
        by_line.append({"root": r, "depth": gen[r], "ancestors": sorted(members)})
    by_line.sort(key=lambda e: (-e["depth"], str(e["root"])))
    by_line = by_line[:top]
    return {
        "anchor": anchor.handle,
        "max_depth": max_depth,
        "by_line": by_line,
        "furthest": furthest,
    }


def lineages(
    db: DbReadBase,
    *,
    birth_only: bool = False,
    group_by: str = "root",
    min_size: int = 1,
    g: TreeGraph | None = None,
) -> dict[str, Any]:
    g = g or build_tree_graph(db)
    pmap = g.parents_birth if birth_only else g.parents
    up_depth = gp.ancestry_depth(pmap)
    max_tree_depth = max(up_depth.values(), default=0)
    # children map = reverse of pmap; longest chain DOWN via ancestry_depth on it.
    children: dict[str, list[str]] = {n: [] for n in pmap}
    for child, ps in pmap.items():
        for p in ps:
            children.setdefault(p, []).append(child)
    down_depth = gp.ancestry_depth(children)  # generations of descendants below each node

    def closure_size(r: str) -> int:
        seen = {r}
        stack = [r]
        while stack:
            n = stack.pop()
            for c in children.get(n, ()):
                if c not in seen:
                    seen.add(c)
                    stack.append(c)
        return len(seen)

    root_set = gp.roots(pmap)  # brick walls
    if group_by == "surname":
        from .ru_surnames import get_family_surname

        buckets: dict[str, dict[str, Any]] = {}
        for r in root_set:
            person = _safe_person(db, r)
            sn = (get_family_surname(person.primary_name) if person else "") or "?"
            slot = buckets.setdefault(
                sn, {"surname": sn, "roots": [], "size": 0, "depth": 0}
            )
            slot["roots"].append(r)
            slot["size"] += closure_size(r)
            slot["depth"] = max(slot["depth"], down_depth.get(r, 0))
        lines = [s for s in buckets.values() if s["size"] >= min_size]
        lines.sort(key=lambda e: (-e["depth"], -e["size"], e["surname"]))
    else:  # group_by == "root"
        lines = []
        for r in root_set:
            size = closure_size(r)
            if size < min_size:
                continue
            lines.append({"root": r, "depth": down_depth.get(r, 0), "size": size})
        lines.sort(key=lambda e: (-e["depth"], -e["size"], str(e["root"])))

    return {
        "max_tree_depth": max_tree_depth,
        "root_count": len(root_set),
        "lineages": lines,
    }


def integrity(
    db: DbReadBase,
    *,
    checks=("one_sided_refs", "dangling"),
    max_examples: int = 0,
) -> dict[str, Any]:
    want = set(checks)
    out: dict[str, list[dict[str, Any]]] = {}

    if "one_sided_refs" in want:
        probs: list[dict[str, Any]] = []
        # family -> member: member must reference the family back
        for family in db.iter_families():
            fh = family.handle
            for role, ph in (
                ("father", family.get_father_handle()),
                ("mother", family.get_mother_handle()),
            ):
                if not ph:
                    continue
                p = _safe_person(db, ph)
                if p is not None and fh not in p.get_family_handle_list():
                    probs.append(
                        {
                            "handle": ph,
                            "gramps_id": p.gramps_id,
                            "detail": f"{role} of family {family.gramps_id} but "
                            "family missing from their spouse list",
                        }
                    )
            for cref in family.get_child_ref_list():
                c = _safe_person(db, cref.ref)
                if c is not None and fh not in c.get_parent_family_handle_list():
                    probs.append(
                        {
                            "handle": cref.ref,
                            "gramps_id": c.gramps_id,
                            "detail": f"child of family {family.gramps_id} but "
                            "family missing from their parent-family list",
                        }
                    )
        # person -> family: family must list the person back
        for person in db.iter_people():
            for fh in person.get_family_handle_list():
                fam = _safe_family(db, fh)
                if fam is not None and person.handle not in (
                    fam.get_father_handle(),
                    fam.get_mother_handle(),
                ):
                    probs.append(
                        {
                            "handle": person.handle,
                            "gramps_id": person.gramps_id,
                            "detail": f"lists family {fam.gramps_id} as spouse-family "
                            "but is neither father nor mother",
                        }
                    )
            for fh in person.get_parent_family_handle_list():
                fam = _safe_family(db, fh)
                if fam is not None and person.handle not in {
                    cr.ref for cr in fam.get_child_ref_list()
                }:
                    probs.append(
                        {
                            "handle": person.handle,
                            "gramps_id": person.gramps_id,
                            "detail": f"lists family {fam.gramps_id} as parent-family "
                            "but is not among its children",
                        }
                    )
        out["one_sided_refs"] = probs

    if "dangling" in want:
        dang: list[dict[str, Any]] = []
        for person in db.iter_people():
            for fh in (
                person.get_family_handle_list() + person.get_parent_family_handle_list()
            ):
                if not db.has_family_handle(fh):
                    dang.append(
                        {
                            "handle": person.handle,
                            "gramps_id": person.gramps_id,
                            "detail": f"references missing family handle {fh}",
                        }
                    )
            for er in person.get_event_ref_list():
                if not db.has_event_handle(er.ref):
                    dang.append(
                        {
                            "handle": person.handle,
                            "gramps_id": person.gramps_id,
                            "detail": f"references missing event handle {er.ref}",
                        }
                    )
        out["dangling"] = dang

    if "thin_records" in want:  # opt-in only (noisy)
        thin = [
            {"handle": p.handle, "gramps_id": p.gramps_id, "detail": "no events recorded"}
            for p in db.iter_people()
            if not p.get_event_ref_list()
        ]
        out["thin_records"] = thin

    counts = {k: len(v) for k, v in out.items()}
    if max_examples:
        out = {k: v[:max_examples] for k, v in out.items()}
    return {"checks": sorted(out), "counts": counts, "problems": out}


def centrality(
    db: DbReadBase,
    *,
    metric: str = "betweenness",
    top: int = 10,
    max_nodes: int = 5000,
    g: TreeGraph | None = None,
) -> dict[str, Any]:
    g = g or build_tree_graph(db)
    adj = g.undirected
    capped = False
    if metric == "degree":
        ranked = gp.degree_centrality(adj)
    elif metric == "articulation":
        aps = gp.articulation_points(adj)
        # rank cut vertices by degree so the "most structural" come first
        deg = dict(gp.degree_centrality(adj))
        ranked = sorted(((h, deg[h]) for h in aps), key=lambda kv: (-kv[1], str(kv[0])))
    elif metric in ("betweenness", "closeness"):
        fn = (
            gp.betweenness_centrality
            if metric == "betweenness"
            else gp.closeness_centrality
        )
        if len(adj) > max_nodes:
            capped = True
            ranked = []
            for comp in gp.connected_components(adj):
                if len(comp) > max_nodes:
                    continue  # skip oversized component with a note
                sub = {n: adj[n] & comp for n in comp}
                ranked.extend(fn(sub))
            ranked.sort(key=lambda kv: (-kv[1], str(kv[0])))
        else:
            ranked = fn(adj)
    else:
        raise ValueError(f"unknown metric {metric!r}")
    return {
        "metric": metric,
        "capped": capped,
        "people": [{"handle": h, "score": s} for h, s in ranked[:top]],
    }
