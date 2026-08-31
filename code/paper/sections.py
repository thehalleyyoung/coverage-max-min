"""Parse a source paper into blocks addressable by heading text.

Addressing by section NUMBER was fragile: inserting a section renumbers every
one below it, and the outline then silently points at whichever section moved
into that slot. That failure is invisible — the build succeeds and publishes the
wrong content under the wrong heading. Addressing by title fails loudly instead,
because a title that no longer exists cannot be found.
"""
import difflib
import re

HEAD = re.compile(r"^(#{2,3})\s+(?:([0-9]+(?:\.[0-9]+)?[a-z]?)\.?\s+)?(.*)$")


def slug(title: str) -> str:
    """Match on words, so punctuation and emphasis drift do not break a lookup."""
    t = re.sub(r"[*_`]", "", title.lower())
    return " ".join(re.findall(r"[a-z0-9]+", t))


def parse(md: str) -> dict:
    """Blocks keyed by title-slug, with number and title retained.

    Numbers stay available for cross-reference rewriting; they are no longer the
    address.
    """
    blocks, cur = {}, None
    for ln in md.split("\n"):
        m = HEAD.match(ln)
        if m:
            level, num, title = len(m.group(1)), m.group(2), m.group(3).strip()
            cur = {"level": level, "num": num, "title": title, "body": []}
            key = slug(title)
            if key in blocks:
                raise KeyError(f"duplicate section title {title!r}; titles are the "
                               f"address, so they must be unique within a source")
            blocks[key] = cur
            continue
        if cur is not None:
            cur["body"].append(ln)
    for b in blocks.values():
        b["body"] = "\n".join(b["body"]).strip("\n")
    return blocks


def _find(blocks: dict, key: str) -> dict:
    k = slug(key)
    if k in blocks:
        return blocks[k]
    near = difflib.get_close_matches(k, list(blocks), n=3, cutoff=0.6)
    hint = ""
    if near:
        hint = "\n  did you mean: " + "; ".join(
            repr(blocks[n]["title"]) for n in near)
    raise KeyError(
        f"no section titled {key!r} in this source.{hint}\n"
        f"  Sections are addressed by heading text; if you renamed one, update "
        f"the outline to match.")


def body(blocks: dict, key: str) -> str:
    return _find(blocks, key)["body"]


def title(blocks: dict, key: str) -> str:
    return _find(blocks, key)["title"]


def number(blocks: dict, key: str) -> str | None:
    return _find(blocks, key)["num"]


def by_number(blocks: dict) -> dict:
    """{source number: block} — used only to rewrite § cross-references."""
    return {b["num"]: b for b in blocks.values() if b["num"]}
