"""Parse a source paper into numbered blocks so it can be re-ordered."""
import re

HEAD = re.compile(r"^(#{2,3})\s+(?:([0-9]+(?:\.[0-9]+)?[a-z]?)\.?\s+)?(.*)$")


def parse(md: str) -> dict:
    """{key: {'level', 'num', 'title', 'body'}} keyed by number, else by title."""
    lines = md.split("\n")
    blocks, cur = {}, None
    for ln in lines:
        m = HEAD.match(ln)
        if m:
            level, num, title = len(m.group(1)), m.group(2), m.group(3).strip()
            key = num or title
            cur = {"level": level, "num": num, "title": title, "body": []}
            blocks[key] = cur
            continue
        if cur is not None:
            cur["body"].append(ln)
    for b in blocks.values():
        b["body"] = "\n".join(b["body"]).strip("\n")
    return blocks


def body(blocks: dict, key: str) -> str:
    if key not in blocks:
        raise KeyError(f"no such block: {key!r} (have {sorted(blocks)[:12]}…)")
    return blocks[key]["body"]


def title(blocks: dict, key: str) -> str:
    return blocks[key]["title"]
