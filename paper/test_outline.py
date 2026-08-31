#!/usr/bin/env python3
"""Guard the assembler against the failure that shipped the wrong section.

Sections used to be addressed by number. Inserting one renumbered everything
below it, the outline kept pointing at the old slot, and the build happily
published a different section under the wrong heading. Nothing failed; the PDF
was simply wrong. These checks make that loud.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import outline
import sections

SRC = Path(__file__).resolve().parent.parent.parent / "research" / "infinite_horizon_diversity"


def load():
    from part_three import PART_THREE
    return {"mm": sections.parse((SRC / "PAPER.md").read_text()),
            "cv": sections.parse((SRC / "coverage" / "PAPER.md").read_text()),
            "iii": sections.parse(PART_THREE)}


def main() -> int:
    src, fails = load(), []

    # 1. every outline reference resolves to a real heading
    used = {k: set() for k in src}
    for _, items in outline.OUTLINE:
        for item in items:
            if item[0] == "text":
                continue
            kind, key = (item[2], item[3]) if item[0] == "sub" else (item[0], item[1])
            if kind == "text":
                continue
            try:
                sections.body(src[kind], key)
                used[kind].add(sections.slug(key))
            except KeyError as e:
                fails.append(f"unresolved {kind}:{key!r} — {str(e).splitlines()[0]}")

    # 2. no source section is silently dropped
    # Sections the unified paper deliberately does not carry, because authored
    # replacements cover them or they are container headings whose children are
    # used individually. Anything NOT on this list that goes unused is a bug:
    # that is how the Notes section was silently dropped.
    KNOWN_UNUSED = {
        "mm": {
            "abstract",
            "why a fixed prompt has a finite novelty budget and what to spend instead",
            "introduction",          # replaced by the authored INTRO
            "contributions",         # replaced by the authored CONTRIB
            "theory conditional dimension governs saturation",   # container
            "scoring candidate axes",                            # container
            "relation to the coverage problem",  # absorbed into the relation section
            "limitations",           # replaced by the merged LIMITATIONS
            "conclusion",            # replaced by the merged CONCLUSION
        },
        "cv": {
            "abstract",
            "introduction",          # merged into the authored INTRO
            "theory",                # container
            "relation to part i",    # absorbed into the relation section
            "limitations",           # merged
            "conclusion",            # merged
        },
        "iii": set(),
    }
    for kind, blocks in src.items():
        for slug_, b in blocks.items():
            if slug_ not in used[kind] and slug_ not in KNOWN_UNUSED.get(kind, set()):
                fails.append(f"orphan {kind}: {b['title']!r} exists in the source "
                             f"but no outline entry uses it")

    # 3. titles are unique within a source (parse() enforces; assert it stays)
    for kind in src:
        pass  # parse() raises on duplicates, so reaching here means unique

    if fails:
        print("OUTLINE CHECK FAILED")
        for f in fails:
            print(f"  - {f}")
        return 1
    n = sum(len(v) for v in used.values())
    print(f"outline OK: {n} sections resolved by title, no orphans")
    return 0


if __name__ == "__main__":
    sys.exit(main())
