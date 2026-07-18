"""
The closed vocabulary: relations, their cardinality, and the natural-language
prompts used to score them zero-shot with SigLIP.

This is the video analogue of the text system's hand-built extraction
vocabulary — and it carries the same honest limitation: anything outside the
bank is invisible (see the OOV limitation probe in the text project's test
battery, reproduced here for video).

Each relation declares:
  * cardinality  - "single" (exclusive: at most one object at a time) or
                   "multi" (objects coexist)
  * objects      - the closed object vocabulary for that relation
  * template     - how to render (relation, object) as a caption for SigLIP

Cardinality is what feeds `exclusive_relations` in resolve.py, so this file is
where the sweep-line's conflict behaviour is actually decided.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Relation:
    name: str
    cardinality: str          # "single" | "multi"
    objects: tuple[str, ...]
    template: str             # must contain "{object}"

    def prompts(self) -> list[tuple[str, str]]:
        """Return [(object, caption), ...] for this relation."""
        return [(o, self.template.format(object=o)) for o in self.objects]


# --------------------------------------------------------------------------
# The default bank. Deliberately small: a closed vocabulary you can actually
# characterize beats a large one you cannot.
# --------------------------------------------------------------------------
DEFAULT_BANK: tuple[Relation, ...] = (
    Relation(
        name="HOLDS",
        cardinality="single",
        objects=(
            "a cup", "a phone", "a book", "a bottle", "a knife",
            "a laptop", "a bag", "a pen", "nothing",
        ),
        template="a photo of a person holding {object}",
    ),
    Relation(
        name="LOCATED_IN",
        cardinality="single",
        objects=(
            "a kitchen", "a bedroom", "an office", "a bathroom",
            "a living room", "a store", "a street", "a garden",
        ),
        template="a photo taken in {object}",
    ),
    Relation(
        name="ACTIVITY",
        cardinality="single",
        objects=(
            "cooking", "walking", "typing", "reading", "eating",
            "cleaning", "driving", "talking to someone", "sitting still",
        ),
        template="a photo of a person {object}",
    ),
    Relation(
        name="NEAR",
        cardinality="multi",
        objects=(
            "a table", "a chair", "a door", "a window", "a sink",
            "a refrigerator", "a television", "a bed",
        ),
        template="a photo containing {object}",
    ),
)


def exclusive_relations(bank: tuple[Relation, ...] = DEFAULT_BANK) -> set[str]:
    """The set handed to resolve() to decide which slots conflict."""
    return {r.name for r in bank if r.cardinality == "single"}


def flatten(bank: tuple[Relation, ...] = DEFAULT_BANK) -> tuple[list[str], list[tuple[str, str]]]:
    """Flatten the bank into a single prompt list for one batched text encode.

    Returns:
        captions: list[str]                      -- what SigLIP encodes
        index:    list[(relation_name, object)]  -- parallel to captions
    """
    captions: list[str] = []
    index: list[tuple[str, str]] = []
    for rel in bank:
        for obj, caption in rel.prompts():
            captions.append(caption)
            index.append((rel.name, obj))
    return captions, index


def by_name(bank: tuple[Relation, ...] = DEFAULT_BANK) -> dict[str, Relation]:
    return {r.name: r for r in bank}
