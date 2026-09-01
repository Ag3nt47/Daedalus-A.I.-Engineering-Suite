from __future__ import annotations

import json
import re
from pathlib import Path
from urllib.parse import urlparse

import pytest

from daedalus.resources import RESOURCE_FILES, load_json

ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOT = ROOT / "src" / "daedalus" / "resources"
DOC_ROOT = ROOT / "docs"
ID_PATTERN = re.compile(r"^[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*$")
GLOSSARY_MODES = {"plain", "math", "code", "diagnostic"}
CONTENT_LEVELS = {"beginner", "builder", "advanced", "capstone"}


@pytest.fixture(scope="module")
def resources() -> dict[str, dict]:
    return {name: load_json(name) for name in RESOURCE_FILES}


def _items(resources: dict[str, dict], name: str, key: str) -> list[dict]:
    value = resources[name][key]
    assert isinstance(value, list) and value, f"{name}:{key} must be a nonempty list"
    return value


def _all_modules(learning: dict) -> list[dict]:
    return [module for track in learning["tracks"] for module in track["modules"]]


def _assert_unique_ids(items: list[dict], label: str) -> set[str]:
    identifiers = [item.get("id") for item in items]
    assert all(isinstance(value, str) and ID_PATTERN.fullmatch(value) for value in identifiers)
    assert len(identifiers) == len(set(identifiers)), f"duplicate {label} ID"
    return set(identifiers)


def test_expected_files_are_packaged_and_loadable(resources: dict[str, dict]) -> None:
    assert RESOURCE_FILES == {
        "error_cards.json",
        "glossary.json",
        "learning_paths.json",
        "project_recipes.json",
        "sources.json",
    }
    for name, data in resources.items():
        assert (RESOURCE_ROOT / name).is_file()
        assert data["schema_version"] == 1
        assert re.fullmatch(r"\d{4}\.\d{2}", data["resource_version"])
        on_disk = json.loads((RESOURCE_ROOT / name).read_text(encoding="utf-8"))
        assert data == on_disk

    with pytest.raises(ValueError, match="Unknown Daedalus resource"):
        load_json("../private.json")


def test_all_ids_are_unique_and_namespaced(resources: dict[str, dict]) -> None:
    learning = resources["learning_paths.json"]
    tracks = _items(resources, "learning_paths.json", "tracks")
    modules = _all_modules(learning)
    recipes = _items(resources, "project_recipes.json", "recipes")
    glossary = _items(resources, "glossary.json", "entries")
    cards = _items(resources, "error_cards.json", "cards")
    sources = _items(resources, "sources.json", "sources")
    checkpoints = [module["checkpoint"] for module in modules]

    groups = {
        "track": _assert_unique_ids(tracks, "track"),
        "module": _assert_unique_ids(modules, "module"),
        "recipe": _assert_unique_ids(recipes, "recipe"),
        "glossary": _assert_unique_ids(glossary, "glossary"),
        "error": _assert_unique_ids(cards, "error card"),
        "source": _assert_unique_ids(sources, "source"),
        "checkpoint": _assert_unique_ids(checkpoints, "checkpoint"),
    }
    combined = set().union(*groups.values())
    assert len(combined) == sum(len(values) for values in groups.values())
    for prefix in ("track", "module", "recipe", "checkpoint"):
        assert all(identifier.startswith(prefix + ".") for identifier in groups[prefix])


def test_source_schema_https_and_original_summaries(resources: dict[str, dict]) -> None:
    sources = _items(resources, "sources.json", "sources")
    urls: set[str] = set()
    for source in sources:
        assert {
            "id",
            "title",
            "url",
            "publisher",
            "kind",
            "levels",
            "topics",
            "offline_summary",
        } <= source.keys()
        parsed = urlparse(source["url"])
        assert parsed.scheme == "https", f"HTTPS expected for {source['id']}"
        assert parsed.netloc and " " not in source["url"]
        assert source["url"] not in urls
        urls.add(source["url"])
        assert set(source["levels"]) <= CONTENT_LEVELS
        assert source["levels"] and source["topics"]
        summary = source["offline_summary"].strip()
        assert 40 <= len(summary) <= 400
        assert summary.casefold() != source["title"].strip().casefold()

    kinds = {source["kind"] for source in sources}
    assert {"official-documentation", "official-course", "standard", "original-paper"} <= kinds
    assert sum(source["kind"] == "original-paper" for source in sources) >= 6


def test_learning_track_module_lab_and_checkpoint_schema(resources: dict[str, dict]) -> None:
    learning = resources["learning_paths.json"]
    tracks = learning["tracks"]
    assert learning["levels"] == ["beginner", "builder", "advanced", "capstone"]
    assert len(tracks) >= 9

    for track in tracks:
        assert track["level"] in CONTENT_LEVELS
        assert track["title"].strip() and len(track["summary"].strip()) >= 40
        assert isinstance(track["prerequisite_track_ids"], list)
        assert isinstance(track["modules"], list) and track["modules"]
        for module in track["modules"]:
            objectives = module["objectives"]
            assert len(objectives) >= 3 and len(objectives) == len(set(objectives))
            assert all(len(value.strip()) >= 20 for value in objectives)

            lab = module["lab"]
            assert lab["title"].strip()
            assert len(lab["instructions"]) >= 3
            assert all(len(value.strip()) >= 15 for value in lab["instructions"])
            assert len(lab["deliverable"].strip()) >= 25
            assert lab["recipe_id"] is None or isinstance(lab["recipe_id"], str)

            checkpoint = module["checkpoint"]
            assert 0 < checkpoint["required_score"] <= 1
            assert len(checkpoint["checks"]) >= 3
            assert all(len(value.strip()) >= 20 for value in checkpoint["checks"])

            assert module["source_ids"]
            assert module["glossary_ids"]
            assert isinstance(module["error_card_ids"], list)


def test_internal_references_and_track_graph(resources: dict[str, dict]) -> None:
    learning = resources["learning_paths.json"]
    tracks = learning["tracks"]
    modules = _all_modules(learning)
    recipes = resources["project_recipes.json"]["recipes"]
    glossary = resources["glossary.json"]["entries"]
    cards = resources["error_cards.json"]["cards"]
    sources = resources["sources.json"]["sources"]

    track_ids = {item["id"] for item in tracks}
    module_ids = {item["id"] for item in modules}
    recipe_ids = {item["id"] for item in recipes}
    glossary_ids = {item["id"] for item in glossary}
    card_ids = {item["id"] for item in cards}
    source_ids = {item["id"] for item in sources}

    for track in tracks:
        prerequisites = set(track["prerequisite_track_ids"])
        assert prerequisites <= track_ids
        assert track["id"] not in prerequisites
        assert track["capstone_recipe_id"] in recipe_ids
    for module in modules:
        if module["lab"]["recipe_id"] is not None:
            assert module["lab"]["recipe_id"] in recipe_ids
        assert set(module["source_ids"]) <= source_ids
        assert set(module["glossary_ids"]) <= glossary_ids
        assert set(module["error_card_ids"]) <= card_ids
    for recipe in recipes:
        assert set(recipe["prerequisite_module_ids"]) <= module_ids
        assert set(recipe["source_ids"]) <= source_ids
    for entry in glossary:
        assert set(entry["related_ids"]) <= glossary_ids
        assert entry["id"] not in entry["related_ids"]
        assert set(entry["source_ids"]) <= source_ids
    for card in cards:
        assert set(card["glossary_ids"]) <= glossary_ids
        assert set(card["source_ids"]) <= source_ids

    remaining = {track["id"]: set(track["prerequisite_track_ids"]) for track in tracks}
    completed: set[str] = set()
    while remaining:
        ready = {track_id for track_id, needs in remaining.items() if needs <= completed}
        assert ready, "track prerequisite cycle detected"
        completed |= ready
        remaining = {track_id: needs for track_id, needs in remaining.items() if track_id not in ready}


def test_recipe_schema_and_gates(resources: dict[str, dict]) -> None:
    recipes = _items(resources, "project_recipes.json", "recipes")
    for recipe in recipes:
        assert recipe["level"] in CONTENT_LEVELS
        assert isinstance(recipe["estimated_hours"], int) and recipe["estimated_hours"] > 0
        assert len(recipe["summary"].strip()) >= 40
        assert recipe["prerequisite_module_ids"]
        assert len(recipe["steps"]) >= 4
        assert len(recipe["gates"]) >= 3
        assert all(len(value.strip()) >= 20 for value in recipe["steps"] + recipe["gates"])
        assert recipe["source_ids"]


def test_glossary_supports_all_four_modes(resources: dict[str, dict]) -> None:
    glossary = _items(resources, "glossary.json", "entries")
    assert resources["glossary.json"]["mode_order"] == [
        "plain",
        "math",
        "code",
        "diagnostic",
    ]
    assert len(glossary) >= 30
    for entry in glossary:
        assert entry["term"].strip()
        assert set(entry["levels"]) <= CONTENT_LEVELS
        assert entry["topics"] and entry["source_ids"]
        assert set(entry["modes"]) == GLOSSARY_MODES
        assert all(len(entry["modes"][mode].strip()) >= 20 for mode in GLOSSARY_MODES)
        assert isinstance(entry["aliases"], list)


def test_error_cards_are_deterministic_safe_and_complete(resources: dict[str, dict]) -> None:
    cards = _items(resources, "error_cards.json", "cards")
    expected = {
        "numpy.broadcast-mismatch",
        "numpy.matmul-core-mismatch",
        "numeric.nonfinite",
        "numeric.unstable-softmax",
        "autograd.gradient-accumulation",
        "autograd.broadcast-reduction",
        "training.diverging-loss",
        "workspace.path-boundary",
        "git.ignored-but-tracked",
        "git.push-conflict",
        "security.secret-detected",
        "security.unfixed-advisory",
        "checkpoint.untrusted-pickle",
    }
    assert {card["id"] for card in cards} == expected
    assert "Never capture raw secrets" in resources["error_cards.json"]["redaction_policy"]

    for card in cards:
        assert card["level"] in CONTENT_LEVELS
        assert card["severity"] in {"warning", "error", "blocking"}
        assert len(card["plain_cause"].strip()) >= 30
        assert len(card["likely_causes"]) >= 2
        assert len(card["evidence"]) >= 3
        assert len(card["checks"]) >= 2
        assert card["safe_fixes"] and card["never_actions"]
        assert card["glossary_ids"] and card["source_ids"]
        assert card["triggers"]
        for trigger in card["triggers"]:
            assert trigger["kind"].strip()
            re.compile(trigger["pattern"], flags=re.IGNORECASE)


def test_primary_source_topic_coverage_and_no_orphans(resources: dict[str, dict]) -> None:
    sources = resources["sources.json"]["sources"]
    modules = _all_modules(resources["learning_paths.json"])
    recipes = resources["project_recipes.json"]["recipes"]
    glossary = resources["glossary.json"]["entries"]
    cards = resources["error_cards.json"]["cards"]

    topics = {topic for source in sources for topic in source["topics"]}
    assert {
        "python",
        "math",
        "numpy",
        "autograd",
        "neural-networks",
        "git",
        "github",
        "security",
        "packaging",
        "ui",
    } <= topics

    all_source_ids = {source["id"] for source in sources}
    used_source_ids = {
        source_id
        for collection in (modules, recipes, glossary, cards)
        for item in collection
        for source_id in item["source_ids"]
    }
    assert used_source_ids == all_source_ids, f"orphan sources: {all_source_ids - used_source_ids}"


def test_operator_documentation_is_present_and_substantive() -> None:
    expected = {
        "GETTING_STARTED.md",
        "LEARNING_PATHS.md",
        "PROJECT_RECIPES.md",
        "ARCHITECTURE.md",
        "SECURITY_MODEL.md",
        "BACKUP_RESTORE.md",
        "RELEASE_GUIDE.md",
    }
    assert {path.name for path in DOC_ROOT.glob("*.md")} >= expected
    for name in expected:
        text = (DOC_ROOT / name).read_text(encoding="utf-8")
        assert len(text) >= 2_000, f"{name} is unexpectedly thin"
        assert "TODO" not in text and "TBD" not in text

