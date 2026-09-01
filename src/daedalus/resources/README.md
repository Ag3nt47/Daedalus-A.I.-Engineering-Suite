# Daedalus packaged resources

These JSON files are the stable, offline-readable content layer for the Learning
Atlas and deterministic diagnostic helper. They contain original summaries and
links to authoritative sources; they do not mirror or reproduce external
documentation.

- `learning_paths.json`: tracks, modules, labs, and measurable checkpoints.
- `project_recipes.json`: reusable projects referenced by the curriculum.
- `sources.json`: verified official documentation and primary papers.
- `glossary.json`: Plain, Math, Code, and Diagnostic explanations.
- `error_cards.json`: deterministic troubleshooting cards and safe actions.

Load them through `daedalus.resources.load_json()` so the same code works in a
source checkout and an installed wheel. Schema changes require a version bump,
updated tests, and a migration note in the release guide.
