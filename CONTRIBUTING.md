# Contributing

Thank you for improving Daedalus.

1. Create a focused branch.
2. Install `.[dev]` in an isolated virtual environment.
3. Add tests for behavioral changes.
4. Run `python -m ruff check .`, `python -m bandit -r src -ll`, and
   `python -m pytest`.
5. Run `python -m pip_audit -r requirements.txt --progress-spinner=off`.
6. Run `python -m daedalus.services.release_guard --repo . scan`.
7. Submit a pull request explaining the learning or engineering value and complete
   the safety/custody checklist.

Core neural-network behavior must remain inspectable and NumPy-only. High-level
ML frameworks may be used only in optional comparison tooling and must never
become a runtime requirement of the core engine.
