## What changed

Describe the user-visible outcome and the problem it solves.

## Safety and custody

- [ ] No credentials, private workspaces, datasets, checkpoints, logs, or backup content are included.
- [ ] New network, process, retention, restore, or publishing behavior is documented.
- [ ] Security-sensitive workflow and automation changes use least privilege and pinned actions.

## Verification

- [ ] `python -m ruff check .` passes.
- [ ] `python -m pytest` passes, or the exact limitation is documented below.
- [ ] Release Guard and relevant security/backup checks pass.
- [ ] UI changes were checked at the minimum supported size and with reduced motion.

## Screenshots / notes

Use synthetic or manually redacted data only. Include known limitations and follow-up work.
