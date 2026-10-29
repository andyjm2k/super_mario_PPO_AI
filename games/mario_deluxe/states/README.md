# Multi-level start states
Drop additional PyBoy `.state` snapshots here (for example mid-level or world 2+ starts).
Training randomly samples from `ROMs/*.state` and `ROMs/states/*.state` so the agent
generalizes across stages instead of overfitting one spawn.
