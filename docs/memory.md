# CPU memory controls

Bricks AI V10 adds CPU/RAM controls for local training sessions.

Useful command:

```bash
cargo run -- memory
```

Useful `.env` keys:

```env
BRICKS_AI_MEMORY_MODE=low
BRICKS_AI_GRID_COUNT=4
BRICKS_AI_CASES_PER_GRID=256
BRICKS_AI_MAX_QUEUE_JOBS=64
BRICKS_AI_MAX_CANDIDATE_CACHE=64
BRICKS_AI_MAX_PATH_CACHE=64
```

`low` keeps the trainer compact and limits queue/cache growth. `balanced` is the default. `performance` keeps larger local caches.

Checkpoints are compacted before being written: transient signals, gradients and correlation activity are cleared from the saved checkpoint.
