# Architecture

## Core modules

The current prototype is implemented inside `bricks-ai-core`.

Main entities:

- `RawAITrainer`
- `NodeId`
- `Case`
- `GridPage`
- `Grid`
- `Correlation`
- `PreferredPath`
- `CriticalZone`
- `TrainingPack`
- `TeacherSettings`

## Training cycle

```text
reset temporary state
-> inject inputs
-> forward propagation
-> compute loss
-> seed error
-> backward error propagation
-> apply gradients
-> reinforce used links
-> update confidence
-> score preferred paths
-> optionally engrave or fork candidates
```

## Local revalidation

When a zone drifts:

1. Mark the cell as critical.
2. Fork a candidate page from the official page.
3. Train only a local radius of cells.
4. Compare old loss and new loss.
5. Promote the candidate only if it clearly wins.
6. Otherwise retry with a new candidate page.
