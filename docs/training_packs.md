# Training Packs

A training pack is a queue of knowledge jobs.

Bricks AI supports a universal pack:

```text
root domain
-> expand subthemes
-> generate training data
-> validate
-> train
-> engrave
```

The pack can run in several modes:

- `Manual`: fixed themes
- `Broad`: broad roots
- `Universal`: recursive expansion

The queue contains two job kinds:

- `ExpandTheme`
- `GenerateTrainingData`

The trainer exposes prompts that external provider clients can send to AI services.
