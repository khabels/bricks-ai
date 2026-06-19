# Bricks AI Runtime

The runtime crate reads `engraved_model.json` and interprets engraved cases and correlations without needing the trainer checkpoint.

Use:

```bash
cargo run -- inspect-model --model engraved_model.json
cargo run -- predict --model engraved_model.json --input "Bricks AI probe"
cargo run -- export-runtime-model --model engraved_model.json --out runtime_model.json
```

`bricks_ai_checkpoint.bin` is for resuming training. `engraved_model.json` or `runtime_model.json` is for parallel applications that need to interpret the model.
