# Contributing to Bricks AI

Thanks for your interest in Bricks AI.

## Principles

- Keep the project experimental but honest.
- Do not claim that Bricks AI is a production LLM trainer.
- Validate AI-generated data before using it for training.
- Preserve the rule: never overwrite validated weights without a competing local candidate.
- Prefer small, well-tested modules.

## Development

```bash
cargo check
cargo test
cargo run -p bricks-ai-app -- demo
```

## Pull requests

A good pull request should include:

- clear motivation
- tests when possible
- docs for new public concepts
- no committed API keys or secrets

## Areas needing help

- provider integrations
- UI implementation
- benchmarking
- data validation
- documentation
- safety review
