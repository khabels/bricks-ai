# UI Contract

The first UI should expose three main panels.

## 1. Settings

Purpose: manage provider keys and provider roles.

Fields:

- Provider name
- Enabled/disabled
- Model name
- Role: Generator, Validator, Judge
- API key input field
- Key status: present/missing
- Consensus threshold
- Minimum confidence
- Cross-validation required

Keys should be saved to an OS keyring or secret manager, not committed to disk in plain text.

## 2. Training Pack

Purpose: start or pause pack training.

Controls:

- Start Training button
- Pause button
- Resume button
- Pack mode: Manual, Broad, Universal
- Max depth
- Items per theme
- Enabled root domains

Progress indicators:

- Status
- Queue length
- Current job
- Accepted items
- Rejected items
- Trained items

## 3. Engraving Monitor

Purpose: inspect validation and local correction.

Views:

- Engraved cases
- Engraved correlations
- Active critical zones
- Candidate pages
- Promoted pages
- Rejected pages
