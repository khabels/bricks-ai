# Engraved Weights

An engraved weight is a protected weight.

It means:

```text
this weight has been validated enough to preserve it
```

Engraving does not mean eternal truth. It means the model will not casually overwrite the value.

If drift occurs, Bricks AI forks a local candidate page instead of modifying the engraved weight directly.

```text
engraved zone
-> drift
-> candidate page
-> local training
-> validation
-> promotion or rejection
```
