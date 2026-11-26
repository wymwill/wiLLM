# Architecture Decision Records

Short notes on decisions that would be expensive to change later. One file per
decision, numbered, never rewritten. A later ADR can supersede an earlier one.

Template:

```
# NNNN Title

Status: accepted | superseded by NNNN
Date: YYYY-MM-DD

## Context
What problem forced a decision.

## Decision
What we chose.

## Consequences
What gets easier, what gets harder, what we will revisit.
```

| ADR | Title |
| --- | --- |
| [0001](0001-one-library-per-module.md) | One CMake library per module, one way dependencies |
| [0002](0002-tensor-view.md) | Non owning tensor view with runtime dtype and element strides |
| [0003](0003-error-handling.md) | Exceptions at boundaries, no exceptions in hot loops |
| [0004](0004-doctest-via-fetchcontent.md) | doctest via FetchContent, header only |
