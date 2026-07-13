# Core Infrastructure Boundary

This repository owns only the deterministic media pipeline:

- Bilibili input normalization and subtitle-track retrieval;
- low-resolution fallback video download;
- screenshots sampled no more often than once every 3 seconds;
- OCR, conservative sentence reconstruction, caching, and file output.

Do not add character personality inference, relationship analysis, writing rules,
or WebGAL formatting here. Downstream analyzers must consume the files documented
in `README.md` and must not modify the media pipeline.

Keep the output schema backward-compatible. Add fields instead of changing the
meaning of existing fields, and increment `SCHEMA_VERSION` for incompatible
changes.
