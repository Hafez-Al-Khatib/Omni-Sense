"""
DEPRECATED — IEP1/YAMNet embedding extractor.

This script used to either:
  • Load YAMNet from TensorFlow Hub in-process, or
  • POST WAVs to the IEP1 microservice (port 8001)

IEP1 has been decommissioned.  See `archive/iep1/README.md` for the ADR.

The new serving & training path uses the 39-d physics feature vector
produced by `omni/eep/features.py` — equivalent DSP features are
computed by `scripts/extract_dsp_features.py` for offline training.

Running this file now prints a migration hint and exits non-zero so
any stale pipeline script that still invokes it fails loudly instead
of silently producing garbage parquet files.
"""

from __future__ import annotations

import sys
import textwrap


def main() -> int:
    msg = textwrap.dedent(
        """
        [extract_embeddings.py] DECOMMISSIONED.

        The YAMNet/IEP1 embedding pipeline was removed because its 1024-d
        airborne-audio embedding is architecturally incompatible with the
        39-d structure-borne physics feature space used by IEP2/Omni.

        Use instead:
            python scripts/extract_dsp_features.py \\
                --input-dir data/synthesized \\
                --output    data/synthesized/dsp_features.parquet

        Programmatic callers should import:
            from omni.eep.features import extract_features

        If you have an automated pipeline still calling this script,
        update the reference and re-run.  Failing here on purpose.
        """
    ).strip()
    print(msg, file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
