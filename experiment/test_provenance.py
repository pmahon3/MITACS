"""Integrity test for the result-provenance envelope (make_result).

Load-bearing properties this guards:
  1. round-trip: a written result loads back with identical body;
  2. tamper-evidence: editing the body after write is DETECTED on load
     (the body_sha256 contract — the analogue of freeze.load_verified);
  3. the two hashes are distinct concerns: changing inputs moves
     inputs_fingerprint but a byte-identical body keeps body_sha256;
  4. the strict integrity policy: a dirty tree is refused for
     CLAIM/METHOD; allow_dirty is rejected for non-INSPECTION grades.

Run::

    python -m experiment.test_provenance
"""
from __future__ import annotations

import tempfile
import unittest.mock as mock
from pathlib import Path

from experiment.provenance import (
    Grade,
    build_header,
    load_verified_result,
    make_result,
)

_BODY = "headline: MAE = 786 MW  MAPE = 4.78%\n  (some table)\n"


def _run(name, fn):
    try:
        fn()
    except AssertionError as e:
        print(f"FAIL {name}: {e}")
        return False
    print(f"PASS {name}")
    return True


def test_roundtrip_and_tamper():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.txt"
        # git_clean patched True so the test is independent of tree state
        with mock.patch("experiment.provenance.git_clean", return_value=True):
            hdr = make_result(
                path=p, grade=Grade.CLAIM, title="t", body=_BODY,
                inputs={"data_fingerprint": "abc"}, seeds={},
            )
        h2, body = load_verified_result(p)
        assert body == _BODY, "body changed across round-trip"
        assert h2["body_sha256"] == hdr["body_sha256"], "header mismatch"
        assert h2["grade"] == "CLAIM-GRADE", "grade not recorded"
        # tamper: edit the body, keep the header -> must be detected
        txt = p.read_text().replace("786 MW", "111 MW")
        p.write_text(txt)
        try:
            load_verified_result(p)
            raise AssertionError("tampered body NOT detected")
        except ValueError as e:
            assert "integrity compromised" in str(e), f"wrong error: {e}"


def test_two_hashes_are_distinct_concerns():
    # same body, different inputs -> body_sha256 stable, fingerprint moves
    h1 = build_header(
        grade=Grade.CLAIM, title="t", inputs={"k": "v1"}, seeds={},
        body=_BODY,
    )
    h2 = build_header(
        grade=Grade.CLAIM, title="t", inputs={"k": "v2"}, seeds={},
        body=_BODY,
    )
    assert h1["body_sha256"] == h2["body_sha256"], (
        "body hash must not depend on inputs"
    )
    assert h1["inputs_fingerprint"] != h2["inputs_fingerprint"], (
        "inputs fingerprint must change when inputs change"
    )


def test_strict_dirty_policy():
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "r.txt"
        with mock.patch(
            "experiment.provenance.git_clean", return_value=False
        ):
            try:
                make_result(
                    path=p, grade=Grade.CLAIM, title="t", body=_BODY,
                )
                raise AssertionError("dirty tree NOT refused for CLAIM")
            except SystemExit as e:
                assert "dirty tree" in str(e), f"wrong refusal: {e}"
            # allow_dirty is rejected for non-INSPECTION grades
            try:
                make_result(
                    path=p, grade=Grade.METHOD, title="t", body=_BODY,
                    allow_dirty=True,
                )
                raise AssertionError("allow_dirty accepted for METHOD")
            except ValueError as e:
                assert "INSPECTION-ONLY" in str(e), f"wrong error: {e}"
            # INSPECTION + allow_dirty is permitted (scratch capture)
            make_result(
                path=p, grade=Grade.INSPECTION, title="t", body=_BODY,
                allow_dirty=True,
            )
            assert p.exists(), "INSPECTION allow_dirty should write"


def main():
    ok = True
    ok &= _run("roundtrip_and_tamper", test_roundtrip_and_tamper)
    ok &= _run(
        "two_hashes_distinct", test_two_hashes_are_distinct_concerns
    )
    ok &= _run("strict_dirty_policy", test_strict_dirty_policy)
    print("ALL PASS" if ok else "FAILURES")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
