"""Audit log: chain validity, signature presence, tampering detection."""
import pytest

from omni.audit import log as audit


def _fresh_chain():
    """Patch in a clean chain for each test."""
    old = audit.CHAIN[:]
    audit.CHAIN.clear()
    yield
    audit.CHAIN.clear()
    audit.CHAIN.extend(old)


@pytest.fixture(autouse=True)
def isolated_chain():
    saved = list(audit.CHAIN)
    audit.CHAIN.clear()
    yield
    audit.CHAIN.clear()
    audit.CHAIN.extend(saved)


def test_empty_chain_is_valid():
    ok, bad = audit.verify_chain()
    assert ok is True
    assert bad is None


def test_single_record_valid():
    audit._record("test", "test_action", "resource", "id-1", {"key": "val"})
    ok, bad = audit.verify_chain()
    assert ok is True


def test_multi_record_chain_valid():
    for i in range(10):
        audit._record("actor", "action", "type", f"id-{i}", {"i": i})
    ok, bad = audit.verify_chain()
    assert ok is True
    assert len(audit.CHAIN) == 10


def test_tampered_hash_detected():
    for i in range(5):
        audit._record("actor", "action", "type", f"id-{i}", {"i": i})
    # Tamper with the middle record's prev_hash link
    audit.CHAIN[2] = audit.CHAIN[2].model_copy(
        update={"prev_hash": "0" * 64}
    )
    ok, bad = audit.verify_chain()
    assert ok is False
    assert bad == 2


def test_each_record_has_signature():
    audit._record("actor", "action", "type", "id-sig", {"x": 1})
    ev = audit.CHAIN[-1]
    assert len(ev.signature_ed25519) > 10
    assert len(ev.payload_hash_sha256) == 64


def test_chain_grows_monotonically():
    for i in range(3):
        audit._record("a", "b", "c", str(i), {})
    assert len(audit.CHAIN) == 3
    hashes = [e.payload_hash_sha256 for e in audit.CHAIN]
    assert len(set(hashes)) == 3  # all unique
