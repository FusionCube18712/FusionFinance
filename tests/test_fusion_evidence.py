from __future__ import annotations

from dataclasses import FrozenInstanceError

from alpha.verifier.evidence import (
    EvidenceRecord,
    EvidenceSource,
    audit_evidence,
)


DOC_ID = "filing:AMD:2026Q2"
QUOTE = "Revenue was $12.50 billion."


def _source(*, available_at: str = "2026-07-08T20:10:00Z") -> EvidenceSource:
    return EvidenceSource(
        document_id=DOC_ID,
        available_at=available_at,
        text=f"Management discussion. {QUOTE} Cash flow improved.",
        numeric_values={"revenue_billion": 12.5},
    )


def _evidence(
    *,
    document_id: str = DOC_ID,
    quote: str = QUOTE,
    asserted_value: float | None = 12.5,
) -> EvidenceRecord:
    return EvidenceRecord(
        document_id=document_id,
        quoted_text=quote,
        numeric_key="revenue_billion" if asserted_value is not None else "",
        asserted_value=asserted_value,
    )


def test_complete_evidence_is_approval_ready_and_inputs_stay_immutable() -> None:
    raw_numbers = {"revenue_billion": 12.5}
    source = EvidenceSource(
        document_id=DOC_ID,
        available_at="2026-07-08T20:10:00Z",
        text=QUOTE,
        numeric_values=raw_numbers,
    )
    evidence = _evidence(asserted_value=12.5005)
    sources = [source]
    records = [evidence]

    result = audit_evidence(
        records,
        sources,
        thesis_as_of="2026-07-09T19:00:00Z",
        absolute_tolerance=0.001,
    )

    assert result.evidence_valid is True
    assert result.citation_coverage == 1.0
    assert result.numeric_reconciliation == 1.0
    assert result.timestamp_integrity is True
    assert result.errors == ()
    assert (result.valid_citations, result.total_citations) == (1, 1)
    assert (result.reconciled_numbers, result.total_numbers) == (1, 1)
    assert sources == [source]
    assert records == [evidence]

    raw_numbers["revenue_billion"] = 999.0
    assert source.numeric_values["revenue_billion"] == 12.5
    try:
        source.numeric_values["revenue_billion"] = 999.0
    except TypeError:
        pass
    else:
        raise AssertionError("source numeric values must be immutable")
    try:
        evidence.document_id = "changed"
    except FrozenInstanceError:
        pass
    else:
        raise AssertionError("evidence records must be immutable")


def test_tampered_number_fails_numeric_reconciliation() -> None:
    result = audit_evidence(
        [_evidence(asserted_value=13.0)],
        [_source()],
        thesis_as_of="2026-07-09T19:00:00Z",
        absolute_tolerance=0.001,
    )

    assert result.evidence_valid is False
    assert result.citation_coverage == 1.0
    assert result.numeric_reconciliation == 0.0
    assert result.timestamp_integrity is True
    assert result.errors == (f"NUMERIC_MISMATCH {DOC_ID} revenue_billion",)


def test_source_available_after_thesis_timestamp_fails_closed() -> None:
    result = audit_evidence(
        [_evidence()],
        [_source(available_at="2026-07-10T00:00:00Z")],
        thesis_as_of="2026-07-09T19:00:00Z",
    )

    assert result.evidence_valid is False
    assert result.citation_coverage == 1.0
    assert result.numeric_reconciliation == 1.0
    assert result.timestamp_integrity is False
    assert result.errors == (f"SOURCE_AFTER_AS_OF {DOC_ID}",)


def test_unknown_document_fails_all_unverifiable_dimensions() -> None:
    result = audit_evidence(
        [_evidence(document_id="filing:UNKNOWN:2026Q2")],
        [_source()],
        thesis_as_of="2026-07-09T19:00:00Z",
    )

    assert result.evidence_valid is False
    assert result.citation_coverage == 0.0
    assert result.numeric_reconciliation == 0.0
    assert result.timestamp_integrity is False
    assert result.errors == ("UNKNOWN_DOCUMENT filing:UNKNOWN:2026Q2",)


def test_missing_exact_quote_fails_citation_coverage() -> None:
    result = audit_evidence(
        [_evidence(quote="Revenue was $125.00 billion.", asserted_value=None)],
        [_source()],
        thesis_as_of="2026-07-09T19:00:00Z",
    )

    assert result.evidence_valid is False
    assert result.citation_coverage == 0.0
    assert result.numeric_reconciliation == 1.0
    assert result.timestamp_integrity is True
    assert result.errors == (f"QUOTE_NOT_FOUND {DOC_ID}",)


def test_invalid_duplicate_and_empty_document_sets_fail_closed() -> None:
    invalid = EvidenceSource(
        document_id="../.env",
        available_at="2026-07-08T20:10:00Z",
        text=QUOTE,
    )
    duplicate = _source()
    result = audit_evidence(
        [_evidence()],
        [invalid, duplicate, duplicate],
        thesis_as_of="2026-07-09T19:00:00Z",
    )

    assert result.evidence_valid is False
    assert "INVALID_DOCUMENT_ID source[0]" in result.errors
    assert f"DUPLICATE_DOCUMENT_ID {DOC_ID}" in result.errors

    empty = audit_evidence([], [], thesis_as_of="2026-07-09T19:00:00Z")
    assert empty.evidence_valid is False
    assert empty.citation_coverage == 0.0
    assert empty.numeric_reconciliation == 0.0
    assert empty.timestamp_integrity is False
    assert empty.errors == ("NO_EVIDENCE",)
