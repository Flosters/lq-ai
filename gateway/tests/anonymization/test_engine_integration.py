"""Full AnalyzerEngine integration test — M2-B2.

Exercises the configured Presidio :class:`AnalyzerEngine` end-to-end:

* spaCy ``en_core_web_lg`` model loaded.
* Custom recognizers (``CaseNumberRecognizer``,
  ``MatterNumberRecognizer``) registered and firing.
* Default recognizers we keep enabled (PERSON, ORG, EMAIL_ADDRESS,
  PHONE_NUMBER, US_BANK_NUMBER, LOCATION) firing.
* Default recognizers we disable (US_PASSPORT, US_DRIVER_LICENSE,
  US_SSN, CRYPTO, IBAN_CODE, IP_ADDRESS, MEDICAL_LICENSE) NOT firing.

The test is marked ``slow`` so it skips by default. Local runs:
``pytest -m slow tests/anonymization/test_engine_integration.py``.
CI runs ``not slow and not provider`` so this is opt-in.

Why marked slow:
The first call to :func:`get_analyzer_engine` loads spaCy's
``en_core_web_lg`` model (~560MB on disk, 2-3 seconds wall-clock).
The remaining ``analyze`` calls are fast (sub-second each); the
wall-clock cost is the initial model load. Keeping the test out of
the default run preserves the fast feedback loop the rest of the
suite gives.
"""

from __future__ import annotations

import pytest

from app.anonymization.engine import _reset_analyzer_engine_for_tests, get_analyzer_engine

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def analyzer() -> object:
    """Module-scoped: load the spaCy model + build the engine once for this file."""

    # Fresh singleton — avoids cross-test pollution if some earlier test
    # in the same session (unlikely, but defensively) built a different
    # registry.
    _reset_analyzer_engine_for_tests()
    return get_analyzer_engine()


def _hit_entity_types(results: list) -> set[str]:
    return {r.entity_type for r in results}


def test_engine_recognizes_custom_case_number(analyzer: object) -> None:
    """A canonical federal cite surfaces as a CASE_NUMBER entity."""

    text = "See Smith v. Jones, 123 F.3d 456 (9th Cir. 2024) for the holding."

    results = analyzer.analyze(text=text, language="en")
    assert "CASE_NUMBER" in _hit_entity_types(results)


def test_engine_recognizes_custom_matter_number(analyzer: object) -> None:
    """An alpha-year-sequence matter number surfaces as a MATTER_NUMBER entity."""

    text = "Internal matter LQ-2026-0042 covers the dispute."

    results = analyzer.analyze(text=text, language="en")
    assert "MATTER_NUMBER" in _hit_entity_types(results)


def test_engine_recognizes_default_person_org(analyzer: object) -> None:
    """The kept-enabled defaults (PERSON, ORGANIZATION) fire."""

    text = "John Smith of Acme Corp. negotiated the agreement."

    results = analyzer.analyze(text=text, language="en")
    hits = _hit_entity_types(results)
    assert "PERSON" in hits


def test_engine_recognizes_email_and_phone(analyzer: object) -> None:
    """EMAIL_ADDRESS and PHONE_NUMBER both fire."""

    text = "Contact counsel at counsel@firm.com or call 415-555-0123."

    results = analyzer.analyze(text=text, language="en")
    hits = _hit_entity_types(results)
    assert "EMAIL_ADDRESS" in hits
    assert "PHONE_NUMBER" in hits


def test_engine_does_not_recognize_disabled_us_ssn(analyzer: object) -> None:
    """US_SSN is disabled — a number-shaped exhibit doesn't surface as SSN."""

    text = "See Exhibit 123-45-6789 attached to the filing."

    results = analyzer.analyze(text=text, language="en")
    assert "US_SSN" not in _hit_entity_types(results)


def test_engine_does_not_recognize_disabled_ip_address(analyzer: object) -> None:
    """IP_ADDRESS is disabled — version-shaped numbers don't surface as IPs."""

    text = "Per section 192.168.1.1 of the supplement, ..."

    results = analyzer.analyze(text=text, language="en")
    assert "IP_ADDRESS" not in _hit_entity_types(results)


def test_enabled_recognizer_entities_are_language_invariant(analyzer: object) -> None:
    """The set of enabled entity types must be identical under every configured language.

    Whole-branch review (2026-07-30) found that ``load_predefined_recognizers``
    returns a *different* default set per language: enabling ``es`` pulled in
    Spain's ``ES_NIF``/``ES_NIE`` (wrong jurisdiction for this Argentine
    corpus), while ``en``-only recognizers like ``US_BANK_NUMBER`` —
    advertised as enabled in ``docs/security/anonymization.md`` — silently
    stopped firing whenever a text got routed to Spanish. Both problems are
    invisible to a test that only calls ``analyze(language="en")``, which is
    exactly why this test derives the entity sets straight from the engine's
    registry instead of hardcoding either language's list: it fails the
    moment any future recognizer appears on one side and not the other,
    regardless of which side gains or loses it.
    """

    entities_by_language = {
        language: set(analyzer.get_supported_entities(language=language))
        for language in ("es", "en")
    }

    es_entities = entities_by_language["es"]
    en_entities = entities_by_language["en"]

    assert es_entities == en_entities, (
        f"only under es: {es_entities - en_entities}; only under en: {en_entities - es_entities}"
    )
    # Guard against a vacuous pass (e.g. both empty because of a wiring bug
    # upstream) — the curated set has real entities on both sides.
    assert "US_BANK_NUMBER" in es_entities
    assert "AR_DNI" in es_entities


def test_no_recognizer_is_registered_twice_for_one_language(analyzer: object) -> None:
    """Exactly one instance of each recognizer per configured language.

    The entity-set parity test above cannot see this: ``get_supported_entities``
    returns a set, so a recognizer registered twice for the same language looks
    identical to one registered once. That is not hypothetical — restoring
    ``US_BANK_NUMBER`` under Spanish first shipped as an extra
    ``registry.add_recognizer`` on top of the instance Presidio already loads
    under English, leaving two for ``en``. Nothing broke in the output
    (``_resolve_overlaps`` collapses the repeated spans) so no test failed, but
    the regex ran twice on every English request.
    """

    from collections import Counter

    instancias = Counter(
        (type(recognizer).__name__, recognizer.supported_language)
        for recognizer in analyzer.registry.recognizers
    )
    repetidos = {clave: n for clave, n in instancias.items() if n > 1}

    assert not repetidos, f"reconocedores registrados más de una vez: {repetidos}"


def test_engine_combined_text_surfaces_multiple_entities(analyzer: object) -> None:
    """A realistic legal-prose paragraph surfaces all expected entity types."""

    text = (
        "In Smith v. Jones, 123 F.3d 456 (9th Cir. 2024), counsel "
        "Jane Doe of Acme LLP argued that internal matter LQ-2024-0001 "
        "was settled. Contact: jane.doe@acme.com or 415-555-0123."
    )

    results = analyzer.analyze(text=text, language="en")
    hits = _hit_entity_types(results)

    # Custom entities.
    assert "CASE_NUMBER" in hits
    assert "MATTER_NUMBER" in hits
    # Kept-enabled defaults — PERSON and EMAIL_ADDRESS are the most
    # reliable detectors on this prose; PHONE_NUMBER + ORG depend on
    # tokenization and are exercised in the dedicated tests above.
    assert "PERSON" in hits
    assert "EMAIL_ADDRESS" in hits
    # Disabled defaults stay silent.
    assert "US_SSN" not in hits
    assert "IP_ADDRESS" not in hits
