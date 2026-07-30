"""Anonymizer façade + module-level AnalyzerEngine — M2-A3 → M2-B2.

The :class:`Anonymizer` is the entry point the gateway middleware
(M2-B3) will use to pseudonymize an outbound prompt and rehydrate the
returning response. M2-A3 shipped the class shape; M2-B2 (this task)
adds:

* :func:`get_analyzer_engine` — module-level singleton that
  constructs a Presidio :class:`AnalyzerEngine`, registers the
  custom legal recognizers (``CaseNumberRecognizer``,
  ``MatterNumberRecognizer``), and disables the noisy default
  recognizers that don't pay off on legal-document corpus.
* :data:`ENABLED_DEFAULT_RECOGNIZERS` and
  :data:`DISABLED_DEFAULT_RECOGNIZERS` — the recognizer-list
  configuration the singleton applies. Documented inline so the
  rationale is alongside the code.

M2-B3 will wire :meth:`Anonymizer.pseudonymize` and
:meth:`Anonymizer.rehydrate` to call the engine + the Presidio
:class:`AnonymizerEngine`. Today those methods still raise
:class:`NotImplementedError` because the request/response middleware
path isn't built yet.

Why a module-level singleton?
-----------------------------

Constructing an :class:`AnalyzerEngine` loads a spaCy model per
configured language (``en_core_web_lg`` is ~560MB on disk alone;
see "Bilingual" below for the per-language breakdown). Doing that
per-request would dominate gateway latency. The
middleware allocates one mapper per request (in-process, drops on
response) but **reuses the analyzer** across requests. Same pattern
Presidio's own examples and FastAPI integrations follow.

The singleton is lazy: it's only constructed on first call. The
test suite that just exercises the custom recognizers in isolation
(via ``recognizer.analyze(...)`` directly) never triggers it.

Bilingual since the anonimización-es-y-prueba-con-contratos plan
--------------------------------------------------------------------

The corpus is Argentine contracts, not English briefs, so the engine
now loads one spaCy model per configured language (``app.anonymization
.languages.SPACY_MODELS``) and caches one ``AnalyzerEngine`` per
distinct ``languages`` tuple (``_analyzer_singletons``, keyed by
tuple — "singleton" above still holds per language set). Each text is
analyzed **once**, against the single language
:func:`app.anonymization.language_detect.detect_language` picks — not
against every configured language with the findings merged. That
union design was tried and measured worse: ``_resolve_overlaps`` keeps
the longest span, so English false positives over Spanish prose beat
the clean Spanish findings. See :meth:`Anonymizer.pseudonymize_into`
for the detail.

The custom pattern recognizers (email, phone, ``CASE_NUMBER``,
``MATTER_NUMBER``, and the Argentine ``ArTaxIdRecognizer`` /
``ArBankRecognizer`` / ``ArDniRecognizer``) are registered once per
configured language regardless — they're regex, not NER, so a
detector mistake never costs an identifier.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, cast

from app.anonymization.language_detect import detect_language
from app.anonymization.languages import DEFAULT_LANGUAGES, SPACY_MODELS
from app.anonymization.mapper import PseudonymMapper
from app.anonymization.recognizers.ar_bank import ArBankRecognizer
from app.anonymization.recognizers.ar_dni import ArDniRecognizer
from app.anonymization.recognizers.ar_tax_id import ArTaxIdRecognizer
from app.anonymization.recognizers.case_number import CaseNumberRecognizer
from app.anonymization.recognizers.matter_number import MatterNumberRecognizer

if TYPE_CHECKING:
    from presidio_analyzer import AnalyzerEngine


class _AnalyzerProtocol(Protocol):
    """Subset of :class:`presidio_analyzer.AnalyzerEngine` we depend on.

    Lets :class:`Anonymizer` accept either the real Presidio engine
    (the production path) or a test double in unit tests, without
    importing Presidio just for type-checking. The real engine's
    ``analyze`` returns ``list[RecognizerResult]``; we only read
    ``entity_type``, ``start``, ``end`` (and ``score`` for overlap
    tie-breaks), which both shapes expose.
    """

    def analyze(self, *, text: str, language: str = "en") -> list[Any]: ...


# Default-recognizer configuration for legal-document corpus.
#
# **Enabled** — these recognizers pay off on legal prose; the
# false-positive rate is acceptable and the entities they catch are
# the ones in-house lawyers actually want pseudonymized:
#
# * ``PERSON`` — names of parties, judges, counsel, witnesses.
# * ``ORG`` — corporate entities, firms, agencies.
# * ``EMAIL_ADDRESS`` — counsel email, party email.
# * ``PHONE_NUMBER`` — contact numbers in correspondence.
# * ``US_BANK_NUMBER`` — bank account numbers that show up in
#   settlement statements, escrow docs. Surfaces under Presidio's
#   built-in ``US_BANK_NUMBER`` entity type.
# * ``LOCATION`` — addresses, courthouses, jurisdictions. Mapped to
#   ``ADDRESS`` in the pseudonym domain to match the operator's
#   mental model.
# * Custom entities from this task — ``CASE_NUMBER``,
#   ``MATTER_NUMBER``.
ENABLED_DEFAULT_RECOGNIZERS: tuple[str, ...] = (
    "PERSON",
    "ORGANIZATION",
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "US_BANK_NUMBER",
    "LOCATION",
)

# **Disabled** — these recognizers ship in Presidio's default set
# but produce a high false-positive rate on legal corpus, or cover
# entity types that are irrelevant for in-house legal work. We
# remove them from the analyzer so they don't fire even when an
# operator's text accidentally pattern-matches:
#
# * ``US_PASSPORT`` / ``US_DRIVER_LICENSE`` / ``US_SSN`` — high
#   false-positive rate in contract numbers, dates, and exhibit
#   indexes. The downside risk of redacting "Exhibit A-123-45-6789"
#   as an SSN outweighs the small probability of an actual SSN
#   appearing in a brief.
# * ``CRYPTO`` — irrelevant for legal corpus; the patterns
#   (Bitcoin/Ethereum addresses) collide with random hex strings.
# * ``IBAN_CODE`` — US-centric deployments rarely see them; when
#   they do, the bank-number recognizer covers the use case.
# * ``IP_ADDRESS`` — incidental in evidence logs but extremely
#   high false-positive rate against version numbers, page
#   references, and dotted numeric identifiers.
# * ``MEDICAL_LICENSE`` — niche to healthcare practice areas; the
#   shape collides with case numbers in unrelated corpora.
# * ``EsNifRecognizer`` / ``EsNieRecognizer`` — Spain's tax ID and
#   foreigner ID. Wrong jurisdiction for an Argentine corpus; these
#   only exist in the registry because Presidio ships a different
#   predefined set per language and ``es`` happens to include them
#   (see the whole-branch review note below).
# * ``UsItinRecognizer`` — US taxpayer ID. Not in the advertised
#   entity set for this corpus; same as above, an artifact of the
#   ``en`` predefined set rather than a deliberate inclusion.
# * ``NhsRecognizer`` — UK National Health Service number. Wrong
#   jurisdiction; same artifact of the ``en`` predefined set.
#
# Operators whose corpus benefits from these (e.g. a healthcare
# practice that needs ``MEDICAL_LICENSE``) re-enable per-recognizer
# in their deployment config; see ``docs/security/anonymization.md``.
#
# Why these last four are here and not just "not registered": Presidio's
# ``load_predefined_recognizers`` returns a *different* default set per
# language — ``es`` pulls in Spain's NIF/NIE, ``en`` pulls in US ITIN and
# the UK NHS number, neither side sees the other's extras. Left alone,
# enabling a language would silently add or remove entity types as a side
# effect of routing, not a deliberate choice. Disabling them by name here
# makes the enabled set language-invariant: whichever language(s) are
# configured, the same recognizers exist. (Whole-branch review, 2026-07-30;
# see also the ``UsBankRecognizer`` registration below, which fixes the
# mirror-image problem — an entity advertised in the docs that stopped
# firing under Spanish routing.)
DISABLED_DEFAULT_RECOGNIZERS: tuple[str, ...] = (
    "UsPassportRecognizer",
    "UsLicenseRecognizer",
    "UsSsnRecognizer",
    "CryptoRecognizer",
    "IbanRecognizer",
    "IpRecognizer",
    "MedicalLicenseRecognizer",
    "EsNifRecognizer",
    "EsNieRecognizer",
    "UsItinRecognizer",
    "NhsRecognizer",
)


_analyzer_singletons: dict[tuple[str, ...], AnalyzerEngine] = {}


def get_analyzer_engine(
    languages: tuple[str, ...] = DEFAULT_LANGUAGES,
) -> AnalyzerEngine:
    """Return a configured :class:`AnalyzerEngine`, constructing once per language set.

    First call for a given ``languages`` tuple builds the NLP engine (one
    spaCy model per language), loads the predefined recognizers for those
    languages, drops the disabled defaults, and registers each custom
    recognizer once per language. Subsequent calls with the same tuple return
    the cached instance — ``analyze`` is read-only and thread-safe.

    Three things have to agree on the language list or Presidio raises
    ``"Misconfigured engine"``: the NLP engine's models, the registry's
    ``supported_languages``, and the engine's own. That's why this is one
    function and not three call sites.

    Registering the custom recognizers per language is load-bearing and
    silent when forgotten: ``PatternRecognizer`` defaults to
    ``supported_language="en"``, so a recognizer registered once would stop
    firing the moment we analyze in Spanish — no error, just fewer hits.
    """

    cached = _analyzer_singletons.get(languages)
    if cached is not None:
        return cached

    from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
    from presidio_analyzer.nlp_engine import NlpEngineProvider
    from presidio_analyzer.predefined_recognizers import UsBankRecognizer

    nlp_engine = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": language, "model_name": SPACY_MODELS[language]}
                for language in languages
            ],
        }
    ).create_engine()

    registry = RecognizerRegistry(supported_languages=list(languages))
    registry.load_predefined_recognizers(languages=list(languages), nlp_engine=nlp_engine)

    # Remove the noisy default recognizers (see
    # DISABLED_DEFAULT_RECOGNIZERS above for the per-name rationale).
    registry.recognizers = [
        r for r in registry.recognizers if type(r).__name__ not in DISABLED_DEFAULT_RECOGNIZERS
    ]

    # One instance per language. See the docstring: skipping the loop is a
    # silent failure, not a loud one.
    for language in languages:
        registry.add_recognizer(CaseNumberRecognizer(supported_language=language))
        registry.add_recognizer(MatterNumberRecognizer(supported_language=language))
        registry.add_recognizer(ArTaxIdRecognizer(supported_language=language))
        registry.add_recognizer(ArBankRecognizer(supported_language=language))
        registry.add_recognizer(ArDniRecognizer(supported_language=language))
        # ``UsBankRecognizer`` is a pure pattern recognizer — its regex
        # doesn't care what language the surrounding prose is in, so its
        # relevance doesn't depend on language routing either. Presidio's
        # predefined set only ships it under ``en``, though, which made the
        # ``US_BANK_NUMBER`` entity documented in
        # ``docs/security/anonymization.md`` silently stop firing whenever a
        # text got routed to Spanish — a non-Argentine bank account number
        # in Spanish prose would reach the provider in cleartext. Registering
        # it explicitly per language, the same way the custom recognizers
        # above are, closes that gap.
        registry.add_recognizer(UsBankRecognizer(supported_language=language))

    engine = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=list(languages),
    )
    _analyzer_singletons[languages] = engine
    return engine


def _reset_analyzer_engine_for_tests() -> None:
    """Drop every cached singleton. Tests use this to start from a clean state."""

    _analyzer_singletons.clear()


@dataclass(slots=True)
class AnonymizationResult:
    """Outcome of a pseudonymization pass.

    ``text`` is the substituted text the gateway forwards to the
    provider. ``mapper`` carries the assignments so the response path
    can rehydrate originals via :meth:`PseudonymMapper.reverse`.
    """

    text: str
    mapper: PseudonymMapper


class Anonymizer:
    """Pseudonymize entities in outbound text; rehydrate on the response path.

    Instances are lightweight and stateless beyond the (optionally
    injected) analyzer. The middleware allocates one Anonymizer +
    one :class:`PseudonymMapper` per request; the analyzer dependency
    is the module-level singleton (``get_analyzer_engine``) by default
    so spaCy stays loaded across requests, but tests inject a stub to
    keep the fast-feedback path off the spaCy model.

    Two entry points:

    * :meth:`pseudonymize_into` — extends an existing mapper with
      substitutions from ``text``. The middleware uses this so the
      same name appearing across multiple messages resolves to the
      same pseudonym.
    * :meth:`pseudonymize` — one-shot convenience that wraps a fresh
      mapper in an :class:`AnonymizationResult`. Useful in tests and
      single-text callers; the middleware does NOT use this.
    """

    def __init__(
        self,
        analyzer: _AnalyzerProtocol | None = None,
        languages: tuple[str, ...] = DEFAULT_LANGUAGES,
    ) -> None:
        """Inject an analyzer or fall back to the module singleton lazily.

        Passing ``analyzer=None`` (the default) defers the analyzer lookup to
        the first ``pseudonymize_into`` call — construction never triggers a
        spaCy load on its own. ``languages`` is both the set the engine loads
        and the set each text is analyzed against.
        """

        self._analyzer = analyzer
        self._languages = languages

    def _resolve_analyzer(self) -> _AnalyzerProtocol:
        analyzer = self._analyzer
        if analyzer is None:
            # ``AnalyzerEngine`` (the real Presidio type) satisfies
            # ``_AnalyzerProtocol`` structurally — both expose
            # ``analyze(text, language)`` returning a list. mypy can't
            # verify that because Presidio's types are untyped at the
            # third-party boundary, so we cast at the import edge.
            analyzer = cast(_AnalyzerProtocol, get_analyzer_engine(self._languages))
            self._analyzer = analyzer
        return analyzer

    @property
    def languages(self) -> tuple[str, ...]:
        """Idiomas configurados, para que el middleware detecte una vez por request."""

        return self._languages

    def pseudonymize(self, text: str) -> AnonymizationResult:
        """One-shot: pseudonymize ``text`` against a fresh mapper.

        Returns an :class:`AnonymizationResult` carrying the substituted
        text + the freshly populated mapper. The middleware does NOT
        use this — it allocates one mapper per request and threads it
        through :meth:`pseudonymize_into` for each message — but
        single-text callers (tests, one-off rehydration scripts) get a
        clean façade.
        """

        mapper = PseudonymMapper()
        substituted = self.pseudonymize_into(text, mapper)
        return AnonymizationResult(text=substituted, mapper=mapper)

    def pseudonymize_into(
        self, text: str, mapper: PseudonymMapper, *, language: str | None = None
    ) -> str:
        """Extend ``mapper`` with substitutions from ``text``; return the result.

        Walks the analyzer's spans, resolves overlapping detections to
        the longer span (ties broken by score), then substitutes
        right-to-left so earlier offsets stay valid. Calling
        :meth:`PseudonymMapper.assign` for an already-known
        ``(entity_type, original)`` reuses the prior pseudonym, so the
        same name across multiple ``pseudonymize_into`` calls on the
        same mapper resolves to the same pseudonym.

        Empty text short-circuits; the analyzer is never called.

        ``language`` lo pasa el middleware, que ve el request entero y detecta
        una sola vez (ver ``_detect_request_language``). Cuando viene ``None``
        —tests y llamadores de un solo texto— se detecta sobre este texto.
        """

        if not text:
            return text

        analyzer = self._resolve_analyzer()
        if language is None:
            language = detect_language(text, candidates=self._languages)
        # Presidio no detecta idioma, así que hay que elegirlo. Se elige uno y
        # se analiza UNA vez — no se unen los idiomas.
        #
        # La unión se probó y se descartó con medición (2026-07-30): analizar
        # español con el modelo inglés produce ocho falsos positivos
        # destructivos (``'por'`` y ``'en adelante'`` como ORGANIZATION,
        # ``'se celebra el presente'`` como PERSON), y ``_resolve_overlaps``
        # elige el span más largo, así que la basura del inglés le gana a los
        # hallazgos limpios del español. Unir salía peor que elegir bien.
        #
        # El detector está sesgado al español porque los costos de errar son
        # asimétricos ~8 a 1; ver ``language_detect.py``. Y sólo decide la
        # mitad NER: los reconocedores de patrón están registrados bajo todos
        # los idiomas, así que un CUIT o un DNI se detecta con independencia
        # de lo que el detector haya elegido.
        results = analyzer.analyze(text=text, language=language)
        spans = _resolve_overlaps(results)

        # Two-pass substitution. Pass 1 walks spans left-to-right and
        # calls ``mapper.assign`` so the per-entity-type counter
        # increments in *reading* order (``PERSON_0001`` is the first
        # name in the text, not the last). Pass 2 splices substitutions
        # in right-to-left order so earlier ``(start, end)`` offsets
        # stay valid as the text length changes around each splice.
        ordered = sorted(spans, key=lambda s: s.start)
        pseudonyms: list[tuple[Any, str]] = [
            (span, mapper.assign(span.entity_type, text[span.start : span.end])) for span in ordered
        ]

        out = text
        for span, pseudonym in reversed(pseudonyms):
            out = out[: span.start] + pseudonym + out[span.end :]
        return out

    def rehydrate(self, text: str, mapper: PseudonymMapper, *, json_safe: bool = False) -> str:
        """Walk pseudonyms in ``text`` and substitute originals.

        One pass over ``mapper.reverse()`` items, ``str.replace`` for each
        pseudonym ordered by descending length. The ordering is load-bearing:
        without it a shorter pseudonym (``PERSON_0001``) would
        match-and-replace inside a longer one (``PERSON_00010``) and mangle
        the output.

        ``json_safe=True`` is for responses whose ``content`` is a serialized
        JSON document (any request carrying ``response_format``). There the
        pseudonym sits *inside* a JSON string literal, so splicing a raw
        original that contains ``"``, ``\\`` or a newline produces a document
        that no longer parses — and a multi-line address block, which this
        layer explicitly supports detecting, is exactly that case. In that
        mode each original is escaped with the JSON string rules before
        substitution, so ``json.loads`` on the result yields the original
        byte-for-byte.

        Empty mapper, empty text, and text containing no pseudonyms all
        return cleanly (an empty ``reverse()`` table makes the loop a no-op).
        """

        if not text:
            return text
        for pseudonym, original in sorted(
            mapper.reverse().items(), key=lambda kv: len(kv[0]), reverse=True
        ):
            replacement = json.dumps(original)[1:-1] if json_safe else original
            text = text.replace(pseudonym, replacement)
        return text


def _resolve_overlaps(results: list[Any]) -> list[Any]:
    """Collapse overlapping analyzer spans to one per region.

    Presidio's :class:`AnalyzerEngine` returns every recognizer's hit;
    two recognizers detecting the same span (e.g. ``PERSON`` and a
    false-positive ``US_BANK_NUMBER`` on ``John Smith``) surface as
    two results. The substitution loop must see one span per region or
    it will try to splice inside an already-substituted pseudonym.

    Resolution: sort by ``(span_length, score)`` descending and walk;
    for each span, drop any later span whose ``[start, end)`` overlaps
    one already kept. Longest wins; same length → higher score wins.
    """

    if not results:
        return []
    ordered = sorted(
        results,
        key=lambda r: (r.end - r.start, getattr(r, "score", 0.0)),
        reverse=True,
    )
    kept: list[Any] = []
    for span in ordered:
        if any(_overlaps(span, k) for k in kept):
            continue
        kept.append(span)
    return kept


def _overlaps(a: Any, b: Any) -> bool:
    """True iff ``[a.start, a.end)`` and ``[b.start, b.end)`` share any char."""

    return bool(a.start < b.end and b.start < a.end)
