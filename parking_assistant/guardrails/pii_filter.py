"""Guardrails for PII detection and filtering using Microsoft Presidio."""

from typing import Dict, List, Tuple

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PIIGuardrail:
    """
    Detects and filters PII from text using Presidio.

    Protects against exposing secrets from vector DB retrieval or in LLM output.
    """

    # Deliberately narrow: only genuine secrets. PERSON, PHONE_NUMBER and LOCATION are
    # excluded because they false-positive on Belgrade street names, on the user's own
    # booking details (name, car plate) which the reservation flow echoes back, and on
    # parking data (spot counts like "1800/2500" were once redacted as phone numbers).
    SECRET_ENTITIES = ["EMAIL_ADDRESS", "CREDIT_CARD", "IBAN_CODE", "IP_ADDRESS"]

    def __init__(self, language: str = "en"):
        """
        Initialize Presidio analyzer and anonymizer.

        Args:
            language: Language code (en, sr for Serbian if available)
        """
        # Create NLP engine
        nlp_configuration = {
            "nlp_engine_name": "spacy",
            "models": [{"lang_code": language, "model_name": "en_core_web_sm"}]
        }

        provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
        nlp_engine = provider.create_engine()

        # Create analyzer
        self.analyzer = AnalyzerEngine(
            nlp_engine=nlp_engine,
            supported_languages=[language]
        )

        # Create anonymizer
        self.anonymizer = AnonymizerEngine()

        self.language = language

    def detect_pii(self, text: str, entities: List[str] = None) -> List[Dict]:
        """
        Detect PII entities in text.

        Args:
            text: Text to analyze
            entities: List of entity types to detect (uses SECRET_ENTITIES if None)

        Returns:
            List of detected PII entities with metadata
        """
        if entities is None:
            entities = self.SECRET_ENTITIES

        results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language=self.language
        )

        return [
            {
                "entity_type": r.entity_type,
                "start": r.start,
                "end": r.end,
                "score": r.score,
                "text": text[r.start:r.end]
            }
            for r in results
        ]

    def anonymize_pii(self, text: str, entities: List[str] = None) -> Tuple[str, bool]:
        """
        Anonymize PII in text by replacing with placeholders.

        Args:
            text: Text to anonymize
            entities: Entity types to anonymize (uses SECRET_ENTITIES if None)

        Returns:
            Tuple of (anonymized_text, was_modified)
        """
        if entities is None:
            entities = self.SECRET_ENTITIES

        # Analyze
        results = self.analyzer.analyze(
            text=text,
            entities=entities,
            language=self.language
        )

        if not results:
            return text, False

        # Anonymize
        anonymized = self.anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators={
                "DEFAULT": OperatorConfig("replace", {"new_value": "[REDACTED]"}),
                "EMAIL_ADDRESS": OperatorConfig("replace", {"new_value": "[EMAIL]"}),
            }
        )

        return anonymized.text, True

    def filter_retrieved_context(self, context: str, threshold: float = 0.5) -> str:
        """
        Filter PII from retrieved context before sending to LLM.

        Aggressively filters to prevent leaking sensitive data from vector DB.

        Args:
            context: Retrieved context text
            threshold: Confidence threshold for filtering (lower = more aggressive)

        Returns:
            Filtered context
        """
        # Detect only genuine secrets
        pii_entities = self.detect_pii(context)

        high_conf_entities = [e for e in pii_entities if e["score"] >= threshold]

        if not high_conf_entities:
            return context

        # Anonymize just those entity types
        filtered, _ = self.anonymize_pii(context)
        return filtered

    def filter_llm_output(self, text: str) -> str:
        """Redact secrets (see SECRET_ENTITIES) from the LLM response before the user sees it."""
        filtered, _ = self.anonymize_pii(text)
        return filtered


# Global instance (lazy-initialized)
_guardrail_instance = None


def get_guardrail() -> PIIGuardrail:
    """Get or create global guardrail instance."""
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = PIIGuardrail()
    return _guardrail_instance
