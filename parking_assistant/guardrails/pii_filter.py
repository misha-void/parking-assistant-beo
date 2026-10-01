"""Guardrails for PII detection and filtering using Microsoft Presidio."""

from typing import List, Dict, Tuple
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig


class PIIGuardrail:
    """
    Detects and filters PII from text using Presidio.
    
    Protects against accidentally exposing sensitive information
    from vector DB retrieval or user inputs.
    """
    
    # PII entities to detect
    DEFAULT_ENTITIES = [
        "PERSON",           # Names
        "EMAIL_ADDRESS",    # Emails
        "PHONE_NUMBER",     # Phone numbers
        "CREDIT_CARD",      # Credit card numbers
        "IBAN_CODE",        # Bank accounts
        "IP_ADDRESS",       # IP addresses
        "LOCATION",         # Specific addresses (we allow general location names)
        "DATE_TIME",        # Don't filter - needed for parking context
        "NRP",              # National registry numbers
    ]
    
    # Entities to actually anonymize (subset of detected)
    ANONYMIZE_ENTITIES = [
        "PERSON",
        "EMAIL_ADDRESS",
        "PHONE_NUMBER",
        "CREDIT_CARD",
        "IBAN_CODE",
        "NRP"
    ]

    # Entities to filter from RETRIEVED CONTEXT only — genuine secrets. Deliberately
    # excludes PHONE_NUMBER/PERSON/LOCATION, which false-positive on parking data
    # (e.g. spot counts like "1800/2500" were being redacted as phone numbers) and
    # on Serbian street/location names.
    CONTEXT_ENTITIES = [
        "EMAIL_ADDRESS",
        "CREDIT_CARD",
        "IBAN_CODE",
    ]
    
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
            entities: List of entity types to detect (uses DEFAULT_ENTITIES if None)
        
        Returns:
            List of detected PII entities with metadata
        """
        if entities is None:
            entities = self.DEFAULT_ENTITIES
        
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
            entities: Entity types to anonymize (uses ANONYMIZE_ENTITIES if None)
        
        Returns:
            Tuple of (anonymized_text, was_modified)
        """
        if entities is None:
            entities = self.ANONYMIZE_ENTITIES
        
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
                "PERSON": OperatorConfig("replace", {"new_value": "[NAME]"}),
                "PHONE_NUMBER": OperatorConfig("replace", {"new_value": "[PHONE]"}),
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
        # Detect only genuine secrets (not numbers/names that collide with parking data)
        pii_entities = self.detect_pii(context, self.CONTEXT_ENTITIES)

        high_conf_entities = [e for e in pii_entities if e["score"] >= threshold]

        if not high_conf_entities:
            return context

        # Anonymize just those entity types
        filtered, _ = self.anonymize_pii(context, self.CONTEXT_ENTITIES)
        return filtered
    
    def validate_user_input(self, user_input: str, allow_entities: List[str] = None) -> Tuple[bool, List[Dict]]:
        """
        Validate user input for unexpected PII.
        
        Used to warn if user accidentally shares sensitive info they shouldn't.
        During reservation, we EXPECT names/car numbers, so those are allowed.
        
        Args:
            user_input: User's message
            allow_entities: Entities that are expected/allowed in this context
        
        Returns:
            Tuple of (is_safe, detected_unexpected_pii)
        """
        if allow_entities is None:
            allow_entities = []
        
        detected = self.detect_pii(user_input)
        
        # Filter out allowed entities
        unexpected = [
            e for e in detected
            if e["entity_type"] not in allow_entities
        ]
        
        is_safe = len(unexpected) == 0
        
        return is_safe, unexpected


# Global instance (lazy-initialized)
_guardrail_instance = None


def get_guardrail() -> PIIGuardrail:
    """Get or create global guardrail instance."""
    global _guardrail_instance
    if _guardrail_instance is None:
        _guardrail_instance = PIIGuardrail()
    return _guardrail_instance
