# model_registry.py
# ---------------------------------------------------------------
# Single source of truth for all shared model instances.
# Every module imports from here instead of instantiating its own.
#
# Lifecycle:
#   1. main.py calls register("embedding_model", model) during
#      load_ai_models() so the model is already warm before first use.
#   2. Any module that needs the model calls get_embedding_model().
#      If main.py already registered it, the cached instance is
#      returned immediately — zero extra load time, zero extra RAM.
#   3. If a module is used standalone (e.g. KnowledgeBase_training
#      run directly), get_embedding_model() lazy-loads on first call.
# ---------------------------------------------------------------

_registry: dict = {}


def register(key: str, instance) -> None:
    """
    Pre-populate the registry with an already-loaded instance.
    Called by main.py during startup so other modules never pay
    the cold-load cost.
    """
    _registry[key] = instance


def get_embedding_model():
    """
    Returns the shared all-MiniLM-L6-v2 SentenceTransformer instance.
    Lazy-loads on first call if main.py hasn't pre-populated it yet.
    """
    if "embedding_model" not in _registry:
        from sentence_transformers import SentenceTransformer
        _registry["embedding_model"] = SentenceTransformer("all-MiniLM-L6-v2")
    return _registry["embedding_model"]


def get_nlp():
    """
    Returns the shared spaCy en_core_web_sm instance.
    Lazy-loads on first call.

    Install once: python -m spacy download en_core_web_sm
    """
    if "nlp" not in _registry:
        try:
            import spacy
            _registry["nlp"] = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )
    return _registry["nlp"]