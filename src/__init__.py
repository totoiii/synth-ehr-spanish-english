"""
Fine-tuning and generation modules for synthetic clinical text generation
using language models and the CARES dataset.
"""

# Prefer the modules that exist in this repo.
# Keep legacy names optional to avoid import-time failures.
try:
    from .trainer_unsloth import UnslothFineTuner as FineTuner  # type: ignore
except Exception:  # pragma: no cover
    FineTuner = None  # type: ignore

try:
    from .generator_unsloth import TextGenerator  # type: ignore
except Exception:  # pragma: no cover
    TextGenerator = None  # type: ignore

__all__ = [name for name in ["FineTuner", "TextGenerator"] if globals().get(name) is not None]
__version__ = "1.0.0"
