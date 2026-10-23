"""Unifying demonstration for *The Language of Thought Is a Design Choice*.

Importing this package pins every library cache inside the repository, so
``import ltd`` must come before ``torch``/``transformers``/``vllm``.
"""

from . import env as env  # noqa: F401  (side effect: configures caches)

__version__ = "0.1.0"
