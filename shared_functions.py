"""
Backward-compatible facade for shared ChromaDB helpers.

Prefer importing from `food_rag.database` in new code.
"""

from food_rag.database import (  # noqa: F401
    create_similarity_search_collection,
    load_food_data,
    perform_filtered_similarity_search,
    perform_similarity_search,
    populate_similarity_collection,
)
