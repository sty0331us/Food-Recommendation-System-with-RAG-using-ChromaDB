"""ChromaDB operations: load data, create collections, similarity & hybrid search."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils import embedding_functions

from food_rag.config import Settings, get_settings

_client: chromadb.ClientAPI | None = None


def get_chroma_client(settings: Settings | None = None) -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client (singleton)."""
    global _client
    if _client is not None:
        return _client

    settings = settings or get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(settings.chroma_persist_dir))
    return _client


def reset_chroma_client() -> None:
    """Reset the cached client (useful in tests)."""
    global _client
    _client = None


def load_food_data(file_path: str | Path) -> list[dict[str, Any]]:
    """Load and normalize food data from a JSON file."""
    path = Path(file_path)
    try:
        with path.open("r", encoding="utf-8") as file:
            food_data = json.load(file)

        if not isinstance(food_data, list):
            raise ValueError("Food dataset must be a JSON array of objects")

        for i, item in enumerate(food_data):
            if "food_id" not in item:
                item["food_id"] = str(i + 1)
            else:
                item["food_id"] = str(item["food_id"])

            item.setdefault("food_ingredients", [])
            item.setdefault("food_description", "")
            item.setdefault("cuisine_type", "Unknown")
            item.setdefault("food_calories_per_serving", 0)
            item.setdefault("food_health_benefits", "")
            item.setdefault("cooking_method", "")

            if "food_features" in item and isinstance(item["food_features"], dict):
                taste_features = [
                    str(value) for value in item["food_features"].values() if value
                ]
                item["taste_profile"] = ", ".join(taste_features)
            else:
                item.setdefault("taste_profile", "")

        print(f"Successfully loaded {len(food_data)} food items from {path}")
        return food_data

    except Exception as exc:
        print(f"Error loading food data: {exc}")
        return []


def _embedding_function(model_name: str):
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=model_name
    )


def create_similarity_search_collection(
    collection_name: str,
    collection_metadata: dict | None = None,
    settings: Settings | None = None,
) -> Collection:
    """Create (or recreate) a ChromaDB collection with sentence-transformer embeddings."""
    settings = settings or get_settings()
    client = get_chroma_client(settings)

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    metadata = {"hnsw:space": "cosine"}
    if collection_metadata:
        # Chroma metadata values must be str/int/float/bool
        for key, value in collection_metadata.items():
            metadata[key] = value if isinstance(value, (str, int, float, bool)) else str(value)

    return client.create_collection(
        name=collection_name,
        metadata=metadata,
        embedding_function=_embedding_function(settings.embedding_model),
    )


def _build_document_text(food: dict[str, Any]) -> str:
    """Build rich text used for embedding generation."""
    parts = [
        f"Name: {food['food_name']}.",
        f"Description: {food.get('food_description', '')}.",
        f"Ingredients: {', '.join(food.get('food_ingredients', []))}.",
        f"Cuisine: {food.get('cuisine_type', 'Unknown')}.",
        f"Cooking method: {food.get('cooking_method', '')}.",
    ]

    taste_profile = food.get("taste_profile", "")
    if taste_profile:
        parts.append(f"Taste and features: {taste_profile}.")

    health_benefits = food.get("food_health_benefits", "")
    if health_benefits:
        parts.append(f"Health benefits: {health_benefits}.")

    nutrition = food.get("food_nutritional_factors")
    if isinstance(nutrition, dict):
        nutrition_text = ", ".join(f"{k}: {v}" for k, v in nutrition.items())
        parts.append(f"Nutrition: {nutrition_text}.")

    return " ".join(parts)


def populate_similarity_collection(
    collection: Collection,
    food_items: list[dict[str, Any]],
) -> None:
    """Populate a collection with food documents, metadata, and embeddings."""
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
    used_ids: set[str] = set()

    for i, food in enumerate(food_items):
        base_id = str(food.get("food_id", i))
        unique_id = base_id
        counter = 1
        while unique_id in used_ids:
            unique_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(unique_id)

        documents.append(_build_document_text(food))
        ids.append(unique_id)
        metadatas.append(
            {
                "name": food["food_name"],
                "cuisine_type": food.get("cuisine_type", "Unknown"),
                "ingredients": ", ".join(food.get("food_ingredients", [])),
                "calories": int(food.get("food_calories_per_serving", 0) or 0),
                "description": food.get("food_description", ""),
                "cooking_method": food.get("cooking_method", ""),
                "health_benefits": food.get("food_health_benefits", ""),
                "taste_profile": food.get("taste_profile", ""),
            }
        )

    # Batch add for larger datasets
    batch_size = 100
    for start in range(0, len(documents), batch_size):
        end = start + batch_size
        collection.add(
            documents=documents[start:end],
            metadatas=metadatas[start:end],
            ids=ids[start:end],
        )

    print(f"Added {len(food_items)} food items to collection")


def _format_search_results(results: dict[str, Any]) -> list[dict[str, Any]]:
    """Normalize Chroma query results into RAG-friendly dictionaries."""
    if not results or not results.get("ids") or not results["ids"][0]:
        return []

    formatted: list[dict[str, Any]] = []
    for i in range(len(results["ids"][0])):
        meta = results["metadatas"][0][i]
        distance = results["distances"][0][i]
        similarity_score = 1 - distance

        ingredients_raw = meta.get("ingredients", "")
        ingredients = (
            [part.strip() for part in ingredients_raw.split(",") if part.strip()]
            if isinstance(ingredients_raw, str)
            else ingredients_raw
        )

        formatted.append(
            {
                "food_id": results["ids"][0][i],
                "food_name": meta.get("name", ""),
                "food_description": meta.get("description", ""),
                "cuisine_type": meta.get("cuisine_type", "Unknown"),
                "food_calories_per_serving": meta.get("calories", 0),
                "food_ingredients": ingredients,
                "food_health_benefits": meta.get("health_benefits", ""),
                "cooking_method": meta.get("cooking_method", ""),
                "taste_profile": meta.get("taste_profile", ""),
                "similarity_score": similarity_score,
                "distance": distance,
            }
        )

    return formatted


def perform_similarity_search(
    collection: Collection,
    query: str,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Perform vector similarity search and return formatted results."""
    try:
        results = collection.query(query_texts=[query], n_results=n_results)
        return _format_search_results(results)
    except Exception as exc:
        print(f"Error in similarity search: {exc}")
        return []


def perform_filtered_similarity_search(
    collection: Collection,
    query: str,
    cuisine_filter: str | None = None,
    max_calories: int | None = None,
    n_results: int = 5,
) -> list[dict[str, Any]]:
    """Hybrid search: vector similarity + metadata constraints."""
    filters: list[dict[str, Any]] = []
    if cuisine_filter:
        filters.append({"cuisine_type": {"$eq": cuisine_filter}})
    if max_calories is not None:
        filters.append({"calories": {"$lte": int(max_calories)}})

    where_clause: dict[str, Any] | None = None
    if len(filters) == 1:
        where_clause = filters[0]
    elif len(filters) > 1:
        where_clause = {"$and": filters}

    try:
        query_kwargs: dict[str, Any] = {
            "query_texts": [query],
            "n_results": n_results,
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        results = collection.query(**query_kwargs)
        return _format_search_results(results)
    except Exception as exc:
        print(f"Error in filtered search: {exc}")
        return []


def ensure_collection_ready(
    collection_name: str,
    food_items: list[dict[str, Any]],
    collection_metadata: dict | None = None,
    settings: Settings | None = None,
    force_rebuild: bool = False,
) -> Collection:
    """
    Get an existing collection or create and populate a new one.

    When force_rebuild is False and the collection already has documents,
    reuse it to avoid re-embedding on every startup.
    """
    settings = settings or get_settings()
    client = get_chroma_client(settings)

    if not force_rebuild:
        try:
            existing = client.get_collection(
                name=collection_name,
                embedding_function=_embedding_function(settings.embedding_model),
            )
            if existing.count() > 0:
                print(
                    f"Reusing existing collection '{collection_name}' "
                    f"({existing.count()} items)"
                )
                return existing
        except Exception:
            pass

    collection = create_similarity_search_collection(
        collection_name, collection_metadata, settings=settings
    )
    populate_similarity_collection(collection, food_items)
    return collection
