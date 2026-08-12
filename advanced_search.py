"""Advanced hybrid search CLI: vector similarity + metadata filters."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Any

from food_rag.config import get_settings
from food_rag.database import (
    ensure_collection_ready,
    load_food_data,
    perform_filtered_similarity_search,
    perform_similarity_search,
)


# Must match cuisine_type values in data/FoodDataSet.json
CUISINE_HINTS = {
    "middle eastern": "Middle Eastern",
    "latin american": "Latin American",
    "italian": "Italian",
    "american": "American",
    "australian": "Australian",
    "british": "British",
    "canadian": "Canadian",
    "chinese": "Chinese",
    "french": "French",
    "german": "German",
    "greek": "Greek",
    "indian": "Indian",
    "international": "International",
    "japanese": "Japanese",
    "korean": "Korean",
    "mexican": "Mexican",
    "southern": "Southern",
    "spanish": "Spanish",
    "thai": "Thai",
    "universal": "Universal",
}


def parse_natural_filters(query: str) -> tuple[str, str | None, int | None]:
    """
    Extract optional cuisine / calorie constraints from free text.

    Returns (cleaned_query, cuisine_filter, max_calories).
    """
    cuisine_filter = None
    max_calories = None
    cleaned = query

    calorie_match = re.search(
        r"(?:under|below|less than|max(?:imum)?)\s+(\d+)\s*(?:cal(?:ories)?)?",
        query,
        flags=re.IGNORECASE,
    )
    if calorie_match:
        max_calories = int(calorie_match.group(1))
        cleaned = cleaned.replace(calorie_match.group(0), " ").strip()

    lowered = cleaned.lower()
    for hint, canonical in sorted(CUISINE_HINTS.items(), key=lambda item: len(item[0]), reverse=True):
        if hint in lowered:
            cuisine_filter = canonical
            cleaned = re.sub(re.escape(hint), " ", cleaned, flags=re.IGNORECASE)
            break

    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned or query, cuisine_filter, max_calories


def display_results(results: list[dict[str, Any]], title: str = "Results") -> None:
    """Pretty-print search hits."""
    print(f"\n📊 {title}")
    print("-" * 50)
    if not results:
        print("No matching food items found.")
        return

    for i, result in enumerate(results, 1):
        score = float(result["similarity_score"]) * 100
        print(f"{i}. 🍽️  {result['food_name']}")
        print(
            f"   📍 {result['cuisine_type']} | "
            f"🔥 {result['food_calories_per_serving']} cal | "
            f"📈 {score:.1f}% match"
        )
        description = result.get("food_description") or ""
        if description:
            print(f"   📝 {description[:120]}{'...' if len(description) > 120 else ''}")
        print()


def show_search_help() -> None:
    print("\n📖 ADVANCED SEARCH HELP")
    print("=" * 40)
    print("Hybrid search combines:")
    print("  • Vector similarity (semantic meaning)")
    print("  • Metadata filters (cuisine, max calories)")
    print("\nExamples:")
    print("  • spicy healthy dinner")
    print("  • Italian pasta under 400 calories")
    print("  • filter cuisine=Japanese calories=300 ramen")
    print("\nCommands:")
    print("  • help  - show this menu")
    print("  • quit  - exit")


def interactive_loop(collection: Any, top_k: int) -> None:
    """Interactive hybrid search loop with error handling."""
    print("\n" + "=" * 70)
    print("🔎 ADVANCED HYBRID FOOD SEARCH")
    print("   Vector Similarity + Metadata Filtering (ChromaDB)")
    print("=" * 70)
    print("Type a natural-language query, or use explicit filters.")
    print("Commands: help | quit")
    print("-" * 70)

    while True:
        try:
            user_input = input("\n🔎 Query: ").strip()
            if not user_input:
                print("Please enter a food search query.")
                continue

            lowered = user_input.lower()
            if lowered in {"quit", "exit", "q"}:
                print("Goodbye! 👋")
                break
            if lowered in {"help", "h"}:
                show_search_help()
                continue

            cuisine = None
            max_calories = None
            query = user_input

            # Explicit filter syntax: filter cuisine=Italian calories=400 pasta
            if lowered.startswith("filter "):
                remainder = user_input[7:].strip()
                cuisine_m = re.search(r"cuisine=([A-Za-z ]+)", remainder, re.I)
                cal_m = re.search(r"calories?=(\d+)", remainder, re.I)
                if cuisine_m:
                    cuisine = cuisine_m.group(1).strip().title()
                    remainder = remainder.replace(cuisine_m.group(0), " ")
                if cal_m:
                    max_calories = int(cal_m.group(1))
                    remainder = remainder.replace(cal_m.group(0), " ")
                query = re.sub(r"\s+", " ", remainder).strip() or "food"
            else:
                query, cuisine, max_calories = parse_natural_filters(user_input)

            print(f"\n🔍 Searching for: '{query}'")
            if cuisine or max_calories is not None:
                print(
                    f"   Filters → cuisine={cuisine or 'any'}, "
                    f"max_calories={max_calories if max_calories is not None else 'any'}"
                )
                results = perform_filtered_similarity_search(
                    collection,
                    query=query,
                    cuisine_filter=cuisine,
                    max_calories=max_calories,
                    n_results=top_k,
                )
                display_results(results, "Hybrid Filtered Results")
            else:
                results = perform_similarity_search(collection, query, top_k)
                display_results(results, "Similarity Search Results")

        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as exc:
            print(f"❌ Search error: {exc}")


def main(argv: list[str] | None = None) -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Advanced hybrid food search (ChromaDB similarity + metadata filters)"
    )
    parser.add_argument("--data", default=str(settings.data_path))
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument(
        "--query",
        help="Run a single query non-interactively and exit",
    )
    parser.add_argument("--cuisine", help="Optional cuisine metadata filter")
    parser.add_argument(
        "--max-calories",
        type=int,
        help="Optional maximum calories metadata filter",
    )
    args = parser.parse_args(argv)

    try:
        print("🔎 Advanced Hybrid Food Search")
        print("   ChromaDB vector similarity + metadata constraints")
        print("=" * 55)

        food_items = load_food_data(args.data)
        if not food_items:
            print("❌ No food data loaded.")
            return 1
        print(f"✅ Loaded {len(food_items)} food items")

        collection = ensure_collection_ready(
            settings.search_collection_name,
            food_items,
            collection_metadata={"description": "Hybrid similarity + metadata search"},
            settings=settings,
            force_rebuild=args.rebuild_index,
        )
        print("✅ Vector database ready")

        if args.query:
            results = perform_filtered_similarity_search(
                collection,
                query=args.query,
                cuisine_filter=args.cuisine,
                max_calories=args.max_calories,
                n_results=args.top_k,
            )
            display_results(results)
            return 0

        interactive_loop(collection, top_k=args.top_k)
        return 0

    except Exception as error:
        print(f"❌ Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
