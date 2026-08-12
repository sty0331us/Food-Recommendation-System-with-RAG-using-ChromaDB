"""RAG pipeline: context building, generation, and fallbacks."""

from __future__ import annotations

from typing import Any

from food_rag.llm import generate_text


def prepare_context_for_llm(query: str, search_results: list[dict[str, Any]]) -> str:
    """Prepare structured context from search results for the LLM."""
    if not search_results:
        return "No relevant food items found in the database."

    context_parts = [
        "Based on your query, here are the most relevant food options from our database:",
        "",
    ]

    for i, result in enumerate(search_results[:3], 1):
        food_context = [
            f"Option {i}: {result['food_name']}",
            f"  - Description: {result.get('food_description', '')}",
            f"  - Cuisine: {result.get('cuisine_type', 'Unknown')}",
            f"  - Calories: {result.get('food_calories_per_serving', 0)} per serving",
        ]

        ingredients = result.get("food_ingredients")
        if ingredients:
            if isinstance(ingredients, list):
                food_context.append(
                    f"  - Key ingredients: {', '.join(ingredients[:5])}"
                )
            else:
                food_context.append(f"  - Key ingredients: {ingredients}")

        if result.get("food_health_benefits"):
            food_context.append(
                f"  - Health benefits: {result['food_health_benefits']}"
            )

        if result.get("cooking_method"):
            food_context.append(f"  - Cooking method: {result['cooking_method']}")

        if result.get("taste_profile"):
            food_context.append(f"  - Taste profile: {result['taste_profile']}")

        score = float(result.get("similarity_score", 0)) * 100
        food_context.append(f"  - Similarity score: {score:.1f}%")
        food_context.append("")
        context_parts.extend(food_context)

    # query is accepted for API compatibility / future prompt tuning
    _ = query
    return "\n".join(context_parts)


def generate_fallback_response(
    query: str, search_results: list[dict[str, Any]]
) -> str:
    """Generate a retrieval-only response when the LLM is unavailable."""
    if not search_results:
        return (
            "I couldn't find any food items matching your request. "
            "Try describing what you're in the mood for with different words!"
        )

    top = search_results[0]
    cuisine = str(top.get("cuisine_type", "Unknown"))
    article = "an" if cuisine[:1].lower() in {"a", "e", "i", "o", "u"} else "a"
    parts = [
        f"Based on your request for '{query}', I'd recommend {top['food_name']}.",
        (
            f"It's {article} {cuisine} dish with "
            f"{top.get('food_calories_per_serving', 0)} calories per serving."
        ),
    ]

    if len(search_results) > 1:
        second = search_results[1]
        parts.append(f"Another great option would be {second['food_name']}.")

    return " ".join(parts)


def generate_llm_rag_response(
    query: str, search_results: list[dict[str, Any]]
) -> str:
    """Generate a response using Granite with retrieved context (true RAG)."""
    context = prepare_context_for_llm(query, search_results)

    prompt = f'''You are a helpful food recommendation assistant. A user is asking for food recommendations, and I've retrieved relevant options from a food database.

User Query: "{query}"

Retrieved Food Information:
{context}

Please provide a helpful, short response that:
1. Acknowledges the user's request
2. Recommends 2-3 specific food items from the retrieved options
3. Explains why these recommendations match their request
4. Includes relevant details like cuisine type, calories, or health benefits
5. Uses a friendly, conversational tone
6. Keeps the response concise but informative

Response:'''

    text = generate_text(prompt)
    if text and len(text) >= 50:
        return text
    return generate_fallback_response(query, search_results)


def generate_simple_comparison(
    query1: str,
    query2: str,
    results1: list[dict[str, Any]],
    results2: list[dict[str, Any]],
) -> str:
    """Simple comparison fallback without an LLM."""
    if not results1 and not results2:
        return "No results found for either query."
    if not results1:
        return f"Found results for '{query2}' but none for '{query1}'."
    if not results2:
        return f"Found results for '{query1}' but none for '{query2}'."

    return (
        f"For '{query1}', I recommend {results1[0]['food_name']}. "
        f"For '{query2}', {results2[0]['food_name']} would be perfect."
    )


def generate_llm_comparison(
    query1: str,
    query2: str,
    results1: list[dict[str, Any]],
    results2: list[dict[str, Any]],
) -> str:
    """Generate an AI-powered comparison between two food preference queries."""
    context1 = prepare_context_for_llm(query1, results1[:3])
    context2 = prepare_context_for_llm(query2, results2[:3])

    comparison_prompt = f'''You are analyzing and comparing two different food preference queries. Please provide a thoughtful comparison.

Query 1: "{query1}"
Top Results for Query 1:
{context1}

Query 2: "{query2}"
Top Results for Query 2:
{context2}

Please provide a short comparison that:
1. Highlights the key differences between these two food preferences
2. Notes any similarities or overlaps
3. Explains which query might be better for different situations
4. Recommends the best option from each query
5. Keeps the analysis concise but insightful

Comparison:'''

    text = generate_text(comparison_prompt)
    if text:
        return text
    return generate_simple_comparison(query1, query2, results1, results2)
