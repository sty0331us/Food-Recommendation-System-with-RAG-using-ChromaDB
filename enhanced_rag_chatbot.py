"""Enhanced RAG food recommendation chatbot (CLI)."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from food_rag.config import get_settings
from food_rag.database import ensure_collection_ready, load_food_data, perform_similarity_search
from food_rag.llm import healthcheck_llm
from food_rag.rag import generate_llm_comparison, generate_llm_rag_response


def show_enhanced_rag_help() -> None:
    """Display help information for the enhanced RAG chatbot."""
    print("\n📖 ENHANCED RAG CHATBOT HELP")
    print("=" * 45)
    print("🧠 This chatbot uses IBM Granite to understand your")
    print("   food preferences and provide intelligent recommendations.")
    print("\nHow to get the best recommendations:")
    print("  • Be specific: 'healthy Italian pasta under 350 calories'")
    print("  • Mention preferences: 'spicy comfort food for cold weather'")
    print("  • Include context: 'light breakfast for busy morning'")
    print("  • Ask about benefits: 'protein-rich foods for workout recovery'")
    print("\nSpecial features:")
    print("  • 🔍 Vector similarity search finds relevant foods")
    print("  • 🧠 AI analysis provides contextual explanations")
    print("  • 📊 Detailed nutritional and cuisine information")
    print("  • 🔄 Smart comparison between different preferences")
    print("\nCommands:")
    print("  • 'compare' - AI-powered comparison of two queries")
    print("  • 'help' - Show this help menu")
    print("  • 'quit' - Exit the chatbot")
    print("\nTips for better results:")
    print("  • Use natural language - talk like you would to a friend")
    print("  • Mention dietary restrictions or preferences")
    print("  • Include meal timing (breakfast, lunch, dinner)")
    print("  • Specify if you want healthy, comfort, or indulgent options")


def handle_enhanced_rag_query(
    collection: Any,
    query: str,
    conversation_history: list[str],
    top_k: int,
) -> None:
    """Handle a user query with the full RAG pipeline."""
    print(f"\n🔍 Searching vector database for: '{query}'...")
    if conversation_history:
        # Light context: prefer the latest turns when phrasing is ambiguous
        print(f"   (conversation context: {len(conversation_history)} prior queries)")

    search_results = perform_similarity_search(collection, query, top_k)

    if not search_results:
        print("🤖 Bot: I couldn't find any food items matching your request.")
        print("      Try describing what you're in the mood for with different words!")
        return

    print(f"✅ Found {len(search_results)} relevant matches")
    print("🧠 Generating AI-powered response...")

    ai_response = generate_llm_rag_response(query, search_results)
    print(f"\n🤖 Bot: {ai_response}")

    print("\n📊 Search Results Details:")
    print("-" * 45)
    for i, result in enumerate(search_results[:top_k], 1):
        score = float(result["similarity_score"]) * 100
        print(f"{i}. 🍽️  {result['food_name']}")
        print(
            f"   📍 {result['cuisine_type']} | "
            f"🔥 {result['food_calories_per_serving']} cal | "
            f"📈 {score:.1f}% match"
        )
        if i < min(len(search_results), top_k):
            print()


def handle_enhanced_comparison_mode(collection: Any, top_k: int) -> None:
    """Compare recommendations for two different natural-language queries."""
    print("\n🔄 ENHANCED COMPARISON MODE")
    print("   Powered by AI Analysis")
    print("-" * 35)

    query1 = input("Enter first food query: ").strip()
    query2 = input("Enter second food query: ").strip()

    if not query1 or not query2:
        print("❌ Please enter both queries for comparison")
        return

    print(f"\n🔍 Analyzing '{query1}' vs '{query2}' with AI...")

    results1 = perform_similarity_search(collection, query1, top_k)
    results2 = perform_similarity_search(collection, query2, top_k)
    comparison_response = generate_llm_comparison(query1, query2, results1, results2)

    print(f"\n🤖 AI Analysis: {comparison_response}")

    print("\n📊 DETAILED COMPARISON")
    print("=" * 60)
    left_header = f"Query 1: {query1[:20]}..." if len(query1) > 20 else f"Query 1: {query1}"
    right_header = f"Query 2: {query2[:20]}..." if len(query2) > 20 else f"Query 2: {query2}"
    print(f"{left_header:<30} | {right_header}")
    print("-" * 60)

    for i in range(min(max(len(results1), len(results2), 0), top_k)):
        left = (
            f"{results1[i]['food_name']} ({results1[i]['similarity_score'] * 100:.0f}%)"
            if i < len(results1)
            else "---"
        )
        right = (
            f"{results2[i]['food_name']} ({results2[i]['similarity_score'] * 100:.0f}%)"
            if i < len(results2)
            else "---"
        )
        print(f"{left[:30]:<30} | {right[:30]}")


def enhanced_rag_food_chatbot(collection: Any, top_k: int, history_limit: int) -> None:
    """Interactive RAG chatbot loop."""
    print("\n" + "=" * 70)
    print("🤖 ENHANCED RAG FOOD RECOMMENDATION CHATBOT")
    print("   Powered by IBM's Granite Model + ChromaDB")
    print("=" * 70)
    print("💬 Ask me about food recommendations using natural language!")
    print("\nExample queries:")
    print("  • 'I want something spicy and healthy for dinner'")
    print("  • 'What Italian dishes do you recommend under 400 calories?'")
    print("  • 'I'm craving comfort food for a cold evening'")
    print("  • 'Suggest some protein-rich breakfast options'")
    print("\nCommands:")
    print("  • 'help' - Show detailed help menu")
    print("  • 'compare' - Compare recommendations for two different queries")
    print("  • 'quit' - Exit the chatbot")
    print("-" * 70)

    conversation_history: list[str] = []

    while True:
        try:
            user_input = input("\n👤 You: ").strip()

            if not user_input:
                print("🤖 Bot: Please tell me what kind of food you're looking for!")
                continue

            lowered = user_input.lower()
            if lowered in {"quit", "exit", "q"}:
                print("\n🤖 Bot: Thank you for using the Enhanced RAG Food Chatbot!")
                print("      Hope you found some delicious recommendations! 👋")
                break

            if lowered in {"help", "h"}:
                show_enhanced_rag_help()
                continue

            if lowered == "compare":
                handle_enhanced_comparison_mode(collection, top_k)
                continue

            handle_enhanced_rag_query(
                collection, user_input, conversation_history, top_k
            )
            conversation_history.append(user_input)
            if len(conversation_history) > history_limit:
                conversation_history = conversation_history[-max(3, history_limit // 2) :]

        except KeyboardInterrupt:
            print("\n\n🤖 Bot: Goodbye! Hope you find something delicious! 👋")
            break
        except Exception as exc:
            print(f"❌ Bot: Sorry, I encountered an error: {exc}")


def main(argv: list[str] | None = None) -> int:
    """Entrypoint for the enhanced RAG chatbot."""
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Enhanced RAG Food Recommendation Chatbot (ChromaDB + IBM Granite)"
    )
    parser.add_argument(
        "--data",
        default=str(settings.data_path),
        help="Path to FoodDataSet.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=settings.default_top_k,
        help="Number of retrieved food items",
    )
    parser.add_argument(
        "--rebuild-index",
        action="store_true",
        help="Force rebuild of the ChromaDB collection",
    )
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM health check and run with retrieval fallbacks",
    )
    args = parser.parse_args(argv)

    try:
        print("🤖 Enhanced RAG-Powered Food Recommendation Chatbot")
        print("   Powered by IBM Granite & ChromaDB")
        print("=" * 55)

        food_items = load_food_data(args.data)
        if not food_items:
            print("❌ No food data loaded. Check --data path.")
            return 1
        print(f"✅ Loaded {len(food_items)} food items")

        collection = ensure_collection_ready(
            settings.rag_collection_name,
            food_items,
            collection_metadata={
                "description": "Enhanced RAG chatbot with IBM watsonx.ai integration"
            },
            settings=settings,
            force_rebuild=args.rebuild_index,
        )
        print("✅ Vector database ready")

        if args.retrieval_only:
            print("ℹ️  Retrieval-only mode enabled (LLM responses use fallbacks)")
        else:
            print("🔗 Testing LLM connection...")
            if healthcheck_llm(settings):
                print("✅ LLM connection established")
            else:
                if not settings.allow_llm_fallback:
                    print("❌ LLM connection failed and fallbacks are disabled")
                    return 1
                print(
                    "⚠️  LLM unavailable — continuing with retrieval-based fallbacks"
                )

        enhanced_rag_food_chatbot(
            collection,
            top_k=args.top_k,
            history_limit=settings.max_conversation_history,
        )
        return 0

    except Exception as error:
        print(f"❌ Error: {error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
