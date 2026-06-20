#!/usr/bin/env python3
"""
Batch Evaluation Runner for NASA RAG System

Loads test_questions.json, runs each question through the RAG pipeline,
evaluates responses with RAGAS, and prints per-question metrics + means.

Usage:
    python batch_eval.py --openai-key YOUR_KEY
    python batch_eval.py --openai-key YOUR_KEY --chroma-dir ./chroma_db_openai --collection nasa_space_missions_text
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import rag_client
import llm_client
import ragas_evaluator


def load_test_questions(path: str = "test_questions.json") -> List[Dict]:
    """Load test questions from JSON file."""
    questions_path = Path(path)
    if not questions_path.exists():
        print(f"ERROR: Test questions file not found: {path}", file=sys.stderr)
        sys.exit(1)

    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if not isinstance(questions, list) or len(questions) == 0:
        print("ERROR: test_questions.json must be a non-empty list", file=sys.stderr)
        sys.exit(1)

    return questions


def run_batch_evaluation(
    questions: List[Dict],
    openai_key: str,
    chroma_dir: str,
    collection_name: str,
    model: str = "gpt-3.5-turbo",
    n_docs: int = 3,
) -> List[Dict]:
    """Run evaluation for all test questions and return per-question results."""
    os.environ["OPENAI_API_KEY"] = openai_key

    collection, success, error = rag_client.initialize_rag_system(chroma_dir, collection_name)
    if not success:
        print(f"ERROR: Could not connect to ChromaDB: {error}", file=sys.stderr)
        sys.exit(1)

    results = []

    for i, item in enumerate(questions, 1):
        question = item.get("question", "").strip()
        mission = item.get("mission", "all")
        qid = item.get("id", i)

        print(f"\n[{i}/{len(questions)}] Q{qid}: {question}")
        print(f"  Mission filter: {mission}")

        # Retrieve documents
        docs_result = rag_client.retrieve_documents(
            collection, question, n_results=n_docs, mission_filter=mission
        )

        contexts: List[str] = []
        context_str = ""
        if docs_result and docs_result.get("documents"):
            contexts = docs_result["documents"][0]
            context_str = rag_client.format_context(
                docs_result["documents"][0], docs_result["metadatas"][0]
            )

        if not contexts:
            print("  WARNING: No documents retrieved — skipping evaluation for this question.")
            results.append({"id": qid, "question": question, "metrics": {"error": "no context retrieved"}})
            continue

        # Generate answer
        answer = llm_client.generate_response(openai_key, question, context_str, [], model)
        print(f"  Answer (truncated): {answer[:120]}...")

        # Evaluate
        metrics = ragas_evaluator.evaluate_response_quality(question, answer, contexts)
        print("  Metrics:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"    {metric}: {value:.4f}")
            else:
                print(f"    {metric}: {value}")

        results.append({"id": qid, "question": question, "metrics": metrics})

    return results


def print_summary(results: List[Dict]) -> None:
    """Print mean per metric across all evaluated questions."""
    metric_totals: Dict[str, List[float]] = {}

    for item in results:
        for metric, value in item["metrics"].items():
            if isinstance(value, float):
                metric_totals.setdefault(metric, []).append(value)

    if not metric_totals:
        print("\nNo numeric metrics to summarise.")
        return

    print("\n" + "=" * 50)
    print("MEAN METRICS ACROSS ALL QUESTIONS")
    print("=" * 50)
    for metric, values in metric_totals.items():
        mean = sum(values) / len(values)
        print(f"  {metric}: {mean:.4f}  (n={len(values)})")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch RAGAS evaluation runner for NASA RAG system")
    parser.add_argument("--openai-key", required=True, help="OpenAI API key")
    parser.add_argument("--chroma-dir", default="./chroma_db_openai", help="ChromaDB persist directory")
    parser.add_argument("--collection", default="nasa_space_missions_text", help="ChromaDB collection name")
    parser.add_argument("--questions", default="test_questions.json", help="Path to test questions JSON")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model for answer generation")
    parser.add_argument("--n-docs", type=int, default=3, help="Documents to retrieve per question")
    args = parser.parse_args()

    print("NASA RAG Batch Evaluator")
    print(f"  ChromaDB: {args.chroma_dir} / {args.collection}")
    print(f"  Questions: {args.questions}")
    print(f"  Model: {args.model}")

    questions = load_test_questions(args.questions)
    print(f"\nLoaded {len(questions)} test questions.")

    results = run_batch_evaluation(
        questions=questions,
        openai_key=args.openai_key,
        chroma_dir=args.chroma_dir,
        collection_name=args.collection,
        model=args.model,
        n_docs=args.n_docs,
    )

    print_summary(results)


if __name__ == "__main__":
    main()
