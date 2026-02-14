"""
RAGAS Evaluation Pipeline
Evaluates RAG system quality using industry-standard metrics:
  - Faithfulness: Is the answer grounded in context? (no hallucination)
  - Answer Relevancy: Does the answer address the question?
  - Context Precision: Are retrieved chunks relevant to the question?
  - Context Recall: Did we retrieve all necessary information?
"""

import json
import logging
from pathlib import Path
from typing import Optional

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithoutReference,
    LLMContextRecall,
)

logger = logging.getLogger(__name__)

# Instantiate metrics
faithfulness = Faithfulness()
answer_relevancy = ResponseRelevancy()
context_precision = LLMContextPrecisionWithoutReference()
context_recall = LLMContextRecall()


def load_test_dataset(filepath: str = "app/evaluation/test_queries.json") -> dict:
    """
    Load the test dataset from a JSON file.

    Args:
        filepath: Path to the test queries JSON file.

    Returns:
        Dictionary with 'questions', 'ground_truths', etc.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Test dataset not found: {filepath}")

    with open(path) as f:
        data = json.load(f)

    logger.info(f"Loaded {len(data.get('questions', []))} test queries from {filepath}")
    return data


def evaluate_rag(
    questions: list[str],
    ground_truths: list[str],
    contexts: list[list[str]],
    answers: list[str],
) -> dict:
    """
    Run RAGAS evaluation on the RAG pipeline outputs.

    Metrics explained:
      - Faithfulness: Proportion of claims in the answer that are
        supported by the retrieved context (higher = less hallucination)
      - Answer Relevancy: How well the generated answer addresses
        the original question (semantic similarity)
      - Context Precision: Ratio of relevant chunks among all retrieved
        chunks (higher = better retrieval quality)
      - Context Recall: Proportion of the ground truth that can be
        attributed to the retrieved context (higher = better coverage)

    Args:
        questions: List of test questions.
        ground_truths: List of reference answers.
        contexts: List of lists (retrieved context strings per question).
        answers: List of generated answers.

    Returns:
        Dictionary of metric scores.
    """
    logger.info(f"Running RAGAS evaluation on {len(questions)} queries")

    dataset = Dataset.from_dict({
        "user_input": questions,
        "reference": ground_truths,
        "retrieved_contexts": contexts,
        "response": answers,
    })

    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
    )

    scores = {
        "faithfulness": float(result["faithfulness"]),
        "answer_relevancy": float(result["response_relevancy"]),
        "context_precision": float(result["llm_context_precision_without_reference"]),
        "context_recall": float(result["llm_context_recall"]),
    }

    logger.info("=== RAGAS Evaluation Results ===")
    for metric, score in scores.items():
        logger.info(f"  {metric:>20s}: {score:.3f}")

    return scores


def run_evaluation_from_file(
    filepath: str = "app/evaluation/test_queries.json",
    retriever=None,
    generator=None,
) -> Optional[dict]:
    """
    Run end-to-end evaluation using the test dataset file.

    This function loads test data, runs the retrieval + generation pipeline,
    and evaluates the results using RAGAS metrics.

    Args:
        filepath: Path to the test queries JSON file.
        retriever: ResearchRetriever instance.
        generator: ReportGenerator instance.

    Returns:
        Dictionary of RAGAS metric scores, or None if evaluation fails.
    """
    try:
        data = load_test_dataset(filepath)
    except FileNotFoundError as e:
        logger.error(str(e))
        return None

    questions = data["questions"]
    ground_truths = data["ground_truths"]

    if retriever is None or generator is None:
        logger.error("Retriever and generator must be provided for evaluation")
        return None

    all_contexts: list[list[str]] = []
    all_answers: list[str] = []

    for question in questions:
        # Retrieve
        docs = retriever.retrieve(question)
        contexts = [doc.page_content for doc in docs]
        all_contexts.append(contexts)

        # Generate
        answer = generator.generate(question, docs)
        all_answers.append(answer)

    return evaluate_rag(questions, ground_truths, all_contexts, all_answers)
