"""RAGAS scoring script over evaluation/testset.json. TODO: Day 4."""
"""
Offline RAGAS scoring script. Runs the RAG pipeline against a small labeled
test set and reports faithfulness, answer relevancy, and context precision --
turning "is our RAG actually grounded, not hallucinating" into a number
instead of a hunch.

Run from the project root:
    python -m evaluation.evaluate
"""

import json
from dotenv import load_dotenv

from ragas import SingleTurnSample, EvaluationDataset, evaluate
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

from retrieval.retriever import get_retriever
from generation.llm import get_llm
from generation.prompt import RAG_PROMPT
from generation.parser import answer_parser
from ingestion.embed_store import _get_embeddings


def load_testset(path: str = "evaluation/testset.json"):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline_on_question(question: str, retriever, llm):
    """
    Runs the same document-route pipeline as backend/main.py: retrieve,
    format context, generate a structured answer. Returns the plain-text
    answer and the raw retrieved chunk texts (what RAGAS calls
    retrieved_contexts).
    """
    docs = retriever.invoke(question)
    contexts = [doc.page_content for doc in docs]
    context_str = "\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}" for d in docs
    )
    chain = RAG_PROMPT | llm | answer_parser
    result = chain.invoke({"context": context_str, "question": question})
    return result.answer, contexts


def main():
    load_dotenv()

    testset = load_testset()
    retriever = get_retriever()
    llm = get_llm()

    print(f"Running the RAG pipeline on {len(testset)} test questions...")
    samples = []
    for item in testset:
        question = item["question"]
        ground_truth = item["ground_truth"]
        answer, contexts = run_pipeline_on_question(question, retriever, llm)
        samples.append(
            SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=contexts,
                reference=ground_truth,
            )
        )
        print(f"  done: {question[:60]}...")

    dataset = EvaluationDataset(samples=samples)

    # RAGAS needs its own LLM/embeddings to *judge* the answers -- reusing
    # the same chat model and the same local embeddings the project already
    # uses, so no extra API key is needed just to run evaluation.
    evaluator_llm = LangchainLLMWrapper(llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(_get_embeddings())

    metrics = [
        Faithfulness(llm=evaluator_llm),
        AnswerRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ContextPrecision(llm=evaluator_llm),
    ]

    print("\nScoring with RAGAS (faithfulness, answer relevancy, context precision)...")
    result = evaluate(dataset=dataset, metrics=metrics)

    print("\n=== RAGAS Evaluation Results ===")
    print(result)

    df = result.to_pandas()
    df.to_csv("evaluation/results.csv", index=False)
    print("\nPer-question breakdown saved to evaluation/results.csv")


if __name__ == "__main__":
    main()
