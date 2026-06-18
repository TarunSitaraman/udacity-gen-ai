from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from typing import Dict, List, Optional

# RAGAS imports
try:
    from ragas import SingleTurnSample
    from ragas.metrics import BleuScore, NonLLMContextPrecisionWithReference, ResponseRelevancy, Faithfulness, RougeScore
    from ragas import evaluate
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

def evaluate_response_quality(question: str, answer: str, contexts: List[str]) -> Dict[str, float]:
    """Evaluate response quality using RAGAS metrics"""
    if not RAGAS_AVAILABLE:
        return {"error": "RAGAS not available"}
    
    from datasets import Dataset
    
    # Create evaluator LLM with model gpt-3.5-turbo
    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-3.5-turbo"))
    
    # Create evaluator_embeddings with model text-embedding-3-small
    evaluator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))
    
    # Define an instance for each metric to evaluate
    metrics = [
        ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        Faithfulness(llm=evaluator_llm)
    ]
    
    # Create a dataset for evaluation
    dataset_dict = {
        "user_input": [question],
        "response": [answer],
        "retrieved_contexts": [contexts]
    }
    dataset = Dataset.from_dict(dataset_dict)
    
    # Evaluate the response using the metrics
    results = evaluate(dataset=dataset, metrics=metrics)
    
    # Convert results to a dictionary
    return results.to_pandas().iloc[0].to_dict()
