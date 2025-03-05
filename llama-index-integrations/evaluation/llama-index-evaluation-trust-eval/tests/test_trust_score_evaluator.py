import os
from unittest.mock import patch

import pytest
from trust_eval.evaluator import Evaluator
from trust_eval.response_generator import ResponseGenerator

from llama_index.evaluation.trust_eval import TrustScoreEvaluator


@pytest.fixture()
def evaluator():
    """Fixture to initialize TrustScoreEvaluator."""
    return TrustScoreEvaluator(
        generator_config="generator_config.yaml", eval_config="eval_config.yaml"
    )


@patch("llama_index.evaluation.trust_eval.trust_score.subprocess.run")
@patch("llama_index.evaluation.trust_eval.trust_score.requests.get")
def test_download_data(mock_requests_get, mock_subprocess_run, evaluator):
    """Test that necessary files are downloaded correctly."""
    mock_requests_get.return_value.status_code = 200
    mock_requests_get.return_value.content = b"dummy data"

    evaluator._download_data()

    # # Ensure subprocess was called for wget and gzip
    # mock_subprocess_run.assert_any_call(["wget", "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"], check=True)
    # mock_subprocess_run.assert_any_call(["gzip", "-dv", "psgs_w100.tsv.gz"], check=True)

    # Ensure environment variables are set
    assert os.getenv("DPR_WIKI_TSV") is not None
    assert os.getenv("GTR_EMB") is not None


@patch("llama_index.evaluation.trust_eval.trust_score.retrieve")
@patch("llama_index.evaluation.trust_eval.trust_score.construct_data")
def test_create_dataset(mock_construct_data, mock_retrieve, evaluator):
    """Test dataset creation with mock retrieval and data construction."""
    mock_retrieve.return_value = ["retrieved_doc1", "retrieved_doc2"]
    mock_construct_data.return_value = [
        {"question": "What is AI?", "answer": "Artificial Intelligence"}
    ]

    questions = ["What is AI?"]
    answers = ["Artificial Intelligence"]

    dataset = evaluator.create_dataset(questions, answers)

    assert dataset == [{"question": "What is AI?", "answer": "Artificial Intelligence"}]
    mock_retrieve.assert_called_once_with(questions, top_k=5)
    mock_construct_data.assert_called_once()


@patch.object(
    ResponseGenerator,
    "generate_responses",
    return_value=[{"question": "What is AI?", "generated_response": "AI is ..."}],
)
@patch.object(ResponseGenerator, "save_responses")
def test_run(mock_save_responses, mock_generate_responses, evaluator):
    """Test response generation and saving."""
    evaluator.data = [{"question": "What is AI?", "answer": "Artificial Intelligence"}]

    generated_data = evaluator.run(output_path="output/test_output.json")

    assert generated_data == [
        {"question": "What is AI?", "generated_response": "AI is ..."}
    ]
    mock_generate_responses.assert_called_once()
    mock_save_responses.assert_called_once_with(output_path="output/test_output.json")


@patch.object(Evaluator, "compute_metrics", return_value={"accuracy": 0.9})
@patch.object(Evaluator, "save_results")
def test_evaluate(mock_save_results, mock_compute_metrics, evaluator):
    """Test evaluation and results saving."""
    results = evaluator.evaluate(output_path="results/test_results.json")

    assert results == {"accuracy": 0.9}
    mock_compute_metrics.assert_called_once()
    mock_save_results.assert_called_once_with(output_path="results/test_results.json")


if __name__ == "__main__":
    pytest.main()
