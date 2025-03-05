"""LLM trustworthiness evaluation."""

import os
import subprocess
import zipfile
from typing import Dict, List, Optional

import requests
from trust_eval.config import EvaluationConfig, ResponseGeneratorConfig
from trust_eval.data import construct_data
from trust_eval.evaluator import Evaluator
from trust_eval.logging_config import logger
from trust_eval.response_generator import ResponseGenerator
from trust_eval.retrieval import retrieve


class TrustScoreEvaluator:
    def __init__(
        self,
        generator_config: Optional[str] = "generator_config.yaml",
        eval_config: Optional[str] = "eval_config.yaml",
    ) -> None:
        self.generator_config = ResponseGeneratorConfig.from_yaml(
            yaml_path=generator_config
        )
        logger.info(self.generator_config)
        self.evaluation_config = EvaluationConfig.from_yaml(yaml_path=eval_config)
        logger.info(self.evaluation_config)
        self._download_data()
        self.data = None

    def _download_data(self) -> None:
        dpr_wiki_url = (
            "https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"
        )
        gtr_emb_url = "https://huggingface.co/datasets/princeton-nlp/gtr-t5-xxl-wikipedia-psgs_w100-index/resolve/main/gtr_wikipedia_index.pkl"
        prompts_zip_url = "https://github.com/shanghongsim/trust_eval/raw/main/docs/quickstart/prompts.zip"

        dpr_wiki_path, compressed_dpr_wiki_path = "psgs_w100.tsv", "psgs_w100.tsv.gz"
        gtr_emb_path = "gtr_wikipedia_index.pkl"
        prompts_parent_dir = os.getcwd()
        prompts_dir = "prompts"
        prompts_zip_path = "prompts.zip"

        if not os.path.exists(dpr_wiki_path):
            logger.info("Downloading DPR Wiki TSV...")
            subprocess.run(["wget", dpr_wiki_url], check=True)
            subprocess.run(["gzip", "-dv", compressed_dpr_wiki_path], check=True)
            os.remove(compressed_dpr_wiki_path)

        if not os.path.exists(gtr_emb_path):
            logger.info("Downloading GTR Wikipedia index...")
            subprocess.run(["wget", gtr_emb_url], check=True)

        if not os.path.exists(prompts_dir) or not os.listdir(prompts_dir):
            logger.info("Downloading prompts zip file...")
            response = requests.get(prompts_zip_url)
            if response.status_code == 200:
                with open(prompts_zip_path, "wb") as f:
                    f.write(response.content)
            else:
                logger.error(
                    f"Failed to download prompts zip file, status code: {response.status_code}"
                )
                return
            with zipfile.ZipFile(prompts_zip_path, "r") as zip_ref:
                zip_ref.extractall(prompts_parent_dir)
            os.remove(prompts_zip_path)

        os.environ["DPR_WIKI_TSV"] = os.path.abspath(dpr_wiki_path)
        os.environ["GTR_EMB"] = os.path.abspath(gtr_emb_path)
        logger.info(
            f"Environment variables set: DPR_WIKI_TSV={os.environ['DPR_WIKI_TSV']}, GTR_EMB={os.environ['GTR_EMB']}"
        )

    def create_dataset(
        self, questions: Optional[List[str]], answers: Optional[List[str]]
    ) -> List[Dict]:
        raw_docs = retrieve(questions, top_k=5)
        self.data = construct_data(questions, answers, raw_docs, self.generator_config)
        return self.data

    def run(self, output_path: Optional[str] = "output/custom_data.json") -> List[Dict]:
        generator = ResponseGenerator(self.generator_config)
        self.data = generator.generate_responses(self.data)
        generator.save_responses(output_path=output_path)
        return self.data

    def evaluate(self, output_path: Optional[str] = "results/custom_data.json") -> Dict:
        evaluator = Evaluator(self.evaluation_config)
        results = evaluator.compute_metrics()
        evaluator.save_results(output_path=output_path)
        return results
