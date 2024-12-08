import os
import vessl.serving
import yaml
from typing import Any, Optional
from pydantic import BaseModel

import vessl
from vessl.util.config import VesslConfigLoader
from vessl.util.exception import VesslApiException
from llama_index.llms.vesslai.utils import wait_for_gateway_enabled, read_service
from llama_index.llms.openai_like import OpenAILike


class VesslAILLM(OpenAILike,BaseModel):
    """VesslAI LLM.

    Examples:
        `pip install llama-index-llms-vesslai`

        ```python
        from llama_index.llms.vesslai import VesslAILLM

        # set api key in env or in llm
        # import os

        # vessl configure
        llm = VesslAILLM()

        #1 Serve with methods 
        llm.serve(
            service_name = "llama-index-vesslai-test",
            model_name = "mistralai/Mistral-7B-Instruct-v0.3",
        )

        #2 Serve with yaml
        llm.serve(
            service_name = "llama-index-vesslai-test",
            yaml_path="/root/vesslai/vesslai_vllm.yaml",
        )

        #3 Pre-served endpoint
        llm.connect(
            served_model_name="mistralai/Mistral-7B-Instruct-v0.3",
            endpoint="https://serve-api.vessl.ai/api/v1/services/endpoint/v1",
        )

        resp = llm.complete("Who is Paul Graham?")
        print(resp)
        ```
    """
    organization_name: str = None
    default_service_yaml: str = "vesslai_vllm.yaml"

    def __init__(
        self,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._configure()
    
    def _configure(self):
        vessl.configure()
        if vessl.vessl_api.is_in_run_exec_context():
            vessl.vessl_api.set_access_token(no_prompt=True)
            user = vessl.vessl_api.user
            organization_name = vessl.vessl_api.set_organization()
            project_name = vessl.vessl_api.set_project()
        else:
            config = VesslConfigLoader()
            user = None
            if config.access_token:
                vessl.vessl_api.api_client.set_default_header(
                    "Authorization", f"Token {config.access_token}"
                )

                try:
                    user = vessl.vessl_api.get_my_user_info_api()
                except VesslApiException:
                    pass

            organization_name = config.default_organization
            project_name = config.default_project

            if user is None or organization_name is None:
                print("Please run `vessl configure` first.")
                return
        
        self.organization_name = organization_name
    
    def serve(
        self,
        service_name: str,
        model_name: Optional[str] = None,
        yaml_path: Optional[str] = None,
        is_chat_model: bool = True,
        serverless: bool = False,
        api_key: str = None,
        **kwargs: Any,
    ) -> None:
        self.organization_name = kwargs.get('organization_name', self.organization_name)
        if not model_name and not yaml_path:
            raise ValueError("You must provide either 'model_name' or 'yaml_path', but not both")
        if model_name and yaml_path:
            raise ValueError("You must provide only one of 'model_name' or 'yaml_path', not both")
        
        hf_token = kwargs.get("hf_token", os.environ.get("HF_TOKEN"))
        if hf_token is None:
            raise ValueError("HF_TOKEN must be set either as a parameter or environment variable")

        if not api_key and os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("You must set OPENAI_API_KEY or api_key")
        
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is not None:
            self.api_key = api_key
        
        default_yaml_path = self._get_default_yaml_path()
        
        # serve with model name
        if model_name != None:
            with open(default_yaml_path, 'r') as file:
                service_config = yaml.safe_load(file)
            service_config['env']['MODEL_NAME'] = model_name
            service_config['env']['HF_TOKEN'] = hf_token

            with open(default_yaml_path, 'w') as file:
                yaml.dump(service_config, file)
            
            self.model = model_name
            self.is_chat_model = is_chat_model
            self.api_base = self._launch_service_revision_from_yaml(
                organization_name = self.organization_name,
                yaml_path = default_yaml_path,
                service_name = service_name,
                serverless = serverless,
            )

        # serve with custom service yaml file
        if yaml_path != None:
            with open(yaml_path, 'r') as file:
                service_config = yaml.safe_load(file)
            model_name = service_config['env']['MODEL_NAME']
            service_config['env']['HF_TOKEN'] = hf_token

            with open(yaml_path, 'w') as file:
                yaml.dump(service_config, file)

            self.model = model_name
            self.is_chat_model = is_chat_model
            self.api_base = self._launch_service_revision_from_yaml(
                organization_name = self.organization_name,
                yaml_path = yaml_path,
                service_name = service_name,
                serverless = serverless,
            )
    
    def _launch_service_revision_from_yaml(
        self,
        organization_name: str,
        yaml_path: str,
        service_name: str,
        serverless: bool,
    ) -> str:
        assert organization_name is not None
        assert yaml_path is not None

        with open(yaml_path, "r") as f:
            yaml_body = f.read()
        print(yaml_body)

        revision = vessl.serving.create_revision_from_yaml_v2(
                organization=organization_name,
                service_name=service_name,
                yaml_body=yaml_body,
                serverless=serverless,
                arguments=None,
            )
        vessl.serving.create_active_revision_replacement_rollout(
                organization=organization_name,
                model_service_name=revision.model_service_name,
                desired_active_revisions_to_weight_map={revision.number: 100},
            )
        service_url = f"https://app.vessl.ai/{organization_name}/services/{service_name}"
        print(f"Check your Service at: {service_url}")

        gateway = read_service(service_name=service_name).gateway_config
        wait_for_gateway_enabled(gateway=gateway, service_name=revision.model_service_name, print_output=True)

        print("Endpoint is enabled.")
        gateway = read_service(service_name=service_name).gateway_config
        print(gateway)
        gateway_endpoint = f"https://{gateway.endpoint}/v1"
        print(f"You can test your service via {gateway_endpoint}")

        return gateway_endpoint
    
    def _get_default_yaml_path(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        default_yaml_path = os.path.join(current_dir, self.default_service_yaml)
        return default_yaml_path
    
    def connect(
        self,
        served_model_name: str,
        endpoint: str,
        is_chat_model: bool = True,
        api_key: str = None,
        **kwargs: Any,
    ) -> None:
        if api_key is None and os.environ.get('OPENAI_API_KEY') is None:
            raise ValueError("Set OPENAI_API_KEY or api_key")
        
        self.api_key = os.environ.get('OPENAI_API_KEY')
        if api_key is not None:
            self.api_key = api_key
        
        self.model = served_model_name
        self.api_base = endpoint
        self.is_chat_model = is_chat_model
