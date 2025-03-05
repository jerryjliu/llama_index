import os
import vessl.serving
import yaml
import asyncio
import nest_asyncio
from typing import Any, Optional
from pydantic import BaseModel

import vessl
from vessl.util.config import VesslConfigLoader
from vessl.util.exception import VesslApiException
from llama_index.llms.vesslai.utils import (
    wait_for_gateway_enabled,
    read_service,
    abort_in_progress_rollout_by_name,
    ensure_service_idempotence,
)
from llama_index.llms.openai_like import OpenAILike


class VesslAILLM(OpenAILike, BaseModel):
    """VesslAI LLM.

    Examples:
        `pip install llama-index-llms-vesslai`

        ```python
        from llama_index.llms.vesslai import VesslAILLM

        # set api key in env or in llm
        # import os

        # vessl configure
        llm = VesslAILLM()

        #1 Serve hf model name
        llm.serve(
            service_name="llama-index-vesslai-test",
            model_name="mistralai/Mistral-7B-Instruct-v0.3",
        )

        #2 Serve with yaml
        llm.serve(
            service_name="llama-index-vesslai-test",
            yaml_path="/root/vesslai/vesslai_vllm.yaml",
        )

        #3 Connect with pre-served endpoint
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

    def __init__(self, force_update_access_token: bool = False, **kwargs: Any) -> None:
        nest_asyncio.apply()
        super().__init__()

        self._configure(force_update_access_token=force_update_access_token)

    def _configure(self, force_update_access_token) -> None:
        vessl.configure(force_update_access_token=force_update_access_token)
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
        service_auth_key: str = None,
        force_relaunch: bool = False,
        **kwargs: Any,
    ) -> None:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            task = loop.create_task(
                self._serve(
                    service_name=service_name,
                    model_name=model_name,
                    yaml_path=yaml_path,
                    is_chat_model=is_chat_model,
                    serverless=serverless,
                    api_key=api_key,
                    service_auth_key=service_auth_key,
                    force_relaunch=force_relaunch,
                    **kwargs,
                )
            )
            loop.run_until_complete(task)
        else:
            asyncio.run(
                self._serve(
                    service_name=service_name,
                    model_name=model_name,
                    yaml_path=yaml_path,
                    is_chat_model=is_chat_model,
                    serverless=serverless,
                    api_key=api_key,
                    service_auth_key=service_auth_key,
                    force_relaunch=force_relaunch,
                    **kwargs,
                )
            )

    async def _serve(
        self,
        service_name: str,
        model_name: Optional[str] = None,
        yaml_path: Optional[str] = None,
        is_chat_model: bool = True,
        serverless: bool = False,
        api_key: str = None,
        service_auth_key: str = None,
        force_relaunch: bool = False,
        **kwargs: Any,
    ) -> None:
        print("Launching VesslAI LLM Service")
        self.organization_name = kwargs.get("organization_name", self.organization_name)
        self._validate_openai_key(api_key=api_key)

        if not model_name and not yaml_path:
            raise ValueError(
                "You must provide either 'model_name' or 'yaml_path', but not both"
            )
        if model_name and yaml_path:
            raise ValueError(
                "You must provide only one of 'model_name' or 'yaml_path', not both"
            )

        serve_yaml_path = self._get_default_yaml_path()
        with open(serve_yaml_path) as file:
            serve_config = yaml.safe_load(file)

        serve_model_name = None
        # serve with custom service yaml file
        if yaml_path:
            serve_model_name = serve_config["env"]["MODEL_NAME"]
            serve_yaml_path = yaml_path

        # serve with model name
        if model_name:
            serve_model_name = model_name
            hf_token = kwargs.get("hf_token", os.environ.get("HF_TOKEN"))
            if hf_token is None:
                raise ValueError(
                    "HF_TOKEN must be set either as a parameter or environment variable"
                )
            serve_config = self._build_model_serve_config(
                serve_model_name, serve_config, service_auth_key, hf_token
            )
            serve_yaml_path = self._get_temporary_yaml_path()
            with open(serve_yaml_path, "w") as file:
                yaml.dump(serve_config, file)

        self.model = serve_model_name
        self.is_chat_model = is_chat_model
        if not force_relaunch:
            with open(serve_yaml_path) as file:
                yaml_str = file.read()
            gateway_endpoint = ensure_service_idempotence(
                service_name=service_name, yaml_str=yaml_str
            )
            if gateway_endpoint is not None:
                print(
                    f'Model Name "{self.model}" is already being served. Connecting with Endpoint: {gateway_endpoint} \nIf you want to abort running model_service, please provide force_relaunch = True'
                )
                self.connect(
                    served_model_name=self.model,
                    endpoint=gateway_endpoint,
                    is_chat_model=self.is_chat_model,
                    api_key=api_key,
                )
                return

        self.api_base = await self._launch_service_revision_from_yaml(
            organization_name=self.organization_name,
            yaml_path=serve_yaml_path,
            service_name=service_name,
            serverless=serverless,
        )

        if model_name:
            os.remove(serve_yaml_path)

    def connect(
        self,
        served_model_name: str,
        endpoint: str,
        is_chat_model: bool = True,
        api_key: str = None,
        **kwargs: Any,
    ) -> None:
        self._validate_openai_key(api_key=api_key)

        self.model = served_model_name
        self.api_base = endpoint
        self.is_chat_model = is_chat_model

    def _validate_openai_key(self, api_key: str):
        if api_key is None and os.environ.get("OPENAI_API_KEY") is None:
            raise ValueError("Set OPENAI_API_KEY or api_key")

        self.api_key = os.environ.get("OPENAI_API_KEY")
        if api_key is not None:
            self.api_key = api_key

    def _build_model_serve_config(
        self,
        model_name: str,
        service_config: dict,
        service_auth_key: str,
        hf_token: str,
    ) -> str:
        if hf_token.startswith("hf_"):
            service_config["env"]["HF_TOKEN"] = hf_token
        else:
            service_config["env"]["HF_TOKEN"] = {
                "secret": hf_token,
            }
        if service_auth_key:
            service_config["env"]["SERVICE_AUTH_KEY"] = service_auth_key

        service_config["env"]["MODEL_NAME"] = model_name
        return service_config

    async def _launch_service_revision_from_yaml(
        self,
        organization_name: str,
        yaml_path: str,
        service_name: str,
        serverless: bool,
    ) -> str:
        assert organization_name is not None
        assert yaml_path is not None

        with open(yaml_path) as f:
            yaml_body = f.read()
        print(yaml_body)

        abort_in_progress_rollout_by_name(service_name=service_name)

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
        service_url = (
            f"https://app.vessl.ai/{organization_name}/services/{service_name}"
        )
        print(f"Check your Service at: {service_url}")

        gateway = read_service(service_name=service_name).gateway_config
        await wait_for_gateway_enabled(
            gateway=gateway, service_name=revision.model_service_name
        )

        print("Endpoint is enabled.")
        gateway = read_service(service_name=service_name).gateway_config
        gateway_endpoint = f"https://{gateway.endpoint}/v1"
        print(f"You can test your service via {gateway_endpoint}")

        return gateway_endpoint

    def _get_default_yaml_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, self.default_service_yaml)

    def _get_temporary_yaml_path(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "tmp_vesslai_vllm.yaml")
