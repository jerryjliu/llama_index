# LlamaIndex Llms Integration: VESSL AI

[VESSL AI](https://vessl.ai/) is an MLOps platform for professional AI & LLM teams. VESSL supports both Multi-Cloud and On-Premise systems, helping ML teams seamlessly manage their workloads and data.

## LlamaIndex VESSL AI LLM Provider

The VESSL AI LLM Provider helps AI Engineers serve their models efficiently using the `serve` method. If your team has already deployed an LLM model with VESSL and has an endpoint, you can easily connect to that endpoint and integrate it into your LlamaIndex workflow. The VESSL AI LLM Provider functions similarly to an OPENAI-like object.

1. To serve a Hugging Face model, use the `serve` method:

```python
# 1 Serve with Hugging Face model name
llm.serve(
    service_name="llama-index-vesslai-test",
    model_name="mistralai/Mistral-7B-Instruct-v0.3",
    hf_token="HF_TOKEN",
    api_key="OPENAI-API-KEY",
)
```

2. If you have your own VESSL YAML file for serving, you can deploy the workload using the YAML file:

```python
# 2 Serve with YAML file
llm.serve(
    service_name="llama-index-vesslai-test",
    yaml_path="/your/own/service_yaml_file.yaml",
    api_key="OPENAI-API-KEY",
)
```

3. If you have a pre-served LLM, you can simply connect to the endpoint:

```python
# 3 Connect with pre-served endpoint
llm.connect(
    served_model_name="mistralai/Mistral-7B-Instruct-v0.3",
    endpoint="https://model-service-gateway-abc.oregon.google-cluster.vessl.ai/v1",
)
```
