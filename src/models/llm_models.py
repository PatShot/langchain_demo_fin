# This file is for loading the singular instance of the model for all experiments in the `src` folder.
#
# Pick which server answers by setting LLM_PROVIDER in `.env`:
#   ollama   - a local Ollama server              (default, see ollama_model.py)
#   llamacpp - a local llama.cpp `llama-server`   (see llamacpp_model.py)
#   bedrock  - AWS Bedrock, needs AWS credentials and model access
#
# Both agents import `llm_main_model` from here, so switching provider needs no
# change to the agent code.

import os

from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()


def _build_bedrock_model():
    """AWS Bedrock - kept as an option, but needs an account with model access."""
    from langchain_aws import ChatBedrockConverse

    # Model ids differ per account and region. To see yours, run:
    #   uv run python -c "import boto3;print([m['modelId'] for m in \
    #     boto3.client('bedrock').list_foundation_models()['modelSummaries']])"
    return ChatBedrockConverse(
        model=os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-haiku-5-5"),
        region_name=os.getenv("AWS_REGION", "us-east-1"),
        max_tokens=1024,
    )


if LLM_PROVIDER == "ollama":
    from src.models.ollama_model import build_ollama_model

    llm_main_model = build_ollama_model()
elif LLM_PROVIDER == "llamacpp":
    from src.models.llamacpp_model import build_llamacpp_model

    llm_main_model = build_llamacpp_model()
elif LLM_PROVIDER == "bedrock":
    llm_main_model = _build_bedrock_model()
else:
    raise ValueError(
        f"Unknown LLM_PROVIDER '{LLM_PROVIDER}'. Use ollama, llamacpp or bedrock."
    )
