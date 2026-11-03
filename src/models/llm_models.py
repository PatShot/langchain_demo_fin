# This file is for loading the singular instance of the model for all experiments in the `src` folder.

import os

from dotenv import load_dotenv
from langchain_aws import ChatBedrockConverse

# Reads AWS_BEARER_TOKEN_BEDROCK / AWS_REGION from the `.env` file.
# Copy `.env.example` to `.env` and fill it in before running anything.
load_dotenv()

# A small, cheap model is plenty for this demo. Point BEDROCK_MODEL_ID at any
# other model your AWS account has access to, for example:
#   "anthropic.claude-sonnet-5"                     - stronger, costs more
#   "anthropic.claude-haiku-4-5-20251001-v1:0"      - previous light model
#   "us.anthropic.claude-haiku-5-5"                 - same model, cross-region
# Model ids differ per account and region. To see yours, run:
#   uv run python -c "import boto3;print([m['modelId'] for m in \
#     boto3.client('bedrock').list_foundation_models()['modelSummaries']])"
LLM_MODEL_LIGHT = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-haiku-5-5")

llm_main_model = ChatBedrockConverse(
    model=LLM_MODEL_LIGHT,
    region_name=os.getenv("AWS_REGION", "us-east-1"),
    max_tokens=1024,
    # Older models also accept `temperature=0.2`. The newest Claude models do
    # not take it, so it is left out here to keep the demo warning-free.
)
