from langchain_aws import ChatBedrock
import os
from dotenv import load_dotenv

load_dotenv()

def get_llm():
    return ChatBedrock(
        model_id=os.getenv("LLM_MODEL", "anthropic.claude-3-sonnet-20240229-v1:0"),
        region_name=os.getenv("AWS_REGION", "ap-south-1"),
        model_kwargs={
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 1024,
            "temperature": 0.3,
            "top_k": 250,
            "top_p": 0.999,
            "stop_sequences": ["\n\nHuman:"]
        }
    )
