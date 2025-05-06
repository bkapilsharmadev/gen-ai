# common/llm_chain.py
from dotenv import load_dotenv
import os
from langchain_aws import ChatBedrock

load_dotenv()

def get_llm():
    return ChatBedrock(
        model_id=os.getenv("BEDROCK_MODEL"),
        region_name=os.getenv("AWS_REGION"),
        model_kwargs={
            "temperature": 0.5,
            "maxTokenCount": 1000,
            "stopSequences": []
        }
    )
