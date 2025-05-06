from langchain_aws import BedrockEmbeddings
import os
from dotenv import load_dotenv

load_dotenv()

def get_embedder():
    return BedrockEmbeddings(
        region_name=os.getenv("AWS_REGION", "ap-south-1"),
        model_id=os.getenv("EMBED_MODEL", "amazon.titan-embed-text-v2:0")
    )
