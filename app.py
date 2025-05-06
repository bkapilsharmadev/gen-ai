# app.py

import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from common.llm_chain import get_llm
import tiktoken

# Load environment variables from .env
load_dotenv()

# ─────────────────────────────────────
# Prompt Templates
# ─────────────────────────────────────

def get_combine_prompt():
    return PromptTemplate(
        template="""
Write a concise summary of the following text delimited by triple backquotes.
Return your response in bullet points which covers the key points of the text.

```{text}```

BULLET POINT SUMMARY:
""",
        input_variables=["text"]
    )

def get_map_prompt():
    return PromptTemplate(
        template="""
Write a concise summary of the following:

{text}

CONCISE SUMMARY:
""",
        input_variables=["text"]
    )

# ─────────────────────────────────────
# Document Loader
# ─────────────────────────────────────

def get_offering_docs(file_path="offering.json"):
    loader = JSONLoader(file_path=file_path, jq_schema=".", text_content=False)
    return loader.load()

# ─────────────────────────────────────
# Summarization Chain
# ─────────────────────────────────────

# Load the tokenizer (GPT-4/Claude-style)
encoding = tiktoken.get_encoding("cl100k_base")  # Close match to Claude/Titan

def count_tokens(text: str) -> int:
    return len(encoding.encode(text))

def get_summarize_chain(docs):
    llm = get_llm()
    map_prompt = get_map_prompt()
    combine_prompt = get_combine_prompt()

    # Split large docs into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.split_documents(docs)
    print(f"📝 Split into {len(docs)} document chunks...")

    # Create map-reduce summarization chain
    chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt,
        combine_prompt=combine_prompt
    )

    # Run summarization
    result = chain.invoke(docs)

    # Token counting
    input_tokens = sum(count_tokens(doc.page_content) for doc in docs)
    output_text = result["output_text"] if isinstance(result, dict) else str(result)
    output_tokens = count_tokens(output_text)

    print(f"\n🔢 Estimated Token Usage:")
    print(f"  Input Tokens   : {input_tokens}")
    print(f"  Output Tokens  : {output_tokens}")
    print(f"  Total Tokens   : {input_tokens + output_tokens}")

    # Cost estimation (Amazon Titan Lite: $0.0003 per 1K tokens)
    cost_per_1k = 0.0003
    estimated_cost = (input_tokens + output_tokens) / 1000 * cost_per_1k
    print(f"  💰 Estimated Cost: ${estimated_cost:.6f}")

    return result


# ─────────────────────────────────────
# Entry Point
# ─────────────────────────────────────

if __name__ == "__main__":
    docs = get_offering_docs()
    result = get_summarize_chain(docs)

    print("\n📄 Summary:\n")
    if isinstance(result, dict) and "output_text" in result:
        print(result["output_text"])
    else:
        print(result)
