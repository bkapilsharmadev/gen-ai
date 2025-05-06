import json
import re
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from common.llm_chain import get_llm  # Your own Claude wrapper (via Bedrock)

# Updated JSON key search utility
def find_key_recursively(data, key):
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                yield v
            yield from find_key_recursively(v, key)
    elif isinstance(data, list):
        for item in data:
            yield from find_key_recursively(item, key)
            
# Load and split structured JSON with enriched metadata
VALID_TYPES = ["pickpackrules", "mezzdelivery", "offeringfulfill", "mezzcreation", "csdelivery"]

def load_documents(path="offering.json"):
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    splitter = RecursiveJsonSplitter(max_chunk_size=800)
    split_docs = splitter.create_documents(texts=[json_data], convert_lists=True)

    enriched_docs = []
    last_valid_task_type = None

    def find_key_recursively(data, key):
        if isinstance(data, dict):
            for k, v in data.items():
                if k == key:
                    return v
                result = find_key_recursively(v, key)
                if result:
                    return result
        elif isinstance(data, list):
            for item in data:
                result = find_key_recursively(item, key)
                if result:
                    return result
        return None

    for i, doc in enumerate(split_docs):
        content = doc.page_content
        doc.metadata["chunkIndex"] = i

        task_type = None
        payload_id = None

        try:
            parsed_json = json.loads(content)
            task_type_candidate = find_key_recursively(parsed_json, "type")
            if task_type_candidate in VALID_TYPES:
                task_type = task_type_candidate
            payload_id = find_key_recursively(parsed_json, "payloadId")
        except json.JSONDecodeError:
            pass

        # Fallback with regex
        if not task_type:
            match = re.search(r'"type"\s*:\s*"(\w+)"', content)
            if match and match.group(1) in VALID_TYPES:
                task_type = match.group(1)

        if not payload_id:
            match = re.search(r'"payloadId"\s*:\s*"([^"]+)"', content)
            if match:
                payload_id = match.group(1)

        if task_type:
            doc.metadata["taskType"] = task_type
            last_valid_task_type = task_type
        else:
            doc.metadata["taskType"] = "unknown"
            if last_valid_task_type:
                doc.metadata["slideTaskType"] = last_valid_task_type

        doc.metadata["payloadId"] = payload_id if payload_id else "N/A"

        enriched_docs.append(doc)

        print(f"\n--- Chunk {i + 1} ---")
        print(doc.page_content)
        print("Metadata:", doc.metadata)

    print(f"✅ Split into {len(enriched_docs)} chunks from {path}")
    return enriched_docs


# Ask a question using Bedrock Embeddings + Claude + LangChain RAG
def ask_question(question):
    docs = load_documents()

    # return
    # Embed with Amazon Titan via Bedrock
    embedder = BedrockEmbeddings(
        model_id="amazon.titan-embed-text-v2:0",
        region_name="ap-south-1"
    )

    # # Create vector store and retriever
    vectordb = Chroma.from_documents(docs, embedding=embedder)
    retriever = vectordb.as_retriever(search_kwargs={
        "k": 3,
        "filter": {"taskType": "pickpackrules"}  # 👈 Filter on metadata
    })

    # Use your Claude LLM wrapper
    llm = get_llm()

    # Optional: prompt formatting
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are an intelligent assistant. Use the context below to answer the question.
If you cannot find the answer, say: "I couldn't find that information."

Context:
{context}

Question: {question}
Answer:
"""
    )
    
    #  # Manually retrieve and format
    retrieved_docs = retriever.invoke(question)
    for doc in retrieved_docs:
        print("📄 Chunk:\n", doc.page_content)
        print("🧷 Metadata:", doc.metadata)
    
    print(f"🔍 Retrieved {len(retrieved_docs)} chunks")
    
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    print("📄 Context:\n", context)
    
    return
    formatted_prompt = prompt.format(context=context, question=question)
    
    # Stuff all chunks as context (assumes they're relevant)
    # context = "\n\n".join(doc.page_content for doc in docs)
    # formatted_prompt = prompt.format(context=context, question=question)

    # Claude inference (direct)
    response = llm.invoke([{"role": "user", "content": formatted_prompt}])
    
    # Print answer
    print(f"\n💬 Question: {question}")
    print("📄 Answer:\n", response.content)
    
    # DEBUG full response
    print("\n📦 Full Raw Metadata:")
    print(response.response_metadata)
    

    # # Create RAG chain
    # qa_chain = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff",
    #     chain_type_kwargs={"prompt": prompt}
    # )

    # # Run the question through the chain
    # result = qa_chain.invoke({"query": question})

    # print(f"\n💬 Question: {question}")
    # print(f"📄 Answer:\n{result}")
   

if __name__ == "__main__":
    ask_question("Is pickpack complete for the task?")
