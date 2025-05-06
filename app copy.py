import json
from langchain_text_splitters import RecursiveJsonSplitter
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from common.llm_chain import get_llm  # Your own Claude wrapper (via Bedrock)

# Load and split structured JSON using RecursiveJsonSplitter
def load_documents(path="offering.json"):
    # Load raw JSON content
    with open(path, "r", encoding="utf-8") as f:
        json_data = json.load(f)

    # Initialize splitter
    splitter = RecursiveJsonSplitter(max_chunk_size=800)

    # Correct usage: pass the raw JSON as a list
    # docs = splitter.create_documents(texts=[json_data])
    split_docs = splitter.create_documents(texts=[json_data], convert_lists=True)
    
    
    # for i, doc in enumerate(split_docs):
    #     print(f"\n--- Chunk {i + 1} ---")
    #     print(doc.page_content[:300])  # print a preview
    #     print("Metadata:", doc.metadata)
        
    for i, doc in enumerate(split_docs):
        print(f"\n--- Chunk {i + 1} ---\n")
        print(doc.page_content)           # Full chunk content
        print("Metadata:", doc.metadata)  # Optional: if you're using metadata


    print(f"✅ Split into {len(split_docs)} chunks from {path}")
    return split_docs

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
    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

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
    context = "\n".join(doc.page_content for doc in retrieved_docs)
    print(f"🔍 Retrieved {len(retrieved_docs)} chunks")
    print("📄 Context:\n", context)  # Print a preview of the context
    
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
