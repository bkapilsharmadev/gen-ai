import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from common.embeddings import get_embedder
from common.llm_chain import get_llm

def load_chunks(path="offering.json"):
    loader = JSONLoader(file_path=path, jq_schema=".", text_content=False)
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_documents(docs)

def cosine_similarity(vec1, vec2):
    v1, v2 = np.array(vec1), np.array(vec2)
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)

def build_prompt(context, question):
    return f"""
You are an intelligent assistant. Use the context below to answer the question.
If the answer is not present, reply with "I couldn't find that information."

Context:
\"\"\"
{context}
\"\"\"

Question: {question}
Answer:
"""

def ask_question(question):
    embedder = get_embedder()
    llm = get_llm()

    chunks = load_chunks()
    chunk_texts = [doc.page_content for doc in chunks]

    chunk_vectors = embedder.embed_documents(chunk_texts)
    query_vector = embedder.embed_query(question)

    scores = [cosine_similarity(v, query_vector) for v in chunk_vectors]
    top_chunks = [text for _, text in sorted(zip(scores, chunk_texts), reverse=True)[:3]]
    context = "\n".join(top_chunks)

    prompt = build_prompt(context, question)
    response = llm.invoke([{"role": "user", "content": prompt}])

    print(f"\n💬 Question: {question}")
    print("📄 Answer:\n" + response.content)

if __name__ == "__main__":
    ask_question("Why is this offering for?")
