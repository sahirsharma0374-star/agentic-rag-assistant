import os
import chromadb


def load_documents(docs_dir="docs"):
    """Load all markdown files from the docs directory."""
    documents = []
    for filename in os.listdir(docs_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(docs_dir, filename)
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"content": content, "source": filename})
    return documents


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks 

def ingest_documents(docs_dir="docs"):
    """Load, chunk, and store documents in ChromaDB."""
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_or_create_collection(name="research_docs")

    documents = load_documents(docs_dir)
    all_chunks = []
    all_ids = []
    all_metadatas = []

    for doc in documents:
        chunks = chunk_text(doc["content"])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['source']}_chunk_{i}"
            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadatas.append({"source": doc["source"], "chunk_index": i})

    collection.upsert(documents=all_chunks, ids=all_ids, metadatas=all_metadatas)
    print(f"Ingested {len(all_chunks)} chunks from {len(documents)} documents.")
    return collection






def search_documents(query, n_results=3):
    """Search the document collection for relevant chunks."""

    # STEP 1: Connect to the same persistent database used during ingestion.
    client = chromadb.PersistentClient(path="./chroma_db")

    # STEP 2: Open the collection containing our document chunks.
    collection = client.get_or_create_collection(
        name="research_docs"
    )

    # STEP 3: Search for chunks relevant to the user's question.
    # query_texts accepts a list of questions, so we wrap query in [].
    # n_results controls the maximum number of results requested.
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )

    
    formatted_results = []


    for i in range(len(results["ids"][0])):

        
        formatted_results.append({
            "content": results["documents"][0][i],
            "source": results["metadatas"][0][i]["source"],
            "distance": results["distances"][0][i]
        })

    
    return formatted_results



if __name__ == "__main__":

    
    ingest_documents()

    print("\nTesting search...")

    # Search for information about medallion architecture.
    results = search_documents(
        "WhY IS CLOUD COMPUTING IMPORTANT?",
        n_results=3
    )

    # 
    for r in results:
        print(f"\n[{r['source']}] (distance: {r['distance']:.4f})")
        print(r["content"][:200])





