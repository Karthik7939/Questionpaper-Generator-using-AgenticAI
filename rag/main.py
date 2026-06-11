from src.pipeline.rag_pipeline import RAGPipeline
from src.generation.context_builder import format_chunks_for_prompt


def main():
    pipeline = RAGPipeline()

    print("\n=== RAG Chunking & Retrieval Service ===")
    print("1. Ingest new document")
    print("2. Load existing index and retrieve")
    choice = input("Choice (1/2): ").strip()

    if choice == "1":
        path = input("Enter file or directory path: ").strip()
        update = input("Update existing index? (y/n): ").strip().lower() == "y"
        pipeline.ingest(path, update=update)
    else:
        pipeline.load()

    print(f"\nRAG system ready ({len(pipeline.all_chunks)} chunks). Type 'exit' to quit.\n")
    while True:
        query = input("Retrieval query: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue
        result = pipeline.retrieve(query, verbose=True)
        print("\n--- RETRIEVED CHUNKS ---")
        print(format_chunks_for_prompt(result["chunks"]))
        print("\n--- SOURCES ---")
        for s in result["sources"]:
            print(f"  [{s['chunk_id']}] {s['source']} | {s['first_line'][:80]}")
        print()


if __name__ == "__main__":
    main()
