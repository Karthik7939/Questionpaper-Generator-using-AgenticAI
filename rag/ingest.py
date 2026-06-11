import argparse
from src.pipeline.rag_pipeline import RAGPipeline
from src.utils.logger import logger

def main():
    parser = argparse.ArgumentParser(description="Ingest documents into RAG system")
    parser.add_argument("path", help="Path to file or directory to ingest")
    parser.add_argument("--update", action="store_true", help="Add to existing index instead of rebuilding")
    args = parser.parse_args()

    pipeline = RAGPipeline()
    pipeline.ingest(args.path, update=args.update)
    logger.info("Ingestion done!")

if __name__ == "__main__":
    main()