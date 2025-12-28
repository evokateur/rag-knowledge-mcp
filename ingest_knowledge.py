#!/usr/bin/env python3
"""
Knowledge Base Ingestion Script

Ingests documents from the knowledge base directory using the RagBackend.
This ensures consistency between ingestion and querying.

Usage:
    python ingest_knowledge.py [--rebuild]

Options:
    --rebuild    Delete existing collection and rebuild from scratch
"""

import argparse
import asyncio
import logging
import sys

from config import RAG_KNOWLEDGE_DIR

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main():
    """Main ingestion workflow."""
    parser = argparse.ArgumentParser(description="Ingest knowledge base into ChromaDB")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Delete existing collection and rebuild from scratch",
    )
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Knowledge Base Ingestion")
    logger.info("=" * 60)
    logger.info(f"Knowledge directory: {RAG_KNOWLEDGE_DIR}")
    logger.info(f"Rebuild: {args.rebuild}")
    logger.info("=" * 60)

    # Import and initialize backend using factory
    from config import create_rag_backend

    backend = create_rag_backend()

    try:
        await backend.initialize()

        # Ingest directory using backend
        stats = await backend.ingest_directory(
            directory=RAG_KNOWLEDGE_DIR,
            rebuild=args.rebuild,
        )

        logger.info("=" * 60)
        logger.info("Ingestion Statistics:")
        logger.info(f"  Documents processed: {stats['documents_processed']}")
        logger.info(f"  Total chunks: {stats['total_chunks']}")
        logger.info("=" * 60)
        logger.info("Done! You can now use the MCP server to query the knowledge base.")
        logger.info("=" * 60)

    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error(f"\nPlease create the directory: {RAG_KNOWLEDGE_DIR}")
        logger.error("Add markdown files to it, then run this script again.")
        sys.exit(1)

    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        sys.exit(1)

    finally:
        await backend.close()


if __name__ == "__main__":
    asyncio.run(main())
