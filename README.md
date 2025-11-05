# Multi-Stage RAG Project

## Vector Database Module (`app/vdb/vectordb.py`)

This module handles the creation and population of a Qdrant vector database collection for a multi-stage Retrieval-Augmented Generation (RAG) system. It processes CSV files containing chunked text data with dense and sparse embeddings, upserting them into a Qdrant collection optimized for efficient vector search.

### Key Components

- **Configuration**: Loads environment variables, sets data directory paths, and defines Qdrant client parameters including host, API key, collection name, and vector names.
- **Qdrant Client**: Initializes a connection to a cloud-hosted Qdrant instance with a 60-second timeout.
- **Collection Management**: Checks for existing collections and either updates configuration or creates a new collection with specified vector parameters.
- **Data Processing**: Recursively finds CSV files in the data directory, processes each file to extract embeddings and metadata, and batches upserts to the collection.

### Data Structure and Logic

The module processes CSV files with the following key columns:
- `chunked_text.dense_embedding`: Dense vector representations (e.g., from models like BERT or Sentence Transformers)
- `chunked_text.sparse_indices` and `chunked_text.sparse_values`: Sparse vector components (e.g., from BM25 or SPLADE)
- Metadata fields including chunk ID, text content, source information, and hierarchical folder structure

**Vector Collection Creation Process:**
1. **Vector Size Determination**: Reads the first CSV file to infer dense vector dimensions from the `chunked_text.dense_embedding` column.
2. **Collection Configuration**:
   - Dense vectors: Configured with cosine distance, stored on disk
   - Sparse vectors: Indexed with on-disk storage
   - Scalar quantization: INT8 type with 0.99 quantile, always in RAM for fast access
   - Optimizer settings: Indexing threshold of 10,000, no default segments, memmap threshold of 20,000
3. **Batch Upsert**: Processes data in batches of 100 points, creating PointStruct objects with dual vector representations and comprehensive payload metadata.
4. **Verification**: Retrieves and logs collection statistics, quantization status, and memory compression estimates.

**Critical Factors:**
- **Dual Vector Support**: Combines dense and sparse embeddings for hybrid search capabilities
- **Quantization**: Reduces memory footprint by ~4x through INT8 scalar quantization
- **Batch Processing**: Efficiently handles large datasets with configurable batch sizes
- **Error Handling**: Gracefully skips malformed data and provides detailed logging
- **On-Disk Storage**: Optimizes for large-scale collections with disk-based vector storage

The collection supports advanced RAG workflows by storing semantically rich text chunks with their vector representations, enabling fast similarity search across both dense and sparse vector spaces.