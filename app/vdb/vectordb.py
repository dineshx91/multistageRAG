import ast
import pandas as pd
from dotenv import load_dotenv
import os
from qdrant_client import QdrantClient, models
#from qdrant_client.models import Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct, SparseIndexParams
from qdrant_client.models import (
    Distance, VectorParams, SparseVectorParams, SparseVector, PointStruct, SparseIndexParams, OptimizersConfigDiff
)

# Load environment variables
load_dotenv()

# ==============================
# Configuration
# ==============================
# Get the directory of the current script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIRECTORY = os.path.join(SCRIPT_DIR, "..", "..", "src_data", "dataframes") # Points to the actual CSV data directory

# Qdrant Client Configuration

QDRANT_HOST = os.getenv("QDRANT_HOST_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY_M")
COLLECTION_NAME = "multi_stageRAG"
DENSE_VECTOR_NAME = "dense_vector"
SPARSE_VECTOR_NAME = "sparse_vector"



QDRANT_UPSERT_BATCH_SIZE = 100


def process_csv_batch(data_directory: str):
    # ==============================
    # Step 1: Find all CSV files
    # ==============================
    file_paths = []
    for root, _, files in os.walk(data_directory):
        for file in files:
            if file.endswith(".csv"):
                file_paths.append(os.path.join(root, file))
    
    if not file_paths:
        print(f"No CSV files found in {data_directory} or its subdirectories. Exiting.")
        return

    # ==============================
    # Step 2: Initialize Qdrant Client
    # ==============================
    client = QdrantClient(host=QDRANT_HOST, api_key=QDRANT_API_KEY, timeout=60)

    # ==============================
    # Step 3: Determine vector size and create/update collection
    # ==============================
    try:
        temp_df = pd.read_csv(file_paths[0], nrows=1)
        if "chunked_text.dense_embedding" not in temp_df.columns:
            print(f"Error: 'chunked_text.dense_embedding' column not found in {file_paths[0]}.")
            return
        
        vector_size = len(ast.literal_eval(temp_df["chunked_text.dense_embedding"].iloc[0]))
    except Exception as e:
        print(f"Error determining vector size from {file_paths[0]}: {e}")
        return

    scalar_config = models.ScalarQuantization(
        scalar=models.ScalarQuantizationConfig(
            type=models.ScalarType.INT8,
            quantile=0.99,
            always_ram=True,
        )
    )

    # Check if collection exists
    if client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"Collection '{COLLECTION_NAME}' already exists. Updating configuration...")
        try:
            # Update collection configuration without deleting data
            client.update_collection(
                collection_name=COLLECTION_NAME,
                optimizers_config=models.OptimizersConfigDiff(
                    indexing_threshold=10000,  # Updated from 1000
                    default_segment_number=0,
                    memmap_threshold=20000
                ),
            )
            print(f"✅ Collection '{COLLECTION_NAME}' configuration updated.")
        except Exception as e:
            print(f"⚠️ Could not update collection config: {e}")
            print(f"Continuing with existing collection configuration...")
    else:
        # Create new collection if it doesn't exist
        print(f"Collection '{COLLECTION_NAME}' does not exist. Creating new collection...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={DENSE_VECTOR_NAME: VectorParams(
                size=vector_size, distance=Distance.COSINE, on_disk=True)},
            sparse_vectors_config={SPARSE_VECTOR_NAME: SparseVectorParams(
                index=SparseIndexParams(on_disk=True))},
            quantization_config=scalar_config,
            optimizers_config=OptimizersConfigDiff(
                indexing_threshold=10000,  # Updated from 1000
                default_segment_number=0,
                memmap_threshold=20000
            )
        )
        print(f"✅ Collection '{COLLECTION_NAME}' created with vector size {vector_size}.")

    # ==============================
    # Step 4: Process files and upsert in batches
    # ==============================
    points_to_upsert_batch = []
    total_points_inserted = 0

    for f_idx, f in enumerate(file_paths):
        print(f"--> Processing file {f_idx + 1}/{len(file_paths)}: {f}")
        try:
            df = pd.read_csv(f)
            df["embedding_list"] = df["chunked_text.dense_embedding"].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else None
            )
            df["sparse_indices"] = df["chunked_text.sparse_indices"].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else None
            )
            df["sparse_values"] = df["chunked_text.sparse_values"].apply(
                lambda x: ast.literal_eval(x) if pd.notna(x) else None
            )
            df["source_file"] = f

            for _, row in df.iterrows():
                if row["embedding_list"] is None or row["sparse_indices"] is None or row["sparse_values"] is None:
                    continue

                payload = {
                    "chunk_id": row["chunked_text.chunk_id"],
                    "text": row["chunked_text.text"],
                    "docs_source": row["chunked_text.source_data_docs_source"],
                    "folder_name": row["chunked_text.source_data_folder_name"],
                    "sub_folder": row["chunked_text.source_data_sub_folder"],
                    "sub_sub_folder": row["chunked_text.source_data_sub_sub_folder"],
                    "source_file_name": row["chunked_text.source_data_source_file_name"],
                    "file_id": row["chunked_text.source_data_file_id"],
                    "title": row["title"],
                    "path": row["path"],
                    "chunk_index": row["chunked_text.chunk_index"],
                    "source_file": row["source_file"],
                }

                point = PointStruct(
                    id=row["chunked_text.chunk_id"],
                    vector={
                        DENSE_VECTOR_NAME: row["embedding_list"],
                        SPARSE_VECTOR_NAME: SparseVector(
                            indices=row["sparse_indices"],
                            values=row["sparse_values"]
                        )
                    },
                    payload=payload
                )

                points_to_upsert_batch.append(point)

                if len(points_to_upsert_batch) >= QDRANT_UPSERT_BATCH_SIZE:
                    client.upsert(
                        collection_name=COLLECTION_NAME,
                        wait=True,
                        points=points_to_upsert_batch
                    )
                    total_points_inserted += len(points_to_upsert_batch)
                    print(f"    ... Upserted batch of {len(points_to_upsert_batch)} points. Total inserted: {total_points_inserted}")
                    points_to_upsert_batch = []

        except Exception as e:
            print(f"    !!! Error processing file {f}: {e}. Skipping this file.")

    if points_to_upsert_batch:
        client.upsert(
            collection_name=COLLECTION_NAME,
            wait=True,
            points=points_to_upsert_batch
        )
        total_points_inserted += len(points_to_upsert_batch)
        print(f"    ... Upserted final batch of {len(points_to_upsert_batch)} points.")

    print(f"\n✅ Finished. Inserted a total of {total_points_inserted} vectors from {len(file_paths)} CSV files into Qdrant.")

    # Verify quantization status
    try:
        collection_info = client.get_collection(collection_name=COLLECTION_NAME)
        print(f"✅ Collection '{COLLECTION_NAME}' details:")
        print(f"   Vector count: {collection_info.vectors_count}")
        print(f"   Points count: {collection_info.points_count}")
        print(f"   Status: {collection_info.status}")

        # Log quantization configuration
        if hasattr(collection_info.config, 'quantization_config'):
            quant_config = collection_info.config.quantization_config
            if quant_config:
                print(f"   Quantization enabled: {quant_config}")
                if hasattr(quant_config, 'scalar'):
                    scalar_config = quant_config.scalar
                    print(f"   Quantization type: {scalar_config.type}")
                    print(f"   Quantization quantile: {scalar_config.quantile}")
                    print(f"   Always RAM: {scalar_config.always_ram}")
            else:
                print("   Quantization: Disabled")

        # Log memory compression estimation
        if hasattr(collection_info.config, 'quantization_config') and collection_info.config.quantization_config:
            if hasattr(collection_info.config.quantization_config, 'scalar'):
                original_memory = vector_size * 4  # 32-bit floats
                quantized_memory = vector_size * 1  # 8-bit integers
                compression_ratio = original_memory / quantized_memory
                print(f"   Expected memory compression: {compression_ratio:.1f}x reduction")
                print(f"   Memory per vector: {original_memory} bytes -> {quantized_memory} bytes")

        # Log optimizer configuration if available
        if hasattr(collection_info.config, 'optimizer_config'):
            optimizer = collection_info.config.optimizer_config
            print(f"   Optimizer - Default segment number: {optimizer.default_segment_number}")
            print(f"   Optimizer - Max segment size: {optimizer.max_segment_size}")
            print(f"   Optimizer - Indexing threshold: {optimizer.indexing_threshold}")
            print(f"   Optimizer - Memmap threshold: {optimizer.memmap_threshold}")

    except Exception as e:
        print(f"❌ Error retrieving collection info: {e}")


# ==============================
# Main execution block
# ==============================
# Note: Search functionality has been moved to search_query.py
if __name__ == "__main__":
    print("Starting data processing...")
    process_csv_batch(DATA_DIRECTORY)
    print("Data processing completed. Use search_query.py for vector search functionality.")