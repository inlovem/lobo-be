import logging
import sys
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pymilvus import DataType

from embeddings.constants.embeddings_constants import EmbeddingConstants
from embeddings.embedding_model import EmbeddingModelWrapper
from milvus_db.milvus_service import MilvusService, ColorHandler
from service.models.requests_models import CreateEmbeddingPayload, CreateCollectionPayload, InsertPayload, SearchPayload

##############################################################################
# Configure Logger (ColorHandler)
##############################################################################


logger = logging.getLogger("milvus_service_ logger")
logger.setLevel(logging.DEBUG)  # or INFO, depending on desired verbosity

# Remove existing handlers if any to avoid duplicates
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add our custom color handler
color_handler = ColorHandler()
color_handler.setLevel(logging.DEBUG)  # or any desired level
logger.addHandler(color_handler)

# Optional: set a logging formatter (for timestamp, module name, etc.)
formatter = logging.Formatter(
    fmt="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
color_handler.setFormatter(formatter)

###############################################################################
# 1. Basic Setup: Startup routine for Milvus & Embeddings
###############################################################################
app = FastAPI()

logger.info("===== [Startup] Server is configuring... =====")

try:
    # 1a. Connect to Milvus
    service = MilvusService(connection_alias="default", db_name="default")
    logger.info("Successfully connected to MilvusService.")

    # 1b. Check Milvus health by calling get_server_version() or listing collections
    version = service.client.get_server_version()
    logger.info(f"Milvus server version: {version}")

    # 1c. Simulate a wait time for readiness
    time.sleep(2)
    logger.info("Waited 2 seconds for system readiness.")

    # 1d. We assume "knowledge_base" is already created/loaded externally
    collection_name = "knowledge_base"

    # 1e. Load and example the embedding model
    logger.info("Loading embedding model and testing a sample embed...")
    embedding_model = EmbeddingModelWrapper()
    test_vector = embedding_model.encode("This is a sample text to verify embeddings.")
    logger.info(f"Test embed successful. Length of embedding: {len(test_vector)}")

    logger.info("===== [Startup] Configuration complete. Server ready. =====")

except Exception as e:
    logger.error(f"Startup/Configuration failed: {e}", exc_info=True)
    sys.exit(1)


###############################################################################
# 3. Endpoints
###############################################################################

@app.post("/create_embedding")
def create_embedding(payload: CreateEmbeddingPayload):
    """
    Return the 1536-d embedding for a given text without inserting into Milvus.
    Useful for debugging or advanced usage.
    """
    vector = embedding_model.encode(payload.text)
    if hasattr(vector, "tolist"):
        vector = vector.tolist()
    return {
        "text": payload.text,
        "embedding": vector
    }


###############################################################################
# New Endpoint: Create a Collection
###############################################################################
@app.post("/create_collection")
def create_collection(payload: CreateCollectionPayload):
    """
    Creates a new collection in the Milvus database based on the provided JSON payload.

    Example JSON payload:
    {
        "collection_name": "knowledge_base",
        "schema_fields": [
            {
                "field_name": "id",
                "datatype": "INT64",          // Primary key (supported: INT64, VARCHAR)
                "is_primary": true,
                "auto_id": true
            },
            {
                "field_name": "vector",
                "datatype": "FLOAT_VECTOR",   // Vector field for embeddings
                "dim": 1536
            },
            {
                "field_name": "varchar",
                "datatype": "VARCHAR",        // Scalar field of type VARCHAR
                "max_length": 3000
            },
            {
                "field_name": "flag",
                "datatype": "BOOL"            // Scalar field of type BOOL
            },
            {
                "field_name": "score",
                "datatype": "FLOAT"           // Scalar field of type FLOAT
            }
        ],
        "index_params": [
            {
                "field_name": "vector",
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            }
        ]
    }
    """
    collection_name_arg = payload.collection_name
    schema_fields = payload.schema_fields
    index_params = payload.index_params

    # Check if the collection already exists.
    if service.client.has_collection(collection_name_arg):
        logger.warning(f"Collection '{collection_name_arg}' already exists.")
        raise HTTPException(
            status_code=400,
            detail=f"Collection '{collection_name_arg}' already exists."
        )

    # Mapping of string representations to DataType values.
    datatype_mapping = {
        # Primary key fields (also valid as scalar fields)
        "INT64": DataType.INT64,
        "VARCHAR": DataType.VARCHAR,
        # Scalar field types
        "BOOL": DataType.BOOL,
        "INT8": DataType.INT8,
        "INT16": DataType.INT16,
        "INT32": DataType.INT32,
        "FLOAT": DataType.FLOAT,
        "DOUBLE": DataType.DOUBLE,
        "JSON": DataType.JSON,
        "Array": DataType.ARRAY,  # Use this if your Milvus client supports an array type.
        # Vector field type (if needed)
        "FLOAT_VECTOR": DataType.FLOAT_VECTOR,
    }

    # Update each schema field's "datatype" from a string to a DataType enum.
    mapped_schema_fields = []
    for field in schema_fields:
        dt_str = field.get("datatype")
        if dt_str not in datatype_mapping:
            raise HTTPException(status_code=400, detail=f"Unsupported datatype: {dt_str}")
        # Replace the string with the corresponding DataType enum.
        field["datatype"] = datatype_mapping[dt_str]
        mapped_schema_fields.append(field)

    try:
        # Create the collection using the mapped schema fields and index parameters.
        service.create_collection(
            collection_name=collection_name_arg,
            schema_fields=mapped_schema_fields,
            index_params=index_params
        )
        logger.info(f"Collection '{collection_name_arg}' created successfully.")
        return {
            "status": "success",
            "collection_name": collection_name_arg,
            "message": f"Collection '{collection_name_arg}' created successfully."
        }
    except Exception as e:
        logger.error(f"Failed to create collection '{collection_name_arg}': {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to create collection: {e}"
        )

@app.post("/insert_entity")
def insert_entity(payload: InsertPayload):
    """
    Endpoint that takes an entity string, embeds it, and inserts (vector, entity) into Milvus.
    """
    entity_input = payload.entity
    partition_name = payload.partition_name

    # 1) Generate embedding
    vector = embedding_model.encode(entity_input)

    # 2) Prepare row for Milvus
    row = {
        "vector": vector,
        "entity": entity_input
    }

    # 3) Insert into partition or default partition
    if partition_name:
        result = service.insert_into_partition(
            collection_name=collection_name,
            fields_data=[row],
            partition_name=partition_name
        )
        logger.info(f"Inserted 1 row into partition '{partition_name}'. Result: {result}")
    else:
        result = service.client.insert(
            collection_name=collection_name,
            data=[row]
        )
        logger.info(
            f"Inserted 1 row into default partition of collection '{collection_name}'. "
            f"Result: {result}"
        )

    return {
        "status": "success",
        "insert_count": getattr(result, "insert_count", "unknown"),
        "message": f"Inserted entity: '{entity_input[:50]}...' (truncated if very long)"
    }


@app.post("/search")
def search(payload: SearchPayload):
    """
    Searches the 'knowledge_base' collection by embedding the `query` text, then performing a
    vector similarity search. Returns top_n results (with 'id' and 'entity' fields).
    If you want to search specific partitions, pass them in 'partition_names'.
    """
    # 1) Embed the query text
    query_vector = embedding_model.encode(payload.query)

    # 2) Perform flexible_search
    results = service.flexible_search(
        collection_name=collection_name,
        query_vector=query_vector,
        output_fields=["id", "entity"],  # or specify other fields
        top_n=payload.top_n,
        partition_names=payload.partition_names,
        metric_type="COSINE"
    )

    return {
        "query": payload.query,
        "top_n": payload.top_n,
        "results": results
    }


###############################################################################
# 4. Run via Uvicorn
###############################################################################
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)