# Milvus Embedding API

This repository provides a FastAPI-based service for generating text embeddings (via an `EmbeddingModelWrapper`) and interacting with a Milvus database (`MilvusService`). The service allows you to insert new embeddings into Milvus, create embeddings on-the-fly (without inserting), and perform similarity searches against a specified collection.

---

## Table of Contents
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running the Service](#running-the-service)
- [API Endpoints](#api-endpoints)
  - [1. `POST /create_embedding`](#1-post-create_embedding)
  - [2. `POST /insert_entity`](#2-post-insert_entity)
  - [3. `POST /search`](#3-post-search)
- [Usage Examples](#usage-examples)
  - [Create Embedding Example](#create-embedding-example)
  - [Insert Entity Example](#insert-entity-example)
  - [Search Example](#search-example)

---

## Features

1. **Create Embeddings**: Generate vector embeddings for a given text using the `EmbeddingModelWrapper`.
2. **Insert into Milvus**: Insert the embeddings (and their associated text entities) into a Milvus collection.
3. **Search**: Perform a vector similarity search against the Milvus collection to find the most similar text entities.

---

## Prerequisites

- **Python 3.10+** (for running the FastAPI application).
- A running **[Milvus](https://milvus.io/)** instance.
- (Optional) Familiarity with Docker if you plan on containerizing the application.

---

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/milvus-embedding-service.git
   cd milvus-embedding-service

	2.	Create a virtual environment (recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


	3.	Install dependencies (adjust as needed if you have a custom requirements.txt):

pip install -r requirements.txt
# plus any other dependencies like your embedding model libraries


	4.	Configure Milvus:
	•	Ensure your Milvus server is running.

Be sure to navigate to docker folder otherwise create one
in the docker folder run:
```bash
  curl -sfL https://raw.githubusercontent.com/milvus-io/milvus/master/scripts/standalone_embed.sh -o standalone_embed.sh
```
Then run:
```bash
  bash standalone_embed.sh start
```
	•	Update the connection alias and db_name if necessary in the code (MilvusService(connection_alias="dg_strato", db_name="dg_strato")).
	•	Make sure you have a collection named knowledge_base in Milvus, or adjust the code to match your existing collection name.

Running the Service

1.1 Create a collection:

- Use this as an example to create a collection in the Milvus vector database 
```json
{
    "collection_name": "knowledge_base",
    "schema_fields": [
        {
            "field_name": "id",
            "datatype": "INT64",
            "is_primary": true,
            "auto_id": true
        },
        {
            "field_name": "vector",
            "datatype": "FLOAT_VECTOR",
            "dim": 1536
        },
        {
            "field_name": "varchar",
            "datatype": "VARCHAR",
            "max_length": 3000
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
```
Use the API to do so:
---

### **Using curl**

```bash
curl -X POST "http://localhost:8000/create_collection" \
  -H "Content-Type: application/json" \
  -d '{
        "collection_name": "knowledge_base",
        "schema_fields": [
            {
                "field_name": "id",
                "datatype": "INT64",          // Use string representations; these will be mapped to the correct DataType
                "is_primary": true,
                "auto_id": true
            },
            {
                "field_name": "vector",
                "datatype": "FLOAT_VECTOR",
                "dim": 1536
            },
            {
                "field_name": "varchar",
                "datatype": "VARCHAR",
                "max_length": 3000
            }
        ],
        "index_params": [
            {
                "field_name": "vector",
                "index_type": "AUTOINDEX",
                "metric_type": "COSINE"
            }
        ]
    }'
```

---

### **Explanation**

- **Endpoint URL:**  
  The API is accessible at `http://localhost:8000/create_collection`.

- **HTTP Method:**  
  A `POST` request is used to create a new collection.

- **Headers:**  
  The `Content-Type: application/json` header tells the server that the request body is in JSON format.

- **Payload:**  
  The JSON data in the `-d` parameter includes:  
  - **collection_name:** The name of the collection to be created (`knowledge_base` in this example).  
  - **schema_fields:** An array defining each field in the collection:
    - The `"id"` field is defined as an `INT64` primary key with auto-generated IDs.
    - The `"vector"` field is a `FLOAT_VECTOR` with a dimension of 1536.
    - The `"varchar"` field is a `VARCHAR` field with a maximum length of 3000 characters.
  - **index_params:** Index parameters for the `"vector"` field, using an `AUTOINDEX` and a `COSINE` metric.

Run this command in your terminal (assuming your FastAPI server is running on port 8000) to create the collection with the specified schema and indexing options.
Run the FastAPI application via Uvicorn:

Below is an example of how to configure and run the API endpoint using Postman:

Steps to Use Postman
	1.	Open Postman and Create a New Request:
	•	Click “New” and then “Request”.
	•	Name your request (e.g., Create Collection) and save it in a collection if desired.
	2.	Set the Request Method and URL:
	•	Change the request method to POST.
	•	Enter the URL:

http://localhost:8000/create_collection


	3.	Set the Request Headers:
	•	Under the “Headers” tab, add a new header:
	•	Key: Content-Type
	•	Value: application/json
	4.	Set the Request Body:
	•	Click on the “Body” tab.
	•	Select “raw”.
	•	Choose “JSON” from the dropdown (usually appears as Text by default).
	•	Paste the following JSON payload into the body editor:

```json
{
    "collection_name": "knowledge_base",
    "schema_fields": [
        {
            "field_name": "id",
            "datatype": "INT64",          // Use string representations; these will be mapped to the correct DataType
            "is_primary": true,
            "auto_id": true
        },
        {
            "field_name": "vector",
            "datatype": "FLOAT_VECTOR",
            "dim": 1536
        },
        {
            "field_name": "varchar",
            "datatype": "VARCHAR",
            "max_length": 3000
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

```

	Note: Although the inline comments (// ...) are shown here for explanation, JSON does not support comments. Remove them before sending the request. The final JSON should look like this:

	5.	Send the Request:
	•	Click the “Send” button.
	•	You should see a response from the server in the “Response” section indicating whether the collection was created successfully or if there was an error.

Summary
	•	Method: POST
	•	URL: http://localhost:8000/create_collection
	•	Headers: Content-Type: application/json
	•	Body: (Raw JSON payload as provided)
# Now:

cd endpoints

python api.py

Or:

uvicorn main:app --host 0.0.0.0 --port 8000

After starting, the service should be available at:
http://0.0.0.0:8000

You can also explore and test your API directly with the interactive docs at:
http://0.0.0.0:8000/docs (Swagger UI)

API Endpoints

1. POST /create_embedding

Purpose: Generate the embedding vector for a given text without inserting it into Milvus.

Request Body:

{
  "text": "string"
}

Response:

{
  "text": "string",
  "embedding": [/* 1536-d float array */]
}

Example

curl -X POST -H "Content-Type: application/json" \
-d '{"text": "Hello world"}' \
http://localhost:8000/create_embedding

2. POST /insert_entity

Purpose: Generate the embedding for a given entity string and insert it into the Milvus collection. An optional partition_name can be specified.

Request Body:

{
  "entity": "string",
  "partition_name": "optional_partition_name"
}

	•	entity (required): The text/string you want to embed and store.
	•	partition_name (optional): If not provided, data will be inserted into the default partition of the collection.

Response:

{
  "status": "success",
  "insert_count": "<number_of_inserted_entities>",
  "message": "Inserted entity: 'entity_text...'"
}

Example

curl -X POST -H "Content-Type: application/json" \
-d '{"entity": "Information about a new topic"}' \
http://localhost:8000/insert_entity

3. POST /search

Purpose: Vector similarity search against the knowledge_base collection. Returns top N results with id and entity fields.

Request Body:

{
  "query": "string",
  "top_n": 3,
  "partition_names": ["optional", "list", "of", "partitions"]
}

	•	query: The text to embed and search for similar entities in Milvus.
	•	top_n: The maximum number of search results to return (default: 3).
	•	partition_names (optional): List of partition names to search within. If not provided, searches across all partitions.

Response:

{
  "query": "string",
  "top_n": 3,
  "results": [
    {
      "id": <milvus_id>,
      "entity": "some entity text",
      "vector": <vector>
    },
    ...
  ]
}

Example

curl -X POST -H "Content-Type: application/json" \
-d '{"query": "A piece of text for searching", "top_n": 5}' \
http://localhost:8000/search

Usage Examples

Below are a few quick usage examples (using curl) to illustrate how to interact with the endpoints.

Create Embedding Example

curl -X POST -H "Content-Type: application/json" \
-d '{
  "text": "This is a test sentence"
}' \
http://localhost:8000/create_embedding

Insert Entity Example

curl -X POST -H "Content-Type: application/json" \
-d '{
  "entity": "Learning about Milvus and FastAPI"
}' \
http://localhost:8000/insert_entity

Search Example

curl -X POST -H "Content-Type: application/json" \
-d '{
  "query": "Milvus tutorial",
  "top_n": 3
}' \
http://localhost:8000/search

