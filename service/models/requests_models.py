from typing import List, Optional
from pydantic import BaseModel


# ---------------------------------------------------
# Search Service Models
# ---------------------------------------------------

class SearchRequest(BaseModel):
    """
    Request model for searching.

    Attributes:
        query (str): The search query text.
        top_k (Optional[int]): The number of top search results to return (default is 10).
    """
    query: str
    top_k: Optional[int] = 10


class SearchResponse(BaseModel):
    """
    Response model for a search request.

    Attributes:
        vectors (List[List[float]]): A list of dense vectors (each vector is a list of floats)
                                     corresponding to search results.
    """
    vectors: List[List[float]]


# ---------------------------------------------------
# Ingest Service Models
# ---------------------------------------------------

class IngestRequest(BaseModel):
    """
    Request model for ingesting documents.

    Attributes:
        documents (List[str]): A list of document texts to be ingested.
    """
    documents: List[str]


class IngestResponse(BaseModel):
    """
    Response model for an ingest request.

    Attributes:
        message (str): Confirmation message about the ingestion process.
    """
    message: str


# ---------------------------------------------------
# CreateVector Service Models
# ---------------------------------------------------

class CreateVectorRequest(BaseModel):
    """
    Request model for creating a dense vector from text.

    Attributes:
        text (str): The text input to be converted into a dense vector.
    """
    text: str


class CreateVectorResponse(BaseModel):
    """
    Response model for vector creation.

    Attributes:
        vector (List[float]): The dense embedding vector as a list of floats.
    """
    vector: List[float]


# ---------------------------------------------------
# Request Status Service Models
# ---------------------------------------------------

class RequestStatusRequest(BaseModel):
    """
    Request model for checking the status of a previous request.

    Attributes:
        request_hash (str): The hash (or unique identifier) of the request.
    """
    request_hash: str


class RequestStatusResponse(BaseModel):
    """
    Response model for the request status endpoint.

    Attributes:
        request_hash (str): The hash of the request.
        status (str): The current status of the request (e.g., 'processing', 'completed', 'failed').
    """
    request_hash: str
    status: str