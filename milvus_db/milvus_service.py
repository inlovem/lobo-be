import logging
import os
from enum import Enum
from typing import List, Dict, Optional

import numpy as np
from dotenv import load_dotenv
from numpy import ndarray
from pymilvus import connections, db, MilvusClient, MilvusException, Collection
import threading


class QueryStatus(Enum):
    IDLE = "IDLE"
    BUSY = "BUSY"


##############################################################################
# Custom ColorHandler
##############################################################################
class ColorHandler(logging.StreamHandler):
    # https://en.wikipedia.org/wiki/ANSI_escape_code#Colors
    GRAY8 = "38;5;8"
    GRAY7 = "38;5;7"
    ORANGE = "33"
    RED = "31"
    WHITE = "37"  # Using 37 for plain white text
    RESET = "0"

    def emit(self, record):
        # Map log levels to colors
        level_color_map = {
            logging.DEBUG: self.GRAY8,
            logging.INFO: self.GRAY7,
            logging.WARNING: self.WHITE,  # user wants white for warnings
            logging.ERROR: self.RED  # user wants red for errors
        }

        csi = f"{chr(27)}["  # control sequence introducer
        color = level_color_map.get(record.levelno, self.RESET)

        # We can format the message with the regular formatter if we want
        # But for simplicity, just color the raw message
        formatted_msg = self.format(record)

        # Print colored message, then reset
        print(f"{csi}{color}m{formatted_msg}{csi}{self.RESET}m")


##############################################################################
# Configure Logger
##############################################################################
logger = logging.getLogger("milvus_service_logger")
logger.setLevel(logging.DEBUG)  # or INFO, depending on desired verbosity

# Remove existing handlers if any (helps avoid duplicate logs in some IDEs)
for handler in logger.handlers[:]:
    logger.removeHandler(handler)

# Add our color handler
color_handler = ColorHandler()
color_handler.setLevel(logging.DEBUG)  # or any desired level
logger.addHandler(color_handler)

# Optionally, set a logging formatter (for timestamp, module name, etc.)
formatter = logging.Formatter(
    fmt="%(asctime)s [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
color_handler.setFormatter(formatter)


##############################################################################
# MilvusService class
##############################################################################
class MilvusService:
    def __init__(self, connection_alias: str = "default", db_name: str = None):
        """
        Initialize the MilvusService.

        Parameters:
            connection_alias (str): Alias for the Milvus connection.
            db_name (str): The database name to use. If provided and it doesn't exist, it will be created.
        """
        self.db_name = db_name
        self.lock = threading.Lock()
        self.status = QueryStatus.IDLE  # Initialize status as IDLE


        # Class connection alias to ensure connection consistent and efficiency
        self.connection_alias = connection_alias
        self.db_name = db_name


        # Retrieve connection parameters from environment variables.
        load_dotenv()  # Load environment variables if not already loaded
        host = os.getenv("MILVUS_HOST", "127.0.0.1")
        port = os.getenv("MILVUS_PORT", "19530")
        token = os.getenv("MILVUS_TOKEN")  # Optional token for RBAC

        # Connect to Milvus, specifying the db_name
        connect_args = {
            "host": host,
            "port": port,
            "db_name": db_name
        }
        if token:
            connect_args["token"] = token

        connections.connect(alias=connection_alias, **connect_args)
        logger.info(f"Connected to Milvus at {host}:{port} with alias '{connection_alias}' using database '{db_name}'")

        # If db_name is provided and doesn't exist, create it
        if db_name:
            existing_dbs = db.list_database(using=connection_alias)
            if db_name not in existing_dbs:
                logger.warning(f"Database '{db_name}' does not exist. Creating it...")
                db.create_database(db_name, using=connection_alias)
                logger.info(f"Database '{db_name}' created successfully.")

        # Create MilvusClient with the same db_name
        uri = f"http://{host}:{port}"
        if token:
            self.client = MilvusClient(uri=uri, token=token, db_name=db_name)
        else:
            self.client = MilvusClient(uri=uri, db_name=db_name)
        logger.debug("MilvusClient instance created.")

    def flexible_search(
            self,
            collection_name: str,
            query_vector: np.ndarray,
            output_fields: List[str],
            top_n: int = 5,
            partition_names: Optional[List[str]] = None,
            expr: Optional[str] = None,
            timeout: float = 5.0,
            metric_type: str = "COSINE",
            # Range-based parameters for optional annular (range) search:
            radius: Optional[float] = None,
            range_filter: Optional[float] = None,
            vector_field_name: str = "vector"
    ) -> List[Dict]:
        """
        Execute a vector similarity search (or range-based search if radius/range_filter given).
        Try to retrieve all requested fields, including vector/scalar fields, with multiple fallback approaches.

        Returns a list of dicts, each containing only the requested fields actually retrieved.

        Parameters
        ----------
        collection_name : str
            Name of the collection.
        query_vector : np.ndarray
            Single query embedding. Must match 'dim' in schema.
        output_fields : List[str]
            Fields to be retrieved (e.g., ["id", "varchar", "vector"]).
        top_n : int
            Max number of results to return (after optional range filter).
        partition_names : Optional[List[str]]
            Partitions to search in (if any).
        expr : Optional[str]
            Scalar filter expression, e.g. "price > 100".
        timeout : float
            Search timeout in seconds.
        metric_type : str
            "COSINE", "IP", "L2", etc.
        radius : Optional[float]
            Range search parameter for the lower boundary (COSINE/IP => radius < distance).
        range_filter : Optional[float]
            Range search parameter for the upper boundary (COSINE/IP => distance <= range_filter).
        vector_field_name : str
            Name of the vector field in the schema.

        Returns
        -------
        List[Dict]
            e.g. [{"id": 123, "varchar": "...", "vector": [...], "distance": ...}, ...]
        """
        try:
            # 1. Get the collection using the same alias you connected with
            collection = Collection(
                name=collection_name,
                using=self.connection_alias
            )

            # 2. Load the collection (or specific partitions)
            if partition_names:
                collection.load(partition_names=partition_names)
            else:
                collection.load()

            # 3. Construct the search_params
            search_params = {
                "metric_type": metric_type,
                "params": {}
            }
            if radius is not None:
                search_params["params"]["radius"] = radius
            if range_filter is not None:
                search_params["params"]["range_filter"] = range_filter

            # 4. Prepare query data (list-of-lists)
            if hasattr(query_vector, "tolist"):
                data_to_search = [query_vector.tolist()]
            else:
                data_to_search = [query_vector]

            # 5. Perform the search
            results = collection.search(
                data=data_to_search,
                anns_field=vector_field_name,
                param=search_params,
                limit=top_n,
                expr=expr if expr else None,
                partition_names=partition_names if partition_names else None,
                output_fields=output_fields if output_fields else None,
                timeout=timeout,
            )

            # 6. Build final results
            formatted_results = []
            if results and len(results) > 0:
                hits = results[0]  # single query vector => list of hits
                for hit in hits:
                    row = self._extract_hit_fields(hit, output_fields)
                    # Optionally include distance or score:
                    # row["distance"] = hit.distance
                    # row["score"] = hit.score  # if your PyMilvus version has .score
                    if row:
                        formatted_results.append(row)

            logger.info(f"Returning {len(formatted_results)} search results.")
            return formatted_results

        except MilvusException as e:
            logger.error(f"Failed to execute (range) search in Milvus: {e}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error during search: {e}")
            return []
        finally:
            # Optional: release collection
            # collection.release()
            pass

    def _extract_hit_fields(self, hit, output_fields: List[str]) -> Dict:
        row = {}

        # Method 1: Try 'hit.to_dict()'
        entity_dict = {}
        if hasattr(hit, "to_dict"):
            entity_dict = hit.to_dict() or {}

        for field in output_fields:
            if field == "id":
                # Check if 'id' is in entity_dict, else fallback to hit.id
                if field not in entity_dict and hasattr(hit, "id"):
                    row["id"] = hit.id
                elif field in entity_dict:
                    row["id"] = entity_dict[field]
            else:
                # For scalar or vector fields
                if field in entity_dict:
                    row[field] = entity_dict[field]
                else:
                    # Method 2: Check if 'hit.entity' has a single-arg get() method
                    if hasattr(hit, "entity") and hit.entity:
                        try:
                            fallback_val = hit.entity.get(field)  # no second arg here
                            if fallback_val is not None:
                                row[field] = fallback_val
                        except Exception:
                            pass  # ignore if old versions can't handle 'get' at all

                    # Method 3: Check 'fields_data' or 'entity_fields'
                    if field not in row:
                        if hasattr(hit, "fields_data"):
                            fd = getattr(hit, "fields_data", {})
                            if field in fd:
                                row[field] = fd[field]
                        if hasattr(hit, "entity_fields"):
                            ef = getattr(hit, "entity_fields", {})
                            if isinstance(ef, dict) and field in ef:
                                row[field] = ef[field]

        return row

    def create_collection(self,
                          collection_name: str,
                          schema_fields: list,
                          index_params: list = None):
        """
        Create a collection with the specified schema (and optional index parameters)
        in the currently active database.

        Parameters:
            collection_name (str): Name of the collection.
            schema_fields (list): List of field definitions, for example:
                                  [{"field_name": "id", "datatype": DataType.INT64, "is_primary": True}, ...]
            index_params (list): List of index configurations, for example:
                                 [{"field_name": "id", "index_type": "STL_SORT"}, ...]
        Returns:
            The load state of the collection after creation.
        """
        logger.info(f"Creating schema for collection '{collection_name}'.")
        schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
        for field in schema_fields:
            schema.add_field(**field)
            logger.debug(f"Added field: {field}")

        # Prepare index parameters if provided.
        idx_params = None
        if index_params:
            logger.debug("Preparing index parameters.")
            idx_params = self.client.prepare_index_params()
            for index_conf in index_params:
                idx_params.add_index(**index_conf)
                logger.debug(f"Added index configuration: {index_conf}")

        # Create the collection
        logger.info(f"Creating collection '{collection_name}' in database '{self.db_name}'.")
        self.client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=idx_params
        )

        # Check and log the collection's load state.
        load_state = self.client.get_load_state(collection_name=collection_name)
        logger.info(f"Collection '{collection_name}' load state: {load_state}")
        return load_state

    # ----------------------------------------
    # Partition Management Methods
    # ----------------------------------------

    def list_partitions(self, collection_name: str) -> list:
        """
        List all partitions in a given collection.

        Parameters:
            collection_name (str): The name of the collection.

        Returns:
            list: A list of partition names.
        """
        return self.client.list_partitions(collection_name=collection_name)

    def create_partition(self, collection_name: str, partition_name: str) -> None:
        """
        Create a new partition in a given collection.

        Parameters:
            collection_name (str): The name of the collection.
            partition_name (str): The name of the partition to create.

        Returns:
            None
        """
        self.client.create_partition(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def has_partition(self, collection_name: str, partition_name: str) -> bool:
        """
        Check if a given partition exists in a collection.

        Parameters:
            collection_name (str): The name of the collection.
            partition_name (str): The partition name to check.

        Returns:
            bool: True if the partition exists, False otherwise.
        """
        return self.client.has_partition(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def load_partitions(self, collection_name: str, partition_names: list) -> None:
        """
        Load one or more partitions of a given collection into memory.

        Parameters:
            collection_name (str): The name of the collection.
            partition_names (list): A list of partition names to load.

        Returns:
            None
        """
        self.client.load_partitions(
            collection_name=collection_name,
            partition_names=partition_names
        )

    def release_partitions(self, collection_name: str, partition_names: list) -> None:
        """
        Release one or more partitions from memory.

        Parameters:
            collection_name (str): The name of the collection.
            partition_names (list): A list of partition names to release.

        Returns:
            None
        """
        self.client.release_partitions(
            collection_name=collection_name,
            partition_names=partition_names
        )

    def get_partition_load_state(self, collection_name: str, partition_name: str) -> dict:
        """
        Get the load state for a particular partition in a collection.

        Parameters:
            collection_name (str): The name of the collection.
            partition_name (str): The name of the partition to check.

        Returns:
            dict: Contains "state" key, e.g. {"state": "<LoadState: Loaded>"}
        """
        return self.client.get_load_state(
            collection_name=collection_name,
            partition_name=partition_name
        )

    def drop_partition(self, collection_name: str, partition_name: str) -> None:
        """
        Drop (delete) a partition from a collection *only if* it exists.
        Automatically releases the partition if it's loaded.

        Parameters:
            collection_name (str): The name of the collection.
            partition_name (str): The partition name to drop.

        Returns:
            None
        """
        # 1. Check if the collection exists
        if not self.client.has_collection(collection_name):
            logger.warning(
                f"Cannot drop partition '{partition_name}' because collection '{collection_name}' does not exist. Skipping."
            )
            return

        # 2. Check if the partition exists
        if not self.has_partition(collection_name=collection_name, partition_name=partition_name):
            logger.info(
                f"Partition '{partition_name}' does not exist in collection '{collection_name}'. Skipping drop."
            )
            return

        # 3. Release the partition if it's loaded
        #    (This is a no-op if the partition is not loaded.)
        try:
            logger.debug(
                f"Releasing partition '{partition_name}' in collection '{collection_name}' before dropping.")
            self.client.release_partitions(collection_name, [partition_name])
        except Exception as release_err:
            logger.warning(
                f"Failed to release partition '{partition_name}'. Proceeding with drop, but errors may occur. "
                f"Details: {release_err}"
            )

        # 4. Drop the partition
        self.client.drop_partition(collection_name=collection_name, partition_name=partition_name)
        logger.info(f"Partition '{partition_name}' dropped from collection '{collection_name}'.")

    # ----------------------------------------
    # Data Operations Within Partitions
    # ----------------------------------------
    # (These are examples of how to insert, upsert, delete, search, or query
    #  in a particular partition by specifying the partition name or names.)

    def insert_into_partition(self,
                              collection_name: str,
                              fields_data: list,
                              partition_name: str,
                              **kwargs):
        """
        Insert entities into a specific partition.
        (Equivalent to normal insert but specifying a partition_name.)

        Parameters:
            collection_name (str): The name of the collection.
            fields_data (list): List of dicts or a dict containing field data.
            partition_name (str): The partition in which to insert.
            **kwargs: Other optional parameters for insert().

        Returns:
            MutationResult: Details on the insertion (e.g., how many rows were inserted).
        """
        return self.client.insert(
            collection_name=collection_name,
            data=fields_data,  # <-- Use "data" instead of "fields_data"
            partition_name=partition_name,
            **kwargs
        )

    def upsert_into_partition(self,
                              collection_name: str,
                              fields_data: list,
                              partition_name: str,
                              **kwargs):
        """
        Upsert entities into a specific partition. (Available in newer Milvus versions.)

        Parameters:
            collection_name (str): The name of the collection.
            fields_data (list): List of dicts or a dict containing field data.
            partition_name (str): The partition in which to upsert.
            **kwargs: Additional optional parameters for upsert().

        Returns:
            MutationResult: Information on upsert results.
        """
        # Make sure your Milvus version / PyMilvus client supports upsert!
        return self.client.upsert(
            collection_name=collection_name,
            partition_name=partition_name,
            fields_data=fields_data,
            **kwargs
        )

    def delete_from_partition(self,
                              collection_name: str,
                              expr: str,
                              partition_name: str,
                              **kwargs):
        """
        Delete entities from a specific partition using an expression.

        Parameters:
            collection_name (str): The name of the collection.
            expr (str): A Boolean expression to match entities, e.g. "id in [1,2,3]".
            partition_name (str): The partition from which to delete.
            **kwargs: Additional optional parameters for delete().

        Returns:
            MutationResult: Information on delete results.
        """
        return self.client.delete(
            collection_name=collection_name,
            partition_name=partition_name,
            expr=expr,
            **kwargs
        )

    def search_in_partitions(self,
                             collection_name: str,
                             data,
                             anns_field: str,
                             param: dict,
                             limit: int,
                             partition_names: list,
                             output_fields: list = None,
                             **kwargs):
        """
        Perform a vector search within specific partitions.

        Parameters:
            collection_name (str): The name of the collection to search.
            data: The vector data to search for, e.g. a list of vectors.
            anns_field (str): The vector field to be searched.
            param (dict): Search parameters (e.g., {"metric_type": "L2", "params": {"nprobe": 10}}).
            limit (int): Maximum number of hits to return.
            partition_names (list): The target partition(s) to search in.
            output_fields (list, optional): Additional scalar fields to return.
            **kwargs: Other optional parameters for search().

        Returns:
            list: A list of QueryResults or similar objects (depends on PyMilvus version).
        """
        return self.client.search(
            collection_name=collection_name,
            partition_names=partition_names,
            data=data,
            anns_field=anns_field,
            param=param,
            limit=limit,
            output_fields=output_fields,
            **kwargs
        )

    def query_in_partitions(self,
                            collection_name: str,
                            expr: str,
                            partition_names: list,
                            output_fields: list = None,
                            **kwargs):
        """
        Query entities in specific partitions using scalar filtering.

        Parameters:
            collection_name (str): The name of the collection to query.
            expr (str): A Boolean expression, e.g. "id > 100".
            partition_names (list): The target partition(s) to query in.
            output_fields (list, optional): Additional fields to return in results.
            **kwargs: Other optional parameters for query().

        Returns:
            list: A list of dicts representing matching entities.
        """
        return self.client.query(
            collection_name=collection_name,
            expr=expr,
            partition_names=partition_names,
            output_fields=output_fields,
            **kwargs
        )

    def search_in_partition(self,
                            collection_name: str,
                            partition_names: list,
                            query_vector: ndarray,
                            limit: int = 3,
                            anns_field: str = "vector",
                            metric_type: str = "IP"):
        """
        ANN Search in a specific partition (or multiple partitions).

        Parameters:
            collection_name (str): Target collection name.
            partition_names (list): A list of partition names to narrow the search.
            query_vector (list): The single query vector to search for.
            limit (int): The number of top results to return. Defaults to 3.
            anns_field (str): The vector field to match on (e.g., "vector").
            metric_type (str): The distance metric (e.g., "IP", "L2", "COSINE"). Defaults to "IP".

        Returns:
            list: A list of hits (where each element is a list of the top-K hits for each query).
        """
        search_params = {"metric_type": metric_type}

        # Perform search only in specified partitions
        res = self.client.search(
            collection_name=collection_name,
            partition_names=partition_names,
            data=[query_vector],
            anns_field=anns_field,
            limit=limit,
            search_params=search_params
        )
        return res

    def search_with_output_fields(self,
                                  collection_name: str,
                                  query_vector: ndarray,
                                  output_fields: list,
                                  limit: int = 3,
                                  anns_field: str = "vector",
                                  metric_type: str = "IP"):
        """
        Search with specified output fields to include in the search result.

        Parameters:
            collection_name (str): Target collection name.
            query_vector (list): The single query vector to search for.
            output_fields (list): Additional fields to return, e.g., ["color", "description"].
            limit (int): Number of top results to return. Defaults to 3.
            anns_field (str): The vector field to match on. Defaults to "vector".
            metric_type (str): Distance metric. Defaults to "IP".

        Returns:
            list: A list of hits (where each element is a list of the top-K hits for each query).
        """
        search_params = {"metric_type": metric_type}

        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=anns_field,
            limit=limit,
            search_params=search_params,
            output_fields=output_fields
        )
        return res

    def search_single_vector(self,
                             collection_name: str,
                             query_vector: ndarray,
                             limit: int = 3,
                             anns_field: str = "vector",
                             metric_type: str = "IP"):
        """
        A simple single-vector ANN search that does not restrict partitions or request extra fields.

        Parameters:
            collection_name (str): Target collection name.
            query_vector (list): The single query vector to search for.
            limit (int): Number of top results to return. Defaults to 3.
            anns_field (str): The vector field to match on. Defaults to "vector".
            metric_type (str): Distance metric. Defaults to "IP".

        Returns:
            list: A list of hits (where each element is a list of the top-K hits for each query).
        """
        search_params = {"metric_type": metric_type}

        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            anns_field=anns_field,
            limit=limit,
            search_params=search_params
        )
        return res