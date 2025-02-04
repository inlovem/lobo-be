import logging
import numpy as np
from typing import List

import torch
from torch import Tensor
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

from embeddings.constants.embeddings_constants import EmbeddingConstants

# Basic logging configuration
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingModelWrapper:
    def __init__(
        self,
        model: str = EmbeddingConstants.STELLA_EN_1_5B,
        encoding_dimensions: int = EmbeddingConstants.FITTING_DIMENSIONS,
        verbose: bool = False,
    ):
        """
        Initialize the dense embedding model.

        Args:
            model (str): The pretrained model name or path.
            encoding_dimensions (int): Maximum token length for inputs.
            verbose (bool): Toggle detailed logging.
        """
        logger.disabled = not verbose
        self.encoding_dimensions = encoding_dimensions

        # Device selection: Use CUDA if available, otherwise CPU.
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("CUDA available. Using GPU.")
        else:
            self.device = torch.device("cpu")
            logger.info("CUDA not available. Using CPU.")

        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model)

        # Load model configuration with output_hidden_states enabled.
        config = AutoConfig.from_pretrained(model)
        config.output_hidden_states = True

        # Load the model in evaluation mode.
        self.model = AutoModelForCausalLM.from_pretrained(model, config=config).eval().to(self.device)
        logger.info("Dense embedding model loaded successfully.")

    def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
        """
        Extract the hidden state of the last valid token for each sequence.

        Args:
            last_hidden_states (Tensor): The tensor of hidden states from the model.
            attention_mask (Tensor): The attention mask indicating valid tokens.

        Returns:
            Tensor: A tensor containing the pooled embeddings.
        """
        # Compute the index of the last non-padded token in each sequence.
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_indices = torch.arange(last_hidden_states.size(0), device=last_hidden_states.device)
        return last_hidden_states[batch_indices, sequence_lengths]

    def encode_dense(self, texts: List[str], batch_size: int = 8, verbose: bool = False) -> np.ndarray:
        """
        Encode a list of texts into dense embeddings using batched processing.
        Returns a NumPy array of shape (n_texts, embedding_dim).

        Args:
            texts (List[str]): The list of texts to be encoded.
            batch_size (int): Number of texts to process per batch.
            verbose (bool): Toggle progress display via tqdm.

        Returns:
            np.ndarray: Dense embedding vectors.
        """
        batches = []

        for i in tqdm(range(0, len(texts), batch_size),
                      desc="Encoding Dense Embeddings",
                      disable=not verbose):
            batch_texts = texts[i:i + batch_size]
            # Tokenize the batch of texts.
            batch_inputs = self.tokenizer(
                batch_texts,
                max_length=self.encoding_dimensions,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            # Move tensors to the appropriate device.
            batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

            with torch.no_grad():
                outputs = self.model(**batch_inputs)

            # Retrieve hidden states from the outputs.
            if hasattr(outputs, "last_hidden_state"):
                last_hidden = outputs.last_hidden_state
            elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                last_hidden = outputs.hidden_states[-1]
            elif isinstance(outputs, tuple):
                last_hidden = outputs[0]
            else:
                raise ValueError("Unexpected model output format.")

            # Pool the hidden state corresponding to the last valid token.
            pooled = self.last_token_pool(last_hidden, batch_inputs["attention_mask"])
            # Normalize the embeddings.
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            # Detach and move to CPU as a NumPy array.
            batches.append(normalized.detach().cpu().numpy())

        if batches:
            embeddings_array = np.vstack(batches)
        else:
            embeddings_array = np.empty((0,))  # Return an empty array if no texts were provided

        logger.info(f"Encoded {len(texts)} texts into dense embeddings of shape {embeddings_array.shape}.")
        return embeddings_array

    def encode(self, text: str) -> np.ndarray:
        """
        Convenience method to encode a single text input.
        Returns a 1D NumPy array representing the dense embedding.

        Args:
            text (str): The text to be encoded.

        Returns:
            np.ndarray: The dense embedding vector.
        """
        batch_inputs = self.tokenizer(
            text,
            max_length=self.encoding_dimensions,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        batch_inputs = {k: v.to(self.device) for k, v in batch_inputs.items()}

        with torch.no_grad():
            outputs = self.model(**batch_inputs)

        if hasattr(outputs, "last_hidden_state"):
            last_hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        elif isinstance(outputs, tuple):
            last_hidden = outputs[0]
        else:
            raise ValueError("Unexpected model output format.")

        pooled = self.last_token_pool(last_hidden, batch_inputs["attention_mask"])
        normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
        # Return a 1D NumPy array (flattened embedding for the single input).
        embedding = normalized.detach().cpu().numpy()[0]
        return embedding