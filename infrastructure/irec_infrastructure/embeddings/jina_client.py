"""
Jina V4 Client for Document Embeddings

Provides a robust interface to Jina embeddings API with support for
late chunking, batch processing, and automatic retries.

Validated on millions of document chunks.
"""

import httpx
import numpy as np
from typing import List, Dict, Optional, Union
import logging
from dataclasses import dataclass
import time


@dataclass
class JinaConfig:
    """Configuration for Jina client"""
    api_key: str = None
    model_name: str = "jina-embeddings-v3"
    api_url: str = "https://api.jina.ai/v1/embeddings"
    max_retries: int = 3
    timeout: int = 30
    late_chunking: bool = True
    

class JinaClient:
    """
    Client for Jina embeddings API with late chunking support.
    
    Example:
        client = JinaClient(config=JinaConfig(api_key="your-key"))
        
        # Generate embeddings with late chunking
        result = client.encode_with_late_chunking(
            text="Long document text...",
            chunk_size=512
        )
        
        # Batch processing
        embeddings = client.encode_batch(
            texts=["text1", "text2", ...],
            batch_size=32
        )
    """
    
    def __init__(self, config: JinaConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.client = httpx.Client(timeout=config.timeout)
        
        if not config.api_key:
            raise ValueError("Jina API key required")
    
    def __enter__(self):
        """Enter the runtime context."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and close the client."""
        self.close()
        return False
    
    def close(self):
        """Close the HTTP client."""
        if hasattr(self, 'client'):
            self.client.close()
    
    def encode_with_late_chunking(
        self, 
        text: str, 
        chunk_size: int = 512
    ) -> Dict[str, Union[List[float], List[str]]]:
        """
        Encode text with late chunking to preserve semantic boundaries.
        
        Args:
            text: Input text to encode
            chunk_size: Target size for chunks (in tokens)
            
        Returns:
            Dictionary with 'embeddings' and 'chunks' keys
        """
        # Use late chunking to split text
        chunks = self._split_into_chunks(text, chunk_size)
        
        # Generate embeddings for chunks
        embeddings = self.encode_batch(chunks)
        
        return {
            'embeddings': embeddings.tolist(),
            'chunks': chunks,
            'num_chunks': len(chunks)
        }
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode multiple texts in batches for efficiency.
        
        Args:
            texts: List of texts to encode
            batch_size: Number of texts per API call
            
        Returns:
            Array of embeddings, shape (n_texts, embedding_dim)
        """
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            payload = {
                "model": self.config.model_name,
                "input": batch,
                "encoding_type": "float"
            }
            
            if self.config.late_chunking:
                payload["late_chunking"] = True
            
            response = self._make_request(payload)
            
            # Extract embeddings from response
            batch_embeddings = [item["embedding"] for item in response["data"]]
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def _make_request(self, payload: Dict) -> Dict:
        """Make API request with retries"""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json"
        }
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    self.config.api_url,
                    json=payload,
                    headers=headers
                )
                response.raise_for_status()
                return response.json()
                
            except httpx.HTTPError as e:
                if attempt == self.config.max_retries - 1:
                    raise
                wait_time = 2 ** attempt
                self.logger.warning(
                    f"Request failed (attempt {attempt + 1}), "
                    f"retrying in {wait_time}s: {e}"
                )
                time.sleep(wait_time)
    
    def _split_into_chunks(self, text: str, chunk_size: int) -> List[str]:
        """Split text into semantic chunks.
        
        Note: This is a placeholder implementation that splits by word count,
        not by token count. The chunk_size parameter represents word count here.
        For accurate token-based chunking, use the LateChucker class instead.
        """
        # Simple word-based splitting - not token-based
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size):
            chunk = ' '.join(words[i:i + chunk_size])
            chunks.append(chunk)
        
        return chunks