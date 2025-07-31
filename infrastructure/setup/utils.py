#!/usr/bin/env python3
"""
Shared utilities for infrastructure setup scripts
"""

import re
from pathlib import Path
from typing import Optional


def extract_arxiv_id(filename: str) -> Optional[str]:
    """
    Extract arXiv ID from filename.
    
    Args:
        filename: The filename to extract ID from (can be full path or just filename)
        
    Returns:
        Validated arXiv ID or None if invalid
        
    Examples:
        "2301.00001.pdf" -> "2301.00001"
        "2301_00001.pdf" -> "2301.00001"
        "invalid.pdf" -> None
    """
    # Get just the filename without path
    if isinstance(filename, Path):
        filename = filename.name
    else:
        filename = Path(filename).name
        
    # Remove .pdf extension
    if filename.endswith('.pdf'):
        filename = filename[:-4]
    
    # Replace underscore with dot (common in downloaded PDFs)
    if '_' in filename and '.' not in filename:
        filename = filename.replace('_', '.', 1)
    
    # Validate arXiv ID format
    # Modern format: YYMM.NNNNN or YYMM.NNNNNN (optionally with version like v1, v2)
    # Legacy format: archive/YYMMNNN or archive-name/YYMMNNN (e.g., hep-th/9901001)
    arxiv_pattern = re.compile(
        r'^(\d{4}\.\d{4,6}(v\d+)?|[a-z-]+/\d{7}(v\d+)?)$',
        re.IGNORECASE
    )
    
    if arxiv_pattern.match(filename):
        return filename
    
    return None


def validate_disk_space(path: str, required_gb: float = 10.0) -> tuple[bool, float]:
    """
    Check if a path has sufficient free disk space.
    
    Args:
        path: Directory path to check
        required_gb: Required free space in GB (default: 10.0)
        
    Returns:
        Tuple of (has_enough_space, available_gb)
        
    Raises:
        ValueError: If path doesn't exist or isn't accessible
    """
    import shutil
    
    path_obj = Path(path)
    
    # Create directory if it doesn't exist
    if not path_obj.exists():
        path_obj.mkdir(parents=True, exist_ok=True)
    
    if not path_obj.is_dir():
        raise ValueError(f"Path {path} is not a directory")
    
    # Get disk usage statistics
    stat = shutil.disk_usage(str(path_obj))
    available_gb = stat.free / (1024 ** 3)  # Convert bytes to GB
    
    return available_gb >= required_gb, available_gb