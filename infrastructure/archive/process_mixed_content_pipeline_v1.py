#!/usr/bin/env python3
"""
Mixed Content Processing Pipeline V1 - PDFs, Code, and Config Files
Extends PDF pipeline to handle code repositories for CONVEYANCE analysis
- Processes PDFs with Docling (papers)
- Processes code files with AST extraction
- Processes config files (YAML, JSON, TOML)
- Creates semantic bridges between paper concepts and code implementations
"""

# Standard library imports
import argparse
import ast
import json
import logging
import multiprocessing as mp
import os
import queue
import re
import sys
import threading
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union

# Third-party imports
import lmdb
import numpy as np
import toml
import yaml
from arango import ArangoClient
from tqdm import tqdm

# Import torch and transformers at top level
try:
    import torch
except ImportError:
    torch = None
    print("WARNING: PyTorch not installed. Some features will be disabled.")

try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    AutoTokenizer = None
    AutoModel = None
    print("WARNING: Transformers not installed. Embedding features will be disabled.")

# Set GPU visibility BEFORE any imports
def set_worker_gpu(gpu_id: int):
    """Set GPU visibility for worker process"""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Check if docling is installed
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.datamodel.base_models import InputFormat
    DOCLING_AVAILABLE = True
except ImportError:
    print("WARNING: Docling not installed. PDF processing will be disabled.")
    print("Install with: pip install docling")

# GPU monitoring
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False
    print("WARNING: pynvml not available, GPU monitoring disabled")

# Suppress specific warnings only
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', message='.*torch.distributed.*')
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# Database password function
def get_db_password() -> str:
    """Securely retrieve database password from environment variable."""
    password = os.getenv("ARANGO_PASSWORD", "")
    if not password:
        raise ValueError("ARANGO_PASSWORD environment variable not set")
    return password

# Configure logging
LOG_FILE_PATH = os.getenv('PIPELINE_LOG_FILE', 'mixed_content_pipeline_v1.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE_PATH)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for mixed content pipeline"""
    # Directories
    pdf_dir: str = "/mnt/data/arxiv_data/pdf"
    code_dir: str = ""  # Repository directory to analyze
    checkpoint_dir: str = "checkpoints/mixed_content_v1"
    
    # Database
    db_name: str = "conveyance_analysis"
    db_host: str = "localhost"
    db_port: int = 8529
    db_username: str = "root"
    db_batch_size: int = 300
    
    # GPU Configuration
    docling_gpu: int = 0
    embedding_gpu: int = 1
    code_processing_gpu: int = -1  # -1 for CPU, 0 or 1 for GPU
    
    # Workers - optimized for continuous GPU utilization
    pdf_workers: int = 2
    code_workers: int = 4  # More workers since code parsing is lighter
    late_workers: int = 3  # Increased to keep GPU 1 busy
    
    # Memory allocation
    pdf_worker_memory: float = 0.45
    code_worker_memory: float = 0.45
    late_worker_memory: float = 0.45
    
    # Power management
    gpu_power_limit: int = 300000
    inter_batch_delay: float = 0.1
    thermal_check_interval: int = 30
    gradual_startup_delay: float = 2.0
    
    # Batching
    embedding_batch_size: int = 6
    pdf_batch_size: int = 3
    code_batch_size: int = 10  # More code files per batch
    
    # Queue sizes
    pdf_queue_size: int = 100
    code_queue_size: int = 200  # New queue for code files
    document_queue_size: int = 200
    output_queue_size: int = 300
    
    # Late Chunking
    max_context_length: int = 32768
    chunk_size_tokens: int = 512
    chunk_stride_tokens: int = 256
    
    # Memory management
    max_document_chars: int = 300000
    max_code_file_chars: int = 100000  # Smaller limit for code files
    batch_timeout_seconds: float = 3.0
    
    # Performance
    use_tf32: bool = True
    compile_model: bool = False
    
    # Limits
    max_pdfs: Optional[int] = None
    max_file_size_mb: float = 75.0
    
    # Resume
    resume: bool = True
    clean_start: bool = False
    
    # CONVEYANCE analysis
    enable_conveyance_analysis: bool = True
    semantic_similarity_threshold: float = 0.7


@dataclass
class CodeUnit:
    """Represents a semantic unit from code"""
    type: str  # 'function', 'class', 'method', 'config'
    name: str
    file_path: str
    content: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    ast_context: Optional[Dict] = None
    line_number: int = 0
    language: str = "python"


@dataclass
class ConfigUnit:
    """Represents configuration data"""
    type: str  # 'config', 'schema', 'manifest'
    format: str  # 'json', 'yaml', 'toml'
    file_path: str
    keys: List[str]
    values: Dict[str, Any]
    context: str


@dataclass
class DocumentWork:
    """Document ready for late chunking"""
    content_id: str  # Can be arxiv_id or file path
    content_type: str  # 'paper', 'code', 'config'
    full_text: str
    metadata: Dict[str, Any]
    extraction_time: float
    char_count: int
    semantic_units: List[Union[CodeUnit, ConfigUnit]] = field(default_factory=list)


@dataclass
class LateChunkOutput:
    """Output from late chunking"""
    content_id: str
    content_type: str
    chunk_embeddings: List[np.ndarray]
    chunk_texts: List[str]
    chunk_metadata: List[Dict[str, Any]]
    total_tokens: int
    processing_time: float
    semantic_units: List[Union[CodeUnit, ConfigUnit]] = field(default_factory=list)


class ImportAnalyzer:
    """Analyze import statements and dependencies"""
    
    def analyze_imports(self, tree: ast.AST) -> Dict[str, Any]:
        """Extract comprehensive import information"""
        imports = {
            'internal': [],
            'external': [],
            'from_imports': defaultdict(list),
            'import_aliases': {},
            'star_imports': []
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    import_alias = alias.asname or alias.name
                    
                    imports['import_aliases'][import_alias] = module_name
                    
                    # Classify as internal or external
                    if self._is_standard_library(module_name) or self._is_third_party(module_name):
                        imports['external'].append(module_name)
                    else:
                        imports['internal'].append(module_name)
                        
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    module = node.module
                    
                    for alias in node.names:
                        if alias.name == '*':
                            imports['star_imports'].append(module)
                        else:
                            imports['from_imports'][module].append(alias.name)
                            if alias.asname:
                                imports['import_aliases'][alias.asname] = f"{module}.{alias.name}"
                    
                    # Classify module
                    if self._is_standard_library(module) or self._is_third_party(module):
                        if module not in imports['external']:
                            imports['external'].append(module)
                    else:
                        if module not in imports['internal']:
                            imports['internal'].append(module)
                            
        # Calculate import metrics
        imports['metrics'] = {
            'total_imports': len(imports['internal']) + len(imports['external']),
            'external_ratio': len(imports['external']) / (len(imports['internal']) + len(imports['external'])) if imports['internal'] or imports['external'] else 0,
            'has_star_imports': bool(imports['star_imports']),
            'import_depth': self._calculate_import_depth(imports)
        }
        
        return imports
    
    def _is_standard_library(self, module: str) -> bool:
        """Check if module is from Python standard library"""
        import sys
        # Use sys.stdlib_module_names if available (Python 3.10+)
        if hasattr(sys, 'stdlib_module_names'):
            return module.split('.')[0] in sys.stdlib_module_names
        else:
            # Fallback for older Python versions
            stdlib_modules = {
                'os', 'sys', 'time', 'datetime', 'json', 're', 'math',
                'collections', 'itertools', 'functools', 'pathlib',
                'typing', 'dataclasses', 'enum', 'abc', 'asyncio',
                'logging', 'argparse', 'configparser', 'subprocess',
                'multiprocessing', 'threading', 'queue', 'warnings',
                'builtins', 'io', 'importlib', 'pickle', 'hashlib'
            }
            return module.split('.')[0] in stdlib_modules
    
    def _is_third_party(self, module: str) -> bool:
        """Check if module is a third-party library"""
        try:
            import importlib.metadata
            # Get all installed packages
            installed_packages = set()
            for dist in importlib.metadata.distributions():
                if dist.name:
                    installed_packages.add(dist.name.lower().replace('-', '_'))
                # Also check module names from the distribution
                if dist.files:
                    for file in dist.files:
                        if file.suffix == '.py' and '/' not in str(file):
                            installed_packages.add(file.stem)
            
            module_root = module.split('.')[0].lower().replace('-', '_')
            # Check if it's installed and not standard library
            return module_root in installed_packages and not self._is_standard_library(module)
        except Exception:
            # Fallback for compatibility
            third_party = {
                'numpy', 'pandas', 'torch', 'tensorflow', 'sklearn',
                'matplotlib', 'seaborn', 'requests', 'flask', 'django',
                'transformers', 'scipy', 'pytest', 'tqdm', 'PIL',
                'lmdb', 'yaml', 'toml', 'arango', 'pynvml', 'docling'
            }
            return module.split('.')[0] in third_party
    
    def _calculate_import_depth(self, imports: Dict) -> int:
        """Calculate maximum import depth"""
        max_depth = 0
        for module in imports['internal'] + imports['external']:
            depth = len(module.split('.'))
            max_depth = max(max_depth, depth)
        return max_depth


class SemanticPatternDetector:
    """Detect semantic patterns in code"""
    
    def detect_patterns(self, tree: ast.AST, content: str) -> Dict[str, Any]:
        """Detect various semantic patterns in code"""
        patterns = {
            'algorithms': self._detect_algorithms(tree, content),
            'data_structures': self._detect_data_structures(tree),
            'ml_patterns': self._detect_ml_patterns(tree),
            'io_patterns': self._detect_io_patterns(tree),
            'concurrency_patterns': self._detect_concurrency_patterns(tree),
            'design_patterns': self._detect_design_patterns_global(tree)
        }
        
        return patterns
    
    def _detect_algorithms(self, tree: ast.AST, content: str) -> List[Dict]:
        """Detect algorithmic patterns"""
        algorithms = []
        
        # Look for sorting algorithms
        if re.search(r'def \w*sort\w*|\.sort\(|sorted\(', content):
            algorithms.append({'type': 'sorting', 'confidence': 0.8})
            
        # Look for search algorithms
        if re.search(r'binary_search|linear_search|\.find\(|\.index\(', content):
            algorithms.append({'type': 'searching', 'confidence': 0.7})
            
        # Matrix operations
        if re.search(r'@|\.dot\(|matmul|matrix', content):
            algorithms.append({'type': 'matrix_operations', 'confidence': 0.8})
            
        # Dynamic programming patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if self._has_memoization(node) or 'dp' in node.name.lower():
                    algorithms.append({'type': 'dynamic_programming', 'confidence': 0.7})
                    break
                    
        return algorithms
    
    def _detect_data_structures(self, tree: ast.AST) -> List[str]:
        """Detect data structures used"""
        structures = set()
        
        for node in ast.walk(tree):
            # Lists and arrays
            if isinstance(node, (ast.List, ast.ListComp)):
                structures.add('list')
            # Dictionaries
            elif isinstance(node, (ast.Dict, ast.DictComp)):
                structures.add('dictionary')
            # Sets
            elif isinstance(node, (ast.Set, ast.SetComp)):
                structures.add('set')
            # Look for tree/graph structures
            elif isinstance(node, ast.ClassDef):
                if any(attr in ['left', 'right', 'children', 'parent', 'nodes', 'edges'] 
                      for attr in self._get_class_attributes(node)):
                    if 'left' in self._get_class_attributes(node) or 'right' in self._get_class_attributes(node):
                        structures.add('tree')
                    if 'nodes' in self._get_class_attributes(node) or 'edges' in self._get_class_attributes(node):
                        structures.add('graph')
                        
        return list(structures)
    
    def _detect_ml_patterns(self, tree: ast.AST) -> Dict[str, Any]:
        """Detect machine learning patterns"""
        ml_patterns = {
            'uses_embeddings': False,
            'embedding_ops': [],
            'model_loading': False,
            'training_code': False,
            'inference_code': False,
            'data_preprocessing': False
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    
                    # Embedding operations
                    if attr in ['encode', 'embed', 'transform', 'forward']:
                        ml_patterns['uses_embeddings'] = True
                        ml_patterns['embedding_ops'].append(attr)
                        
                    # Model operations
                    elif attr in ['load_model', 'from_pretrained', 'load_state_dict']:
                        ml_patterns['model_loading'] = True
                    elif attr in ['train', 'fit', 'backward', 'step']:
                        ml_patterns['training_code'] = True
                    elif attr in ['predict', 'inference', 'eval']:
                        ml_patterns['inference_code'] = True
                    elif attr in ['normalize', 'tokenize', 'preprocess']:
                        ml_patterns['data_preprocessing'] = True
                        
        ml_patterns['embedding_ops'] = list(set(ml_patterns['embedding_ops']))
        return ml_patterns
    
    def _detect_io_patterns(self, tree: ast.AST) -> Dict[str, List[str]]:
        """Detect I/O patterns"""
        io_patterns = {
            'reads_files': [],
            'writes_files': [],
            'database_ops': [],
            'api_calls': [],
            'file_formats': set()
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    # File operations
                    if node.func.id == 'open':
                        if len(node.args) > 1:
                            mode_arg = node.args[1]
                            if isinstance(mode_arg, ast.Constant):
                                if 'r' in str(mode_arg.value):
                                    io_patterns['reads_files'].append('generic')
                                elif 'w' in str(mode_arg.value) or 'a' in str(mode_arg.value):
                                    io_patterns['writes_files'].append('generic')
                                    
                elif isinstance(node.func, ast.Attribute):
                    attr = node.func.attr
                    
                    # Specific file operations
                    if attr in ['read_csv', 'read_json', 'read_parquet', 'read_excel']:
                        format_name = attr.replace('read_', '')
                        io_patterns['reads_files'].append(format_name)
                        io_patterns['file_formats'].add(format_name)
                    elif attr in ['to_csv', 'to_json', 'to_parquet', 'dump', 'write']:
                        io_patterns['writes_files'].append('various')
                        
                    # Database operations
                    elif attr in ['execute', 'query', 'insert', 'update', 'delete', 'find', 'aggregate']:
                        io_patterns['database_ops'].append(attr)
                        
                    # API calls
                    elif attr in ['get', 'post', 'put', 'delete', 'request']:
                        io_patterns['api_calls'].append(attr)
                        
        io_patterns['file_formats'] = list(io_patterns['file_formats'])
        return io_patterns
    
    def _detect_concurrency_patterns(self, tree: ast.AST) -> Dict[str, bool]:
        """Detect concurrency patterns"""
        patterns = {
            'uses_threading': False,
            'uses_multiprocessing': False,
            'uses_asyncio': False,
            'has_locks': False,
            'has_queues': False
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                if isinstance(node, ast.Import):
                    modules = [alias.name for alias in node.names]
                else:
                    modules = [node.module] if node.module else []
                    
                for module in modules:
                    if module:
                        if 'threading' in module:
                            patterns['uses_threading'] = True
                        elif 'multiprocessing' in module:
                            patterns['uses_multiprocessing'] = True
                        elif 'asyncio' in module:
                            patterns['uses_asyncio'] = True
                            
            # Check for async functions
            if isinstance(node, ast.AsyncFunctionDef):
                patterns['uses_asyncio'] = True
                
            # Check for locks and queues
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if 'Lock' in node.func.id or 'Semaphore' in node.func.id:
                        patterns['has_locks'] = True
                    elif 'Queue' in node.func.id:
                        patterns['has_queues'] = True
                        
        return patterns
    
    def _detect_design_patterns_global(self, tree: ast.AST) -> List[str]:
        """Detect design patterns at file level"""
        patterns = []
        
        # Check for common patterns
        class_names = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        # Builder pattern
        if any('Builder' in name for name in class_names):
            patterns.append('builder')
            
        # Strategy pattern
        if any('Strategy' in name or 'Policy' in name for name in class_names):
            patterns.append('strategy')
            
        # Adapter pattern
        if any('Adapter' in name or 'Wrapper' in name for name in class_names):
            patterns.append('adapter')
            
        return patterns
    
    def _has_memoization(self, func_node: ast.FunctionDef) -> bool:
        """Check if function uses memoization"""
        # Check for @lru_cache or @cache decorators
        for dec in func_node.decorator_list:
            if isinstance(dec, ast.Name) and dec.id in ['lru_cache', 'cache', 'memoize']:
                return True
            elif isinstance(dec, ast.Attribute) and dec.attr in ['lru_cache', 'cache']:
                return True
                
        # Check for manual memoization pattern
        for node in ast.walk(func_node):
            if isinstance(node, ast.Name) and ('memo' in node.id.lower() or 'cache' in node.id.lower()):
                return True
                
        return False
    
    def _get_class_attributes(self, class_node: ast.ClassDef) -> List[str]:
        """Get class attribute names"""
        attrs = []
        for node in class_node.body:
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                attrs.append(node.target.id)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        attrs.append(target.id)
        return attrs


class FilesystemAnalyzer:
    """Analyze filesystem context and metadata"""
    
    def analyze(self, file_path: Path, content: str, repo_root: Path = None) -> Dict[str, Any]:
        """Extract filesystem metadata"""
        # Basic file stats
        file_stats = self._get_file_stats(file_path, content)
        
        # Path information
        path_info = self._get_path_info(file_path, repo_root)
        
        # Module role
        module_role = self._determine_module_role(file_path, content)
        
        # Git information (if available)
        git_info = self._get_git_info(file_path)
        
        return {
            'file_stats': file_stats,
            'path_info': path_info,
            'module_role': module_role,
            'git_info': git_info
        }
    
    def _get_file_stats(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Get file statistics"""
        lines = content.split('\n')
        
        # Count different types of lines
        code_lines = 0
        comment_lines = 0
        blank_lines = 0
        
        in_docstring = False
        docstring_char = None
        
        for line in lines:
            stripped = line.strip()
            
            # Handle docstrings
            if not in_docstring and (stripped.startswith('"""') or stripped.startswith("'''")):
                in_docstring = True
                docstring_char = stripped[:3]
                comment_lines += 1
                if stripped.endswith(docstring_char) and len(stripped) > 3:
                    in_docstring = False
            elif in_docstring:
                comment_lines += 1
                if stripped.endswith(docstring_char):
                    in_docstring = False
            # Regular comments
            elif stripped.startswith('#'):
                comment_lines += 1
            # Blank lines
            elif not stripped:
                blank_lines += 1
            # Code lines
            else:
                code_lines += 1
                
        try:
            file_stats = file_path.stat()
            return {
                'size_bytes': file_stats.st_size,
                'lines_of_code': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'total_lines': len(lines),
                'created_date': datetime.fromtimestamp(file_stats.st_ctime).isoformat(),
                'modified_date': datetime.fromtimestamp(file_stats.st_mtime).isoformat()
            }
        except (OSError, IOError) as e:
            logger.warning(f"Failed to get file stats for {file_path}: {e}")
            return {
                'size_bytes': 0,
                'lines_of_code': code_lines,
                'comment_lines': comment_lines,
                'blank_lines': blank_lines,
                'total_lines': len(lines),
                'created_date': None,
                'modified_date': None
            }
    
    def _get_path_info(self, file_path: Path, repo_root: Path = None) -> Dict[str, Any]:
        """Get path information"""
        info = {
            'absolute_path': str(file_path.absolute()),
            'relative_path': str(file_path),
            'directory_depth': len(file_path.parts) - 1,
            'is_test': self._is_test_file(file_path),
            'is_example': self._is_example_file(file_path)
        }
        
        # Calculate package path if in Python package
        if file_path.suffix == '.py':
            package_parts = []
            for part in file_path.parts[:-1]:
                if (file_path.parent / part / '__init__.py').exists():
                    package_parts.append(part)
            if package_parts:
                module_name = file_path.stem if file_path.stem != '__init__' else ''
                if module_name:
                    package_parts.append(module_name)
                info['package_path'] = '.'.join(package_parts)
                
        # Relative to repo root
        if repo_root:
            try:
                info['repo_relative_path'] = str(file_path.relative_to(repo_root))
            except ValueError:
                pass
                
        return info
    
    def _determine_module_role(self, file_path: Path, content: str) -> Dict[str, Any]:
        """Determine the role of this module"""
        return {
            'is_entry_point': 'if __name__ == "__main__"' in content,
            'is_init': file_path.name == '__init__.py',
            'is_config': self._is_config_file(file_path, content),
            'is_utility': self._is_utility_file(file_path, content),
            'is_test': self._is_test_file(file_path),
            'module_type': self._classify_module_type(file_path, content)
        }
    
    def _is_test_file(self, file_path: Path) -> bool:
        """Check if file is a test file"""
        path_str = str(file_path).lower()
        return 'test' in path_str or 'spec' in path_str
    
    def _is_example_file(self, file_path: Path) -> bool:
        """Check if file is an example"""
        path_str = str(file_path).lower()
        return 'example' in path_str or 'demo' in path_str or 'sample' in path_str
    
    def _is_config_file(self, file_path: Path, content: str) -> bool:
        """Check if file contains configuration"""
        indicators = ['CONFIG', 'SETTINGS', 'config =', 'settings =']
        return any(ind in content for ind in indicators)
    
    def _is_utility_file(self, file_path: Path, content: str) -> bool:
        """Check if file contains utility functions"""
        name = file_path.stem.lower()
        return name in ['utils', 'helpers', 'common', 'shared', 'tools']
    
    def _classify_module_type(self, file_path: Path, content: str) -> str:
        """Classify the module type"""
        name = file_path.stem.lower()
        path_str = str(file_path).lower()
        
        if self._is_test_file(file_path):
            return 'test'
        elif self._is_example_file(file_path):
            return 'example'
        elif 'model' in name or 'model' in path_str:
            return 'model'
        elif 'api' in name or 'route' in name:
            return 'api'
        elif 'cli' in name or 'command' in name:
            return 'cli'
        elif self._is_utility_file(file_path, content):
            return 'utility'
        elif 'core' in path_str:
            return 'core'
        else:
            return 'general'
    
    def _get_git_info(self, file_path: Path) -> Dict[str, Any]:
        """Get git information if available"""
        # This is a placeholder - in production you'd use GitPython
        return {
            'is_tracked': True,  # Assume tracked for now
            'last_commit': None,
            'authors': []
        }


class FileTypeDetector:
    """Detect and classify file types"""
    
    CODE_EXTENSIONS = {
        '.py': 'python',
        '.js': 'javascript', 
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.rs': 'rust',
        '.go': 'go'
    }
    
    CONFIG_EXTENSIONS = {
        '.json': 'json',
        '.yaml': 'yaml',
        '.yml': 'yaml',
        '.toml': 'toml',
        '.ini': 'ini',
        '.conf': 'conf'
    }
    
    SPECIAL_CONFIG_FILES = {
        'Dockerfile': 'dockerfile',
        'Makefile': 'makefile',
        '.gitignore': 'gitignore',
        'requirements.txt': 'requirements',
        'package.json': 'package_json',
        'pyproject.toml': 'pyproject'
    }
    
    @classmethod
    def detect_file_type(cls, file_path: Path) -> Tuple[str, str]:
        """Returns (type, subtype) where type is 'code', 'config', etc."""
        ext = file_path.suffix.lower()
        name = file_path.name
        
        # Check special config files first
        if name in cls.SPECIAL_CONFIG_FILES:
            return ('config', cls.SPECIAL_CONFIG_FILES[name])
            
        # Check by extension
        if ext in cls.CODE_EXTENSIONS:
            return ('code', cls.CODE_EXTENSIONS[ext])
        elif ext in cls.CONFIG_EXTENSIONS:
            return ('config', cls.CONFIG_EXTENSIONS[ext])
        elif ext == '.pdf':
            return ('pdf', 'pdf')
        elif ext in ['.md', '.rst', '.txt']:
            return ('documentation', ext[1:])
        else:
            return ('unknown', '')


class CodeProcessor:
    """Process code files and extract semantic units with comprehensive metadata"""
    
    def __init__(self):
        self.import_analyzer = ImportAnalyzer()
        self.pattern_detector = SemanticPatternDetector()
        self.filesystem_analyzer = FilesystemAnalyzer()
        
    def extract_python_units(self, file_path: Path, content: str, repo_root: Path = None) -> Tuple[List[CodeUnit], Dict[str, Any]]:
        """Extract semantic units and comprehensive metadata from Python code"""
        units = []
        
        # Get filesystem metadata
        fs_metadata = self.filesystem_analyzer.analyze(file_path, content, repo_root)
        
        try:
            tree = ast.parse(content)
            
            # Extract imports first for dependency analysis
            imports_metadata = self.import_analyzer.analyze_imports(tree)
            
            # Extract all functions and classes
            functions_metadata = []
            classes_metadata = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    func_meta = self._extract_function_metadata(node, content)
                    functions_metadata.append(func_meta)
                    
                    unit = CodeUnit(
                        type='function',
                        name=node.name,
                        file_path=str(file_path),
                        content=ast.get_source_segment(content, node) or "",
                        docstring=ast.get_docstring(node),
                        signature=self._get_function_signature(node),
                        ast_context=func_meta,
                        line_number=node.lineno,
                        language='python'
                    )
                    units.append(unit)
                    
                elif isinstance(node, ast.ClassDef):
                    class_meta = self._extract_class_metadata(node, content)
                    classes_metadata.append(class_meta)
                    
                    unit = CodeUnit(
                        type='class',
                        name=node.name,
                        file_path=str(file_path),
                        content=ast.get_source_segment(content, node) or "",
                        docstring=ast.get_docstring(node),
                        ast_context=class_meta,
                        line_number=node.lineno,
                        language='python'
                    )
                    units.append(unit)
            
            # Detect semantic patterns
            patterns = self.pattern_detector.detect_patterns(tree, content)
            
            # Calculate CONVEYANCE indicators
            conveyance = self._calculate_conveyance_indicators(
                tree, content, units, fs_metadata
            )
            
            # Combine all metadata
            file_metadata = {
                'ast_metadata': {
                    'functions': functions_metadata,
                    'classes': classes_metadata,
                    'imports': imports_metadata,
                    'has_main_block': self._has_main_block(tree),
                    'total_complexity': sum(f.get('complexity', 0) for f in functions_metadata)
                },
                'filesystem_metadata': fs_metadata,
                'semantic_patterns': patterns,
                'conveyance_indicators': conveyance,
                'metrics': {
                    'total_functions': len(functions_metadata),
                    'total_classes': len(classes_metadata),
                    'avg_function_length': np.mean([f['lines'] for f in functions_metadata]) if functions_metadata else 0
                }
            }
                    
        except SyntaxError as e:
            logger.error(f"Syntax error parsing Python file {file_path}: {e}")
            # Fall back with basic metadata
            file_metadata = {
                'ast_metadata': {'parse_error': f'SyntaxError: {e}'},
                'filesystem_metadata': fs_metadata,
                'semantic_patterns': {},
                'conveyance_indicators': {'parseable': False}
            }
            units.append(CodeUnit(
                type='file',
                name=file_path.name,
                file_path=str(file_path),
                content=content[:1000],
                language='python'
            ))
        except (TypeError, AttributeError) as e:
            logger.error(f"AST processing error for {file_path}: {e}")
            # Fall back with basic metadata
            file_metadata = {
                'ast_metadata': {'parse_error': f'{type(e).__name__}: {e}'},
                'filesystem_metadata': fs_metadata,
                'semantic_patterns': {},
                'conveyance_indicators': {'parseable': False}
            }
            units.append(CodeUnit(
                type='file',
                name=file_path.name,
                file_path=str(file_path),
                content=content[:1000],
                language='python'
            ))
        except Exception as e:
            logger.error(f"Unexpected error parsing Python file {file_path}: {e}")
            # Fall back with basic metadata
            file_metadata = {
                'ast_metadata': {'parse_error': f'Unexpected error: {e}'},
                'filesystem_metadata': fs_metadata,
                'semantic_patterns': {},
                'conveyance_indicators': {'parseable': False}
            }
            units.append(CodeUnit(
                type='file',
                name=file_path.name,
                file_path=str(file_path),
                content=content[:1000],
                language='python'
            ))
            
        return units, file_metadata
    
    def _extract_function_metadata(self, node: ast.FunctionDef, content: str) -> Dict[str, Any]:
        """Extract comprehensive function metadata"""
        # Calculate cyclomatic complexity
        complexity = self._calculate_cyclomatic_complexity(node)
        
        # Extract function calls
        calls = []
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    calls.append(child.func.attr)
                    
        # Get parameter details
        params = []
        for arg in node.args.args:
            param_info = {'name': arg.arg}
            if arg.annotation:
                param_info['type'] = self._unparse_annotation(arg.annotation)
            params.append(param_info)
            
        return {
            'name': node.name,
            'signature': self._get_function_signature(node),
            'parameters': params,
            'return_type': self._get_return_annotation(node),
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'is_async': isinstance(node, ast.AsyncFunctionDef),
            'complexity': complexity,
            'calls': list(set(calls)),  # Unique function calls
            'lines': node.end_lineno - node.lineno + 1 if node.end_lineno else 1,
            'has_docstring': ast.get_docstring(node) is not None,
            'raises_exceptions': self._get_raised_exceptions(node),
            'uses_comprehensions': self._uses_comprehensions(node),
            'has_type_hints': bool(node.returns or any(arg.annotation for arg in node.args.args))
        }
    
    def _extract_class_metadata(self, node: ast.ClassDef, content: str) -> Dict[str, Any]:
        """Extract comprehensive class metadata"""
        # Extract methods and their properties
        methods = []
        attributes = []
        properties = []
        
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                method_info = {
                    'name': item.name,
                    'is_private': item.name.startswith('_'),
                    'is_magic': item.name.startswith('__') and item.name.endswith('__'),
                    'decorators': [self._get_decorator_name(d) for d in item.decorator_list],
                    'has_docstring': ast.get_docstring(item) is not None
                }
                methods.append(method_info)
                
                # Check if it's a property
                if any(self._get_decorator_name(d) == 'property' for d in item.decorator_list):
                    properties.append(item.name)
                    
            elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                # Class attribute with type annotation
                attributes.append({
                    'name': item.target.id,
                    'type': self._unparse_annotation(item.annotation) if item.annotation else None
                })
                
        # Detect design patterns
        patterns = self._detect_design_patterns(node, methods)
        
        return {
            'name': node.name,
            'bases': [self._get_base_name(b) for b in node.bases],
            'methods': methods,
            'attributes': attributes,
            'properties': properties,
            'is_abstract': self._is_abstract_class(node),
            'design_patterns': patterns,
            'decorators': [self._get_decorator_name(d) for d in node.decorator_list],
            'has_init': any(m['name'] == '__init__' for m in methods),
            'lines': node.end_lineno - node.lineno + 1 if node.end_lineno else 1,
            'method_count': len(methods),
            'is_dataclass': any(self._get_decorator_name(d) == 'dataclass' for d in node.decorator_list)
        }
    
    def _calculate_cyclomatic_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of a function"""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def _detect_design_patterns(self, class_node: ast.ClassDef, methods: List[Dict]) -> List[str]:
        """Detect common design patterns in class"""
        patterns = []
        
        method_names = [m['name'] for m in methods]
        
        # Singleton pattern
        if '__new__' in method_names or '_instance' in [attr.target.id for attr in class_node.body if isinstance(attr, ast.AnnAssign) and hasattr(attr.target, 'id')]:
            patterns.append('singleton')
            
        # Factory pattern
        if any('create' in name or 'build' in name for name in method_names):
            patterns.append('factory')
            
        # Observer pattern
        if any(name in method_names for name in ['subscribe', 'notify', 'attach', 'detach']):
            patterns.append('observer')
            
        # Iterator pattern
        if '__iter__' in method_names and '__next__' in method_names:
            patterns.append('iterator')
            
        return patterns
    
    def _calculate_conveyance_indicators(self, tree: ast.AST, content: str, 
                                       units: List[CodeUnit], fs_metadata: Dict) -> Dict[str, Any]:
        """Calculate CONVEYANCE indicators for actionability"""
        total_lines = content.count('\n') + 1
        
        # Count docstrings
        docstring_count = sum(1 for unit in units if unit.docstring)
        
        # Check for various quality indicators
        has_logging = any(isinstance(node, ast.Import) and 'logging' in [alias.name for alias in node.names] 
                         for node in ast.walk(tree))
        
        has_argparse = any(isinstance(node, ast.Import) and 'argparse' in [alias.name for alias in node.names] 
                          for node in ast.walk(tree))
        
        # Check for error handling
        exception_handlers = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ExceptHandler))
        
        return {
            'actionability': {
                'has_main_block': self._has_main_block(tree),
                'has_cli_interface': has_argparse,
                'has_examples': fs_metadata.get('module_role', {}).get('is_example', False),
                'completeness_score': self._calculate_completeness_score(tree, units)
            },
            'documentation': {
                'docstring_coverage': docstring_count / len(units) if units else 0,
                'has_type_hints': any(hasattr(unit, 'ast_context') and 
                                    unit.ast_context.get('has_type_hints', False) 
                                    for unit in units),
                'comment_ratio': fs_metadata.get('file_stats', {}).get('comment_lines', 0) / total_lines,
                'has_module_docstring': ast.get_docstring(tree) is not None
            },
            'quality_signals': {
                'has_tests': 'test' in fs_metadata.get('path_info', {}).get('relative_path', '').lower(),
                'has_error_handling': exception_handlers > 0,
                'uses_logging': has_logging,
                'exception_to_code_ratio': exception_handlers / len(units) if units else 0
            },
            'implementation_signals': {
                'imports_ml_libraries': self._imports_ml_libraries(tree),
                'has_data_processing': self._has_data_processing_patterns(tree),
                'has_algorithms': bool(units),  # Simplified - could be enhanced
                'file_executable': fs_metadata.get('module_role', {}).get('is_entry_point', False)
            }
        }
    
    def _calculate_completeness_score(self, tree: ast.AST, units: List[CodeUnit]) -> float:
        """Calculate how complete/actionable the code is"""
        score = 0.0
        
        # Has functions or classes
        if units:
            score += 0.3
            
        # Has main block
        if self._has_main_block(tree):
            score += 0.2
            
        # Has imports (not standalone)
        if any(isinstance(node, (ast.Import, ast.ImportFrom)) for node in tree.body):
            score += 0.2
            
        # Has docstrings
        if any(unit.docstring for unit in units):
            score += 0.2
            
        # Has error handling
        if any(isinstance(node, ast.Try) for node in ast.walk(tree)):
            score += 0.1
            
        return min(score, 1.0)
    
    def _imports_ml_libraries(self, tree: ast.AST) -> bool:
        """Check if code imports ML/AI libraries"""
        ml_libraries = {
            'torch', 'tensorflow', 'transformers', 'sklearn', 
            'numpy', 'pandas', 'scipy', 'keras', 'jax'
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(alias.name in ml_libraries for alias in node.names):
                    return True
            elif isinstance(node, ast.ImportFrom):
                if node.module and any(lib in node.module for lib in ml_libraries):
                    return True
                    
        return False
    
    def _has_data_processing_patterns(self, tree: ast.AST) -> bool:
        """Check for data processing patterns"""
        patterns = ['read_csv', 'to_numpy', 'DataLoader', 'Dataset', 'transform']
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in patterns:
                        return True
                        
        return False
    
    def _has_main_block(self, tree: ast.AST) -> bool:
        """Check if file has if __name__ == '__main__' block"""
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                if (isinstance(node.test, ast.Compare) and
                    isinstance(node.test.left, ast.Name) and
                    node.test.left.id == '__name__' and
                    any(isinstance(op, ast.Eq) for op in node.test.ops)):
                    return True
        return False
    
    def _get_raised_exceptions(self, node: ast.FunctionDef) -> List[str]:
        """Get list of exceptions raised by function"""
        exceptions = []
        for child in ast.walk(node):
            if isinstance(child, ast.Raise):
                if child.exc:
                    if isinstance(child.exc, ast.Call) and isinstance(child.exc.func, ast.Name):
                        exceptions.append(child.exc.func.id)
                    elif isinstance(child.exc, ast.Name):
                        exceptions.append(child.exc.id)
        return list(set(exceptions))
    
    def _uses_comprehensions(self, node: ast.FunctionDef) -> bool:
        """Check if function uses list/dict/set comprehensions"""
        for child in ast.walk(node):
            if isinstance(child, (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp)):
                return True
        return False
    
    def _is_abstract_class(self, node: ast.ClassDef) -> bool:
        """Check if class is abstract"""
        # Check if inherits from ABC
        for base in node.bases:
            if (isinstance(base, ast.Name) and base.id == 'ABC') or \
               (isinstance(base, ast.Attribute) and base.attr == 'ABC'):
                return True
                
        # Check for @abstractmethod decorators
        for item in node.body:
            if isinstance(item, ast.FunctionDef):
                for dec in item.decorator_list:
                    if self._get_decorator_name(dec) == 'abstractmethod':
                        return True
                        
        return False
    
    def _get_function_signature(self, node: ast.FunctionDef) -> str:
        """Extract function signature"""
        args = []
        for arg in node.args.args:
            args.append(arg.arg)
        return f"{node.name}({', '.join(args)})"
    
    def _get_decorator_name(self, decorator) -> str:
        """Get decorator name"""
        if hasattr(decorator, 'id'):
            return decorator.id
        elif hasattr(decorator, 'attr'):
            return decorator.attr
        return str(decorator)
    
    def _get_base_name(self, base) -> str:
        """Get base class name"""
        if hasattr(base, 'id'):
            return base.id
        return str(base)
    
    def _get_return_annotation(self, node) -> Optional[str]:
        """Get return type annotation"""
        if node.returns:
            return self._unparse_annotation(node.returns)
        return None
    
    def _unparse_annotation(self, node) -> str:
        """Unparse AST annotation node with Python < 3.9 compatibility"""
        import sys
        if sys.version_info >= (3, 9):
            return ast.unparse(node)
        else:
            # Fallback for Python < 3.9
            return self._unparse_node(node)
    
    def _unparse_node(self, node) -> str:
        """Simple AST node unparsing for older Python versions"""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Constant):
            return repr(node.value)
        elif isinstance(node, ast.Attribute):
            return f"{self._unparse_node(node.value)}.{node.attr}"
        elif isinstance(node, ast.Subscript):
            return f"{self._unparse_node(node.value)}[{self._unparse_node(node.slice)}]"
        elif isinstance(node, ast.Index):  # Python < 3.9
            return self._unparse_node(node.value)
        elif isinstance(node, ast.Tuple):
            elements = ", ".join(self._unparse_node(elt) for elt in node.elts)
            return f"({elements})"
        elif isinstance(node, ast.List):
            elements = ", ".join(self._unparse_node(elt) for elt in node.elts)
            return f"[{elements}]"
        else:
            # Fallback to string representation of node type
            return f"<{node.__class__.__name__}>"
    
    def extract_generic_code_units(self, file_path: Path, content: str, language: str) -> List[CodeUnit]:
        """Extract units from non-Python code files"""
        # For now, just create a single unit for the whole file
        # Could be extended with language-specific parsers
        # Extract basic patterns that work across languages
        units = []
        
        # Look for function-like patterns (works for many languages)
        function_patterns = {
            'javascript': r'(function\s+(\w+)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>|(\w+)\s*:\s*function)',
            'java': r'(public|private|protected)?\s*(static)?\s*\w+\s+(\w+)\s*\([^)]*\)',
            'cpp': r'(\w+)\s+(\w+)\s*\([^)]*\)\s*{',
            'go': r'func\s+(\w+)\s*\([^)]*\)',
            'rust': r'fn\s+(\w+)\s*\([^)]*\)',
            'typescript': r'(function\s+(\w+)|const\s+(\w+)\s*=\s*\([^)]*\)\s*=>|(\w+)\s*:\s*function)',
        }
        
        # Extract function-like structures if pattern exists for language
        if language in function_patterns:
            pattern = function_patterns[language]
            for match in re.finditer(pattern, content):
                func_name = None
                for group in match.groups():
                    if group and group not in ['public', 'private', 'protected', 'static', 'const', 'function']:
                        func_name = group
                        break
                        
                if func_name:
                    units.append(CodeUnit(
                        type='function',
                        name=func_name,
                        file_path=str(file_path),
                        content=match.group(0),
                        language=language,
                        line_number=content[:match.start()].count('\n') + 1
                    ))
        
        # If no functions found, create a file-level unit
        if not units:
            units.append(CodeUnit(
                type='file',
                name=file_path.name,
                file_path=str(file_path),
                content=content[:5000],  # First 5000 chars
                language=language
            ))
            
        return units


class ConfigProcessor:
    """Process configuration files"""
    
    def extract_json_config(self, file_path: Path, content: str) -> ConfigUnit:
        """Extract configuration from JSON"""
        try:
            data = json.loads(content)
            
            # Detect config type
            if 'dependencies' in data or 'scripts' in data:
                config_type = 'package_manifest'
            elif 'openapi' in data or 'swagger' in data:
                config_type = 'api_schema'
            else:
                config_type = 'config'
                
            return ConfigUnit(
                type=config_type,
                format='json',
                file_path=str(file_path),
                keys=list(data.keys()) if isinstance(data, dict) else [],
                values=self._extract_important_values(data),
                context=self._infer_context(file_path, data)
            )
        except Exception as e:
            logger.error(f"Error parsing JSON {file_path}: {e}")
            return self._fallback_config_unit(file_path, 'json')
    
    def extract_yaml_config(self, file_path: Path, content: str) -> ConfigUnit:
        """Extract configuration from YAML"""
        # Check input size limit (1MB)
        MAX_CONFIG_SIZE = 1024 * 1024  # 1MB
        if len(content) > MAX_CONFIG_SIZE:
            logger.warning(f"YAML file {file_path} exceeds size limit ({len(content)} > {MAX_CONFIG_SIZE}), truncating")
            content = content[:MAX_CONFIG_SIZE]
        
        try:
            data = yaml.safe_load(content)
            
            return ConfigUnit(
                type='config',
                format='yaml',
                file_path=str(file_path),
                keys=list(data.keys()) if isinstance(data, dict) else [],
                values=self._extract_important_values(data),
                context=self._infer_context(file_path, data)
            )
        except Exception as e:
            logger.error(f"Error parsing YAML {file_path}: {e}")
            return self._fallback_config_unit(file_path, 'yaml')
    
    def extract_toml_config(self, file_path: Path, content: str) -> ConfigUnit:
        """Extract configuration from TOML"""
        try:
            data = toml.loads(content)
            
            # Check if it's pyproject.toml
            if file_path.name == 'pyproject.toml':
                config_type = 'pyproject'
            else:
                config_type = 'config'
                
            return ConfigUnit(
                type=config_type,
                format='toml',
                file_path=str(file_path),
                keys=list(data.keys()),
                values=self._extract_important_values(data),
                context=self._infer_context(file_path, data)
            )
        except Exception as e:
            logger.error(f"Error parsing TOML {file_path}: {e}")
            return self._fallback_config_unit(file_path, 'toml')
    
    def _extract_important_values(self, data: Any) -> Dict[str, Any]:
        """Extract semantically important values from config"""
        if not isinstance(data, dict):
            return {}
            
        important = {}
        
        # Look for patterns that might relate to paper concepts
        semantic_keys = [
            'algorithm', 'method', 'model', 'optimizer',
            'architecture', 'layers', 'dimensions',
            'batch_size', 'learning_rate', 'epochs',
            'embedding_size', 'hidden_size', 'num_layers',
            'dropout', 'activation', 'loss_function'
        ]
        
        def search_dict(d: dict, prefix: str = ''):
            for key, value in d.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                # Check if key is semantically important
                if any(sk in key.lower() for sk in semantic_keys):
                    important[full_key] = value
                    
                # Recurse into nested dicts
                if isinstance(value, dict):
                    search_dict(value, full_key)
                    
        search_dict(data)
        return important
    
    def _infer_context(self, file_path: Path, data: Any) -> str:
        """Infer what this config is for"""
        name = file_path.name.lower()
        
        if 'model' in name or 'config' in name:
            return 'model_configuration'
        elif 'train' in name:
            return 'training_configuration'
        elif 'test' in name or 'eval' in name:
            return 'evaluation_configuration'
        else:
            return 'general_configuration'
    
    def _fallback_config_unit(self, file_path: Path, format: str) -> ConfigUnit:
        """Create a fallback config unit when parsing fails"""
        return ConfigUnit(
            type='config',
            format=format,
            file_path=str(file_path),
            keys=[],
            values={},
            context='unknown'
        )


def code_worker_process(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    code_queue: mp.Queue,
    document_queue: mp.Queue,
    checkpoint_dir: str,
    config: PipelineConfig,
    stop_event: mp.Event,
    db_queue: mp.Queue = None
):
    """Code processing worker - can run on CPU or GPU"""
    if gpu_id >= 0:
        set_worker_gpu(gpu_id)
        if torch is not None:
            torch.cuda.set_per_process_memory_fraction(memory_fraction)
        device = f"GPU {gpu_id}"
    else:
        # Run on CPU
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        device = "CPU"
    
    logger.info(f"Code worker {worker_id} starting on {device}")
    
    # Initialize processors
    code_processor = CodeProcessor()
    config_processor = ConfigProcessor()
    
    # Process files
    batch = []
    last_batch_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Collect batch
            while len(batch) < config.code_batch_size and not stop_event.is_set():
                try:
                    file_info = code_queue.get(timeout=0.5)
                    if file_info is None:
                        if batch:
                            process_code_batch(
                                batch, code_processor, config_processor,
                                document_queue, config, worker_id, db_queue
                            )
                        return
                    batch.append(file_info)
                except queue.Empty:
                    if time.time() - last_batch_time > config.batch_timeout_seconds:
                        break
                        
            if batch:
                process_code_batch(
                    batch, code_processor, config_processor,
                    document_queue, config, worker_id, db_queue
                )
                batch = []
                last_batch_time = time.time()
                
                if hasattr(config, 'inter_batch_delay'):
                    time.sleep(config.inter_batch_delay)
                
        except Exception as e:
            logger.error(f"Code worker {worker_id} error: {e}")
            
    logger.info(f"Code worker {worker_id} stopped")


def process_code_batch(batch, code_processor, config_processor, document_queue, config, worker_id, db_queue=None):
    """Process batch of code/config files with comprehensive metadata"""
    logger.info(f"Worker {worker_id} processing batch of {len(batch)} files")
    
    # Determine repo root from first file
    repo_root = None
    if batch:
        first_path = Path(batch[0][0])
        # Walk up to find .git directory
        current = first_path.parent
        while current != current.parent:
            if (current / '.git').exists():
                repo_root = current
                break
            current = current.parent
    
    for file_info in batch:
        file_path, file_type, subtype = file_info
        
        try:
            # Read file content with better error handling
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with a different encoding
                logger.warning(f"UTF-8 decode error for {file_path}, trying latin-1")
                try:
                    with open(file_path, 'r', encoding='latin-1') as f:
                        content = f.read()
                except Exception as e:
                    logger.error(f"Failed to read {file_path} with fallback encoding: {e}")
                    continue
                
            if len(content) > config.max_code_file_chars:
                logger.warning(f"Skipping {file_path}: file too large ({len(content)} chars)")
                continue
                
            start_time = time.time()
            semantic_units = []
            file_metadata = {}
            
            # Extract semantic units and metadata based on file type
            if file_type == 'code':
                if subtype == 'python':
                    units, file_metadata = code_processor.extract_python_units(
                        Path(file_path), content, repo_root
                    )
                    semantic_units = units
                else:
                    semantic_units = code_processor.extract_generic_code_units(
                        Path(file_path), content, subtype
                    )
                    # Basic metadata for non-Python code
                    file_metadata = {
                        'filesystem_metadata': code_processor.filesystem_analyzer.analyze(
                            Path(file_path), content, repo_root
                        ),
                        'language': subtype
                    }
                    
            elif file_type == 'config':
                if subtype == 'json':
                    unit = config_processor.extract_json_config(Path(file_path), content)
                elif subtype in ['yaml', 'yml']:
                    unit = config_processor.extract_yaml_config(Path(file_path), content)
                elif subtype == 'toml':
                    unit = config_processor.extract_toml_config(Path(file_path), content)
                else:
                    # Generic config
                    unit = ConfigUnit(
                        type='config',
                        format=subtype,
                        file_path=str(file_path),
                        keys=[],
                        values={},
                        context='unknown'
                    )
                semantic_units = [unit]
                
                # Basic metadata for config files
                file_metadata = {
                    'filesystem_metadata': code_processor.filesystem_analyzer.analyze(
                        Path(file_path), content, repo_root
                    ),
                    'config_type': unit.type,
                    'config_format': unit.format
                }
                
            extraction_time = time.time() - start_time
            
            # Create document work for embedding - now includes full metadata
            doc_work = DocumentWork(
                content_id=str(file_path),
                content_type=file_type,
                full_text=content,
                metadata={
                    'file_path': str(file_path),
                    'file_type': file_type,
                    'subtype': subtype,
                    'extraction_time': extraction_time,
                    'semantic_units_count': len(semantic_units),
                    **file_metadata  # Include all extracted metadata
                },
                extraction_time=extraction_time,
                char_count=len(content),
                semantic_units=semantic_units
            )
            
            # Queue for embedding
            document_queue.put(doc_work)
            
            # Send to database with comprehensive metadata
            if db_queue:
                # Send file-level metadata
                if file_type == 'code' and 'ast_metadata' in file_metadata:
                    file_record = {
                        '_key': f"file_{Path(file_path).stem}",
                        'file_path': str(file_path),
                        'file_type': file_type,
                        'language': subtype,
                        'ast_metadata': file_metadata.get('ast_metadata', {}),
                        'filesystem_metadata': file_metadata.get('filesystem_metadata', {}),
                        'semantic_patterns': file_metadata.get('semantic_patterns', {}),
                        'conveyance_indicators': file_metadata.get('conveyance_indicators', {}),
                        'metrics': file_metadata.get('metrics', {})
                    }
                    db_queue.put(('file_metadata', file_record))
                
                # Send semantic units with enriched metadata
                for unit in semantic_units:
                    if isinstance(unit, CodeUnit):
                        record = {
                            '_key': f"{Path(file_path).stem}_{unit.type}_{unit.name}",
                            'type': unit.type,
                            'name': unit.name,
                            'file_path': unit.file_path,
                            'language': unit.language,
                            'docstring': unit.docstring,
                            'line_number': unit.line_number,
                            'content_snippet': unit.content[:500] if unit.content else "",
                            'ast_context': unit.ast_context if hasattr(unit, 'ast_context') else {},
                            'signature': unit.signature if hasattr(unit, 'signature') else None
                        }
                        db_queue.put(('code_units', record))
                    elif isinstance(unit, ConfigUnit):
                        record = {
                            '_key': f"{Path(file_path).stem}_config",
                            'type': unit.type,
                            'format': unit.format,
                            'file_path': unit.file_path,
                            'context': unit.context,
                            'keys': unit.keys[:20],  # First 20 keys
                            'important_values': unit.values
                        }
                        db_queue.put(('config_units', record))
                        
            logger.info(
                f"Extracted {len(semantic_units)} units from {file_path} "
                f"with {len(file_metadata)} metadata categories"
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")


def late_chunking_worker_process_mixed(
    worker_id: int,
    gpu_id: int,
    memory_fraction: float,
    document_queue: mp.Queue,
    output_queue: mp.Queue,
    config: PipelineConfig,
    stop_event: mp.Event
):
    """Late chunking worker for mixed content"""
    set_worker_gpu(gpu_id)
    
    if torch is not None:
        torch.cuda.set_per_process_memory_fraction(memory_fraction)
    
    logger.info(f"Late worker {worker_id} starting on GPU {gpu_id}")
    
    # Memory optimization
    torch.cuda.empty_cache()
    
    # Enable optimizations
    if config.use_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
    # Initialize Jina v4
    model_name = "jinaai/jina-embeddings-v4"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(
        model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).cuda()
    
    model.eval()
    
    logger.info(f"Worker {worker_id} initialized for late chunking")
    
    # Process documents
    batch = []
    last_batch_time = time.time()
    
    while not stop_event.is_set():
        try:
            # Collect batch
            while len(batch) < config.embedding_batch_size and not stop_event.is_set():
                try:
                    doc = document_queue.get(timeout=0.5)
                    if doc is None:
                        if batch:
                            process_mixed_late_chunk_batch(
                                batch, model, tokenizer, output_queue, 
                                config, worker_id
                            )
                        return
                    batch.append(doc)
                except queue.Empty:
                    if time.time() - last_batch_time > config.batch_timeout_seconds:
                        break
                        
            if batch:
                process_mixed_late_chunk_batch(
                    batch, model, tokenizer, output_queue, 
                    config, worker_id
                )
                batch = []
                last_batch_time = time.time()
                
                if hasattr(config, 'inter_batch_delay'):
                    time.sleep(config.inter_batch_delay)
                
        except Exception as e:
            logger.error(f"Late worker {worker_id} error: {e}", exc_info=True)
            batch = []
            torch.cuda.empty_cache()
            
    logger.info(f"Late worker {worker_id} stopped")


def process_mixed_late_chunk_batch(batch, model, tokenizer, output_queue, config, worker_id):
    """Process batch with content-appropriate embedding strategies"""
    
    logger.info(f"Worker {worker_id} processing mixed batch of {len(batch)} documents")
    start_time = time.time()
    
    for doc in batch:
        try:
            doc_start_time = time.time()
            
            # Choose task based on content type
            if doc.content_type == 'paper':
                task = 'retrieval.passage'
            elif doc.content_type == 'code':
                task = 'code'
            else:  # config
                task = 'retrieval.passage'
                
            # Late chunk the document
            chunks = late_chunk_document_with_task(doc, model, tokenizer, config, task)
            
            # Also embed semantic units if present
            if doc.semantic_units:
                for unit in doc.semantic_units:
                    if isinstance(unit, CodeUnit):
                        # Build text representation for code unit
                        text = f"{unit.type} {unit.name}: {unit.docstring or unit.content[:200]}"
                        embedding = embed_text(text, model, tokenizer, 'code')
                        
                        # Add as special chunk
                        chunks.append({
                            'embedding': embedding,
                            'text': text,
                            'metadata': {
                                'type': 'semantic_unit',
                                'unit_type': unit.type,
                                'unit_name': unit.name,
                                'file_path': unit.file_path
                            }
                        })
                        
            # Create output
            output = LateChunkOutput(
                content_id=doc.content_id,
                content_type=doc.content_type,
                chunk_embeddings=[c['embedding'] for c in chunks],
                chunk_texts=[c['text'] for c in chunks],
                chunk_metadata=[c['metadata'] for c in chunks],
                total_tokens=sum(c['metadata'].get('tokens', 0) for c in chunks),
                processing_time=time.time() - doc_start_time,
                semantic_units=doc.semantic_units
            )
            
            output_queue.put(output)
            
            logger.info(
                f"Late chunked {doc.content_id} ({doc.content_type}): "
                f"{len(chunks)} chunks"
            )
            
        except Exception as e:
            logger.error(f"Late chunking failed for {doc.content_id}: {e}")
            
    batch_time = time.time() - start_time
    logger.info(f"Worker {worker_id} batch completed in {batch_time:.1f}s")


def late_chunk_document_with_task(doc, model, tokenizer, config, task):
    """Late chunking with appropriate task adapter"""
    
    # Tokenize with limited context
    inputs = tokenizer(
        doc.full_text,
        return_tensors='pt',
        max_length=config.max_context_length,
        truncation=True,
        return_offsets_mapping=True,
        padding=True
    ).to('cuda')
    
    seq_len = inputs['input_ids'].shape[1]
    
    # Process with mixed precision
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model(
                **{k: v for k, v in inputs.items() if k != 'offset_mapping'},
                task=task
            )
            # Handle different output formats
            if hasattr(outputs, 'multi_vec_emb'):
                all_token_embeddings = outputs.multi_vec_emb[0]
            else:
                all_token_embeddings = outputs.last_hidden_state[0]
    
    # Extract chunks
    chunks = []
    offset_mapping = inputs['offset_mapping'][0].cpu().numpy()
    
    chunk_size = config.chunk_size_tokens
    stride = config.chunk_stride_tokens
    
    for start_idx in range(0, seq_len - chunk_size + 1, stride):
        end_idx = min(start_idx + chunk_size, seq_len)
        
        # Mean pool tokens
        chunk_embedding = all_token_embeddings[start_idx:end_idx].mean(dim=0)
        
        # Get text boundaries
        start_char = int(offset_mapping[start_idx][0])
        end_char = int(offset_mapping[end_idx - 1][1])
        
        if start_char < 0 or end_char > len(doc.full_text):
            continue
            
        chunk_text = doc.full_text[start_char:end_char]
        
        if not chunk_text.strip():
            continue
            
        chunks.append({
            'embedding': chunk_embedding.cpu().numpy(),
            'text': chunk_text,
            'metadata': {
                'start_token': start_idx,
                'end_token': end_idx,
                'start_char': start_char,
                'end_char': end_char,
                'tokens': end_idx - start_idx,
                'content_type': doc.content_type,
                'task': task
            }
        })
        
    return chunks


def embed_text(text, model, tokenizer, task):
    """Embed a single text with specified task"""
    
    inputs = tokenizer(
        text,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    ).to('cuda')
    
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            outputs = model(**inputs, task=task)
            if hasattr(outputs, 'embeddings'):
                embedding = outputs.embeddings[0]
            else:
                embedding = outputs.last_hidden_state[0].mean(dim=0)
                
    return embedding.cpu().numpy()


class MixedDatabaseWriter:
    """Database writer for mixed content with CONVEYANCE analysis"""
    
    def __init__(self, config: PipelineConfig, checkpoint_manager):
        self.config = config
        self.checkpoint_manager = checkpoint_manager
        self.client = None
        self.db = None
        self.collections = {}
        
        # Batch buffers for each collection
        self.batch_buffers = {
            'metadata': [],
            'documents': [],
            'chunks': [],
            'code_units': [],
            'config_units': [],
            'file_metadata': [],  # New: comprehensive file metadata
            'semantic_bridges': []  # For CONVEYANCE analysis
        }
        
        # Stats
        self.stats = {
            'total_documents': 0,
            'total_chunks': 0,
            'total_code_units': 0,
            'total_config_units': 0,
            'total_bridges': 0,
            'start_time': time.time()
        }
        
        # For CONVEYANCE analysis
        self.paper_embeddings = {}  # Cache paper chunk embeddings
        self.code_embeddings = {}   # Cache code embeddings
        
        # Initialize database
        self._initialize_db()
        
    def _initialize_db(self) -> bool:
        """Initialize database connection with mixed content collections"""
        try:
            self.client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            self.db = self.client.db(
                self.config.db_name,
                username=self.config.db_username,
                password=get_db_password()
            )
            
            # Initialize collections
            collection_names = [
                'metadata', 'documents', 'chunks',
                'code_units', 'config_units', 'file_metadata',
                'semantic_bridges'
            ]
            
            for name in collection_names:
                if name == 'semantic_bridges':
                    # Edge collection for semantic relationships
                    if not self.db.has_collection(name):
                        self.db.create_collection(name, edge=True)
                else:
                    if not self.db.has_collection(name):
                        self.db.create_collection(name)
                        
                self.collections[name] = self.db.collection(name)
                logger.info(f"{name} collection has {self.collections[name].count()} documents")
                
            # Create graph for CONVEYANCE analysis
            if not self.db.has_graph('conveyance_graph'):
                graph = self.db.create_graph('conveyance_graph')
                graph.create_edge_definition(
                    edge_collection='semantic_bridges',
                    from_vertex_collections=['chunks', 'code_units'],
                    to_vertex_collections=['chunks', 'code_units', 'config_units']
                )
                
            return True
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
            return False
            
    def process_outputs(self, output_queue: mp.Queue, stop_event: mp.Event, db_queue: mp.Queue = None):
        """Process outputs with CONVEYANCE analysis"""
        logger.info("Mixed Database Writer started")
        
        # Start db_queue handler thread
        db_thread = None
        if db_queue:
            db_thread = threading.Thread(
                target=self._process_db_queue,
                args=(db_queue, stop_event),
                daemon=True
            )
            db_thread.start()
            
        while not stop_event.is_set() or not output_queue.empty():
            try:
                output = output_queue.get(timeout=1.0)
                
                if isinstance(output, LateChunkOutput):
                    self._process_late_chunks_with_analysis(output)
                    
                self._check_and_flush()
                
            except queue.Empty:
                self._check_and_flush()
            except Exception as e:
                logger.error(f"Database writer error: {e}")
                
        self._flush_all_batches()
        
        # Final CONVEYANCE analysis if enabled
        if self.config.enable_conveyance_analysis:
            self._perform_conveyance_analysis()
            
        if db_thread:
            db_thread.join()
            
        logger.info(f"Database Writer stopped. Stats: {self.stats}")
        
    def _process_late_chunks_with_analysis(self, output: LateChunkOutput):
        """Process chunks and cache embeddings for CONVEYANCE analysis"""
        content_id = output.content_id
        
        # Update stats
        self.stats['total_chunks'] += len(output.chunk_embeddings)
        self.stats['total_documents'] += 1
        
        # Cache embeddings for CONVEYANCE analysis
        if output.content_type == 'paper':
            self.paper_embeddings[content_id] = [
                (emb, text, meta) for emb, text, meta in 
                zip(output.chunk_embeddings, output.chunk_texts, output.chunk_metadata)
            ]
        elif output.content_type == 'code':
            self.code_embeddings[content_id] = [
                (emb, text, meta) for emb, text, meta in 
                zip(output.chunk_embeddings, output.chunk_texts, output.chunk_metadata)
            ]
            
        # Convert to records and store
        for i in range(len(output.chunk_texts)):
            chunk_record = {
                '_key': f"{content_id}_chunk_{i:04d}",
                'chunk_id': f"{content_id}_chunk_{i:04d}",
                'content_id': content_id,
                'content_type': output.content_type,
                'text': output.chunk_texts[i],
                'embedding': output.chunk_embeddings[i].tolist(),
                'chunk_index': i,
                'chunk_metadata': output.chunk_metadata[i],
                'processed_at': datetime.now().isoformat()
            }
            
            self.batch_buffers['chunks'].append(chunk_record)
            
        # Mark as processed
        if output.content_type == 'paper':
            self.checkpoint_manager.mark_pdf_processed(
                content_id, 
                len(output.chunk_texts), 
                output.total_tokens
            )
            
    def _perform_conveyance_analysis(self):
        """Analyze semantic bridges between papers and code"""
        logger.info("Performing CONVEYANCE analysis...")
        
        bridge_count = 0
        
        # Compare paper chunks with code chunks
        for paper_id, paper_chunks in self.paper_embeddings.items():
            for code_id, code_chunks in self.code_embeddings.items():
                
                # Find semantic bridges between this paper and code file
                for p_idx, (p_emb, p_text, p_meta) in enumerate(paper_chunks):
                    for c_idx, (c_emb, c_text, c_meta) in enumerate(code_chunks):
                        
                        # Calculate cosine similarity
                        similarity = self._cosine_similarity(p_emb, c_emb)
                        
                        if similarity > self.config.semantic_similarity_threshold:
                            # Found a semantic bridge!
                            bridge = {
                                '_from': f'chunks/{paper_id}_chunk_{p_idx:04d}',
                                '_to': f'chunks/{code_id}_chunk_{c_idx:04d}',
                                'similarity': float(similarity),
                                'bridge_type': self._classify_bridge(p_text, c_text, p_meta, c_meta),
                                'paper_id': paper_id,
                                'code_file': code_id,
                                'discovered_at': datetime.now().isoformat()
                            }
                            
                            self.batch_buffers['semantic_bridges'].append(bridge)
                            bridge_count += 1
                            
                            if bridge_count % 100 == 0:
                                logger.info(f"Found {bridge_count} semantic bridges...")
                                
        self.stats['total_bridges'] = bridge_count
        logger.info(f"CONVEYANCE analysis complete. Found {bridge_count} semantic bridges.")
        
    def _cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings using numpy"""
        # Normalize vectors to unit length
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Normalized dot product is cosine similarity
        emb1_normalized = emb1 / norm1
        emb2_normalized = emb2 / norm2
        
        return np.dot(emb1_normalized, emb2_normalized)
        
    def _classify_bridge(self, paper_text: str, code_text: str, 
                        paper_meta: Dict, code_meta: Dict) -> str:
        """Classify the type of semantic bridge"""
        paper_lower = paper_text.lower()
        code_lower = code_text.lower()
        
        # Check for algorithm implementation
        if any(word in paper_lower for word in ['algorithm', 'procedure', 'method']):
            if any(word in code_lower for word in ['def', 'function', 'class']):
                return 'algorithm_implementation'
                
        # Check for equation implementation
        if any(char in paper_text for char in ['∑', '∫', '∂', '=']):
            if any(word in code_lower for word in ['calculate', 'compute', 'sum', 'integral']):
                return 'equation_implementation'
                
        # Check for parameter/config mapping
        if any(word in paper_lower for word in ['parameter', 'hyperparameter', 'configuration']):
            if code_meta.get('content_type') == 'config':
                return 'parameter_specification'
                
        return 'general_semantic_similarity'
        
    def _process_db_queue(self, db_queue: mp.Queue, stop_event: mp.Event):
        """Process items from db_queue"""
        while not stop_event.is_set() or not db_queue.empty():
            try:
                item = db_queue.get(timeout=1.0)
                if item is None:
                    break
                    
                collection_type, record = item
                if collection_type in self.batch_buffers:
                    self.batch_buffers[collection_type].append(record)
                    
                    # Update stats
                    if collection_type == 'code_units':
                        self.stats['total_code_units'] += 1
                    elif collection_type == 'config_units':
                        self.stats['total_config_units'] += 1
                    
                # Check if any buffer needs flushing
                for coll_name, buffer in self.batch_buffers.items():
                    if len(buffer) >= self.config.db_batch_size:
                        self._flush_batch(coll_name)
                        
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"DB queue processing error: {e}")
                
    def _check_and_flush(self):
        """Check and flush full buffers"""
        for collection_name, buffer in self.batch_buffers.items():
            if len(buffer) >= self.config.db_batch_size:
                self._flush_batch(collection_name)
                
    def _flush_all_batches(self):
        """Flush all remaining buffers"""
        for collection_name in self.batch_buffers:
            if self.batch_buffers[collection_name]:
                self._flush_batch(collection_name)
                
    def _flush_batch(self, collection_name: str):
        """Write batch to database"""
        buffer = self.batch_buffers.get(collection_name, [])
        if not buffer:
            return
            
        collection = self.collections.get(collection_name)
        if not collection:
            logger.error(f"Collection {collection_name} not found")
            return
            
        try:
            # Use upsert operations instead of insert_many with overwrite
            from arango import DocumentInsertError
            
            # Prepare documents for upsert
            for doc in buffer:
                # Use _key as unique identifier for upsert
                if '_key' in doc:
                    try:
                        collection.update({'_key': doc['_key']}, doc)
                    except:
                        # Document doesn't exist, insert it
                        collection.insert(doc)
                else:
                    # No _key, just insert
                    collection.insert(doc)
            
            logger.info(f"Upserted {len(buffer)} records to {collection_name}")
            self.batch_buffers[collection_name] = []
        except Exception as e:
            logger.error(f"Failed to write to {collection_name}: {e}")
            # Keep buffer for retry instead of clearing it
            logger.warning(f"Keeping {len(buffer)} records in buffer for retry")


class MixedContentPipeline:
    """Main pipeline for processing mixed content"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config.checkpoint_dir)
        
        # Setup multiprocessing
        try:
            mp.set_start_method('spawn')
        except RuntimeError:
            # Already set, ignore
            pass
        
        # Create queues with balanced sizes for continuous flow
        # Smaller PDF queue to prevent memory buildup
        self.pdf_queue = mp.Queue(maxsize=min(config.pdf_queue_size, 50))
        self.code_queue = mp.Queue(maxsize=config.code_queue_size)
        # Larger document queue to buffer between stages
        self.document_queue = mp.Queue(maxsize=config.document_queue_size * 2)
        self.output_queue = mp.Queue(maxsize=config.output_queue_size)
        self.db_queue = mp.Queue(maxsize=200)
        
        # Control
        self.stop_event = mp.Event()
        
        # Workers
        self.workers = []
        
        # Stats
        self.start_time = None
        self.files_queued = {'pdf': 0, 'code': 0, 'config': 0}
        
        # GPU monitoring
        self.gpu_monitor_thread = None
        
    def setup_database(self) -> bool:
        """Setup database for mixed content"""
        try:
            client = ArangoClient(hosts=f'http://{self.config.db_host}:{self.config.db_port}')
            sys_db = client.db('_system', username=self.config.db_username, password=get_db_password())
            
            if not sys_db.has_database(self.config.db_name):
                sys_db.create_database(self.config.db_name)
                logger.info(f"Created database: {self.config.db_name}")
            else:
                logger.info(f"Using existing database: {self.config.db_name}")
                
            return True
            
        except Exception as e:
            logger.error(f"Database setup failed: {e}")
            return False
            
    def monitor_gpu_utilization(self):
        """Monitor GPU utilization in background thread"""
        if not NVML_AVAILABLE:
            return
            
        while not self.stop_event.is_set():
            try:
                gpu_stats = []
                for gpu_id in [self.config.docling_gpu, self.config.embedding_gpu]:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    
                    gpu_stats.append({
                        'gpu': gpu_id,
                        'utilization': util.gpu,
                        'memory_used': mem.used / mem.total * 100,
                        'temperature': temp
                    })
                
                # Log if utilization is low
                for stat in gpu_stats:
                    if stat['utilization'] < 70:
                        logger.warning(
                            f"GPU {stat['gpu']} underutilized: {stat['utilization']}% "
                            f"(Memory: {stat['memory_used']:.1f}%, Temp: {stat['temperature']}°C)"
                        )
                        
                # Adjust queue sizes if needed
                if self.document_queue.qsize() < 10 and gpu_stats[1]['utilization'] < 70:
                    logger.info("Document queue low, embedding GPU underutilized")
                    
            except Exception as e:
                logger.error(f"GPU monitoring error: {e}")
                
            time.sleep(30)  # Check every 30 seconds
    
    def start(self):
        """Start the pipeline"""
        self.start_time = time.time()
        
        # Start GPU monitoring
        if NVML_AVAILABLE:
            self.gpu_monitor_thread = threading.Thread(
                target=self.monitor_gpu_utilization,
                daemon=True
            )
            self.gpu_monitor_thread.start()
        
        # Start PDF workers if PDF directory is specified
        if self.config.pdf_dir:
            logger.info(f"Starting {self.config.pdf_workers} PDF workers on GPU {self.config.docling_gpu}")
            for i in range(self.config.pdf_workers):
                from process_pdfs_continuous_gpu_v10_fixed import pdf_worker_process
                p = mp.Process(
                    target=pdf_worker_process,
                    args=(
                        i, self.config.docling_gpu, self.config.pdf_worker_memory,
                        self.pdf_queue, self.document_queue,
                        self.config.checkpoint_dir, self.config, self.stop_event,
                        self.db_queue
                    )
                )
                p.start()
                self.workers.append(p)
                time.sleep(self.config.gradual_startup_delay)
                
        # Start code workers if code directory is specified
        if self.config.code_dir:
            gpu_desc = f"GPU {self.config.code_processing_gpu}" if self.config.code_processing_gpu >= 0 else "CPU"
            logger.info(f"Starting {self.config.code_workers} code workers on {gpu_desc}")
            for i in range(self.config.code_workers):
                p = mp.Process(
                    target=code_worker_process,
                    args=(
                        i, self.config.code_processing_gpu, self.config.code_worker_memory,
                        self.code_queue, self.document_queue,
                        self.config.checkpoint_dir, self.config, self.stop_event,
                        self.db_queue
                    )
                )
                p.start()
                self.workers.append(p)
                # Shorter delay for CPU workers
                delay = self.config.gradual_startup_delay if self.config.code_processing_gpu >= 0 else 0.5
                time.sleep(delay)
                
        # Start embedding workers
        logger.info(f"Starting {self.config.late_workers} embedding workers on GPU {self.config.embedding_gpu}")
        for i in range(self.config.late_workers):
            p = mp.Process(
                target=late_chunking_worker_process_mixed,
                args=(
                    i, self.config.embedding_gpu, self.config.late_worker_memory,
                    self.document_queue, self.output_queue,
                    self.config, self.stop_event
                )
            )
            p.start()
            self.workers.append(p)
            time.sleep(self.config.gradual_startup_delay)
            
        # Start database writer with CONVEYANCE analysis
        self.db_writer = MixedDatabaseWriter(self.config, self.checkpoint_manager)
        self.db_thread = threading.Thread(
            target=self.db_writer.process_outputs,
            args=(self.output_queue, self.stop_event, self.db_queue)
        )
        self.db_thread.start()
        
        logger.info("Mixed content pipeline started successfully")
        
    def queue_files(self):
        """Queue all files for processing"""
        # Queue PDFs if specified
        if self.config.pdf_dir:
            pdf_dir = Path(self.config.pdf_dir)
            pdf_files = sorted(pdf_dir.glob("*.pdf"))
            
            if self.config.max_pdfs:
                pdf_files = pdf_files[:self.config.max_pdfs]
                
            logger.info(f"Queueing {len(pdf_files)} PDFs")
            
            for pdf_path in tqdm(pdf_files, desc="Queueing PDFs"):
                if not self.config.resume or not self.checkpoint_manager.is_processed(pdf_path.stem):
                    self.pdf_queue.put(str(pdf_path))
                    self.files_queued['pdf'] += 1
                    
        # Queue code files if specified
        if self.config.code_dir:
            code_dir = Path(self.config.code_dir)
            
            # Find all code and config files
            for file_path in tqdm(code_dir.rglob("*"), desc="Scanning code directory"):
                if file_path.is_file():
                    file_type, subtype = FileTypeDetector.detect_file_type(file_path)
                    
                    if file_type in ['code', 'config']:
                        # Check file size
                        try:
                            file_size = file_path.stat().st_size
                            if file_size > self.config.max_file_size_mb * 1024 * 1024:
                                logger.warning(f"Skipping {file_path}: file too large")
                                continue
                        except (OSError, IOError) as e:
                            logger.warning(f"Failed to get file size for {file_path}: {e}")
                            continue
                            
                        self.code_queue.put((str(file_path), file_type, subtype))
                        self.files_queued[file_type] = self.files_queued.get(file_type, 0) + 1
                        
            logger.info(f"Queued {self.files_queued['code']} code files and {self.files_queued.get('config', 0)} config files")
            
        # Signal end to workers
        for _ in range(self.config.pdf_workers):
            self.pdf_queue.put(None)
        for _ in range(self.config.code_workers):
            self.code_queue.put(None)
            
    def wait_for_completion(self):
        """Wait for all work to complete"""
        # Wait for all extraction workers to finish
        extraction_workers = self.config.pdf_workers + self.config.code_workers
        
        while True:
            time.sleep(10)
            
            # Check if extraction workers are done
            extraction_done = all(not w.is_alive() for w in self.workers[:extraction_workers])
            
            if extraction_done and self.document_queue.empty():
                # Signal embedding workers to stop
                for _ in range(self.config.late_workers):
                    self.document_queue.put(None)
                break
                
            # Log progress
            stats = self.db_writer.stats
            logger.info(
                f"Progress: {stats['total_documents']} docs, "
                f"{stats['total_chunks']} chunks, "
                f"{stats['total_code_units']} code units"
            )
            
        # Wait for all workers
        for worker in self.workers:
            worker.join()
            
        # Stop database writer
        self.stop_event.set()
        self.db_thread.join()
        
        # Final stats
        total_time = time.time() - self.start_time
        final_stats = self.db_writer.stats
        
        logger.info(f"""
============================================================
Mixed Content Pipeline completed in {total_time:.1f} seconds
Documents: {final_stats['total_documents']}
Chunks: {final_stats['total_chunks']}
Code Units: {final_stats['total_code_units']}
Config Units: {final_stats['total_config_units']}
Semantic Bridges: {final_stats['total_bridges']}
Rate: {final_stats['total_documents'] / total_time:.2f} docs/second
============================================================
""")


def main():
    parser = argparse.ArgumentParser(
        description="Mixed Content Processing Pipeline - PDFs, Code, and Config"
    )
    
    # Directories
    parser.add_argument("--pdf-dir", help="Directory containing PDFs")
    parser.add_argument("--code-dir", help="Repository directory to analyze")
    
    # Database
    parser.add_argument("--db-name", default="conveyance_analysis")
    parser.add_argument("--db-host", default="localhost")
    
    # GPU
    parser.add_argument("--docling-gpu", type=int, default=0)
    parser.add_argument("--embedding-gpu", type=int, default=1)
    parser.add_argument("--code-gpu", type=int, default=-1,
                       help="GPU for code processing (-1 for CPU, 0-1 for GPU)")
    
    # Workers
    parser.add_argument("--pdf-workers", type=int, default=2)
    parser.add_argument("--code-workers", type=int, default=4)
    parser.add_argument("--late-workers", type=int, default=3)
    
    # CONVEYANCE analysis
    parser.add_argument("--enable-conveyance", action="store_true", default=True,
                       help="Enable CONVEYANCE analysis between papers and code")
    parser.add_argument("--similarity-threshold", type=float, default=0.7,
                       help="Similarity threshold for semantic bridges")
    
    # Limits
    parser.add_argument("--max-pdfs", type=int, help="Maximum PDFs to process")
    
    # Control
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--clean-start", action="store_true")
    
    args = parser.parse_args()
    
    # Check environment variable early
    arango_password = os.getenv("ARANGO_PASSWORD")
    if not arango_password:
        logger.error("ERROR: ARANGO_PASSWORD environment variable not set!")
        sys.exit(1)
    
    # Validate inputs
    if not args.pdf_dir and not args.code_dir:
        logger.error("ERROR: Must specify at least one of --pdf-dir or --code-dir")
        sys.exit(1)
    
    # Check if PDF processing is requested but docling is not available
    if args.pdf_dir and not DOCLING_AVAILABLE:
        logger.warning("PDF processing requested but docling is not available. PDF files will be skipped.")
        
    # Validate GPU IDs
    if NVML_AVAILABLE:
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            for gpu_name, gpu_id in [("docling", args.docling_gpu), 
                                      ("embedding", args.embedding_gpu), 
                                      ("code", args.code_gpu)]:
                if gpu_id >= 0 and gpu_id >= device_count:
                    logger.error(f"ERROR: {gpu_name}-gpu {gpu_id} does not exist. Available GPUs: 0-{device_count-1}")
                    sys.exit(1)
        except Exception as e:
            logger.warning(f"Could not validate GPU IDs: {e}")
        
    # Create config
    config = PipelineConfig(
        pdf_dir=args.pdf_dir or "",
        code_dir=args.code_dir or "",
        db_name=args.db_name,
        db_host=args.db_host,
        docling_gpu=args.docling_gpu,
        embedding_gpu=args.embedding_gpu,
        code_processing_gpu=args.code_gpu,
        pdf_workers=args.pdf_workers if args.pdf_dir and DOCLING_AVAILABLE else 0,
        code_workers=args.code_workers if args.code_dir else 0,
        late_workers=args.late_workers,
        max_pdfs=args.max_pdfs,
        resume=args.resume and not args.clean_start,
        clean_start=args.clean_start,
        enable_conveyance_analysis=args.enable_conveyance,
        semantic_similarity_threshold=args.similarity_threshold
    )
    
    # Log configuration
    logger.info(f"""
============================================================
Mixed Content Pipeline V1 - CONVEYANCE Analysis
============================================================
PDF Directory: {config.pdf_dir or 'Not specified'}
Code Directory: {config.code_dir or 'Not specified'}
PDF Workers: {config.pdf_workers} on GPU {config.docling_gpu}
Code Workers: {config.code_workers} on {'GPU ' + str(config.code_processing_gpu) if config.code_processing_gpu >= 0 else 'CPU'}
Embedding Workers: {config.late_workers} on GPU {config.embedding_gpu}
CONVEYANCE Analysis: {'Enabled' if config.enable_conveyance_analysis else 'Disabled'}
Similarity Threshold: {config.semantic_similarity_threshold}
============================================================
""")
    
    # Initialize pipeline
    pipeline = MixedContentPipeline(config)
    
    # Setup database
    if not pipeline.setup_database():
        logger.error("Failed to setup database")
        sys.exit(1)
        
    try:
        # Start pipeline
        pipeline.start()
        
        # Queue files
        pipeline.queue_files()
        
        # Wait for completion
        pipeline.wait_for_completion()
        
        logger.info("✅ Mixed Content Pipeline Completed Successfully")
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        pipeline.stop_event.set()
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        

if __name__ == "__main__":
    main()