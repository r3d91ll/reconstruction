#!/usr/bin/env python3
"""
Fix for GPU conflict between Docling OCR and embedding workers
"""

import os

def fix_pdf_worker_init():
    """
    Add this to PDFExtractionWorker.__init__ to force CPU usage for Docling
    """
    # Force Docling/EasyOCR to use CPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    
    # Alternative: Use specific GPU for OCR (if you have 3+ GPUs)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def fix_in_process_pdfs_v4():
    """
    Add to line ~410 in process_pdfs_continuous_gpu_v4.py
    before self.converter = self._init_docling()
    """
    # In PDFExtractionWorker.__init__, add:
    if hasattr(self.config, 'disable_ocr_gpu') and self.config.disable_ocr_gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Command line fix without code changes:
if __name__ == '__main__':
    print("""
Quick fixes without modifying code:

1. Disable OCR entirely (for academic PDFs this is usually fine):
   python3 process_pdfs_continuous_gpu_v4.py \\
     --pdf-dir /mnt/data/arxiv_data/pdf \\
     --max-pdfs 200 \\
     --batch-size 32 \\
     --gpu-devices 0 1 \\
     --db-host 192.168.1.69

2. Run PDF extraction on CPU while embeddings use GPU:
   # Start PDF extraction with CPU only
   CUDA_VISIBLE_DEVICES="" python3 extract_pdfs_only.py
   
   # Then run embeddings on GPU
   python3 embed_chunks_only.py --gpu-devices 0 1

3. Use only GPU 1 for embeddings (leave GPU 0 for OCR):
   python3 process_pdfs_continuous_gpu_v4.py \\
     --gpu-devices 1 \\
     --db-host 192.168.1.69 \\
     --max-pdfs 200
""")