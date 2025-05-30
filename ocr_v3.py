#!/usr/bin/env python3
"""
Multi-Engine OCR Comparison Tool with PDF Support

This script demonstrates how to use multiple OCR engines and compare their results:
- Azure Document Intelligence
- AWS Textract
- Tesseract OCR (with PDF to image conversion)
- EasyOCR (with PDF to image conversion)

Prerequisites:
pip install azure-ai-documentintelligence azure-core boto3 pytesseract easyocr opencv-python pillow pandas matplotlib seaborn difflib Levenshtein pdf2image PyMuPDF

For PDF support:
- Install poppler-utils (Linux/Mac) or download poppler for Windows
- Or use PyMuPDF as alternative PDF processor
"""

import os
import time
import difflib
from datetime import datetime
from typing import Dict, List, Union
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
import io
import dotenv
dotenv.load_dotenv()
# OCR Engine Imports
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential

try:
    import boto3
    from botocore.exceptions import NoCredentialsError, ClientError
    AWS_AVAILABLE = True
except ImportError:
    AWS_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import Levenshtein
    LEVENSHTEIN_AVAILABLE = True
except ImportError:
    LEVENSHTEIN_AVAILABLE = False

# PDF processing imports
try:
    from pdf2image import convert_from_path, convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    PDF2IMAGE_AVAILABLE = False

try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

class OCRResult:
    """Class to store OCR results with metadata"""
    def __init__(self, engine_name: str, text: str, confidence: float = None, 
                 processing_time: float = None, bounding_boxes: List = None):
        self.engine_name = engine_name
        self.text = text
        self.confidence = confidence
        self.processing_time = processing_time
        self.bounding_boxes = bounding_boxes or []
        self.timestamp = datetime.now()
        
    def to_dict(self):
        return {
            'engine_name': self.engine_name,
            'text': self.text,
            'confidence': self.confidence,
            'processing_time': self.processing_time,
            'word_count': len(self.text.split()),
            'char_count': len(self.text),
            'timestamp': self.timestamp.isoformat()
        }

class PDFProcessor:
    """Class to handle PDF to image conversion"""
    
    @staticmethod
    def is_pdf(file_path: str) -> bool:
        """Check if file is a PDF"""
        return file_path.lower().endswith('.pdf')
    
    @staticmethod
    def pdf_to_images(file_path: str, dpi: int = 200) -> List[Image.Image]:
        """Convert PDF to list of PIL Images"""
        if not PDFProcessor.is_pdf(file_path):
            # If not PDF, try to open as image
            try:
                return [Image.open(file_path)]
            except Exception as e:
                raise ValueError(f"Cannot process file {file_path}: {e}")
        
        images = []
        
        # Try pdf2image first
        if PDF2IMAGE_AVAILABLE:
            try:
                images = convert_from_path(file_path, dpi=dpi)
                return images
            except Exception as e:
                print(f"pdf2image failed: {e}, trying PyMuPDF...")
        
        # Fallback to PyMuPDF
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(file_path)
                for page_num in range(len(doc)):
                    page = doc.load_page(page_num)
                    # Convert to PIL Image
                    pix = page.get_pixmap(matrix=fitz.Matrix(dpi/72, dpi/72))
                    img_data = pix.tobytes("ppm")
                    img = Image.open(io.BytesIO(img_data))
                    images.append(img)
                doc.close()
                return images
            except Exception as e:
                raise ValueError(f"PyMuPDF failed: {e}")
        
        raise ValueError("No PDF processing library available. Install pdf2image or PyMuPDF")

class AzureOCREngine:
    """Azure Document Intelligence OCR Engine"""
    def __init__(self, endpoint=None, key=None):
        self.endpoint = endpoint or os.getenv('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT')
        self.key = key or os.getenv('AZURE_DOCUMENT_INTELLIGENCE_KEY')
        
        if not self.endpoint or not self.key:
            raise ValueError("Azure credentials not provided. Set AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT and AZURE_DOCUMENT_INTELLIGENCE_KEY environment variables")
            
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
    
    def extract_text(self, file_path: str, model_id="prebuilt-read") -> OCRResult:
        start_time = time.time()
        
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            poller = self.client.begin_analyze_document(
                model_id,
                file_content,
                content_type="application/octet-stream"
            )
            
            result = poller.result()
            processing_time = time.time() - start_time
            
            # Extract bounding boxes
            bounding_boxes = []
            if result.pages:
                for page in result.pages:
                    if hasattr(page, 'lines') and page.lines:
                        for line in page.lines:
                            if hasattr(line, 'polygon'):
                                bounding_boxes.append({
                                    'text': line.content,
                                    'bbox': line.polygon
                                })
            
            return OCRResult(
                engine_name="Azure Document Intelligence",
                text=result.content or "",
                processing_time=processing_time,
                bounding_boxes=bounding_boxes
            )
            
        except Exception as e:
            return OCRResult(
                engine_name="Azure Document Intelligence",
                text=f"Error: {str(e)}",
                processing_time=time.time() - start_time
            )





class AWSTextractEngine:
    """AWS Textract OCR Engine with S3 async processing support"""
    def __init__(self, region_name='us-east-1', aws_access_key_id=None, aws_secret_access_key=None, aws_session_token=None):
        if not AWS_AVAILABLE:
            raise ImportError("boto3 not available")
        
        try:
            if aws_access_key_id is None:
                aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID', None)
            if aws_secret_access_key is None:
                aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY', None)
            if aws_session_token is None:
                aws_session_token = os.getenv('AWS_SESSION_TOKEN', None)
                
            if aws_access_key_id and aws_secret_access_key:
                self.textract_client = boto3.client(
                    'textract',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token
                )
                self.s3_client = boto3.client(
                    's3',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key_id,
                    aws_secret_access_key=aws_secret_access_key,
                    aws_session_token=aws_session_token
                )
            else:
                # Use default credential chain
                self.textract_client = boto3.client('textract', region_name=region_name)
                self.s3_client = boto3.client('s3', region_name=region_name)
                
        except Exception as e:
            raise ValueError(f"Failed to initialize AWS clients: {e}")
    
    def extract_text(self, file_path: str, s3_bucket: str = None, s3_key: str = None) -> OCRResult:
        """
        Extract text from document. Supports both local files and S3 objects.
        
        Args:
            file_path: Local file path (used if s3_bucket/s3_key not provided)
            s3_bucket: S3 bucket name (for S3-based processing)
            s3_key: S3 object key (for S3-based processing)
        """
        start_time = time.time()
        
        try:
            # If S3 parameters provided, use S3-based processing
            if s3_bucket and s3_key:
                return self._process_from_s3(s3_bucket, s3_key, start_time)
            
            # Otherwise, process local file
            return self._process_local_file(file_path, start_time)
            
        except (NoCredentialsError, ClientError) as e:
            return OCRResult(
                engine_name="AWS Textract",
                text=f"AWS Error: {str(e)}",
                processing_time=time.time() - start_time
            )
        except Exception as e:
            return OCRResult(
                engine_name="AWS Textract",
                text=f"Error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def extract_text_from_s3(self, s3_bucket: str, s3_key: str) -> OCRResult:
        """
        Convenience method for S3-based processing
        """
        return self.extract_text(None, s3_bucket, s3_key)
    
    def _process_from_s3(self, s3_bucket: str, s3_key: str, start_time: float) -> OCRResult:
        """Process document directly from S3"""
        
        # Check if file is PDF by extension or content type
        is_pdf = s3_key.lower().endswith('.pdf')
        
        if not is_pdf:
            # Check content type from S3 metadata
            try:
                response = self.s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
                content_type = response.get('ContentType', '')
                is_pdf = content_type == 'application/pdf'
            except:
                pass  # If we can't check, assume based on extension
        
        if is_pdf:
            # Use async processing for PDFs
            return self._process_pdf_async_s3(s3_bucket, s3_key, start_time)
        else:
            # Use sync processing for images
            return self._process_image_sync_s3(s3_bucket, s3_key, start_time)
    
    def _process_local_file(self, file_path: str, start_time: float) -> OCRResult:
        """Process local file (upload to S3 if needed for large PDFs)"""
        
        with open(file_path, "rb") as f:
            file_content = f.read()
        
        file_size_mb = len(file_content) / (1024 * 1024)
        
        if PDFProcessor.is_pdf(file_path):
            # For PDFs > 5MB, recommend S3 upload for better performance
            if file_size_mb > 5:
                print(f"Large PDF detected ({file_size_mb:.1f}MB). Consider uploading to S3 for optimal performance.")
                print("Converting to images for processing...")
                return self._process_pdf_via_images(file_content, start_time)
            else:
                # Try synchronous processing first
                try:
                    response = self.textract_client.analyze_document(
                        Document={'Bytes': file_content},
                        FeatureTypes=['TABLES', 'FORMS']
                    )
                    return self._extract_text_from_response(response, start_time)
                except ClientError as e:
                    error_code = e.response.get('Error', {}).get('Code', '')
                    if 'InvalidParameterException' in error_code or 'UnsupportedDocumentException' in error_code:
                        print("Direct PDF processing failed, converting to images...")
                        return self._process_pdf_via_images(file_content, start_time)
                    else:
                        raise e
        else:
            # Process image directly
            response = self.textract_client.detect_document_text(
                Document={'Bytes': file_content}
            )
            return self._extract_text_from_response(response, start_time)
    
    def _process_image_sync_s3(self, s3_bucket: str, s3_key: str, start_time: float) -> OCRResult:
        """Process image from S3 using synchronous detection"""
        
        response = self.textract_client.detect_document_text(
            Document={
                'S3Object': {
                    'Bucket': s3_bucket,
                    'Name': s3_key
                }
            }
        )
        return self._extract_text_from_response(response, start_time)
    
    def _process_pdf_async_s3(self, s3_bucket: str, s3_key: str, start_time: float) -> OCRResult:
        """Process PDF from S3 using asynchronous detection"""
        
        try:
            print(f"Starting async processing for s3://{s3_bucket}/{s3_key}")
            
            # Start async job
            response = self.textract_client.start_document_text_detection(
                DocumentLocation={
                    'S3Object': {
                        'Bucket': s3_bucket,
                        'Name': s3_key
                    }
                }
            )
            
            job_id = response['JobId']
            print(f"Job started with ID: {job_id}")
            
            # Poll for completion
            return self._wait_for_job_completion(job_id, start_time)
            
        except Exception as e:
            return OCRResult(
                engine_name="AWS Textract",
                text=f"S3 Async processing error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _wait_for_job_completion(self, job_id: str, start_time: float, 
                                max_wait_time: int = 600, poll_interval: int = 5) -> OCRResult:
        """Wait for async job completion and extract results"""
        
        waited_time = 0
        
        while waited_time < max_wait_time:
            try:
                result = self.textract_client.get_document_text_detection(JobId=job_id)
                status = result['JobStatus']
                
                print(f"Job status: {status} (waited {waited_time}s)")
                
                if status == 'SUCCEEDED':
                    print("Job completed successfully, extracting text...")
                    return self._extract_text_from_async_result(job_id, start_time)
                
                elif status == 'FAILED':
                    error_msg = result.get('StatusMessage', 'Unknown error')
                    return OCRResult(
                        engine_name="AWS Textract",
                        text=f"Async processing failed: {error_msg}",
                        processing_time=time.time() - start_time
                    )
                
                elif status in ['IN_PROGRESS', 'PARTIAL_SUCCESS']:
                    # Still processing, wait and check again
                    time.sleep(poll_interval)
                    waited_time += poll_interval
                else:
                    # Unknown status
                    return OCRResult(
                        engine_name="AWS Textract",
                        text=f"Unknown job status: {status}",
                        processing_time=time.time() - start_time
                    )
                
            except Exception as e:
                return OCRResult(
                    engine_name="AWS Textract",
                    text=f"Error checking job status: {str(e)}",
                    processing_time=time.time() - start_time
                )
        
        # Timeout
        return OCRResult(
            engine_name="AWS Textract",
            text="Error: Async processing timeout",
            processing_time=time.time() - start_time
        )
    
    def _extract_text_from_async_result(self, job_id: str, start_time: float) -> OCRResult:
        """Extract text from completed async job"""
        
        text_blocks = []
        bounding_boxes = []
        next_token = None
        page_count = 0
        
        while True:
            try:
                if next_token:
                    result = self.textract_client.get_document_text_detection(
                        JobId=job_id, 
                        NextToken=next_token
                    )
                else:
                    result = self.textract_client.get_document_text_detection(JobId=job_id)
                
                # Process blocks from this batch
                for block in result.get('Blocks', []):
                    if block['BlockType'] == 'LINE':
                        text_blocks.append(block['Text'])
                        if 'Geometry' in block:
                            bounding_boxes.append({
                                'text': block['Text'],
                                'bbox': block['Geometry']['BoundingBox'],
                                'page': block.get('Page', 1)
                            })
                    elif block['BlockType'] == 'PAGE':
                        page_count += 1
                
                next_token = result.get('NextToken')
                if not next_token:
                    break
                    
            except Exception as e:
                return OCRResult(
                    engine_name="AWS Textract",
                    text=f"Error extracting async results: {str(e)}",
                    processing_time=time.time() - start_time
                )
        
        processing_time = time.time() - start_time
        extracted_text = '\n'.join(text_blocks)
        
        print(f"Extracted text from {page_count} pages")
        
        return OCRResult(
            engine_name="AWS Textract",
            text=extracted_text,
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )
    
    def _process_pdf_via_images(self, file_content: bytes, start_time: float) -> OCRResult:
        """Fallback: Process PDF by converting to images"""
        
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file.write(file_content)
                temp_file_path = temp_file.name
            
            try:
                # Convert PDF to images
                images = PDFProcessor.pdf_to_images(temp_file_path, dpi=200)
                
                all_text = []
                all_bounding_boxes = []
                
                for i, image in enumerate(images):
                    print(f"Processing page {i+1}/{len(images)}...")
                    
                    # Convert PIL Image to bytes for Textract
                    img_buffer = io.BytesIO()
                    image.save(img_buffer, format='PNG')
                    img_bytes = img_buffer.getvalue()
                    
                    # Process image with Textract
                    response = self.textract_client.detect_document_text(
                        Document={'Bytes': img_bytes}
                    )
                    
                    # Extract text from this page
                    page_text, page_boxes = self._extract_text_and_boxes_from_response(response, i+1)
                    all_text.append(page_text)
                    all_bounding_boxes.extend(page_boxes)
                
                processing_time = time.time() - start_time
                combined_text = '\n\n--- Page Break ---\n\n'.join(all_text)
                
                return OCRResult(
                    engine_name="AWS Textract",
                    text=combined_text,
                    processing_time=processing_time,
                    bounding_boxes=all_bounding_boxes
                )
                
            finally:
                os.unlink(temp_file_path)
                
        except Exception as e:
            return OCRResult(
                engine_name="AWS Textract",
                text=f"PDF image processing error: {str(e)}",
                processing_time=time.time() - start_time
            )
    
    def _extract_text_from_response(self, response: dict, start_time: float) -> OCRResult:
        """Extract text from synchronous Textract response"""
        
        processing_time = time.time() - start_time
        text_blocks = []
        bounding_boxes = []
        
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                text_blocks.append(block['Text'])
                if 'Geometry' in block:
                    bounding_boxes.append({
                        'text': block['Text'],
                        'bbox': block['Geometry']['BoundingBox']
                    })
        
        extracted_text = '\n'.join(text_blocks)
        
        return OCRResult(
            engine_name="AWS Textract",
            text=extracted_text,
            processing_time=processing_time,
            bounding_boxes=bounding_boxes
        )
    
    def _extract_text_and_boxes_from_response(self, response: dict, page_num: int = 1):
        """Extract text and bounding boxes from response, return as tuple"""
        
        text_blocks = []
        bounding_boxes = []
        
        for block in response.get('Blocks', []):
            if block['BlockType'] == 'LINE':
                text_blocks.append(block['Text'])
                if 'Geometry' in block:
                    bounding_boxes.append({
                        'text': block['Text'],
                        'bbox': block['Geometry']['BoundingBox'],
                        'page': page_num
                    })
        
        return '\n'.join(text_blocks), bounding_boxes

    

class TesseractOCREngine:
    """Tesseract OCR Engine with PDF support"""
    def __init__(self, tesseract_cmd=None):
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available")
        
        if tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    
    def extract_text(self, file_path: str) -> OCRResult:
        start_time = time.time()
        
        try:
            # Convert PDF to images if needed
            images = PDFProcessor.pdf_to_images(file_path)
            
            all_text = []
            all_confidences = []
            
            for i, image in enumerate(images):
                # Get text with confidence
                data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
                text = pytesseract.image_to_string(image)
                
                all_text.append(text)
                
                # Calculate average confidence for this page
                confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                if confidences:
                    all_confidences.extend(confidences)
            
            processing_time = time.time() - start_time
            
            # Combine all text
            combined_text = '\n\n'.join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return OCRResult(
                engine_name="Tesseract",
                text=combined_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            return OCRResult(
                engine_name="Tesseract",
                text=f"Error: {str(e)}",
                processing_time=time.time() - start_time
            )

class EasyOCREngine:
    """EasyOCR Engine with PDF support"""
    def __init__(self, languages=['en']):
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available")
        
        self.reader = easyocr.Reader(languages)
    
    def extract_text(self, file_path: str) -> OCRResult:
        start_time = time.time()
        
        try:
            # Convert PDF to images if needed
            images = PDFProcessor.pdf_to_images(file_path)
            
            all_text = []
            all_confidences = []
            
            for image in images:
                # Convert PIL Image to numpy array for EasyOCR
                image_array = np.array(image)
                results = self.reader.readtext(image_array)
                
                page_text = ' '.join([result[1] for result in results])
                all_text.append(page_text)
                
                confidences = [result[2] for result in results]
                all_confidences.extend(confidences)
            
            processing_time = time.time() - start_time
            
            # Combine all text
            combined_text = '\n\n'.join(all_text)
            avg_confidence = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            
            return OCRResult(
                engine_name="EasyOCR",
                text=combined_text,
                confidence=avg_confidence,
                processing_time=processing_time
            )
            
        except Exception as e:
            return OCRResult(
                engine_name="EasyOCR",
                text=f"Error: {str(e)}",
                processing_time=time.time() - start_time
            )

class OCRComparator:
    """Class to compare OCR results from multiple engines"""
    
    def __init__(self):
        self.engines = {}
        self.results = []
    
    def add_engine(self, name: str, engine):
        """Add an OCR engine to the comparison"""
        self.engines[name] = engine
    
    def compare_engines(self, file_path: str, ground_truth: str = None, s3_bucket: str= None, s3_key: str = None) -> Dict:
        """Compare all engines on a single document"""
        results = []
        
        print(f"Processing file: {file_path}")
        print("=" * 50)
        
        for name, engine in self.engines.items():
            print(f"Running {name}...")
            try:
                if hasattr(engine, 'extract_text_from_s3') and name == "AWS Textract":
                # Use S3 method for AWS Textract
                    result = engine.extract_text_from_s3(s3_bucket, s3_key)
                    
                else:
                    result = engine.extract_text(file_path)
                results.append(result)
                print(f"✅ {name} completed in {result.processing_time:.2f}s")

                # Export the result in comparison.json file as dict 
                result_dict = result.to_dict()
                with open('comparison.json', 'a', encoding='utf-8') as f:
                    f.write(f"{result_dict}\n")
            except Exception as e:
                print(f"❌ {name} failed: {str(e)}")
                results.append(OCRResult(name, f"Error: {str(e)}"))
        
        # Calculate comparison metrics
        comparison_data = self._calculate_metrics(results, ground_truth)
        
        return {
            'file_path': file_path,
            'results': results,
            'comparison_data': comparison_data,
            'ground_truth': ground_truth
        }
    
    def _calculate_metrics(self, results: List[OCRResult], ground_truth: str = None) -> Dict:
        """Calculate comparison metrics between OCR results"""
        metrics = {}
        
        # Basic metrics
        for result in results:
            if not result.text.startswith("Error:"):
                metrics[result.engine_name] = {
                    'text_length': len(result.text),
                    'word_count': len(result.text.split()),
                    'processing_time': result.processing_time,
                    'confidence': result.confidence
                }
        
        # Cross-comparison metrics
        valid_results = [r for r in results if not r.text.startswith("Error:")]
        if len(valid_results) >= 2:
            for i, result1 in enumerate(valid_results):
                for j, result2 in enumerate(valid_results[i+1:], i+1):
                    comparison_key = f"{result1.engine_name}_vs_{result2.engine_name}"
                    
                    # Character-level similarity
                    char_similarity = self._calculate_similarity(result1.text, result2.text)
                    
                    # Word-level similarity
                    words1 = set(result1.text.lower().split())
                    words2 = set(result2.text.lower().split())
                    word_similarity = len(words1.intersection(words2)) / len(words1.union(words2)) if words1.union(words2) else 0
                    
                    metrics[comparison_key] = {
                        'character_similarity': char_similarity,
                        'word_similarity': word_similarity,
                        'edit_distance': self._calculate_edit_distance(result1.text, result2.text)
                    }
        
        # Ground truth comparison
        if ground_truth:
            for result in valid_results:
                gt_key = f"{result.engine_name}_vs_ground_truth"
                metrics[gt_key] = {
                    'accuracy': self._calculate_accuracy(result.text, ground_truth),
                    'character_similarity': self._calculate_similarity(result.text, ground_truth),
                    'edit_distance': self._calculate_edit_distance(result.text, ground_truth)
                }
        
        return metrics
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity using SequenceMatcher"""
        return difflib.SequenceMatcher(None, text1, text2).ratio()
    
    def _calculate_edit_distance(self, text1: str, text2: str) -> int:
        """Calculate edit distance between two texts"""
        if LEVENSHTEIN_AVAILABLE:
            return Levenshtein.distance(text1, text2)
        else:
            # Simple edit distance calculation
            if len(text1) < len(text2):
                return self._calculate_edit_distance(text2, text1)
            
            if len(text2) == 0:
                return len(text1)
            
            previous_row = list(range(len(text2) + 1))
            for i, c1 in enumerate(text1):
                current_row = [i + 1]
                for j, c2 in enumerate(text2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
    
    def _calculate_accuracy(self, extracted_text: str, ground_truth: str) -> float:
        """Calculate accuracy percentage"""
        if not ground_truth:
            return 0.0
        
        # Character-level accuracy
        total_chars = max(len(extracted_text), len(ground_truth))
        if total_chars == 0:
            return 1.0
        
        edit_dist = self._calculate_edit_distance(extracted_text, ground_truth)
        accuracy = max(0, (total_chars - edit_dist) / total_chars)
        return accuracy
    
    def generate_comparison_report(self, comparison_results: Dict, output_path: str = None):
        """Generate a detailed comparison report"""
        report = []
        report.append("OCR Engine Comparison Report")
        report.append("=" * 50)
        report.append(f"File: {comparison_results['file_path']}")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Results summary
        report.append("Extracted Text Results:")
        report.append("-" * 30)
        for result in comparison_results['results']:
            report.append(f"\n{result.engine_name}:")
            report.append(f"Processing Time: {result.processing_time:.2f}s")
            if result.confidence:
                report.append(f"Confidence: {result.confidence:.2f}")
            report.append(f"Text Length: {len(result.text)} characters")
            report.append(f"Word Count: {len(result.text.split())} words")
            report.append(f"Text Preview: {result.text[:]}")
            report.append("")
        
        # Comparison metrics
        report.append("Comparison Metrics:")
        report.append("-" * 30)
        for key, value in comparison_results['comparison_data'].items():
            if isinstance(value, dict):
                report.append(f"\n{key}:")
                for metric, score in value.items():
                    if isinstance(score, float):
                        report.append(f"  {metric}: {score:.4f}")
                    else:
                        report.append(f"  {metric}: {score}")
        
        report_text = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_text)
            print(f"Report saved to: {output_path}")
        
        return report_text
    
    def create_visual_comparison(self, comparison_results: Dict, output_dir: str = None):
        """Create visual comparison charts"""
        if not output_dir:
            output_dir = "ocr_comparison_charts"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract data for visualization
        engines = []
        processing_times = []
        confidences = []
        text_lengths = []
        
        for result in comparison_results['results']:
            if not result.text.startswith("Error:"):
                engines.append(result.engine_name)
                processing_times.append(result.processing_time or 0)
                confidences.append(result.confidence or 0)
                text_lengths.append(len(result.text))
        
        if not engines:
            print("No successful results to visualize")
            return None
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('OCR Engine Comparison', fontsize=16)
        
        # Processing time comparison
        if processing_times:
            axes[0, 0].bar(engines, processing_times, color='skyblue')
            axes[0, 0].set_title('Processing Time Comparison')
            axes[0, 0].set_ylabel('Time (seconds)')
            axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Confidence comparison
        if any(c > 0 for c in confidences):
            axes[0, 1].bar(engines, confidences, color='lightgreen')
            axes[0, 1].set_title('Confidence Score Comparison')
            axes[0, 1].set_ylabel('Confidence')
            axes[0, 1].tick_params(axis='x', rotation=45)
        else:
            axes[0, 1].text(0.5, 0.5, 'No confidence data available', 
                           ha='center', va='center', transform=axes[0, 1].transAxes)
            axes[0, 1].set_title('Confidence Score Comparison')
        
        # Text length comparison
        if text_lengths:
            axes[1, 0].bar(engines, text_lengths, color='lightcoral')
            axes[1, 0].set_title('Extracted Text Length')
            axes[1, 0].set_ylabel('Characters')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Similarity heatmap
        if len(engines) > 1:
            similarity_matrix = np.zeros((len(engines), len(engines)))
            for i, engine1 in enumerate(engines):
                for j, engine2 in enumerate(engines):
                    if i == j:
                        similarity_matrix[i, j] = 1.0
                    else:
                        key = f"{engine1}_vs_{engine2}"
                        reverse_key = f"{engine2}_vs_{engine1}"
                        if key in comparison_results['comparison_data']:
                            similarity_matrix[i, j] = comparison_results['comparison_data'][key].get('character_similarity', 0)
                        elif reverse_key in comparison_results['comparison_data']:
                            similarity_matrix[i, j] = comparison_results['comparison_data'][reverse_key].get('character_similarity', 0)
            
            sns.heatmap(similarity_matrix, annot=True, xticklabels=engines, yticklabels=engines, 
                       cmap='YlOrRd', ax=axes[1, 1])
            axes[1, 1].set_title('Text Similarity Matrix')
        else:
            axes[1, 1].text(0.5, 0.5, 'Need 2+ engines for similarity comparison', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Text Similarity Matrix')
        
        plt.tight_layout()
        chart_path = os.path.join(output_dir, 'ocr_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {chart_path}")
        
        return chart_path

def check_dependencies():
    """Check and report available dependencies"""
    print("Checking dependencies...")
    print("=" * 30)
    
    dependencies = {
        "Azure Document Intelligence": True,  # Always available if azure packages installed
        "AWS Textract": AWS_AVAILABLE,
        "Tesseract": TESSERACT_AVAILABLE,
        "EasyOCR": EASYOCR_AVAILABLE,
        "PDF2Image": PDF2IMAGE_AVAILABLE,
        "PyMuPDF": PYMUPDF_AVAILABLE,
        "Levenshtein": LEVENSHTEIN_AVAILABLE
    }
    
    for name, available in dependencies.items():
        status = "✅" if available else "❌"
        print(f"{status} {name}")
    
    if not (PDF2IMAGE_AVAILABLE or PYMUPDF_AVAILABLE):
        print("\n⚠️  Warning: No PDF processing library available.")
        print("   Install pdf2image or PyMuPDF for PDF support.")
    
    print()

def main():
    """Main function to demonstrate usage"""
    # Check dependencies
    check_dependencies()
    
    # Initialize comparator
    comparator = OCRComparator()
    
    # Add available engines
    print("Initializing OCR engines...")
    
    # Azure Document Intelligence
    try:
        azure_engine = AzureOCREngine()
        comparator.add_engine("Azure", azure_engine)
        print("✅ Azure Document Intelligence initialized")
    except Exception as e:
        print(f"❌ Azure Document Intelligence failed: {e}")
    
    # AWS Textract
    s3_bucket = os.getenv('AWS_S3_BUCKET', None)
    s3_key= os.getenv('AWS_S3_KEY', None)
    if AWS_AVAILABLE:
        try:
            aws_engine = AWSTextractEngine()
            comparator.add_engine("AWS Textract", aws_engine)
            print("✅ AWS Textract initialized")
        except Exception as e:
            print(f"❌ AWS Textract failed: {e}")
    
    # Tesseract
    if TESSERACT_AVAILABLE:
        try:
            tesseract_engine = TesseractOCREngine()
            comparator.add_engine("Tesseract", tesseract_engine)
            print("✅ Tesseract initialized")
        except Exception as e:
            print(f"❌ Tesseract failed: {e}")
    
    # EasyOCR
    if EASYOCR_AVAILABLE:
        try:
            easyocr_engine = EasyOCREngine()
            comparator.add_engine("EasyOCR", easyocr_engine)
            print("✅ EasyOCR initialized")
        except Exception as e:
            print(f"❌ EasyOCR failed: {e}")
    
    # Example usage
    if comparator.engines:
        print(f"\nReady to compare {len(comparator.engines)} OCR engines")
        print("Available engines:", list(comparator.engines.keys()))
        
        # Example file path (update with your file)
        sample_file = "sample.pdf"  # Update this path
        
        if os.path.exists(sample_file):
            print(f"\nRunning comparison on: {sample_file}")
            
            # Optional: provide ground truth text for accuracy calculation
            ground_truth = ""  # Set this to the correct text if available
            
            # Run comparison
            results = comparator.compare_engines(sample_file, ground_truth, s3_bucket=s3_bucket, s3_key=s3_key)
            
            # Generate report
            report = comparator.generate_comparison_report(results, "ocr_comparison_report.txt")
            print("\nComparison Report:")
            print(report[:500] + "..." if len(report) > 500 else report)
            
            # Create visual comparison
            comparator.create_visual_comparison(results)
            
        else:
            print(f"\nSample file '{sample_file}' not found.")
            print("Please update the sample_file path in the main() function.")
            print("\nSupported formats: PDF, PNG, JPG, TIFF, etc.")
    else:
        print("No OCR engines were successfully initialized.")
        print("Please check your configuration and dependencies.")

if __name__ == "__main__":
    main()