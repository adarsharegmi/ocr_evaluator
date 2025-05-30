#!/usr/bin/env python3
"""
Azure Document Intelligence OCR Test Script

This script demonstrates how to use Azure Document Intelligence service
for OCR and document analysis tasks.

Prerequisites:
1. Install required packages: pip install azure-ai-documentintelligence azure-core
2. Set up Azure Document Intelligence service in Azure portal
3. Get your endpoint and API key
4. Set environment variables or update the config section below
"""

import os
import time
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from azure.core.exceptions import HttpResponseError

class AzureOCRTester:
    def __init__(self, endpoint=None, key=None):
        """
        Initialize Azure Document Intelligence client
        
        Args:
            endpoint: Azure service endpoint (optional, can use env var)
            key: Azure service key (optional, can use env var)
        """
        self.endpoint = os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT', "https://mcp-rag-001.cognitiveservices.azure.com/")
        self.key =  os.environ.get('AZURE_DOCUMENT_INTELLIGENCE_KEY', "ALEEopWcyATjvMix4JVoJiWzxNr4DyTDnZ1YCF5YzaZerNmli19jJQQJ99BEACYeBjFXJ3w3AAALACOGvfX6")
        if not self.endpoint or not self.key:
            raise ValueError(
                "Please provide endpoint and key either as parameters or set environment variables:\n"
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT\n"
                "AZURE_DOCUMENT_INTELLIGENCE_KEY"
            )
        
        self.client = DocumentIntelligenceClient(
            endpoint=self.endpoint,
            credential=AzureKeyCredential(self.key)
        )
    
    def analyze_document_from_url(self, document_url, model_id="prebuilt-read"):
        """
        Analyze document from URL
        
        Args:
            document_url: URL of the document to analyze
            model_id: Model to use (prebuilt-read, prebuilt-layout, prebuilt-document, etc.)
        """
        print(f"Analyzing document from URL: {document_url}")
        print(f"Using model: {model_id}")
        
        try:
            # Create the analyze request body
            analyze_request = {"urlSource": document_url}
            
            poller = self.client.begin_analyze_document(
                model_id,
                analyze_request
            )
            
            print("Analysis started. Waiting for completion...")
            result = poller.result()
            
            return self._process_result(result)
            
        except HttpResponseError as e:
            print(f"HTTP Error: {e}")
            print(f"Status Code: {e.status_code}")
            print(f"Error Details: {e.error}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def analyze_document_from_file(self, file_path, model_id="prebuilt-read"):
        """
        Analyze document from local file
        
        Args:
            file_path: Path to the local file
            model_id: Model to use
        """
        print(f"Analyzing document from file: {file_path}")
        print(f"Using model: {model_id}")
        
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        try:
            with open(file_path, "rb") as f:
                file_content = f.read()
            
            poller = self.client.begin_analyze_document(
                model_id,
                file_content,
                content_type="application/octet-stream"
            )
            
            print("Analysis started. Waiting for completion...")
            result = poller.result()
            
            return self._process_result(result)
            
        except HttpResponseError as e:
            print(f"HTTP Error: {e}")
            print(f"Status Code: {e.status_code}")
            print(f"Error Details: {e.error}")
            return None
        except Exception as e:
            print(f"Error: {e}")
            return None
    
    def _process_result(self, result):
        """Process and display analysis results"""
        analysis_result = {
            'content': result.content,
            'pages': [],
            'tables': [],
            'key_value_pairs': []
        }
        
        print("\n" + "="*50)
        print("ANALYSIS RESULTS")
        print("="*50)
        
        # Display extracted content
        print(f"\nExtracted Content:\n{result.content}")
        
        # Process pages
        if result.pages:
            print(f"\nFound {len(result.pages)} page(s)")
            for page_idx, page in enumerate(result.pages):
                page_info = {
                    'page_number': page.page_number,
                    'width': page.width,
                    'height': page.height,
                    'unit': page.unit,
                    'lines': []
                }
                
                print(f"\nPage {page.page_number}:")
                print(f"  Dimensions: {page.width} x {page.height} {page.unit}")
                
                if page.lines:
                    print(f"  Lines found: {len(page.lines)}")
                    for line in page.lines:
                        line_info = {
                            'content': line.content,
                            'bounding_box': line.polygon if hasattr(line, 'polygon') else None
                        }
                        page_info['lines'].append(line_info)
                        print(f"    - {line.content}")
                
                analysis_result['pages'].append(page_info)
        
        # Process tables
        if result.tables:
            print(f"\nFound {len(result.tables)} table(s)")
            for table_idx, table in enumerate(result.tables):
                table_info = {
                    'row_count': table.row_count,
                    'column_count': table.column_count,
                    'cells': []
                }
                
                print(f"\nTable {table_idx + 1}:")
                print(f"  Rows: {table.row_count}, Columns: {table.column_count}")
                
                for cell in table.cells:
                    cell_info = {
                        'content': cell.content,
                        'row_index': cell.row_index,
                        'column_index': cell.column_index,
                        'is_header': getattr(cell, 'is_header', False)
                    }
                    table_info['cells'].append(cell_info)
                    print(f"    Cell[{cell.row_index},{cell.column_index}]: {cell.content}")
                
                analysis_result['tables'].append(table_info)
        
        # Process key-value pairs (if available)
        if hasattr(result, 'key_value_pairs') and result.key_value_pairs:
            print(f"\nFound {len(result.key_value_pairs)} key-value pair(s)")
            for kv in result.key_value_pairs:
                kv_info = {
                    'key': kv.key.content if kv.key else None,
                    'value': kv.value.content if kv.value else None
                }
                analysis_result['key_value_pairs'].append(kv_info)
                print(f"    {kv.key.content if kv.key else 'N/A'}: {kv.value.content if kv.value else 'N/A'}")
        
        return analysis_result

def main():
    """Main function to demonstrate usage"""
    # Configuration - Update these or set environment variables
    ENDPOINT = "YOUR_ENDPOINT_HERE"  # e.g., "https://your-resource.cognitiveservices.azure.com/"
    KEY = "YOUR_KEY_HERE"
    
    # Sample document URLs for testing
    sample_urls = [
        "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf",
        "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"
    ]
    
    try:
        # Initialize OCR tester
        ocr_tester = AzureOCRTester(endpoint=ENDPOINT, key=KEY)
        
        print("Azure Document Intelligence OCR Test")
        print("="*40)
        
        # Test different models
        models_to_test = [
            # ("prebuilt-read", "General OCR - extracts text"),
            # ("prebuilt-layout", "Layout analysis - extracts text, tables, structure"),
            ("prebuilt-document", "Document analysis - extracts text, tables, key-value pairs")
        ]
        
        for model_id, description in models_to_test:
            print(f"\n{'='*60}")
            print(f"Testing Model: {model_id}")
            print(f"Description: {description}")
            print('='*60)
            
            # Test with sample URL
            result = ocr_tester.analyze_document_from_url(
                sample_urls[0], 
                model_id=model_id
            )
            
            if result:
                print(f"✅ Successfully analyzed document with {model_id}")
            else:
                print(f"❌ Failed to analyze document with {model_id}")
            
            # Add delay between requests
            time.sleep(2)
        
        # Example for local file analysis
        print(f"\n{'='*60}")
        print("Local File Analysis Example")
        print('='*60)
        print("To analyze a local file, use:")
        print("result = ocr_tester.analyze_document_from_file('path/to/your/document.pdf')")
        
    except ValueError as e:
        print(f"Configuration Error: {e}")
        print("\nSetup Instructions:")
        print("1. Create an Azure Document Intelligence resource in Azure portal")
        print("2. Get your endpoint and key from the resource")
        print("3. Either update the ENDPOINT and KEY variables in this script")
        print("   OR set environment variables:")
        print("   - AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")
        print("   - AZURE_DOCUMENT_INTELLIGENCE_KEY")
    
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == "__main__":
    main()