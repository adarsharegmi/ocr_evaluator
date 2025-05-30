# Evaluate OCR

This repository contains tools and scripts for evaluating Optical Character Recognition (OCR) performance.

## Overview

This project aims to provide a framework for assessing the accuracy and effectiveness of various OCR solutions. By comparing OCR outputs against ground truth data, users can make informed decisions about which OCR solution best fits their needs.

## Features

- Benchmarking tools for OCR engines
- Metrics calculation (accuracy, precision, recall, F1 score)
- Visualization of OCR performance results
- Support for multiple OCR engines comparison

## Getting Started

### Prerequisites

- Python 3.7+
- Required Python packages (install via `pip install -r requirements.txt`)

### Installation

```bash
git clone https://github.com/yourusername/evaluate-ocr.git
cd evaluate-ocr
pip install -r requirements.txt
```

### Usage

```bash
python evaluate.py --input images/ --ground-truth data/ground_truth/ --ocr-engine tesseract
```

## Documentation

For detailed information on using this tool, please refer to the [documentation](docs/index.md).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.