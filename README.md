
# Neural Style Transfer with TensorFlow

## Overview

This project implements a deep learning model for neural style transfer using TensorFlow. The goal is to transfer the artistic style of one image onto the content of another, creating visually compelling outputs. The project leverages TensorFlow Hub's Magenta model for arbitrary image stylization and integrates Adaptive Instance Normalization (AdaIN) for enhanced style transfer capabilities.

## Features

- Transfer artistic styles from multiple style images onto a content image.
- Use of TensorFlow Hub's pre-trained models for efficient stylization.
- Implementation of Adaptive Instance Normalization (AdaIN) for improved results.
- User-friendly interface for input images and style selection.

## Getting Started

### Prerequisites

- Python 3.11
- TensorFlow
- TensorFlow Hub
- OpenCV
- Other required libraries can be found in `requirements.txt`

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-style-transfer.git
   cd neural-style-transfer
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage

1. Place your content and style images in the designated folder.
2. Run the style transfer script:
   ```bash
   python style_transfer.py --content_image path/to/content.jpg --style_image path/to/style.jpg --output_image path/to/output.jpg
   ```


## Project Structure

```
neural-style-transfer/
├── style_transfer.py         # Main script for style transfer
├── requirements.txt          # List of required packages
├── images/                   # Directory for content and style images
├── output/                   # Directory for output images
└── README.md                 # Project documentation
```

## Contributing

Contributions are welcome! If you have suggestions for improvements or features, please open an issue or submit a pull request.

## Acknowledgments

- [TensorFlow](https://www.tensorflow.org/)
- [TensorFlow Hub](https://tfhub.dev/)
- [Magenta](https://magenta.tensorflow.org/)
```
