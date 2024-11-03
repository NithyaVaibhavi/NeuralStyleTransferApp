# Necessary imports
from flask import Flask, render_template, request, send_file, jsonify
import os
import tensorflow as tf
import numpy as np
import PIL.Image
import tensorflow_hub as hub

# Initialize Flask app and set up output directory
app = Flask(__name__)
OUTPUT_DIR = 'static/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load the pre-trained TensorFlow Hub model
hub_model = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2')

# Helper function to convert tensor to image
def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor) > 3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Helper function to load and preprocess images
def load_img(path_to_img):
    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

# Route for homepage
@app.route('/')
def home():
    return render_template('home2.html')

@app.route('/home2.html')
def home2():
    return render_template('home2.html')

@app.route('/singleStyleTransfer.html')
def single_style_transfer():
    return render_template('singleStyleTransfer.html')

@app.route('/multiStyleTransfer.html')
def multi_style_transfer():
    return render_template('multiStyleTransfer.html')

# API endpoint for single style transfer
@app.route('/api/style-transfer', methods=['POST'])
def style_transfer():
    content_image = request.files['contentImage']
    style_image = request.files['styleImage']
    content_path = os.path.join(OUTPUT_DIR, 'content.jpg')
    style_path = os.path.join(OUTPUT_DIR, 'style.jpg')
    content_image.save(content_path)
    style_image.save(style_path)
    content_image = load_img(content_path)
    style_image = load_img(style_path)
    stylized_image = hub_model(tf.constant(content_image), tf.constant(style_image))[0]
    output_image = tensor_to_image(stylized_image)
    output_path = os.path.join(OUTPUT_DIR, 'stylized_image.jpg')
    output_image.save(output_path)
    return {'resultImageUrl': f'/static/output/stylized_image.jpg'}

# API endpoint for multi-style transfer
@app.route('/api/multi-style-transfer', methods=['POST'])
def multi_style_transfer_handler():
    # Get the content image and style images from the request
    content_image = request.files['contentImage']
    style_images = request.files.getlist('styleImages')
    
    # Save the content image
    content_path = os.path.join(OUTPUT_DIR, 'content.jpg')
    content_image.save(content_path)
    content_image_tensor = load_img(content_path)

    # Initialize a list to hold style tensors
    style_tensors = []
    
    # Process each style image and load them as tensors
    for idx, style_image in enumerate(style_images):
        style_path = os.path.join(OUTPUT_DIR, f'style_{idx}.jpg')
        style_image.save(style_path)  # Save the style image to disk
        style_image_tensor = load_img(style_path)  # Load and preprocess the style image
        style_tensors.append(style_image_tensor)  # Append the tensor to the list

    # Generate style features for each style tensor
    style_features = [hub_model(tf.constant(content_image_tensor), tf.constant(style_tensor))[0] for style_tensor in style_tensors]
    
    # Blend the style features by averaging them
    blended_style = tf.reduce_mean(tf.stack(style_features), axis=0)
    
    # Apply the blended style to the content image
    stylized_image = hub_model(tf.constant(content_image_tensor), tf.constant(blended_style))[0]
    
    # Convert the tensor to an image and save it
    output_image = tensor_to_image(stylized_image)
    output_path = os.path.join(OUTPUT_DIR, 'stylized_image.jpg')
    output_image.save(output_path)

    # Return the URL for the generated stylized image
    return jsonify({'resultImageUrls': [f'/static/output/stylized_image.jpg']})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Default to 5000 if not set
    app.run(host='0.0.0.0', port=port, debug=True)
