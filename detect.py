import requests
import json
import base64
from PIL import Image
import io
import argparse

def encode_image_to_base64(image_path):
    """
    Convert an image file to base64 string
    """
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA format
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
            # Convert image to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        raise Exception(f"Error encoding image: {str(e)}")

def detect_objects(image_path, model="llama3.2-vision"):
    """
    Detect objects in an image using Ollama's vision model
    """
    # Encode image
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the prompt
    prompt = """
    Analyze the image and provide ONLY a JSON response with two sections:
    1. "detections": array of detected objects with bounding boxes
    2. "vehicles": detailed information about any vehicles

    Required format:
    {
        "detections": [
            {
                "class": "<object type>",
                "confidence": <number between 0 and 1>,
                "bounding_box": [x1, y1, x2, y2]
            }
        ],
        "vehicles": [
            {
                "type": "car/truck/van/etc",
                "make": "manufacturer",
                "model": "model name",
                "color": "primary color",
                "year": "approximate year or year range",
                "license_plate": "plate number if visible",
                "location": "brief position description",
                "confidence": <number between 0 and 1>
            }
        ]
    }

    For vehicles, provide as much detail as you can confidently determine. If any field cannot be determined, use null.
    For license plates, only include if text is clearly readable.
    """
    
    # Prepare the request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        # Make the request
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Parse the response
        result = response.json()
        return result['response']
        
    except requests.exceptions.RequestException as e:
        if "Connection refused" in str(e):
            raise Exception("Could not connect to Ollama. Make sure it's running on localhost:11434")
        raise Exception(f"Error making request to Ollama: {str(e)}")
    

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect objects in an image using Llama vision model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                      help='Model to use (default: llama3.2-vision)')
    
    args = parser.parse_args()
    
    try:
        # Run detection
        print("Analyzing image...")
        result = detect_objects(args.image_path, args.model)
        print("\nDetected objects:")
        print(result)
        
    except Exception as e:
        print(f"Error: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()