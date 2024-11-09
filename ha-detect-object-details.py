#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image
import io
import argparse
import sys

def encode_image_to_base64(image_path):
    """Convert an image file to base64 string"""
    with Image.open(image_path) as img:
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

def detect_objects(image_path, model="llama3.2-vision"):
    base64_image = encode_image_to_base64(image_path)
    
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
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }
    
    response = requests.post(url, json=payload)
    result = response.json()['response']
    
    # Find and parse JSON in response
    start_idx = result.find('{')
    end_idx = result.rfind('}') + 1
    
    if start_idx != -1 and end_idx != -1:
        json_str = result[start_idx:end_idx]
        analysis = json.loads(json_str)
        
        # Create summary for output
        summary = []
        if 'vehicles' in analysis and analysis['vehicles']:
            for vehicle in analysis['vehicles']:
                desc = []
                if vehicle.get('color'):
                    desc.append(vehicle['color'])
                if vehicle.get('make'):
                    desc.append(vehicle['make'])
                if vehicle.get('model'):
                    desc.append(vehicle['model'])
                if vehicle.get('license_plate'):
                    desc.append(f"plate: {vehicle['license_plate']}")
                if desc:
                    summary.append(f"{vehicle['type'].upper()}: {' '.join(desc)}")
        
        # Return both full JSON and summary
        output = {
            "summary": "\n".join(summary) if summary else "No vehicles detected",
            "analysis": analysis
        }
        
        # Print JSON string that HA can parse
        print(json.dumps(output))
        return output

def main():
    parser = argparse.ArgumentParser(description='Detect objects and analyze vehicles in images')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--model', type=str, default='llama3.2-vision', help='Model to use')
    args = parser.parse_args()
    
    try:
        detect_objects(args.image_path, args.model)
    except Exception as e:
        print(json.dumps({"error": str(e), "summary": "Error analyzing image"}))
        sys.exit(1)

if __name__ == "__main__":
    main()