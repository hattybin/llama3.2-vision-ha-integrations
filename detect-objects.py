#!/usr/bin/env python3

import requests
import json
import base64
from PIL import Image, ImageDraw, ImageFont
import io
import argparse
import os
import random
import tempfile
from pathlib import Path

def encode_image_to_base64(image_path):
    """Convert an image file to base64 string"""
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

def draw_detections(image_path, detections_json, output_path=None, filter_classes=None):
    """
    Draw bounding boxes and labels on the image based on JSON detections.
    
    Args:
        image_path (str): Path to the input image
        detections_json (list): List of detection dictionaries
        output_path (str): Optional path for output image
        filter_classes (list): Optional list of classes to include
    """
    # Convert to Path objects
    image_path = Path(image_path)
    
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()

    # Generate random colors for each unique class
    colors = {}

    # Get image dimensions
    width, height = image.size

    # Filter detections if classes specified
    if filter_classes:
        filter_classes = [c.lower() for c in filter_classes]
        detections_json = [d for d in detections_json if d['class'].lower() in filter_classes]

    for detection in detections_json:
        class_name = detection['class']
        confidence = detection['confidence']
        bbox = detection['bounding_box']

        # Generate a random color for this class if we haven't yet
        if class_name not in colors:
            colors[class_name] = (
                random.randint(0, 255),
                random.randint(0, 255),
                random.randint(0, 255)
            )

        # Convert normalized coordinates to pixel coordinates
        x1 = int(bbox[0] * width)
        y1 = int(bbox[1] * height)
        x2 = int(bbox[2] * width)
        y2 = int(bbox[3] * height)

        # Draw bounding box
        box_color = colors[class_name]
        draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)

        # Prepare label text
        label_text = f"{class_name}: {confidence:.2f}"
        
        # Draw label background
        text_bbox = draw.textbbox((x1, y1-25), label_text, font=font)
        draw.rectangle(text_bbox, fill=box_color)
        
        # Draw label text
        draw.text((x1, y1-25), label_text, fill='white', font=font)

    # Generate output path if not provided
    if output_path is None:
        # Use pathlib for reliable path manipulation
        output_filename = f"{image_path.stem}-detect{image_path.suffix}"
        output_path = Path.home() / "Pictures" / output_filename
    else:
        output_path = Path(output_path)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save the annotated image
    image.save(str(output_path))
    print(f"Annotated image saved to: {output_path}")
    return str(output_path)

def detect_objects(image_path, model="llama3.2-vision"):
    """Detect objects in an image using Ollama's vision model with enhanced vehicle detection"""
    print("Starting detection...")
    
    # Encode image
    print("Encoding image...")
    base64_image = encode_image_to_base64(image_path)
    
    # Prepare the prompt - Making it VERY explicit about vehicle details
    prompt = """
    Analyze the image and provide ONLY a JSON response. Focus on precise bounding box coordinates.

    Required format:
    {
        "detections": [
            {
                "class": "<object type>",
                "confidence": <number between 0 and 1>,
                "bounding_box": [x1, y1, x2, y2]  // EXACT normalized coordinates
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

    CRITICAL - Bounding Box Rules:
    1. x1,y1 is the top-left corner
    2. x2,y2 is the bottom-right corner
    3. Use normalized coordinates (0.00 to 1.00)
    4. Be extremely precise in measuring object boundaries
    5. Include ALL of the object, but minimize empty space
    6. For vehicles, ensure the box includes the entire vehicle, side mirrors and bumpers as well as a small margin around the vehicle

    Example of accurate car coordinates: [0.52, 0.29, 0.95, 0.72]

       
    For vehicles, ensure the detection section includes the specific vehicle class and the details of the normal detection, class, confidence, and boounding_box  . provide as much detail as you can confidently determine. If any field cannot be determined, use null.
    For license plates, only include if text is clearly readable.
    """
    
    # Make the request
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        print("Making request to Ollama...")
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # Get the raw response
        print("Got response from Ollama")
        result = response.json()['response']
        
        print("\nRaw response from model:")
        print(result)
        
        # Try to parse JSON response
        try:
            # Find JSON in the response
            start_idx = result.find('{')
            end_idx = result.rfind('}') + 1
            
            if start_idx != -1 and end_idx != -1:
                json_str = result[start_idx:end_idx]
                print("\nExtracted JSON string:")
                print(json_str)
                
                analysis = json.loads(json_str)
                
                # Save the full analysis to a file that HA can read
                with open('/tmp/llama_vision_full_analysis.json', 'w') as f:
                    json.dump(analysis, f, indent=2)
                
                # Save a formatted summary for the notification
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
                
                # Save the summary for HA notifications
                with open('/tmp/llama_vision_last_analysis.txt', 'w') as f:
                    f.write("\n".join(summary) if summary else "No vehicles detected")
                
                return analysis
                
        except json.JSONDecodeError as e:
            print(f"\nJSON parsing error: {str(e)}")
            return None
            
    except requests.exceptions.RequestException as e:
        if "Connection refused" in str(e):
            raise Exception("Could not connect to Ollama. Make sure it's running on localhost:11434")
        raise Exception(f"Error making request to Ollama: {str(e)}")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")
        raise Exception(f"Error processing response: {str(e)}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Detect and visualize objects in images using Llama vision model')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('--output', type=str, help='Output path for annotated image (default: ~/Pictures/)')
    parser.add_argument('--model', type=str, default='llama3.2-vision',
                      help='Model to use (default: llama3.2-vision)')
    parser.add_argument('--classes', nargs='+', help='Optional list of classes to detect (e.g., --classes person car dog)')
    parser.add_argument('--debug', action='store_true', help='Enable debug output')
    
    args = parser.parse_args()
    
    try:
        # Run detection
        print("Analyzing image...")
        detections = detect_objects(args.image_path, args.model)
        
        if not detections:
            print("No objects detected or could not parse detections.")
            return
            
        if args.debug:
            print("\nParsed detections:")
            print(json.dumps(detections, indent=2))
            
        # Draw detections
        output_path = draw_detections(
            args.image_path, 
            detections, 
            output_path=args.output,
            filter_classes=args.classes
        )
        
        # Print detection summary
        print("\nDetected objects:")
        for det in detections:
            if args.classes and det['class'].lower() not in [c.lower() for c in args.classes]:
                continue
            print(f"- {det['class']}: {det['confidence']:.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()