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
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            
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
        detections_json (list/dict): List of detection dictionaries or full JSON response
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

    # Handle both full JSON object and direct detections array
    if isinstance(detections_json, dict) and 'detections' in detections_json:
        detections = detections_json['detections']
    elif isinstance(detections_json, dict):
        # Single detection object
        detections = [detections_json]
    else:
        # Assume it's a list of detections
        detections = detections_json if isinstance(detections_json, list) else [detections_json]

    # Filter detections if classes specified
    if filter_classes:
        filter_classes = [c.lower() for c in filter_classes]
        detections = [d for d in detections if d.get('class', '').lower() in filter_classes]

    for detection in detections:
        try:
            class_name = detection.get('class')
            confidence = detection.get('confidence', 0.0)
            bbox = detection.get('bounding_box')

            if not all([class_name, bbox]):
                print(f"Skipping invalid detection: {detection}")
                continue

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

        except Exception as e:
            print(f"Error drawing detection: {e}")
            continue

    # Generate output path if not provided
    if output_path is None:
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

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize object detections')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('detections', type=str, help='JSON string or path to JSON file containing detections')
    parser.add_argument('--output', type=str, help='Output path for annotated image (default: ~/Pictures/)')
    parser.add_argument('--classes', nargs='+', help='Optional list of classes to visualize')
    
    args = parser.parse_args()
    
    try:
        # Load detections from JSON string or file
        try:
            detections = json.loads(args.detections)
        except json.JSONDecodeError:
            # Try loading as file path
            with open(args.detections, 'r') as f:
                detections = json.load(f)
        
        if not detections:
            print("No valid detections found.")
            return
            
        # Draw detections
        output_path = draw_detections(
            args.image_path, 
            detections, 
            output_path=args.output,
            filter_classes=args.classes
        )
        
        # Print detection summary
        print("\nVisualized detections:")
        if isinstance(detections, dict) and 'detections' in detections:
            for det in detections['detections']:
                if not args.classes or det.get('class', '').lower() in [c.lower() for c in args.classes]:
                    print(f"- {det.get('class')}: {det.get('confidence', 0.0):.2f}")
        else:
            for det in detections if isinstance(detections, list) else [detections]:
                if not args.classes or det.get('class', '').lower() in [c.lower() for c in args.classes]:
                    print(f"- {det.get('class')}: {det.get('confidence', 0.0):.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()
        exit(1)

if __name__ == "__main__":
    main()