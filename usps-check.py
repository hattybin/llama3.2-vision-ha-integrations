import requests
import base64
from PIL import Image
import io
import logging
from typing import Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def encode_image_to_base64(image_path: str) -> str:
    """Convert an image file to base64 string"""
    try:
        with Image.open(image_path) as img:
            # Convert to RGB if image is in RGBA format
            if img.mode in ('RGBA', 'P'):
                img = img.convert('RGB')
            
            # Convert image to bytes
            buffered = io.BytesIO()
            img.save(buffered, format="JPEG", quality=95)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            return img_str
    except Exception as e:
        logger.error(f"Failed to encode image: {str(e)}")
        raise

def detect_usps_truck(image_path: str, model: str = "llama3.2-vision") -> int:
    """
    Detect if a USPS truck is present in an image
    
    Args:
        image_path: Path to the image file
        model: Name of the vision model to use
        
    Returns:
        1 if USPS truck detected, 0 if not
    """
    try:
        base64_image = encode_image_to_base64(image_path)
    except Exception as e:
        logger.error(f"Image encoding failed: {str(e)}")
        raise
    
    prompt = """
    Find a USPS mail truck in the supplied image. 
    Respond with true if a USPS mail truck is detected, false otherwise.
    """
    
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": [base64_image]
    }
    
    try:
        response = requests.post(url, json=payload, timeout=30)
        response.raise_for_status()
        return 1 if "true" in response.json().get('response', '').lower() else 0
            
    except requests.exceptions.RequestException as e:
        if "Connection refused" in str(e):
            logger.error("Could not connect to Ollama service")
            raise Exception("Could not connect to Ollama. Make sure it's running on localhost:11434")
        logger.error(f"Request to Ollama failed: {str(e)}")
        raise

def check_usps_in_snapshot(snapshot_url: str = None, model: str = "llama3.2-vision") -> int:
    """
    Check if a USPS truck is in the snapshot
    Args:
        snapshot_url: Optional URL of the snapshot image
        model: Name of the vision model to use
    Returns:
        1 if USPS truck detected, 0 if not
    """
    # Test image path
    test_image = '/home/mike/Pictures/ai-training/small_old_usps_delivery_truck/10495.jpg'
    
    # temp override for testing
    if snapshot_url:
        return detect_usps_truck(test_image, model)
    
    # Normal operation with URL
    temp_path = "snapshot.jpg"
    try:
        response = requests.get(snapshot_url, timeout=10)
        response.raise_for_status()
        
        with open(temp_path, "wb") as f:
            f.write(response.content)
        
        return detect_usps_truck(temp_path, model)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download snapshot: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"Failed to process snapshot: {str(e)}")
        raise

if __name__ == "__main__":
    snapshot_url = "http://hb-compute-01.hb.local:5100/api/frontright/latest.jpg"
    
    try:
        # For testing with the local image, call without URL:
        # usps_detected = check_usps_in_snapshot()
        
        # For normal operation with URL:
        usps_detected = check_usps_in_snapshot(snapshot_url)
        
        logger.info(f"USPS truck detected: {usps_detected}")
        print(usps_detected)  # Will print 1 or 0 for Home Assistant
    except Exception as e:
        logger.error(f"Script failed: {str(e)}")
        print(0)  # Return 0 on error for Home Assistant
        exit(1)