import cv2
import logging
from roboflow import Roboflow
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import os

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("debug.log"),  # Log to a file
        logging.StreamHandler()           # Log to console
    ]
)

def main():
    try:
        # Initialize Roboflow with your API key
        api_key = "KUP9w62eUcD5PrrRMJsV"  # Replace with your actual API key
        logging.debug(f"Initializing Roboflow with API key: {api_key}")
        rf = Roboflow(api_key=api_key)

        # Load your project and model version
        project_name = "model_verification_project"
        version_number = 1
        logging.debug(f"Loading project '{project_name}' and version '{version_number}'")
        project = rf.workspace().project(project_name)
        model = project.version(version_number).model

        # Path to your input image
        image_path = r"C:\Users\MSI\Desktop\yumi verification\photo_7_2024-12-13_16-14-10.jpg"
        logging.debug(f"Reading image from path: {image_path}")
        image = cv2.imread(image_path)
        if image is None:
            logging.error(f"Failed to read image from {image_path}")
            return

        # Perform inference
        logging.debug("Performing inference on the image")
        prediction = model.predict(image_path, confidence=40, overlap=30).json()
        logging.debug(f"Inference result: {prediction}")

        # Open the image with PIL for text rendering
        pil_image = Image.open(image_path).convert("RGBA")  # Use RGBA for transparency
        draw = ImageDraw.Draw(pil_image)

        # Load custom font
        font_path = r"C:\Users\MSI\Desktop\yumi verification\STEVEHANDWRITING-REGULAR.ttf"
        try:
            assert os.path.exists(font_path), f"Font file does not exist: {font_path}"
        except Exception as e:
            logging.error(f"Could not load font at {font_path}. Error: {e}")
            return

        # Iterate over detected objects
        for obj in prediction['predictions']:
            # Extract bounding box coordinates
            logging.debug(f"Processing detection: {obj}")
            x1 = int(obj['x'] - obj['width'] / 2)
            y1 = int(obj['y'] - obj['height'] / 2)
            x2 = int(obj['x'] + obj['width'] / 2)
            y2 = int(obj['y'] + obj['height'] / 2)
            logging.debug(f"Bounding box coordinates: ({x1}, {y1}), ({x2}, {y2})")

            # Define the text to overlay
            text = "GCHES4"
            logging.debug(f"Overlaying text: '{text}'")

            # Dynamically adjust font size with a larger scaling factor
            box_width = x2 - x1
            box_height = y2 - y1
            font_size = int(min(box_width // len(text), box_height // 2) * 0.8)  # Increase scaling factor
            logging.debug(f"Calculated font size: {font_size}")

            try:
                font = ImageFont.truetype(font_path, size=font_size)
            except Exception as e:
                logging.error(f"Error loading font: {e}")
                font = ImageFont.load_default()

            # Manually calculate text position
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = x1 + (box_width - text_width) // 2
            text_y = y1 + (box_height - text_height) // 2

            # Create a new layer for the text with transparency
            text_layer = Image.new("RGBA", pil_image.size, (255, 255, 255, 0))
            text_draw = ImageDraw.Draw(text_layer)
            text_draw.text((text_x, text_y), text, fill=(0, 0, 0, 180), font=font)  # Slight opacity

            # Apply a slight blur for better blending
            blurred_text_layer = text_layer.filter(ImageFilter.GaussianBlur(radius=1.0))

            # Composite the blurred text layer onto the original image
            pil_image = Image.alpha_composite(pil_image, blurred_text_layer)

        # Save the result
        output_path = r"C:\Users\MSI\Desktop\yumi verification\output_image_blended_font.png"
        logging.debug(f"Saving the output image to: {output_path}")
        pil_image.convert("RGB").save(output_path)

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
