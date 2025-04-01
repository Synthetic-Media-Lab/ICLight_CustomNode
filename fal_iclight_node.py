import os
import io
import base64
from PIL import Image
import numpy as np
import torch
import random
import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FALICLightNode")

# Try importing fal_client - print helpful error if missing
try:
    import fal_client
    logger.info("Successfully imported fal_client module")
except ImportError:
    logger.error("Failed to import fal_client. Please install it using: pip install fal-client")
    raise

class FALICLightNode:
    """
    A ComfyUI node that connects to FAL.ai's ICLight v2 model using the official fal_client.
    
    This node sends images to the ICLight v2 model for relighting based on text prompts.
    """
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "perfume bottle in a volcano surrounded by lava."
                }),
                "negative_prompt": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
                "num_inference_steps": ("INT", {
                    "default": 28,
                    "min": 10,
                    "max": 100,
                    "step": 1
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 5.0,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.1
                }),
                "seed": ("INT", {
                    "default": -1,
                    "min": -1,
                    "max": 2147483647,
                    "step": 1
                }),
                "initial_latent": (["None", "Left", "Right", "Top", "Bottom"],),
                "output_format": (["jpeg", "png"],),
            },
            "optional": {
                "mask": ("IMAGE",),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_image"
    CATEGORY = "external/generation"
    
    def process_image(self, image, prompt, negative_prompt="", num_inference_steps=28, 
                     guidance_scale=5.0, seed=-1, initial_latent="None", 
                     output_format="jpeg", mask=None):
        """
        Process an image using FAL.ai's ICLight v2 model via fal_client
        
        Args:
            image: Input image tensor [B, H, W, C] in range 0-1
            prompt: Text prompt describing desired lighting/scene
            negative_prompt: Text to avoid in generation
            num_inference_steps: Number of diffusion steps
            guidance_scale: How closely to follow the prompt
            seed: Random seed for reproducibility (-1 for random)
            initial_latent: Lighting condition
            output_format: Output image format (jpeg/png)
            mask: Optional mask tensor [B, H, W, C] in range 0-1
            
        Returns:
            Tuple containing the processed image tensor
        """
        try:
            # Get the API key for fal_client from environment variable
            fal_key = os.environ.get("FAL_KEY")
            if not fal_key:
                logger.error("FAL_KEY environment variable not set. Please configure a Modal Secret.")
                raise ValueError("FAL_KEY environment variable not set!")
            # The fal_client typically uses FAL_KEY directly, so we don't need to set it again if the secret is mounted correctly.
            logger.info("Using API key from FAL_KEY environment variable")
            
            # Generate a random seed if not specified
            if seed == -1:
                seed = random.randint(0, 2147483647)
                logger.info(f"Using random seed: {seed}")
            
            # Take the first image if it's a batch
            if len(image.shape) == 4:
                img_data = image[0]
            else:
                img_data = image
                
            # Convert the image tensor to a PIL image
            if isinstance(img_data, torch.Tensor):
                img_data = img_data.cpu().numpy()
            
            img_array = (img_data * 255).astype(np.uint8)
            pil_image = Image.fromarray(img_array)
            
            # Convert PIL image to base64
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            encoded_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            image_url = f"data:image/png;base64,{encoded_image}"
            
            # Process mask if provided
            mask_url = None
            if mask is not None:
                if len(mask.shape) == 4:
                    mask_data = mask[0]
                else:
                    mask_data = mask
                
                if isinstance(mask_data, torch.Tensor):
                    mask_data = mask_data.cpu().numpy()
                
                mask_array = (mask_data * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_array)
                
                # Convert mask to base64
                mask_buffer = io.BytesIO()
                mask_pil.save(mask_buffer, format="PNG")
                encoded_mask = base64.b64encode(mask_buffer.getvalue()).decode("utf-8")
                mask_url = f"data:image/png;base64,{encoded_mask}"
            
            logger.info(f"Sending request to FAL.ai ICLight v2 API with prompt: '{prompt}'")
            
            # Prepare input for fal_client
            input_data = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image_url": image_url,
                "num_inference_steps": num_inference_steps,
                "guidance_scale": guidance_scale,
                "seed": seed,
                "output_format": output_format
            }
            
            # Add mask_url if mask was provided
            if mask_url:
                input_data["mask_image_url"] = mask_url
                logger.info("Including mask in the request")
            
            # Add initial_latent only if it's not "None"
            if initial_latent != "None":
                input_data["initial_latent"] = initial_latent
            
            # Call the FAL.ai API using fal_client
            endpoint = "fal-ai/iclight-v2"
            
            # Submit the request
            logger.info("Submitting request to FAL.ai...")
            request_handler = fal_client.submit(endpoint, arguments=input_data)
            
            # Get the response
            logger.info("Waiting for response...")
            result = request_handler.get()
            
            # Extract the image URL from the result
            if "images" not in result or not result["images"]:
                raise ValueError("No images returned in the response")
                
            image_url = result["images"][0]["url"]
            
            # Download the image
            logger.info(f"Downloading generated image...")
            response = requests.get(image_url)
            response.raise_for_status()
            image_data = response.content
            
            # Convert the image to PIL and then to tensor
            img = Image.open(io.BytesIO(image_data))
            
            # Handle different image modes
            if img.mode != "RGB":
                img = img.convert("RGB")
                
            # Convert PIL image to numpy array (0-1 range)
            img_array = np.array(img).astype(np.float32) / 255.0
            
            # Add batch dimension to match ComfyUI's expected format [B, H, W, C]
            img_tensor = torch.from_numpy(img_array).unsqueeze(0)
            
            logger.info(f"Successfully generated image with FAL.ai ICLight v2")
            return (img_tensor,)
            
        except Exception as e:
            logger.error(f"Error processing image with FAL.ai ICLight: {str(e)}", exc_info=True)
            raise RuntimeError(f"FAL.ai ICLight processing failed: {str(e)}")

# Register the node with ComfyUI
NODE_CLASS_MAPPINGS = {
    "FALICLightNode": FALICLightNode
}

# Add display name mapping
NODE_DISPLAY_NAME_MAPPINGS = {
    "FALICLightNode": "FAL.ai ICLight v2"
}