import asyncio
import aiohttp
from typing import Optional, Tuple
from PIL import Image
from io import BytesIO
import logging

fastapi_logger = logging.getLogger(__name__)


class AsyncImageLoader:
    def __init__(self, timeout: float = 30.0):
        """
        Initialize async image loader.
        
        Args:
            timeout: Maximum time to wait for image download in seconds
        """
        self.timeout = timeout
    
    async def load_image(self, url: str) -> Tuple[Optional[Image.Image], Optional[str]]:
        """
        Asynchronously download and load image from URL.
        
        Args:
            url: URL of the image to download
            
        Returns:
            Tuple of (PIL Image or None, error message or None)
        """
        try:
            # Validate URL format
            if not url or not isinstance(url, str):
                return None, "Invalid URL format: URL must be a non-empty string"
            
            # Trim whitespace
            url = url.strip()
            
            if not (url.startswith('http://') or url.startswith('https://')):
                return None, "URL must start with http:// or https://"
            
            # Basic URL length validation
            if len(url) > 2048:  # RFC compliant max URL length
                return None, "URL too long (max 2048 characters)"
            
            fastapi_logger.info(f"Downloading image from: {url}")
            
            # Create timeout for the entire download process
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
            }
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    # Check HTTP status
                    if response.status != 200:
                        error_msg = f"HTTP {response.status}: {response.reason}"
                        fastapi_logger.error(f"Failed to download image: {error_msg}")
                        return None, f"Failed to download image: {error_msg}"
                    
                    # Check content type
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(img_type in content_type for img_type in ['image/', 'application/octet-stream']):
                        return None, f"Invalid content type: {content_type}. Expected image."
                    
                    # Read image data
                    image_data = await response.read()
                    
                    if not image_data:
                        return None, "Empty image data received"
                    
                    # Try to open image with PIL
                    try:
                        image = Image.open(BytesIO(image_data))
                        
                        # Convert to RGB if needed
                        if image.mode not in ('RGB', 'RGBA'):
                            image = image.convert('RGB')
                        elif image.mode == 'RGBA':
                            # Create white background for RGBA images
                            rgb_image = Image.new('RGB', image.size, (255, 255, 255))
                            rgb_image.paste(image, mask=image.split()[-1])  # Use alpha channel as mask
                            image = rgb_image
                        
                        # Validate image dimensions
                        if image.size[0] == 0 or image.size[1] == 0:
                            return None, "Invalid image dimensions"
                        
                        fastapi_logger.info(f"Successfully loaded image: {image.size[0]}x{image.size[1]} pixels")
                        return image, None
                        
                    except Exception as e:
                        fastapi_logger.error(f"Failed to process image data: {str(e)}")
                        return None, f"Failed to process image: {str(e)}"
        
        except asyncio.TimeoutError:
            error_msg = f"Download timeout after {self.timeout} seconds"
            fastapi_logger.error(error_msg)
            return None, error_msg
            
        except aiohttp.ClientError as e:
            error_msg = f"Network error: {str(e)}"
            fastapi_logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"Unexpected error: {str(e)}"
            fastapi_logger.error(error_msg)
            return None, error_msg