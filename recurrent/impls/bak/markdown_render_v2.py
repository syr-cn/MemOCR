import markdown
import imgkit
import io
from PIL import Image, ImageChops

def markdown_to_image(markdown_text):
    """
    Converts markdown to PIL, removing default browser margins 
    and cropping extra whitespace.
    """
    markdown_text = markdown_text.strip().strip('`')
    html_content = markdown.markdown(markdown_text, extensions=['extra'])
    
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                background-color: white;
                margin: 0; 
                padding: 0;
            }}
            .content {{
                /* Add a small internal padding so text doesn't touch the edge */
                padding: 0px; 
                display: inline-block;
            }}
        </style>
    </head>
    <body>
        <div class="content">
            {html_content}
        </div>
    </body>
    </html>
    """

    # Config options to reduce default window width
    options = {
        # 'quiet': '',
        # Setting a smaller width sometimes helps, but PIL crop (step 3) is more reliable
        # 'width': 600 
    }

    img_bytes = imgkit.from_string(full_html, False, options=options)
    pil_image = Image.open(io.BytesIO(img_bytes))

    # 2. PIL CROPPING FIX:
    # This removes the massive white space on the right/bottom.
    assert pil_image is not None, f'pil_image is None for markdown_text: {markdown_text}'

    pil_image = trim_whitespace(pil_image)

    return pil_image

def trim_whitespace(im):
    """
    Automatically crops the image to remove the background color borders.
    """
    # Create a background image with the same color as the top-left pixel
    bg = Image.new(im.mode, im.size, im.getpixel((0,0)))
    
    # Calculate the difference between the original and the background
    diff = ImageChops.difference(im, bg)
    
    # Get the bounding box of the non-zero difference (the actual content)
    bbox = diff.getbbox()
    if not bbox:
        return im
    bbox = (max(bbox[0] - 30, 0), max(bbox[1] - 30, 0), min(bbox[2] + 30, im.width), min(bbox[3] + 30, im.height))
    
    if bbox:
        return im.crop(bbox)
    return im

if __name__ == "__main__":
    md_text = """
# Clean Image
This image has **no extra margins**.

## Features
- Tightly cropped
- No browser borders
    """

    image = markdown_to_pil(md_text)
    
    output_filename = "output_cropped.png"
    image.save(output_filename)
    
    print(f"Saved to {output_filename}")
    print(f"Final Size: {image.size}")