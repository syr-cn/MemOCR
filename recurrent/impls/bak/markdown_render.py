import markdown
from weasyprint import HTML
from io import BytesIO
from pdf2image import convert_from_bytes


def markdown_to_image(md_text: str) -> bytes:
    # --- 1. Markdown → HTML ---
    md_text = md_text.strip().strip('`')
    html_body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables"]
    )

    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                padding: 12px;
                line-height: 1.6;
            }}
            pre {{
                background: #f4f4f4;
                padding: 12px;
                border-radius: 6px;
            }}
            table {{
                border-collapse: collapse;
            }}
            td, th {{
                border: 1px solid #555;
                padding: 6px 10px;
            }}
        </style>
    </head>
    <body>
        {html_body}
    </body>
    </html>
    """

    pdf_bytes = HTML(string=full_html).write_pdf()
    images = convert_from_bytes(pdf_bytes, dpi=300)

    # Concatenate all images vertically and return as PNG bytes
    from PIL import Image

    widths, heights = zip(*(img.size for img in images))
    total_height = sum(heights)
    max_width = max(widths)

    combined_image = Image.new('RGB', (max_width, total_height), (255, 255, 255))

    y_offset = 0
    for img in images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    return combined_image

if __name__ == "__main__":
    md_text = """
    # Hello from Markdown

    This image was generated **entirely in memory**.

    ## Code Example
    ```python
    print("Hello Image World!")
    ```

    ## Table

    | A   | B |
    | --- | - |
    | 1   | 2 |
    """

    image_object = markdown_to_image(md_text)

    # Save if you want (optional)
    image_object.save("markdown_render.png")