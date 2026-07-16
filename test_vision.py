"""
Quick test of gemma4:e4b's vision capability — the same model your resume
chatbot already runs for text, tried here on an image instead.

Usage:
    python3 test_vision.py path/to/screenshot.png

Try it on a screenshot of a LinkedIn job posting (or anything else) —
since this reads an image you took yourself in your own browser, it's a
completely different thing from scraping LinkedIn's site programmatically,
which is what we backed away from earlier today.
"""
import sys
import os
import ollama

MODEL = "llama3.2-vision:11b"  # same vision+tools model your chatbot already uses


def extract_text_from_image(image_path: str) -> str:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"No file found at {image_path!r} (cwd: {os.getcwd()})")

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    print(f"Read {len(image_bytes)} bytes from {image_path}")
    if len(image_bytes) == 0:
        raise ValueError(f"{image_path!r} exists but is empty — re-save the screenshot.")

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the full job posting text from this screenshot, as "
                    "close to verbatim as possible. Include the job title, company, "
                    "requirements, and any other details visible. If it's not a job "
                    "posting, just describe what you actually see."
                ),
                "images": [image_bytes],  # raw bytes, not a path — more reliably supported
            }
        ],
    )
    return response["message"]["content"]


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 test_vision.py path/to/screenshot.png")
        sys.exit(1)

    image_path = sys.argv[1]
    print(f"Sending {image_path} to {MODEL}...\n")
    result = extract_text_from_image(image_path)
    print("--- Extracted ---")
    print(result)
