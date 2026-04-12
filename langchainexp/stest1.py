import os
import base64
import requests
from IPython.display import Image

engine_id = ""
api_host = ""
api_key = ""

image_description = "computers being tied together"
prompt = f"""
an illustration of {image_description}. in the
style of corporate memphis, white background, professional,
clean lines, warm postel colors
"""

response = requests.post(
    f"{api_host}/v1/generation/{engine_id}/text-to-image",
    headers={
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}"
    },
    json={
        "text_prompts": [
            {
                "text": prompt,
            }
        ],
        "cfg_scale": 7,
        "height": 1024,
        "width": 1024,
        "samples": 1,
        "steps": 30,
    },
)

if response.status_code != 200:
    raise Exception("error not 200")

data = response.json()

for i, image in enumerate(data["artifacts"]):
    filename = f"image-{i}.png"
    with open(filename, "wb") as f:
        f.write(base64.b64decode(image["base64"]))

    image_paths.append(filename)

Image(filename=image_paths[0])

