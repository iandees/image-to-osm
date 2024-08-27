import json
import os

import PIL
from flask import Flask, request, jsonify, render_template
import base64
import io
from PIL import Image
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionUserMessageParam, \
    ChatCompletionContentPartTextParam, ChatCompletionContentPartImageParam
from openai.types.chat.chat_completion_content_part_image_param import ImageURL

app = Flask(__name__)

# Set your OpenAI API key
client = OpenAI()

# Endpoint to show the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint to receive the image
@app.route('/upload', methods=['POST'])
def upload_image():
    data = request.get_json()
    image_data = data['image']

    # Decode the base64 image
    image_data = image_data.split(",")[1]
    try:
        image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    except PIL.UnidentifiedImageError as e:
        return jsonify({"error": f"Don't know how to read that image. {str(e)}"}), 400

    # Resize the image to a maximum dimension of 1024
    # A smaller image is fewer tokens paid to OpenAI and faster processing.
    max_dimension = 1024
    image.thumbnail((max_dimension, max_dimension))

    # Remove alpha channel if present
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    # Convert image to binary data for API request
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)

    # Convert image data to base64 text
    b64image_data = base64.b64encode(buffer.read()).decode('utf-8')

    # Send the image to OpenAI API (using an appropriate image-to-text endpoint)
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini-2024-07-18",
            messages=[
                ChatCompletionUserMessageParam(
                    role="user",
                    content=[
                        ChatCompletionContentPartTextParam(
                            type="text",
                            text="Suggest OpenStreetMap tags for the primary subject in the given image. If there are opening hours visible in the image, return them in OpenStreetMap 'opening_hours' format. Remember: 'opening_hours' format MUST use two letter English abbreviations for days of the week. Only output JSON. Do not make up OpenStreetMap tags. If you find something that should have OpenStreetMap tags, set status to 'ok'. If nothing has OpenStreetMap tags, set status to 'not_found'. Output a JSON object with keys 'status' and 'tags'. 'tags' should be a simple object with tag key and tag value.",
                        ),
                        ChatCompletionContentPartImageParam(
                            type="image_url",
                            image_url=ImageURL(
                                url=f"data:image/jpeg;base64,{b64image_data}",
                                detail="high",
                            ),
                        ),
                    ]
                ),
            ],
        )

        result = response.choices[0].message.content

        # Read the JSON from OpenAI
        result_json_str = result.replace("```json", "").replace("```", "").strip()
        json_data = json.loads(result_json_str)

        return jsonify(json_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
