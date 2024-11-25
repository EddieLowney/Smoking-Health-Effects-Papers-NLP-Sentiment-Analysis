"""
File: myopenai.py
Description: A simple wrapper for calling OpenAI APIs
"""

import openai
import dotenv
import os
import base64

class MyOpenAPI:

    def __init__(self):
        dotenv.load_dotenv()
        self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @staticmethod
    def read_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def ask(self, model="chatgpt-4o-latest", prompt=None, image=None):
        """ Ask chatgpt a question """

        if prompt is not None:
            if image is None:
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ]
                )
            else:  # if an image is provided
                completion = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/png;base64,{MyOpenAPI.read_image(image)}"},
                                },
                            ]
                        }])

            return completion.choices[0].message.content
        else:
            return None






