import base64

def get_image_description(client, uploaded_file, detailed_prompt, user_prompt):
    # Encode the uploaded image in base64
    encoded_image = base64.b64encode(uploaded_file).decode('utf-8')

    # Create the GPT-4 API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": f"{detailed_prompt} Please respond specifically to the user's prompt in JSON format: {user_prompt}."
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{encoded_image}"}
                    },
                ],
            }
        ],
        max_tokens=1500,
        temperature=0.0,
        response_format={"type": "json_object"}  # Turn on JSON mode
    )

    # Extract and return the description
    return response.choices[0].message.content
