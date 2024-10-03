import base64

def get_image_description(client, uploaded_file, system_prompt, user_prompt, is_image_type=False):
    # Ensure that uploaded_file is always treated as a list
    if not isinstance(uploaded_file, list):
        uploaded_file = [uploaded_file]  # Convert single item to a list

    # Encode the content based on the type of the file
    if is_image_type:
        encoded_content_list = [get_image_content(base64.b64encode(i).decode('utf-8')) for i in uploaded_file ]# Handle image bytes/memoryview
    else:
        encoded_content_list = [get_text_content(i) for i in uploaded_file ] # Handle text strings

    # Create the GPT-4 API request
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt
                    },
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Please respond specifically to the user's prompt in JSON format: {user_prompt}."
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    *encoded_content_list
                ],
            }
        ],
        max_tokens=1500,
        temperature=0.3,
        response_format={"type": "json_object"}  # Turn on JSON mode
    )

    # Extract and return the description
    return response.choices[0].message.content


def get_image_content(image_content):
    return {
        "type": "image_url",
        "image_url": {"url": f"data:image/png;base64,{image_content}"}
    }

def get_text_content(text_content):
    return {
        "type": "text",
        "text": text_content
    }