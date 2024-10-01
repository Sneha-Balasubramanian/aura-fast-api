import base64

def get_image_description(client, uploaded_file, detailed_prompt, user_prompt):
    # Encode the uploaded image in base64
    
    if isinstance(uploaded_file, list): 
        encoded_image_list = [get_image_content(base64.b64encode(i).decode('utf-8')) for i in uploaded_file]
    else: 
        encoded_image_list = [get_image_content(base64.b64encode(uploaded_file).decode('utf-8'))]

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
                    *encoded_image_list
                ],
            }
        ],
        max_tokens=1500,
        temperature=0.0,
        response_format={"type": "json_object"}  # Turn on JSON mode
    )

    # Extract and return the description
    return response.choices[0].message.content


def get_image_content(image_content):
    return {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{image_content}"}
            }