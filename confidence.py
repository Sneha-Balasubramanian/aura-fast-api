def get_confidence_level(client, descriptions):
    # Create the GPT-4o API request
    response = client.chat.completions.create(
        model="gpt-4o-mini",
    
        messages=[
            {
                "role": "system",
                "content": 
                """
                    you are a helpful AI assistant specialized in analyzing all the key-value pairs given as input and evaluating the confidence score. I have done three API calls which will also give you the output three times, the result of which is appended in a list (descriptions).

                    Follow the following steps:

                    1. The inputs will be key-value pairs; I want you to analyze them completely.
                    2. Select the most accurate key-value pair for all the fields  among the entries provided.
                    3. For every field's selected key-value pair, evaluate the confidence score.
                    4. Give the output in the following format:
                    "key-value pair", "confidence score": x, where the x value ranges between (0 to 1).
                    5. Don't print the whole process , give only the final result 
                    6. Evaluation of the confidence score is based on the consistency of the outputs. Compare the outputs in all of the jsons, Higher the consistency, the higher the confidence score is.
                """
            },
            {
                "role": "user",
                "content": descriptions
            },
        ],
        max_tokens=1500,
        temperature=0.0
         
    )
    return response.choices[0].message.content
