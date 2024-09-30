import cv2
import numpy as np
import streamlit as st
import openai
import json
import fitz 
from pdf2image import convert_from_bytes

from utils import get_image_description
from confidence import get_confidence_level

import os  # This is the missing import
from dotenv import load_dotenv  # For loading the .env file

# Load the environment variables from the .env file
load_dotenv()
# Access the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPENAI_API_KEY")


# Updated prompt for processing and analyzing textual data
detailed_prompt = (
    """
    "You are a highly skilled and detail-oriented assistant specialized in processing and analyzing textual data extracted from PDFs. "
    "Your primary task is to help the user extract specific details, numerical values, or other relevant information from the text content provided. "
    "The user may ask questions or make requests related to various aspects of the extracted text, including counting items, identifying categories, extracting personal details, and more.\n\n"
    "When responding to the user's requests:\n"
    "1. Understand the Context: Carefully read the user's request to ensure you fully understand what information they need and do not extract the information which is not in the prompt\n"
    "2. Text Extraction and Analysis: Analyze the provided text to extract accurate information, such as numerical values, names, categories, or other details.\n"
    "3. Clear and Concise Responses: Provide clear, concise, and accurate responses based on the text content. Include relevant details and context to ensure the user gets the exact information they need.\n"
    "4. Highlight Key Information: When listing items or details, organize them in a structured format (e.g., bullet points or numbered lists) for easy readability.\n"
    "5. Accuracy and Verification: Double-check your analysis to ensure the accuracy of the extracted information, especially when dealing with numerical data or critical details.\n"
    "6. Handle Complex Queries: If the user's query is complex or involves multiple steps, break down the response into logical parts and guide the user through each step.\n"
    "7. Your answers should strictly like an object with key-value pairs. There can be other data structure contain inside the object as well if necessary. Add descriptions if only user asks for it. If user asks for certain values, provide object like structure.\n\n"
    "8. Extract only the particular key-value pair the user is asking for."
    """
)

# Streamlit app layout
st.title("Image Analysis and Description Tool")

# Image Analysis Section
st.write("Upload an image or PDF and get a description using GPT-4.")

# Textbox for updating the prompt
user_prompt = st.text_input("Enter the prompt for image description", "What’s in this image?")

# Upload image or PDF button
uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

if uploaded_file is not None:
    st.write(f"Uploaded file type: {uploaded_file.type}")

    if uploaded_file.type == "application/pdf":
        # Handle PDF file
        st.write("Processing PDF...")
        
        # Open the PDF with PyMuPDF to get page count
        pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        page_count = pdf_document.page_count
        st.write(f"PDF has {page_count} pages.")

        if page_count == 1:
            # Single-page PDF: Convert the first page to an image and process it
            images = convert_from_bytes(uploaded_file.read(), fmt='jpeg')
            img = np.array(images[0])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB for OpenAI
            
            # Convert image to OpenAI-supported format
            _, img_encoded = cv2.imencode('.jpg', img_rgb)
            img_bytes = img_encoded.tobytes()

            # Get description for the image
            descriptions = []
            descriptions.append(get_image_description(openai, img_bytes, detailed_prompt, user_prompt))

        else:
            # Multi-page PDF: Extract textual data from all pages
            st.write("Extracting text from PDF...")
            all_text = ""
            
            for page_num in range(page_count):
                page = pdf_document.load_page(page_num)
                page_text = page.get_text("text")
                all_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

            # Process the extracted text with OpenAI
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=f"Extract information based on the prompt:\n{user_prompt}\nText:\n{all_text}",
                max_tokens=1000
            )
            descriptions = [response.choices[0].text.strip()]

    else:
        # Read the uploaded image
        img_before = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img_before is None:
            st.error("Error loading image. Please upload a valid image file.")
        else:
            # Display the original uploaded image
            st.image(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)

            # Convert to grayscale
            img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)

            # Edge detection
            img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)

            # Check if the image is readable
            if np.var(img_gray) < 500:  # Adjust the threshold as necessary
                st.error("The image is not readable. Please upload an image with better clarity.")
            else:
                st.write("Classifying...")
                descriptions = []

                # Call the function three times and append results to the list
                for _ in range(3):
                    description = get_image_description(openai, uploaded_file, detailed_prompt, user_prompt)
                    descriptions.append(description)

    description_json = {
        "descriptions": descriptions  # Store the list of descriptions
    }
    st.write(description_json)

    descriptions_str = json.dumps(descriptions)
    result = get_confidence_level(openai, descriptions_str)

    st.write(result)