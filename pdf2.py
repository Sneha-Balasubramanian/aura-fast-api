import cv2
import numpy as np
import streamlit as st
import openai
from openai import OpenAI
import json
import fitz 
from pdf2image import convert_from_bytes

from utils import get_image_description
from confidence import get_confidence_level
import os  # For accessing environment variables
from dotenv import load_dotenv  # For loading the .env file

# Load the environment variables from the .env file
load_dotenv()
# Access the OpenAI API key from the environment variables
openai.api_key = os.getenv("OPEN_API_KEY")

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
st.title("PDF/Image Analysis Tool")

# Image Analysis Section
st.write("Upload an image or PDF and get a description using GPT-4o.")

# Textbox for updating the prompt
user_prompt = st.text_input("Enter the prompt for image description", "What’s in the document?")

# Upload image or PDF button
uploaded_file = st.file_uploader("Choose an image or PDF...", type=["jpg", "jpeg", "png", "pdf"])

# Function to extract text from PDF using PyMuPDF (fitz)
def extract_pdf_text(uploaded_file):
    """Extract text from a PDF file using PyMuPDF and return it as a list."""
    text_descriptions = []  # Store text descriptions for all pages
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text = page.get_text()
            text_descriptions.append(text)  # Collect text for each page
    return text_descriptions

# Function to check if the PDF contains images using PyMuPDF and pdf2image
def has_images_in_pdf(uploaded_file):
    """
    Check if a PDF contains images and verify if they are in typical image modes (RGB/RGBA).
    """
    # Step 1: Check if the PDF contains images using PyMuPDF (fitz)
    pdf_document = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    has_images = False

    for page in pdf_document:
        print('page metadata:::', page)
        images = page.get_images(full=True)
        if images:
            has_images = True
            break
    
    uploaded_file.seek(0)  # Reset the file pointer

    if not has_images:
        return False  # No images found

    # Step 2: If images were found, check their modes using pdf2image
    pdf_images = convert_from_bytes(uploaded_file.getbuffer())
    
    for image in pdf_images:
        if image.mode in ["RGB", "RGBA"]:  # Typical image modes
            return True  # Images found in acceptable modes
    
    return False  # No images with acceptable modes

# Function to process and convert PDF pages with images into image descriptions
def process_pdf_images(uploaded_file):
    """Convert PDF pages to images and get descriptions using GPT."""
    images = convert_from_bytes(uploaded_file.getbuffer(), fmt='jpeg')
    image_bytes_list = []
    
    for img in images:
        img = np.array(img)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image to RGB
        
        # Convert image to OpenAI-supported format
        _, img_encoded = cv2.imencode('.jpg', img_rgb)
        img_bytes = img_encoded.tobytes()
        image_bytes_list.append(img_bytes)
        
        
    return image_bytes_list

# Main processing logic
if uploaded_file is not None:
    if uploaded_file.type == "application/pdf":
        st.write("Processing PDF...")

        # Check if the PDF contains images in acceptable modes
        if has_images_in_pdf(uploaded_file):
            st.write("Images found in the PDF. Processing images...")

            # Convert PDF pages with images to image format and get descriptions
            image_bytes_array = process_pdf_images(uploaded_file)

            list_all_descriptions = []
            for _ in range(3):
                all_descriptions = get_image_description(openai, image_bytes_array, detailed_prompt=detailed_prompt, user_prompt=user_prompt)
                print('buffer all dscriptionssssssssssssssssssss',all_descriptions[0])    #remove
                list_all_descriptions.append(all_descriptions)     

            # Display all image descriptions
            st.write("Image descriptions from the PDF:")
            st.write(list_all_descriptions)

            # Get confidence level for the descriptions
            list_all_descriptions_str = json.dumps(list_all_descriptions)
            confidence_result = get_confidence_level(openai, list_all_descriptions_str)
            st.write(confidence_result)


        else:
            st.write("No images found in the PDF or no valid images. Extracting text...")

            # Extract text from PDF and collect all pages into a list
            all_text_descriptions = extract_pdf_text(uploaded_file)

            if all_text_descriptions:  # Check if there are any extracted texts
                st.write("Extracted text from the PDF:")
                st.write(all_text_descriptions)

                # Convert the list of text descriptions to JSON for confidence analysis
                text_descriptions_str = json.dumps(all_text_descriptions)
                confidence_result = get_confidence_level(openai, text_descriptions_str)
                st.write(confidence_result)
            else:
                st.write("No text found in the PDF.")
    else:
        # Handle image file (JPEG/PNG)
        img_before = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR)

        if img_before is None:
            st.error("Error loading image. Please upload a valid image file.")
        else:
            st.image(cv2.cvtColor(img_before, cv2.COLOR_BGR2RGB), caption='Uploaded Image', use_column_width=True)
            img_gray = cv2.cvtColor(img_before, cv2.COLOR_BGR2GRAY)
            img_edges = cv2.Canny(img_gray, 100, 100, apertureSize=3)

            if np.var(img_gray) < 500:  # Adjust the threshold as necessary
                st.error("The image is not readable. Please upload a clearer image.")
            else:
                st.write("Classifying...")
                descriptions = []

                for _ in range(3):
                    description = get_image_description(openai, uploaded_file.getbuffer(), detailed_prompt, user_prompt)
                    descriptions.append(description)

                st.write(descriptions)

                descriptions_str = json.dumps(descriptions)
                result = get_confidence_level(openai, descriptions_str)
                st.write(result)
