import google.generativeai as genai
import streamlit as st
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel( "gemini-pro")



def generate_mcq_questions(text_input):

    template ="""
    Create multiple choice questions from a given text with options.
    ```
    {text_input}
    ```
    """

    formatted_template = template.format(text_input=text_input)
    response = model.generate_content(formatted_template)
    mcqs = response.text
    mcqs = mcqs.replace("**", "")
    # print(mcqs)
    return mcqs