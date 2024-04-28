import streamlit as st
import textract
import docx
import PyPDF2
import script_nltk
import script_LLM
from PyPDF2 import PdfFileReader

def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
        return text

def extract_text_from_doc(file_path):
    doc = docx.Document(file_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# def extract_text_from_text(file_path):
#     with open(file_path, 'r', encoding='utf-8') as file:
#         return file.read()

def extract_text_from_text(uploaded_file):
    text = uploaded_file.read().decode('utf-8')
    return text

def main():
    st.title("MCQ Generator application")
    uploaded_file = st.file_uploader("Upload Document", type=['pdf', 'docx', 'txt'])

    if uploaded_file is not None:
        file_type = uploaded_file.type.split('/')[-1]
        if file_type == 'pdf':
            text = extract_text_from_pdf(uploaded_file)
        elif file_type == 'docx':
            text = extract_text_from_doc(uploaded_file)
        elif file_type == 'plain' or 'txt':
            text = extract_text_from_text(uploaded_file)
        else:
            st.error("Unsupported file format")

        # st.write(text)

        option = st.selectbox("Choose an option", ["HOME","NLTK1","LLM"])

        if option == "NLTK1":
            # Perform analysis or processing for Option 1
            mcqs = script_nltk.process_text_file(text)
            st.write("You selected Option 1")
        elif option == "LLM":
            # Perform analysis or processing for Option 3
            mcqs = script_LLM.generate_mcq_questions(text)
            st.write(mcqs)


if __name__ == "__main__":
    main()
