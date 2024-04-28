import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return tokens

def understand_text(text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    concepts = [chunk.text for chunk in doc.noun_chunks]
    relationships = []
    for token in doc:
        if token.dep_ in ('amod', 'prep'):
            relationships.append((token.head.text, token.text))

    understanding = {
        'entities': entities,
        'concepts': concepts,
        'relationships': relationships
    }

    return understanding

def generate_questions(text_understanding):
    generated_questions = []
    for concept in text_understanding['concepts']:
        question = f"What is {concept}?"
        generated_questions.append(question)

    return generated_questions

def generate_mcq_questions(text):
    preprocessed_text = preprocess_text(text)
    text_understanding = understand_text(text)
    generated_questions = generate_questions(understand_text)
    print(generated_questions)

