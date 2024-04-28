import script_nltk
text = open('egypt.txt', 'r', encoding='UTF-8').read()
print(script_nltk.generate_mcq_questions(text))
