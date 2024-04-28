import re
import pke
import string
import requests
import random
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from flashtext import KeywordProcessor
from pywsd.similarity import max_similarity
from pywsd.lesk import adapted_lesk
from nltk.corpus import wordnet as wn
from summarizer import Summarizer


class TextProcessor:
    def __init__(self, text):
        self.text = text
        self.summarized_text = ""
        self.keywords = []
        self.filtered_keys = []
        self.keyword_sentence_mapping = {}
        self.key_distractor_list = {}

    def summarize_text(self):
        model = Summarizer()
        result = model(self.text, min_length=60, max_length=500, ratio=0.4)
        self.summarized_text = ''.join(result)

    def get_nouns_multipartite(self):
        extractor = pke.unsupervised.MultipartiteRank()
        extractor.load_document(input=self.text)
        pos = {'PROPN'}
        stoplist = list(string.punctuation)
        stoplist += ['-lrb-', '-rrb-', '-lcb-', '-rcb-', '-lsb-', '-rsb-']
        stoplist += stopwords.words('english')
        extractor.candidate_selection(pos=pos)
        extractor.candidate_weighting(alpha=1.1, threshold=0.75, method='average')
        keyphrases = extractor.get_n_best(n=20)
        self.keywords = [key[0] for key in keyphrases]

    def filter_keywords(self):
        self.filtered_keys = [keyword for keyword in self.keywords if keyword.lower() in self.summarized_text.lower()]

    def tokenize_sentences(self):
        sentences = [sent_tokenize(self.summarized_text)]
        sentences = [y for x in sentences for y in x]
        sentences = [sentence.strip() for sentence in sentences if len(sentence) > 20]
        return sentences

    def get_sentences_for_keyword(self, sentences):
        keyword_processor = KeywordProcessor()
        for word in self.filtered_keys:
            self.keyword_sentence_mapping[word] = []
            keyword_processor.add_keyword(word)
        for sentence in sentences:
            keywords_found = keyword_processor.extract_keywords(sentence)
            for key in keywords_found:
                self.keyword_sentence_mapping[key].append(sentence)
        for key in self.keyword_sentence_mapping.keys():
            values = self.keyword_sentence_mapping[key]
            values = sorted(values, key=len, reverse=True)
            self.keyword_sentence_mapping[key] = values

    def get_distractors_wordnet(self, syn, word):
        distractors = []
        word = word.lower()
        orig_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        hypernym = syn.hypernyms()
        if len(hypernym) == 0:
            return distractors
        for item in hypernym[0].hyponyms():
            name = item.lemmas()[0].name()
            if name == orig_word:
                continue
            name = name.replace("_", " ")
            name = " ".join(w.capitalize() for w in name.split())
            if name is not None and name not in distractors:
                distractors.append(name)
        return distractors

    def get_wordsense(self, sent, word):
        word = word.lower()
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        synsets = wn.synsets(word, 'n')
        if synsets:
            wup = max_similarity(sent, word, 'wup', pos='n')
            adapted_lesk_output = adapted_lesk(sent, word, pos='n')
            lowest_index = min(synsets.index(wup), synsets.index(adapted_lesk_output))
            return synsets[lowest_index]
        else:
            return None

    def get_distractors_conceptnet(self, word):
        word = word.lower()
        original_word = word
        if len(word.split()) > 0:
            word = word.replace(" ", "_")
        distractor_list = []
        url = "http://api.conceptnet.io/query?node=/c/en/%s/n&rel=/r/PartOf&start=/c/en/%s&limit=5" % (word, word)
        obj = requests.get(url).json()
        for edge in obj['edges']:
            link = edge['end']['term']
            url2 = "http://api.conceptnet.io/query?node=%s&rel=/r/PartOf&end=%s&limit=10" % (link, link)
            obj2 = requests.get(url2).json()
            for edge in obj2['edges']:
                word2 = edge['start']['label']
                if word2 not in distractor_list and original_word.lower() not in word2.lower():
                    distractor_list.append(word2)
        return distractor_list

    def generate_distractors(self):
        for keyword in self.keyword_sentence_mapping:
            wordsense = self.get_wordsense(self.keyword_sentence_mapping[keyword][0], keyword)
            if wordsense:
                distractors = self.get_distractors_wordnet(wordsense, keyword)
                if len(distractors) == 0:
                    distractors = self.get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    self.key_distractor_list[keyword] = distractors
            else:
                distractors = self.get_distractors_conceptnet(keyword)
                if len(distractors) != 0:
                    self.key_distractor_list[keyword] = distractors

    def generate_fill_in_the_blank_questions(self):
        index = 1
        for each in self.key_distractor_list:
            sentence = self.keyword_sentence_mapping[each][0]
            pattern = re.compile(each, re.IGNORECASE)
            output = pattern.sub(" _______ ", sentence)
            print("%s)" % (index), output)
            choices = [each.capitalize()] + self.key_distractor_list[each]
            top4choices = choices[:4]
            random.shuffle(top4choices)
            optionchoices = ['a', 'b', 'c', 'd']
            for idx, choice in enumerate(top4choices):
                print("\t", optionchoices[idx], ")", " ", choice)
            index = index + 1

def process_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    text_processor = TextProcessor(text)
    text_processor.summarize_text()
    text_processor.get_nouns_multipartite()
    text_processor.filter_keywords()
    sentences = text_processor.tokenize_sentences()
    text_processor.get_sentences_for_keyword(sentences)
    text_processor.generate_distractors()
    text_processor.generate_fill_in_the_blank_questions()

