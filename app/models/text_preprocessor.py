import re
import pymorphy2

from nltk.corpus import stopwords
from razdel import tokenize

class TextPreprocessor:
    
    def __init__(self):
        self.stopwords = stopwords.words('russian')
        self.morph = pymorphy2.MorphAnalyzer()
        
    def clean_text(self, text):
        if not isinstance(text, str):
            text = str(text)
    
        text = text.lower()
        text = text.strip('\n').strip('\r').strip('\t')
        text = re.sub("-\s\r\n\|-\s\r\n|\r\n", '', str(text))

        text = re.sub("[0-9]|[-—.,:;_%©«»?*!@#№$^•·&()]|[+=]|[[]|[]]|[/]|", '', text)
        text = re.sub(r"\r\n\t|\n|\\s|\r\t|\\n", ' ', text)
        text = re.sub(r'[\xad]|[\s+]', ' ', text.strip())
    
        return text
    
    def lemmatization(self, text):
        if not isinstance(text, str):
            text = str(text)
    
        tokens = list(tokenize(text))
        words = [_.text for _ in tokens]

        words_lem = []
        for w in words:
            if w[0] == '-': 
                w = w[1:]
            if len(w)>1:
                words_lem.append(self.morph.parse(w)[0].normal_form)
    
        words_lem_without_stopwords=[i for i in words_lem if not i in self.stopwords]
    
        return ' '.join(words_lem_without_stopwords)