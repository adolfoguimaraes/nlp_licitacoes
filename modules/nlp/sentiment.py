import spacy
from spacytextblob.spacytextblob import SpacyTextBlob
class Sentiment_analyze():
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')
        self.nlp.add_pipe('spacytextblob')

    def sentiment_analyze_spacy(text):
        doc = nlp(text)
        sentiment = doc._.polarity
        if sentiment > 0:
            return 'positive'
        elif sentiment < 0:
            return 'negative'
        else:
            return 'neutral'
        

