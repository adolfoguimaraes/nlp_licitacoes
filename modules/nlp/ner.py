import spacy
class Ner:
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')
    
    def extract_entities(self, tokens):
        text = ' '.join(tokens)
        doc = self.nlp(text)
        entities = [(entity, entity.label_) for entity in doc.ents]
        return entities
