import spacy
class Ner():
    def __init__(self):
        self.nlp = spacy.load('pt_core_news_lg')

    def extract_entities_tokens(self, tokens):
        text = ' '.join(tokens)
        doc = self.nlp(text)
        entities = [(entity, entity.label_) for entity in doc.ents]
        return entities
            

    def extract_entities_text(self, text):
        doc = self.nlp(text)
        entities = [(entity, entity.label_) for entity in doc.ents]
        return entities
        
    def extract_entities_maritalk(self, text):
        import maritalk

        model = maritalk.MariTalk(
            key="104147566582134244375$db094baa6042418d",
            model="sabia-2-small"  #modelos sabia-2-medium e sabia-2-small
        )

        main_prompt = """
        You are an advanced text analysis model skilled in Named Entity Recognition (NER). Your task is to identify and categorize named entities in a given text and present them in a Python dictionary format. Each category of entities should be a key in the dictionary, and the corresponding value should be a list of entities that fall under that category.

        Follow these steps:
        1. Read the provided text carefully.
        2. Identify the named entities and categorize them (e.g., Person, Organization, Location, Date, etc.).
        3. For each category, list the identified entities.
        4. Present your findings in a Python dictionary format.

        Here is the format of the dictionary you should use:

        ```python
        {
            "Person": ["Entity1", "Entity2", ...],
            "Organization": ["Entity1", "Entity2", ...],
            "Location": ["Entity1", "Entity2", ...],
            "Date": ["Entity1", "Entity2", ...],
            "Money": ["Entity1", "Entity2", ...],
            ...
        }
        ```

        Here is the text that needs NER. Make sure you to only return the python dictionary and nothing more:
        Text: """
        prompt = main_prompt + str(text)
        response  = model.generate(prompt)
        answer = response["answer"]
        return answer
