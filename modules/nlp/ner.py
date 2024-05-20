import spacy
class Ner():
    def __init__(self, text_='token'):
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
            key="",
            model="sabia-2-small"  #modelos sabia-2-medium e sabia-2-small
        )

        main_prompt = """
        You are an expert Named Entity Recognition (NER) system. Your task is to accept Text as input and extract named entities for the set of predefined entity labels. The output should be a python dictionary of named entities for each label.
        From the text input provided, extract named entities for each label in the following format:

        Person: <comma delimited list of strings>
        Organization: <comma delimited list of strings>
        Location: <comma delimited list of strings>
        Date: <comma delimited list of strings>
        Money: <comma delimited list of strings>
        Time: <comma delimited list of strings>

        Below are definitions of each label to help aid you in kinds of named entities to extract for each label.
        Assume these definitions are written by an expert and follow the closely.

        Person: Individual names of people.
        Organization: Names of companies, institutions, etc.
        Location: Geographical places.
        Date: Specific dates or date ranges.
        Time: Time references.
        Money: Monetary values.

        Here is the text that needs labeling. Make sure you to only return the python dictionary and nothing more:

        text: """
        prompt = main_prompt + str(text)
        response  = model.generate(prompt)
        answer = response["answer"]
        return answer
