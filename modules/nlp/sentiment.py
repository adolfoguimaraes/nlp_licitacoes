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
        
    def sentiment_analyze_maritalk(self, text):
        import maritalk

        model = maritalk.MariTalk(
            key="104147566582134244375$db094baa6042418d",
            model="sabia-2-small"  #modelos sabia-2-medium e sabia-2-small
        )

        main_prompt = """
        Você é um analista de texto altamente qualificado. Sua tarefa é realizar uma análise de sentimento do texto fornecido e apresentar o resultado em uma única palavra. O resultado deve ser uma das seguintes opções: "positivo", "negativo" ou "neutro".
        Siga estas etapas:

        Leia o texto fornecido cuidadosamente.
        Realize a análise de sentimento do texto.
        Apresente o resultado da análise de sentimento em uma única palavra: "positivo", "negativo" ou "neutro".

        É importante que a saída seja apenas a análise de sentimento em uma única palavra, sem qualquer outro texto ou explicação.
        Texto:"""
        prompt = main_prompt + str(text)
        response  = model.generate(prompt)
        answer = response["answer"]
        return answer
