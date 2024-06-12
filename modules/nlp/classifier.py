class Classifier():
    def __init__(self):
        import maritalk

        self.model = maritalk.MariTalk(
            key="104147566582134244375$c4ce5b5fbee6978e",
            model="sabia-2-small"  #sabia-2-medium // sabia-2-small
        )

    def classify(self, text):
        
        main_prompt ="""Você é um assistente de classificação especializado em diários oficiais. Dado o texto a seguir de uma página de um diário oficial, classifique-o em uma das seguintes categorias:

        Licitações
        Contratos
        Leis
        Decretos
        Nomeações e Exonerações
        Atos Administrativos
        Publicações Judiciais
        Publicações de Empresas
        Editais Diversos
        Avisos e Comunicados de Interesse Público
        Publicações de Organismos Internacionais
        Matérias de Transparência

        Siga estas etapas:
        Leia o texto fornecido cuidadosamente.
        Classifique o texto em uma das categorias listadas.

        Apresente apenas a categoria de classificação como saída, sem qualquer outro texto ou explicação.
        Texto:"""
        prompt = main_prompt + str(text)
        response  = self.model.generate(prompt)
        answer = response["answer"]
        return answer