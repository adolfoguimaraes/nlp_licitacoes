class Summarizer():
    def __init__(self):
        import maritalk

        self.model = maritalk.MariTalk(
            key="104147566582134244375$c4ce5b5fbee6978e",
            model="sabia-2-small"  #sabia-2-medium // sabia-2-small
        )

    def summary(self, text):
        
        main_prompt ="""Você é um assistente de resumo especializado em diários oficiais. Sua tarefa é ler o texto fornecido de uma página de um diário oficial e fornecer um resumo conciso que capture os principais pontos e informações relevantes.

        Siga estas etapas:
        Leia o texto fornecido cuidadosamente.
        Identifique os principais pontos e informações relevantes.
        Escreva um resumo conciso que capture esses pontos principais.
        Texto:"""
        prompt = main_prompt + str(text)
        response  = self.model.generate(prompt)
        answer = response["answer"]
        return answer
    
    def extract_info(self, text):
        main_prompt="""Você é um assistente especializado em análise de diários oficiais. Sua tarefa é extrair informações de licitações do texto fornecido. Para cada licitação, identifique a empresa, o valor e os objetivos. Apresente suas respostas em formato de dicionário Python, onde cada licitação é representada por um dicionário com as chaves "Empresa", "Valor" e "Objetivos".

        Siga estas etapas:
        1. Leia o texto fornecido cuidadosamente.
        2. Identifique cada licitação mencionada no texto.
        3. Para cada licitação, extraia as informações de "Empresa", "Valor" e "Objetivos".
        4. Apresente os resultados em um dicionário Python, onde cada licitação é representada por um dicionário com as chaves "Empresa", "Valor" e "Objetivos".

        Formato do dicionário:

        ```python
        {
            "Licitação 1": {
                "Empresa": "Nome da Empresa",
                "Valor": "Valor da Licitação",
                "Objetivos": "Objetivos da Licitação"
            },
            "Licitação 2": {
                "Empresa": "Nome da Empresa",
                "Valor": "Valor da Licitação",
                "Objetivos": "Objetivos da Licitação"
            },
            ...
        }
        ```
        Apresente apenas o dicionário python como saída, sem qualquer outro texto ou explicação.
        Texto:"""
        prompt = main_prompt + str(text)
        response  = self.model.generate(prompt)
        answer = response["answer"]
        return answer