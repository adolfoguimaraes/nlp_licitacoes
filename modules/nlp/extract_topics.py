class ExtractTopics():

    def __init__(self,model='default'):

        if model == 'default':
            from bertopic import BERTopic
            self.bert_model = BERTopic(language='portuguese', min_topic_size=5, nr_topics="auto")

        elif model == 'gpt':
            import openai
            from bertopic.representation import OpenAI
            from bertopic import BERTopic

            # Create your representation model
            client = openai.OpenAI(api_key="sk-")
            embedding_model = OpenAI(client) 

            self.bert_model = BERTopic(embedding_model=embedding_model, language='portuguese', min_topic_size=5, nr_topics="auto")

        elif model == 'zeroshot':
            from bertopic import BERTopic
            from bertopic.representation import KeyBERTInspired

            zeroshot_topic_list = ["Aquisições Diretas",
                                    "Contratações Emergenciais",
                                    "Dispensas de Licitação"]

            self.bert_model = BERTopic(
                embedding_model="thenlper/gte-small",
                language="portuguese",
                nr_topics="auto",
                min_topic_size=5,
                zeroshot_topic_list=zeroshot_topic_list,
                zeroshot_min_similarity=.85,
                representation_model=KeyBERTInspired())


    def extract_topics(self, text):

        self.bert_model.fit_transform(text)
        topics = self.bert_model.get_topics()
        topics_info = self.bert_model.get_topic_info()

        return topics, topics_info
    
    def find_topics(self, find):
        similar_topics, similarity = self.bert_model.find_topics(find, top_n=3)
        return similar_topics