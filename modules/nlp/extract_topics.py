class ExtractTopics():

    def __init__(self,model='default'):
        from umap import UMAP
        from bertopic.vectorizers import ClassTfidfTransformer
        umap_model = UMAP(random_state=57)
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
        
        if model == 'default':
            from bertopic import BERTopic
            self.bert_model = BERTopic(language='portuguese', min_topic_size=5, nr_topics="auto", umap_model=umap_model, ctfidf_model=ctfidf_model)

        elif model == 'gpt':
            import openai
            from bertopic.representation import OpenAI
            from bertopic import BERTopic

            client = openai.OpenAI(api_key="sk-")
            prompt = """
            I have a topic that contains the following documents:
            [DOCUMENTS]
            The topic is described by the following keywords: [KEYWORDS]

            Based on the information above, extract a short but highly descriptive topic label of at most 5 words. Make sure it is in the following format:
            topic: <topic label>
            """
            representation_model = OpenAI(client, model="gpt-3.5-turbo", delay_in_seconds=10, chat=True, prompt=prompt)
            self.bert_model = BERTopic(representation_model=representation_model)

        elif model == 'maritalk':
            import maritalk
            from bertopic import BERTopic

            model = maritalk.MariTalk(
                key="104147566582134244375$db094baa6042418d",
                model="sabia-2-small"  #modelos sabia-2-medium e sabia-2-small
            )

            system_prompt = """
            <s>[INST] <<SYS>>
            You are a helpful, respectful and honest assistant for labeling topics.
            <</SYS>>
            """
            example_prompt = """
            I have a topic that contains the following documents:
            - Traditional diets in most cultures were primarily plant-based with a little meat on top, but with the rise of industrial style meat production and factory farming, meat has become a staple food.
            - Meat, but especially beef, is the word food in terms of emissions.
            - Eating meat doesn't make you a bad person, not eating meat doesn't make you a good one.

            The topic is described by the following keywords: 'meat, beef, eat, eating, emissions, steak, food, health, processed, chicken'.

            Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.

            [/INST] Environmental impacts of eating meat
            """
            main_prompt = """
            [INST]
            I have a topic that contains the following documents:
            [DOCUMENTS]

            The topic is described by the following keywords: '[KEYWORDS]'.

            Based on the information about the topic above, please create a short label of this topic. Make sure you to only return the label and nothing more.
            [/INST]
            """

            prompt = system_prompt + example_prompt + main_prompt
            representation_model = model.generate(prompt)
            self.bert_model = BERTopic(representation_model=representation_model)

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