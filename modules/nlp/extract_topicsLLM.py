class ExtractTopicsLLM():
    def __init__(self):
        import maritalk

        self.model = maritalk.MariTalk(
            key="104147566582134244375$db094baa6042418d",
            model="sabia-2-small"  #modelos sabia-2-medium e sabia-2-small
        )

    def extract_topics(self, text):
        main_prompt = """
        You are a highly skilled text analyst. Your task is to extract the main topics from a given text and present them in a Python dictionary format. Each topic should be listed as a key in the dictionary with a nested dictionary as its value containing a brief description and the number of times it is mentioned in the text. 

        Follow these steps:
        1. Read the provided text carefully.
        2. Identify the main topics discussed in the text.
        3. For each topic, provide a brief description.
        4. Count the number of times each topic is mentioned.
        5. Present your findings in a Python dictionary format.

        Here is the format of the dictionary you should use:

        ```python
        {
            "Topic 1": {
                "Description": "Brief description of Topic 1",
                "Frequency": X
            },
            "Topic 2": {
                "Description": "Brief description of Topic 2",
                "Frequency": Y
            },
            ...
        }
        ```

        Example:

        Input Text:
        "Artificial intelligence (AI) is transforming various industries by automating processes and enhancing efficiency. In healthcare, AI algorithms can analyze medical data to assist in diagnosis and treatment plans. The finance sector uses AI for fraud detection and investment predictions. AI's ability to learn and adapt makes it a powerful tool in these fields. Furthermore, the impact of AI on job markets and ethical considerations around its use are hotly debated topics."

        Output Dictionary:

        ```python
        {
            "Artificial Intelligence": {
                "Description": "The overall subject of AI and its applications in different sectors.",
                "Frequency": 4
            },
            "Healthcare": {
                "Description": "Use of AI in analyzing medical data for diagnosis and treatment.",
                "Frequency": 1
            },
            "Finance": {
                "Description": "Application of AI in fraud detection and investment predictions.",
                "Frequency": 1
            },
            "Job Market": {
                "Description": "The impact of AI on employment opportunities and job market dynamics.",
                "Frequency": 1
            },
            "Ethics": {
                "Description": "Ethical considerations and debates surrounding the use of AI.",
                "Frequency": 1
            }
        }
        ```
        Here is the text that needs topic extraction. Make sure you to only return the python dictionary and nothing more:
        Text:
        """
        prompt = main_prompt + str(text)
        response  = self.model.generate(prompt)
        answer = response["answer"]
        return answer