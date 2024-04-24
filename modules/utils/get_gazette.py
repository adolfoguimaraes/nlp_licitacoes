import requests
import json
from urllib import request

class GetGazette():

    def __init__(self, 
                 TERRITORY_ID='2927408', 
                 PUBLISHED_SINCE='2023-05-22', 
                 PUBLISHED_UNTIL='2023-05-22',
                 OUTPUT_DIR="."):
        
        self.TERRITORY_ID = TERRITORY_ID
        self.PUBLISHED_SINCE = PUBLISHED_SINCE
        self.PUBLISHED_UNTIL = PUBLISHED_UNTIL
        self.OUTPUT_DIR = OUTPUT_DIR

        self.URL_SEARCH = f"https://queridodiario.ok.org.br/api/gazettes?territory_ids={TERRITORY_ID}&published_since={PUBLISHED_SINCE}&published_until={PUBLISHED_UNTIL}&excerpt_size=500&number_of_excerpts=1&size=400&sort_by=ascending_date"


    def downloadGazette(self):

        try:
            response = requests.get(self.URL_SEARCH)
            if response.status_code == 200:
                result_json = response.json()
                str_name = self.TERRITORY_ID + "_" + self.PUBLISHED_SINCE.replace("-", "") + "_" + self.PUBLISHED_UNTIL.replace("-","")
                
                with open(self.OUTPUT_DIR + str_name + ".json", 'w') as f:
                    json.dump(result_json, f)

                gazettes_list = result_json.get('gazettes')
                count = 0
                for gazette in gazettes_list:
                    id_ = gazette['territory_id'] + "_" + gazette['date'].replace("-","") + "_" + str(count)
                    txt_file = gazette['txt_url']
                    request.urlretrieve(txt_file, self.OUTPUT_DIR + id_ + ".txt")
                    count += 1
            else:
                raise Exception("Erro na coleta do diário: " + response.status_code)
        except Exception as e:
            print("Erro na requisição:")
            print(e)