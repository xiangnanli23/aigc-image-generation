from typing import Union

import json
import requests



class HTTP:
    @staticmethod
    def get(url: str) -> str:
        res = requests.get(url)
        if res.status_code == 200:
            return res.text
            # return res.content.decode('utf-8')
        else:
            print(f"HTTP GET ERROR: {res.status_code}")
            return None

    @staticmethod
    def post(url: str, data: Union[dict, str]) -> Union[dict, None]:
        if isinstance(data, dict):
            data = json.dumps(data)
        if isinstance(data, str):
            data = data
        
        res = requests.post(url, data, headers={"content-type": "application/json"})
        # res = requests.post(url, data)
                
        if res.status_code == 200:
            return json.loads(res.content)
        else:
            print(f"HTTP POST ERROR: {res.status_code}")
            return None

###############################################################################
def serve_get(serve_url):
    res = requests.get(serve_url)
    return res.text

def serve_post(data, serve_url):
    data = json.dumps(data)
    res = requests.post(serve_url, data=data, headers={"content-type": "application/json"})
    encode_res = json.loads(res.content)
    return encode_res



###############################################################################
if __name__ == '__main__':
    pass
    