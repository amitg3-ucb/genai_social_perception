def getEmbeddings(spec):
    
    import requests
    import pandas as pd
    import numpy as np
    
    model_id = spec['model_id']
    hf_token = spec['hf_token']
    api_url = spec['api_url']
    headers = spec['headers']
    
    def query(texts):
        response = requests.post(api_url, headers=headers, json={"inputs": texts, "options":{"wait_for_model":True}})
        return response.json()
    
    return np.array(query(list(spec['data'].iloc[spec['start']:spec['end']]['abstracts'])))

def addUp(vals):
    
    import numpy as np
    
    return np.sum(vals)