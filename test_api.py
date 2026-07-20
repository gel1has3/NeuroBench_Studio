import json
import urllib.request
import time

payload = {
    "graph": {
        "nodes": [
            {"id": "node_1", "type": "dataset", "fields": {"source": "Local File", "dataset_id": "eeg1.fif,eeg9.fif"}},
            {"id": "node_2", "type": "validation", "fields": {"validation_mode": "development"}},
            {"id": "node_3", "type": "model", "fields": {"architecture": "EEGNet"}}
        ],
        "connections": [
            {"from": "node_1", "to": "node_2"},
            {"from": "node_2", "to": "node_3"}
        ]
    },
    "dataset_id": "eeg1.fif,eeg9.fif",
    "project_name": "Test Run",
    "model_config": {"architecture": "EEGNet"}
}

data = urllib.parse.urlencode({'payload': json.dumps(payload)}).encode('utf-8')
req = urllib.request.Request('http://127.0.0.1:5000/api/run-pipeline', data=data, method='POST')

try:
    with urllib.request.urlopen(req) as response:
        res = json.loads(response.read().decode('utf-8'))
        run_id = res['run_id']
        print(f"Run started: {run_id}")
        
        while True:
            time.sleep(2)
            prog_req = urllib.request.Request(f'http://127.0.0.1:5000/api/pipeline/progress/{run_id}')
            with urllib.request.urlopen(prog_req) as p_res:
                lines = p_res.read().decode('utf-8').split('\n')
                for line in lines:
                    if line.startswith('data: '):
                        p_data = json.loads(line[6:])
                        if p_data.get('stage') in ['complete', 'done', 'error']:
                            if 'results' in p_data:
                                print(f"Has ROC: {'roc_curve' in p_data['results'] and p_data['results']['roc_curve'] is not None}")
                            else:
                                print("Pipeline failed:", p_data)
                            exit(0)
                        print(p_data.get('stage'), p_data.get('progress'))
except Exception as e:
    print(e)
