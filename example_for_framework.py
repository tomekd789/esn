"""
This is a running example of the framework usage

Step 1. Start the data handling service
/opt/anaconda3/envs/algopolis/bin/python3 <path to the ESN project>/esn/flask_data_service.py \
    --andromeda /opt/dane/ssd/data_cache/andromedanew_adjusted/ \
    --year_since 2017 \
    --sequence_len 5865
It will be necessary for this example, and also in case of training;
it will generate random two-week Andromeda sequences (5865 is roughly two weeks of minute ticks)
You can test it from Linux CLI:
curl -X GET http://127.0.0.1:5000/sequence

Step 2. Train a model if you don't have any
Adjust parameters as necessary. See train.py for more comments
/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/train.py
    --id experiment_id \
    --data_url http://127.0.0.1:5000/sequence \
    --epochs 10_000 \
    --cuda_device 3 \
    --batch 1000 \
    --population 50 \
    --model_size 70 \
    --mutation_probability 1.0 \
    --co_probability 0.4 \
    --esn_input_size 20 \
    --max_evaluation_steps 10_000 \
    --take_profit 1.05 \
    --stop_loss 0.95 \
    --save_dir /home/tdryjanski/esn_model
Wait until at least 1st epoch finishes. It will create the ESN population checkpoint

Step 3. Start the predictor service
Run:
/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/flask_predictor_service.py \
    --id experiment_id
    --cuda_device 0
    --esn_input_size 20
    --max_evaluation_steps 10_000
    --take_profit 1.05
    --stop_loss 0.95
    --load_dir /home/tdryjanski/esn_model
You can test it from Linux CLI:
curl -i -H "Content-Type: application/json" -X POST -d '[1.0, 1.0, ...]' http://localhost:5100/predict
"""
import json
import requests

from model import get_rest_data

# Get a single sequence from the data service
while True:
    sequence, = get_rest_data('http://127.0.0.1:5000/sequence', 1)
    sequence = json.dumps(sequence)
    headers = {'Content-Type': 'application/json', 'Accept':'application/json'}
    result = requests.post(url="http://localhost:5100/predict", data=sequence, headers=headers)
    trade, index = json.loads(result.text)
    # if trade == "none":
    #     print("Do nothing")
    # else:
    #     print(f"{trade} at position {index}")
    if trade != "buy" or index != 40:
        print(trade, index)
