###Flask service
Run:
```bash
/opt/anaconda3/envs/algopolis/bin/python3 <path to the ESN project>/esn/flask_data_service.py \
    --andromeda /opt/dane/ssd/data_cache/andromedanew_adjusted/
    --year_since 2017
    --sequence_len 5865
```
Test from CLI:
`curl -X GET http://127.0.0.1:5000/sequence`
It returns a random sequence normalized to start from 1.0 value from a random andromeda ticker
5865 is roughly two weeks of minute ticks

###Creating training file
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/create_training_set.py --andromeda /opt/dane/ssd/data_cache/andromeda_adjusted_new --year_since 2017 --samples 100_000 --target /opt/dane_synology/tdryjanski/<file name>.csv`

###Recent runs
Notes:
- All training in run on bizon; you may want to change path to .py files to your own repo, and change the --save_dir path as needed. Other than that, paths can be kept unchanged, unless you want to use a different training file (.csv)
- Recently I run four experiments, varying only the model size
- Each process takes 14G RAM, because it caches the full training file locally; this is not yet optimal
- For this reason validations should not be run in parallel

Trainings:
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/train.py --id a --recover True --data_url http://127.0.0.1:5000/sequence --epochs 10_000 --cuda_device 0 --batch 200 --population 50 --model_size 40 --mutation_probability 1.0 --co_probability 0.4 --esn_input_size 20 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --save_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/train.py --id b --recover True --data_url http://127.0.0.1:5000/sequence --epochs 10_000 --cuda_device 1 --batch 200 --population 50 --model_size 50 --mutation_probability 1.0 --co_probability 0.4 --esn_input_size 20 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --save_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/train.py --id c --recover True --data_url http://127.0.0.1:5000/sequence --epochs 10_000 --cuda_device 2 --batch 200 --population 50 --model_size 60 --mutation_probability 1.0 --co_probability 0.4 --esn_input_size 20 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --save_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/train.py --id d --recover True --data_url http://127.0.0.1:5000/sequence --epochs 10_000 --cuda_device 3 --batch 200 --population 50 --model_size 70 --mutation_probability 1.0 --co_probability 0.4 --esn_input_size 20 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --save_dir /home/tdryjanski/esn_model`

Validations:
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/evaluate.py --id a --data_url http://127.0.0.1:5000/sequence --sequences 100_000 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --load_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/evaluate.py --id b --data_url http://127.0.0.1:5000/sequence --sequences 100_000 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --load_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/evaluate.py --id c --data_url http://127.0.0.1:5000/sequence --sequences 100_000 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --load_dir /home/tdryjanski/esn_model`
`/opt/anaconda3/envs/algopolis/bin/python3 /home/tdryjanski/projects/esn/evaluate.py --id d --data_url http://127.0.0.1:5000/sequence --sequences 100_000 --max_evaluation_steps 10_000 --take_profit 1.05 --stop_loss 0.95 --load_dir /home/tdryjanski/esn_model`