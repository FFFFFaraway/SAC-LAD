from model import *
from runner import Runner

args = {
    'model_name': 'LAD',
    'use_live_text': True,
    'v_dim': 2048,
    'a_dim': 512,
    't_dim': 768,
    'h_dim': 128,
    'dropout': 0.1,
    'random_seed': 2021,
    'train_r': 0.8,
    'test_r': 0.2,
    'val_r': 0.1,
    'batch_size': 4,
    'lr': 3e-06,
    'epochs': 200,
    'save': True,
    'dif_coef': 1,
    'sim_coef': 1,
    'recon_coef': 1,
    'split_by': 'clip',  # clip or speaker
    'device': 'cpu',  # cpu or cuda
}

model_class = eval(args['model_name'])
model = model_class(args).to(device=args['device'])
runner = Runner(model, args)
runner.train()
