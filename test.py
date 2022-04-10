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
    'batch_size': 4,
    'lr': 3e-06,
    'random_seed': 2021,
    'dif_coef': 1,
    'sim_coef': 1,
    'recon_coef': 1,
    'split_by': 'clip',  # clip or speaker
    'device': 'cpu',  # cpu or cuda
}

model_class = eval(args['model_name'])
model = model_class(args).to(device=args['device'])

model.load_state_dict(torch.load('best_weights/LAD_best.pt', map_location=args['device']))
model.eval()
runner = Runner(model, args)
runner.test()
