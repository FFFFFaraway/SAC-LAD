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
    'split_by': 'clip',  # clip or speaker
    'device': 'cpu',  # cpu or cuda
}

model_class = eval(args['model_name'])
model = model_class(args).to(device=args['device'])

model.load_state_dict(torch.load('best_weights/LAD_best.pt', map_location=args['device']))
model.eval()
runner = Runner(model, args)
runner.test()
