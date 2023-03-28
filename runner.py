import os
import random
import pickle
import torch
import numpy as np
from scipy import stats
from pathlib import Path
from utils import FusionDataset
from torch.nn import MSELoss
from utils import MSE, DiffLoss, CMD
from torch.utils.data import DataLoader
from transformers.optimization import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score, r2_score


def evaluate(ys, ps, phase):
    res = {'mse': mean_squared_error(ys, ps), 'mae': mean_absolute_error(ys, ps)}
    res['corr'], _ = stats.pearsonr(ps, ys)
    rounded_ps = np.array([round(p) for p in ps])
    rounded_ys = np.array([round(y) for y in ys])
    res['acc4'] = accuracy_score(rounded_ys, rounded_ps)
    res['macro4'] = f1_score(rounded_ys, rounded_ps, average='macro')
    res['micro4'] = f1_score(rounded_ys, rounded_ps, average='micro')
    binary_ps = np.array([int(p != 0) for p in rounded_ps])
    binary_ys = np.array([int(y != 0) for y in rounded_ys])
    res['acc2'] = accuracy_score(binary_ys, binary_ps)
    res['macro2'] = f1_score(binary_ys, binary_ps, average='macro')
    res['micro2'] = f1_score(binary_ys, binary_ps, average='micro')
    return {f'{phase}_{k}': v for k, v in res.items()}


class Runner:
    def __init__(self, model, args):
        self.args = args
        print(args)
        self.model = model
        if args['split_by'] == 'clip':
            split_data = self.split_by_clip()
        elif args['split_by'] == 'speaker':
            split_data = self.split_by_speaker()
        else:
            raise NotImplementedError(f'split_by must be clip or speaker not {args["split_by"]}')
        dataset = {k: FusionDataset(split_data[k], device=args['device']) for k in ['train', 'val', 'test']}
        self.dataset = dataset
        self.dl = {key: DataLoader(dataset[key], batch_size=args['batch_size'], shuffle=True) for key in dataset}
        self.loss_func = MSELoss()
        self.loss_diff = DiffLoss()
        self.loss_recon = MSE()
        self.loss_cmd = CMD()
        self.opt = AdamW(model.parameters(), lr=args['lr'])
        self.set_seed()

    def set_seed(self):
        seed = self.args['random_seed']
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)

    def split(self, f):
        args = self.args
        f_train, f_test = train_test_split(f, test_size=args['test_r'], random_state=args['random_seed'])
        f_train, f_val = train_test_split(f_train, test_size=args['val_r'], random_state=args['random_seed'])
        return f_train, f_val, f_test

    def read_data(self):
        args = self.args
        data = {k: np.load(f'data/{k}f.npy') for k in 'vapt'}
        data['y'] = np.load('data/' + args['label_path'])
        data['sp'] = np.load('data/speaker.npy')
        return data

    def split_by_clip(self):
        if os.path.exists('data/split_data_clip.pkl'):
            with open("data/split_data_clip.pkl", "rb") as f:
                data = pickle.load(f)
            return data
        data = self.read_data()
        split_data = {
            'train': {},
            'val': {},
            'test': {}
        }
        for m in 'vapty':
            split_data['train'][m], split_data['val'][m], split_data['test'][m] = self.split(data[m])
        with open("data/split_data_clip.pkl", "wb") as f:
            pickle.dump(split_data, f)
        return split_data

    def split_by_speaker(self):
        if os.path.exists('data/split_data_speaker.pkl'):
            f = open("data/split_data_speaker.pkl", "rb")
            return pickle.load(f)
        split_data = {
            'train': {m: [] for m in 'vapty'},
            'val': {m: [] for m in 'vapty'},
            'test': {m: [] for m in 'vapty'},
        }
        data = self.read_data()
        sp = {}
        sp['train'], sp['val'], sp['test'] = (set(v) for v in self.split(list(set(data['sp']))))
        for i, one_sp in enumerate(data['sp']):
            for phase in ['train', 'val', 'test']:
                if one_sp in sp[phase]:
                    for m in 'vapty':
                        split_data[phase][m].append(data[m][i])
                    break
        for phase in ['train', 'val', 'test']:
            for m in 'vapty':
                split_data[phase][m] = np.array(split_data[phase][m])
        f = open("data/split_data_speaker.pkl", "wb")
        pickle.dump(split_data, f)
        return split_data

    def run(self, phase, g=False):
        args = self.args
        model = self.model
        dl = self.dl[phase]
        ps, ys = [], []
        model.train(mode=g)
        task_losses = []
        diff_losses = []
        recon_losses = []
        cmd_losses = []
        with torch.set_grad_enabled(g):
            for fd, y in dl:
                if g:
                    self.opt.zero_grad()
                logit = model(fd).squeeze(-1)
                task_loss = self.loss_func(logit, y)
                diff_loss = self.get_diff_loss()
                recon_loss = self.get_recon_loss()
                cmd_loss = self.get_cmd_loss()

                task_losses.append(task_loss.item())
                diff_losses.append(args['dif_coef'] * diff_loss.item())
                recon_losses.append(args['recon_coef'] * recon_loss.item())
                cmd_losses.append(args['sim_coef'] * cmd_loss.item())

                loss = task_loss + args['dif_coef'] * diff_loss + \
                       args['sim_coef'] * cmd_loss + args['recon_coef'] * recon_loss
                if g:
                    loss.backward()
                    self.opt.step()
                ps.append(logit.detach().cpu().numpy().reshape(-1, 1))
                ys.append(y.cpu().numpy().reshape(-1, 1))
        ps, ys = np.concatenate(ps, axis=0).squeeze(-1), np.concatenate(ys, axis=0).squeeze(-1)

        res = evaluate(ys, ps, phase)
        length = len(dl)
        losses = {
            f'{phase}_task_loss': sum(task_losses) / length,
            f'{phase}_dif_loss': sum(diff_losses) / length,
            f'{phase}_cmd_loss': sum(cmd_losses) / length,
            f'{phase}_recon_loss': sum(recon_losses) / length,
        }
        res.update(losses)
        print("\t".join(f"{k}:{v:.4f}" for k, v in res.items()))
        return res[f'{phase}_mse']

    def train(self):
        args = self.args
        model = self.model
        for phase in ['train', 'val', 'test']:
            self.run(phase, g=False)
        best_mse = 1000000
        for epoch in range(args['epochs']):
            self.run('train', g=True)
            val_mse = self.run('val', g=False)
            test_mse = self.run('test', g=False)
            if args['save'] and val_mse < best_mse:
                Path('best_weights').mkdir(exist_ok=True)
                torch.save(model.state_dict(), './best_weights/' + str(self.args['model_name']) + '.pt')
                best_mse = test_mse

    def test(self):
        for phase in ['train', 'val', 'test']:
            self.run(phase, g=False)

    def get_cmd_loss(self, ):
        if not hasattr(self.model, 'cmd_pairs'):
            return torch.zeros(1, device=self.args['device'])
        loss = []
        for a, b in self.model.cmd_pairs:
            loss.append(self.loss_cmd(a, b, 5))
        loss = torch.stack(loss, dim=0)
        return torch.sum(loss, dim=0) / 3.0

    def get_diff_loss(self):
        if not hasattr(self.model, 'dif_pairs'):
            return torch.zeros(1, device=self.args['device'])
        loss = []
        for a, b in self.model.dif_pairs:
            loss.append(self.loss_diff(a, b))
        loss = torch.stack(loss, dim=0)
        return torch.sum(loss, dim=0)

    def get_recon_loss(self):
        if not hasattr(self.model, 'recon_pairs'):
            return torch.zeros(1, device=self.args['device'])
        loss = []
        for a, b in self.model.recon_pairs:
            loss.append(self.loss_recon(a, b))
        loss = torch.stack(loss, dim=0)
        return torch.sum(loss, dim=0) / 3.0
