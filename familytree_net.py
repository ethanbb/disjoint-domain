import familytree
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime as dt


def init_torch(device=None, torchfp=None, use_cuda_if_possible=True):
    """Establish floating-point type and device to use with Pytorch"""
    
    if device is None:
        if use_cuda_if_possible and torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(device)
        
    if device.type == 'cuda':
        ttype_ns = torch.cuda
    else:
        ttype_ns = torch
    
    if torchfp is None:
        torchfp = torch.float
    
    if torchfp == torch.float:
        torch.set_default_tensor_type(ttype_ns.FloatTensor)
    elif torchfp == torch.double:
        torch.set_default_tensor_type(ttype_ns.DoubleTensor)
    else:
        raise NotImplementedError(f'No tensor type known for dtype {torchfp}')

    def zeros_fn(size):
        return torch.zeros(size, device=device)

    return device, torchfp, zeros_fn


class FamilyTreeNet(nn.Module):
    """
    Network for learning and testing Hinton's Family Tree domain (replication).
    In this version, for simplicity, all tests are on the "English" tree.
    - single_tree: Train (and test) on English tree alone rather than both English and Italian
    - X_units: Size of X layer
    - param_init_scale: Uniform weight initialization is +/- this value
    - rng_seed: Provide an RNG seed to replicate previous results; if none, uses a random one.
    - device: Override default device ('cuda'/'cpu') for pytorch
    - torchfp: Override default floating-point type (torch.float/torch.double)
    """   
    
    def __init__(self, single_tree=False, person1_repr_units=6, rel_repr_units=6,
                 hidden_units=12, preoutput_units=6, use_biases=False,
                 param_init_type='uniform', param_init_scale=0.3, param_init_offset=0,
                 act_fn=torch.sigmoid, loss_fn=nn.MSELoss, loss_reduction='sum',
                 rng_seed=None, device=None, torchfp=None):
        super(FamilyTreeNet, self).__init__()
        
        self.single_tree = single_tree
        self.device, self.torchfp, self.zeros_fn = init_torch(device, torchfp)
        if self.device.type == 'cuda':
            print('Using CUDA')
        else:
            print('Using CPU')
        
        if rng_seed is None:
            self.seed = torch.seed()
        else:
            self.seed = rng_seed
            torch.manual_seed(rng_seed)
        
        # Get training tensors
        english_tree = familytree.get_hinton_tree()
        person1_mat, rel_mat, person2_mat = english_tree.get_io_mats(zeros_fn=self.zeros_fn, cat_fn=torch.cat)
        
        self.n_inputs_english, self.n_people_english = person1_mat.shape
        
        if not self.single_tree:
            italian_tree = familytree.get_hinton_tree(italian=True)
            person1_italian, rel_italian, person2_italian = italian_tree.get_io_mats(zeros_fn=self.zeros_fn, cat_fn=torch.cat)
            
            person1_mat = torch.block_diag(person1_mat, person1_italian)
            rel_mat = torch.cat((rel_mat, rel_italian), 0)
            person2_mat = torch.block_diag(person2_mat, person2_italian)
        
        person1_units = person1_mat.shape[1]
        rel_units = rel_mat.shape[1]
        person2_units = person2_mat.shape[1] 
        
        self.person1_mat = person1_mat
        self.rel_mat = rel_mat
        self.person2_mat = person2_mat
        
        self.n_inputs = person1_mat.shape[0]
            
        # Make layers
        self.person1_to_repr = nn.Linear(person1_units, person1_repr_units, bias=use_biases).to(self.device)
        self.rel_to_repr = nn.Linear(rel_units, rel_repr_units, bias=use_biases).to(self.device)
        total_repr_units = person1_repr_units + rel_repr_units
        self.repr_to_hidden = nn.Linear(total_repr_units, hidden_units, bias=use_biases).to(self.device)
        self.hidden_to_preoutput = nn.Linear(hidden_units, preoutput_units, bias=use_biases).to(self.device)
        self.preoutput_to_person2 = nn.Linear(preoutput_units, person2_units, bias=use_biases).to(self.device)
        # self.hidden_to_person2 = nn.Linear(hidden_units, person2_units, bias=use_biases).to(self.device)
        
        # Activation function
        self.act_fn = act_fn
        
        # Initialize with small random weights
        if param_init_type != 'default':
            with torch.no_grad():
                for p in self.parameters():
                    if param_init_type == 'uniform':
                        a = -param_init_scale + param_init_offset
                        b = param_init_scale + param_init_offset
                        nn.init.uniform_(p.data, a=a, b=b)
                    elif param_init_type == 'normal':
                        nn.init.normal_(p.data, mean=param_init_offset, std=param_init_scale)
                    else:
                        raise ValueError('Unrecognized param init type')

        # For simplicity, instead of using the "liberal" loss function described in the paper, make the targets 
        # 0.1 (for false) and 0.9 (for true) and use regular mean squared error.
        self.person2_train_target = 0.8 * self.person2_mat + 0.1
#         self.person2_train_target = self.person2_mat
        self.criterion = loss_fn(reduction=loss_reduction)
        
    def forward(self, person1, rel):
        person1_repr_act = self.act_fn(self.person1_to_repr(person1))
        rel_repr_act = self.act_fn(self.rel_to_repr(rel))
        repr_act = torch.cat((person1_repr_act, rel_repr_act), dim=1)
        hidden_act = self.act_fn(self.repr_to_hidden(repr_act))
        preoutput_act = self.act_fn(self.hidden_to_preoutput(hidden_act))
        return torch.sigmoid(self.preoutput_to_person2(preoutput_act))
        # return torch.sigmoid(self.hidden_to_person2(hidden_act))
    
    def b_outputs_correct(self, outputs, batch_inds, threshold=0.2):
        """
        Element-wise function to find which outputs are correct for a batch.
        threshold: the maximum distance from 0 or 1 to be considered correct
        """
        return torch.lt(torch.abs(outputs - self.person2_mat[batch_inds]), threshold)
         
    def train_epoch(self, order, optimizer, batch_size=0):
        """
        Do training on batches of given size of the examples indexed by order.
        Return the total loss and output accuracy for each example (in index order).
        Accuracy for any example that is not used will be nan.
        """
        total_loss = torch.tensor(0.0)
        acc_each = torch.full((self.n_inputs,), np.nan)
        perfect_each = torch.full((self.n_inputs,), np.nan)

        if type(order) != torch.Tensor:
            order = torch.tensor(order, device='cpu', dtype=torch.long)
        
        for batch_inds in torch.split(order, batch_size) if batch_size > 0 else [order]:
            optimizer.zero_grad()
            outputs = self(self.person1_mat[batch_inds], self.rel_mat[batch_inds])
            loss = self.criterion(outputs, self.person2_train_target[batch_inds])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                mean_output = torch.mean(outputs)
                std_output = torch.mean(torch.std(outputs, dim=0))  # mean standard deviation (over output units)
                total_loss += loss
                outputs_correct = self.b_outputs_correct(outputs, batch_inds)
                acc_each[batch_inds] = torch.mean(outputs_correct.to(self.torchfp), dim=1)
                perfect_each[batch_inds] = torch.all(outputs_correct, dim=1).to(self.torchfp)

        return total_loss, acc_each, perfect_each, mean_output, std_output
    
    def prepare_holdout(self, n_holdout):
        """
        Pick n_holdout person1/relationship combinations to test on, in the English tree.
        To avoid any issues with a person or a relationship not being present at all,
        each person and relationship can only be held out at most twice.
        """        
        holdout_inds = torch.empty(n_holdout, device='cpu', dtype=torch.long)
        n_good_inds = 0
        person1_heldout = [0 for _ in range(self.n_people_english)]
        rel_heldout = [0 for _ in range(self.rel_mat.shape[1])]
        person2_heldout = [0 for _ in range(self.n_people_english)]
        
        while n_good_inds < n_holdout:
            ind = torch.randint(self.n_inputs_english, (1,), device='cpu')
            if ind in holdout_inds[:n_good_inds]:
                continue

            person1 = torch.nonzero(torch.squeeze(self.person1_mat[ind]))
            rel = torch.nonzero(torch.squeeze(self.rel_mat[ind]))
            person2 = torch.nonzero(torch.squeeze(self.person2_mat[ind]))
            
            if any((person1_heldout[person1] > 1,
                    rel_heldout[rel] > 1,
                    any([person2_heldout[p2] > 1 for p2 in person2]))):
                continue
            
            holdout_inds[n_good_inds] = ind
            n_good_inds += 1
    
        train_inds = np.setdiff1d(range(self.n_inputs), holdout_inds)
        return train_inds, holdout_inds
    
    def do_training(self, num_epochs=(20, 1480), lr=(0.005, 0.01),
                    momentum=(0.5, 0.9), weight_decay=0.002,
                    n_holdout=4, report_freq=50, batch_size=0):
        """
        Do the training!
        num_epochs is either a scalar or sequence of integers if there are multiple training stages.
        If there are multiple stages, lr, momentum, and weight_decay can either be sequences or scalars
        to use the same hyperparameter for each stage.
        The weight decay here is not scaled by the learning rate (different from PyTorch's definition).
        If batch_size == 0, do full training set batches.
        """
        
        if isinstance(num_epochs, tuple):
            n_stages = len(num_epochs)
        else:
            n_stages = 1
            num_epochs = (num_epochs,)
        
        if isinstance(lr, tuple):
            assert len(lr) == n_stages, 'Wrong number of lr values for number of stages'
        else:
            lr = tuple(lr for _ in range(n_stages))
        
        if isinstance(momentum, tuple):
            assert len(momentum) == n_stages, 'Wrong number of momentum values for number of stages'
        else:
            momentum = tuple(momentum for _ in range(n_stages))
        
        if isinstance(weight_decay, tuple):
            assert len(weight_decay) == n_stages, 'Wrong number of weight decay values for number of stages'
        else:
            weight_decay = tuple(weight_decay for _ in range(n_stages))
        
        #  weight_decay is Hinton's version which is the fraction to reduce the weights by after an update
        #  (independent of the learning rate). To convert to what pytorch means by weight decay, have to 
        #  divide by the learning rate.
        wd_torch = tuple(wd / rate for wd, rate in zip(weight_decay, lr))

        optimizer = torch.optim.SGD(self.parameters(), lr=lr[0], momentum=momentum[0], weight_decay=wd_torch[0])
        
        # Choose the hold-out people/relationships from the English tree
        train_inds, holdout_inds = self.prepare_holdout(n_holdout)
        n_inputs_train = len(train_inds)
        
        total_epochs = sum(num_epochs)
        change_epochs = list(np.cumsum(num_epochs))
        epoch_digits = len(str(total_epochs-1))
        n_report = (total_epochs-1) // report_freq + 1
        reports = {rtype: np.zeros(n_report)
                   for rtype in ['loss', 'accuracy', 'frac_perfect',
                                 'test_accuracy', 'test_frac_perfect',
                                 'mean_output', 'std_output']}
        
        for epoch in range(total_epochs):
            if epoch in change_epochs:
                # Move to new stage of learning
                k_change = change_epochs.index(epoch)
                optimizer.param_groups[0]['lr'] = lr[k_change + 1]
                optimizer.param_groups[0]['momentum'] = momentum[k_change + 1]
                optimizer.param_groups[0]['weight_decay'] = wd_torch[k_change + 1]
            
            order = train_inds[torch.randperm(n_inputs_train, device='cpu')]
            loss, acc_each, perfect_each, mean_out, std_out = self.train_epoch(order, optimizer, batch_size)
            
            # report progress
            if epoch % report_freq == 0:
                k_report = epoch // report_freq
                
                with torch.no_grad():
                    mean_loss = loss.item() / n_inputs_train
                    mean_acc = torch.nansum(acc_each).item() / n_inputs_train
                    frac_perf = torch.nansum(perfect_each).item() / n_inputs_train
                    
                    # testing
                    test_outputs = self(self.person1_mat[holdout_inds], self.rel_mat[holdout_inds])
                    # following the paper, use a more lenient threshold (0.5) for test items
                    test_outputs_correct = self.b_outputs_correct(test_outputs, holdout_inds, threshold=0.5)
                    test_acc = torch.mean(test_outputs_correct.to(self.torchfp)).item()
                    test_frac_perf = torch.mean(torch.all(test_outputs_correct, dim=1).to(self.torchfp), dim=0)
                
                print(f'Epoch {epoch:{epoch_digits}d} end:',
                      f'loss = {mean_loss:7.3f},',
                      f'accuracy = {mean_acc:.3f},',
                      f'{frac_perf*100:3.0f}% perfect (train),',
                      f'{test_frac_perf*100:3.0f}% perfect (test)')
                
                reports['loss'][k_report] = mean_loss
                reports['accuracy'][k_report] = mean_acc
                reports['frac_perfect'][k_report] = frac_perf
                reports['test_accuracy'][k_report] = test_acc
                reports['test_frac_perfect'][k_report] = test_frac_perf
                reports['mean_output'][k_report] = mean_out
                reports['std_output'][k_report] = std_out
                
        return {'reports': reports}

    
def train_n_fam_nets(n=36, run_type='', net_params=None, train_params=None):
    """Do a series of runs and save results"""
    
    net_defaults = {
        'single_tree': False,
        'person1_repr_units': 6,
        'rel_repr_units': 6,
        'hidden_units': 12,
        'preoutput_units': 6,
        'use_biases': False,
        'param_init_type': 'uniform',
        'param_init_scale': 0.3,
        'param_init_offset': 0.0,
        'loss_fn': nn.MSELoss,
        'loss_reduction': 'sum',
        'act_fn': torch.sigmoid
    }
    
    if net_params is None:
        net_params = {}
    net_params = {**net_defaults, **net_params}
    
    train_defaults = {
        'num_epochs': (20, 1480),
        'lr': (0.005, 0.01),
        'momentum': (0.5, 0.9),
        'weight_decay': 0.002,
        'n_holdout': 4,
        'report_freq': 50,
        'batch_size': 0
    }
    
    if train_params is None:
        train_params = {}
    train_params = {**train_defaults, **train_params}
    
    reports_all = []
    seeds_all = []

    net = None
    for i in range(n):
        print(f'Training iteration {i+1}')
        print('----------------------')
        
        net = FamilyTreeNet(**net_params)
        res = net.do_training(**train_params)
        
        seeds_all.append(net.seed)
        reports_all.append(res['reports'])

        print('')
        
    reports = {}
    for report_type in reports_all[0].keys():
        reports[report_type] = np.stack([reports_one[report_type] for reports_one in reports_all])
    
    if run_type != '':
        run_type += '_'
    
    save_name = f'data/familytree/{run_type}res_{dt.now():%Y-%m-%d_%H-%M-%S}.npz'
    np.savez(save_name, reports=reports, net_params=net_params, train_params=train_params, seeds=seeds_all)
    
    return save_name, net
