import torch
import torch.nn as nn
import numpy as np
from datetime import datetime as dt

import familytree
import util


net_defaults = {
    'single_tree': False,
    'share_rel_units': True,
    'person1_repr_units': 6,
    'rel_repr_units': 6,
    'hidden_units': 12,
    'preoutput_units': 6,
    'use_preoutput': False,
    'use_biases': True,
    'target_offset': 0.2,
    'weight_init_type': 'uniform',
    'weight_init_range': 1/12,
    'weight_init_offset': 1/24,
    'bias_init_type': 'uniform',
    'bias_init_range': 0.001,
    'bias_init_offset': 0.0005,
    'act_fn': torch.relu,
    'loss_fn': nn.MSELoss,
    'loss_reduction': 'sum',
    'seed': None,
    'device': None,
    'torchfp': None
}

train_defaults = {
    'num_epochs': 5000,
    'lr': 0.003,
    'momentum': 0,
    'weight_decay': 0,
    'n_holdout': 4,
    'report_freq': 50,
    'snap_freq': 50,
    'batch_size': 0
}


class FamilyTreeNet(nn.Module):
    """
    Network for learning and testing Hinton's Family Tree domain (replication).
    In this version, for simplicity, all tests are on the "English" tree.
    - single_tree: Train (and test) on English tree alone rather than both English and Italian
    - X_units: Size of X layer
    - weight_init_scale: Uniform weight initialization is +/- this value
    - target_offset: If using sigmoid activations, how much to add/subtract to/from targets for loss fn
    - rng_seed: Provide an RNG seed to replicate previous results; if none, uses a random one.
    - device: Override default device ('cuda'/'cpu') for pytorch
    - torchfp: Override default floating-point type (torch.float/torch.double)
    """   
    
    def __init__(self, **net_params):
        super(FamilyTreeNet, self).__init__()

        # Merge default params with overrides and make them properties
        net_params = {**net_defaults, **net_params}
        for key, val in net_params.items():
            setattr(self, key, val)

        self.device, self.torchfp, self.zeros_fn = util.init_torch(self.device, self.torchfp)
        if self.device.type == 'cuda':
            print('Using CUDA')
        else:
            print('Using CPU')
        
        if self.seed is None:
            self.seed = torch.seed()
        else:
            torch.manual_seed(self.seed)
        
        # Get training tensors
        self.english_tree = familytree.get_hinton_tree()
        person1_mat, rel_mat, person2_mat = self.english_tree.get_io_mats(zeros_fn=self.zeros_fn, cat_fn=torch.cat)
        
        self.n_inputs_english, self.n_people_english = person1_mat.shape
        
        if self.single_tree:
            self.full_tree = self.english_tree
        else:
            self.italian_tree = familytree.get_hinton_tree(italian=True)
            self.full_tree = self.english_tree + self.italian_tree
            
            person1_italian, rel_italian, person2_italian = self.italian_tree.get_io_mats(zeros_fn=self.zeros_fn,
                                                                                          cat_fn=torch.cat)
            
            person1_mat = torch.block_diag(person1_mat, person1_italian)
            if self.share_rel_units:
                rel_mat = torch.cat((rel_mat, rel_italian), 0)
            else:
                rel_mat = torch.block_diag(rel_mat, rel_italian)
            person2_mat = torch.block_diag(person2_mat, person2_italian)
        
        self.n_inputs, self.person1_units = person1_mat.shape
        self.rel_units = rel_mat.shape[1]
        self.person2_units = person2_mat.shape[1] 
        
        self.person1_mat = person1_mat
        self.rel_mat = rel_mat
        self.person2_mat = person2_mat
                    
        # Make layers
        def make_layer(in_size, out_size):
            return nn.Linear(in_size, out_size, bias=self.use_biases).to(self.device)

        self.person1_to_repr = make_layer(self.person1_units, self.person1_repr_units)
        self.rel_to_repr = make_layer(self.rel_units, self.rel_repr_units)
        total_repr_units = self.person1_repr_units + self.rel_repr_units
        self.repr_to_hidden = make_layer(total_repr_units, self.hidden_units)

        if self.use_preoutput:
            self.hidden_to_preoutput = make_layer(self.hidden_units, self.preoutput_units)
            self.preoutput_to_person2 = make_layer(self.preoutput_units, self.person2_units)
        else:
            self.hidden_to_person2 = make_layer(self.hidden_units, self.person2_units)
        
        # Initialize with small random weights
        def init_uniform(param, offset, prange):
            a = offset - prange/2
            b = offset + prange/2
            nn.init.uniform_(param.data, a=a, b=b)

        def init_normal(param, offset, prange):
            nn.init.normal_(param.data, mean=offset, std=prange/2)

        def init_default(*_):
            pass

        init_fns = {'default': init_default, 'uniform': init_uniform, 'normal': init_normal}

        with torch.no_grad():
            for layer in self.children():
                try:
                    init_fns[self.weight_init_type](layer.weight, self.weight_init_offset, self.weight_init_range)
                except KeyError:
                    raise ValueError('Weight initialization type not recognized')

                if layer.bias is not None:
                    try:
                        init_fns[self.bias_init_type](layer.bias, self.bias_init_offset, self.bias_init_range)
                    except KeyError:
                        raise ValueError('Bias initialization type notn recognized')

        # For simplicity, instead of using the "liberal" loss function described in the paper, make the targets 
        # 0.1 (for false) and 0.9 (for true) and use regular mean squared error.
        if self.act_fn == torch.sigmoid:
            self.person2_train_target = (1-self.target_offset) * self.person2_mat + self.target_offset/2
        else:
            self.person2_train_target = self.person2_mat

        self.criterion = self.loss_fn(reduction=self.loss_reduction)
        
    def forward(self, person1, rel):
        person1_repr_act = self.act_fn(self.person1_to_repr(person1))
        rel_repr_act = self.act_fn(self.rel_to_repr(rel))
        repr_act = torch.cat((person1_repr_act, rel_repr_act), dim=1)
        hidden_act = self.act_fn(self.repr_to_hidden(repr_act))

        if self.use_preoutput:
            preoutput_act = self.act_fn(self.hidden_to_preoutput(hidden_act))
            return self.act_fn(self.preoutput_to_person2(preoutput_act))
        else:
            return self.act_fn(self.hidden_to_person2(hidden_act))
    
    def b_outputs_correct(self, outputs, batch_inds, threshold=0.2):
        """
        Element-wise function to find which outputs are correct for a batch.
        threshold: the maximum distance from 0 or 1 to be considered correct
        """
        return torch.lt(torch.abs(outputs - self.person2_mat[batch_inds]), threshold)
    
    def weighted_acc(self, outputs, batch_inds, threshold=0.2):
        """
        For each input in the batch, find the average of accuracy for 0s and accuracy for 1s
        (i.e., correct for unbalanced ground truth output)
        """
        n_output_units = self.person2_mat.shape[1]
        set_freq = torch.sum(self.person2_mat[batch_inds], dim=1, keepdim=True)
        unset_freq = n_output_units - set_freq
        
        set_weight = 0.5 / set_freq
        unset_weight = 0.5 / unset_freq
        
        weights = torch.where(self.person2_mat[batch_inds].to(bool), set_weight, unset_weight)
        b_correct = self.b_outputs_correct(outputs, batch_inds, threshold=threshold)
        return torch.sum(weights * b_correct.to(self.torchfp), dim=1)        
         
    def train_epoch(self, order, optimizer, batch_size=0):
        """
        Do training on batches of given size of the examples indexed by order.
        Return the total loss and output accuracy for each example (in index order).
        Accuracy for any example that is not used will be nan.
        """
        total_loss = torch.tensor(0.0)
        acc_each = torch.full((self.n_inputs,), np.nan)
        wacc_each = torch.full((self.n_inputs,), np.nan)
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
                wacc_each[batch_inds] = self.weighted_acc(outputs, batch_inds)
                perfect_each[batch_inds] = torch.all(outputs_correct, dim=1).to(self.torchfp)

        return total_loss, acc_each, wacc_each, perfect_each, mean_output, std_output
    
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
        
        print('Holding out:')
        while n_good_inds < n_holdout:
            ind = torch.randint(self.n_inputs_english, (1,), device='cpu')
            if ind in holdout_inds[:n_good_inds]:
                continue

            person1_ind = torch.nonzero(torch.squeeze(self.person1_mat[ind]))
            rel_ind = torch.nonzero(torch.squeeze(self.rel_mat[ind]))
            person2_inds = torch.nonzero(torch.squeeze(self.person2_mat[ind]))
            
            if any((person1_heldout[person1_ind] > 1,
                    rel_heldout[rel_ind] > 1,
                    any([person2_heldout[p2] > 1 for p2 in person2_inds]))):
                continue
                
            person1_heldout[person1_ind] += 1
            rel_heldout[rel_ind] += 1
            for p2 in person2_inds:
                person2_heldout[p2] += 1
            
            person1 = self.english_tree.members[person1_ind]
            rel = person1.relationship_fns[rel_ind]
            print(f'{person1.name}/{rel.name}')
            
            holdout_inds[n_good_inds] = ind
            n_good_inds += 1
        
        print()
    
        train_inds = np.setdiff1d(range(self.n_inputs), holdout_inds)
        return train_inds, holdout_inds
    
    def prepare_snapshots(self, snap_freq, num_epochs, **_extra):
        """Make tensors to hold representation snapshots"""
        total_epochs = sum(num_epochs)
        snap_epochs = util.calc_snap_epochs(snap_freq, total_epochs, 'lin')
        epoch_digits = len(str(snap_epochs[-1]))
        n_snaps = len(snap_epochs)
        
        snaps = {
            'person1_repr': torch.full((n_snaps, self.person1_units, self.person1_repr_units), np.nan),
            'person1_hidden': torch.full((n_snaps, self.person1_units, self.hidden_units), np.nan),
            'relation_repr': torch.full((n_snaps, self.rel_units, self.rel_repr_units), np.nan),
            'relation_hidden': torch.full((n_snaps, self.rel_units, self.hidden_units), np.nan)
        }
        
        return snap_epochs, epoch_digits, snaps
    
    def do_training(self, **train_params):
        """
        Do the training!
        num_epochs is either a scalar or sequence of integers if there are multiple training stages.
        If there are multiple stages, lr, momentum, and weight_decay can either be sequences or scalars
        to use the same hyperparameter for each stage.
        The weight decay here is not scaled by the learning rate (different from PyTorch's definition).
        If batch_size == 0, do full training set batches.
        """

        # Merge default params with overrides
        train_params = {**train_defaults, **train_params}
            
        num_epochs = train_params['num_epochs']
        if isinstance(num_epochs, tuple):
            n_stages = len(num_epochs)
        else:
            n_stages = 1
            num_epochs = (num_epochs,)

        lr = train_params['lr']
        if isinstance(lr, tuple):
            assert len(lr) == n_stages, 'Wrong number of lr values for number of stages'
        else:
            lr = tuple(lr for _ in range(n_stages))

        momentum = train_params['momentum']
        if isinstance(momentum, tuple):
            assert len(momentum) == n_stages, 'Wrong number of momentum values for number of stages'
        else:
            momentum = tuple(momentum for _ in range(n_stages))

        weight_decay = train_params['weight_decay']
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
        train_inds, holdout_inds = self.prepare_holdout(train_params['n_holdout'])
        n_inputs_train = len(train_inds)
        
        # Prepare snapshots
        snap_epochs, epoch_digits, snaps = self.prepare_snapshots(**train_params)

        total_epochs = sum(num_epochs)
        change_epochs = list(np.cumsum(num_epochs))
        epoch_digits = len(str(total_epochs-1))
        n_report = (total_epochs-1) // train_params['report_freq'] + 1
        reports = {rtype: np.zeros(n_report)
                   for rtype in ['loss', 'mean_output', 'std_output',
                                 'accuracy', 'weighted_acc', 'frac_perfect',
                                 'test_accuracy', 'test_weighted_acc', 'test_frac_perfect']}
        
        for epoch in range(total_epochs):
            if epoch in change_epochs:
                # Move to new stage of learning
                k_change = change_epochs.index(epoch)
                optimizer.param_groups[0]['lr'] = lr[k_change + 1]
                optimizer.param_groups[0]['momentum'] = momentum[k_change + 1]
                optimizer.param_groups[0]['weight_decay'] = wd_torch[k_change + 1]
                
            if epoch in snap_epochs:
                # collect snapshots
                k_snap = snap_epochs.index(epoch)

                with torch.no_grad():
                    people_input = torch.eye(self.person1_units, device=self.device)
                    rels_input = torch.eye(self.rel_units, device=self.device)
                    people_dummy = torch.zeros((self.rel_units, self.person1_units), device=self.device)
                    rels_dummy = people_dummy.T
                    
                    snaps['person1_repr'][k_snap] = self.act_fn(self.person1_to_repr(people_input))
                    dummy_rel_repr = self.act_fn(self.rel_to_repr(rels_dummy))
                    combined_repr = torch.cat((snaps['person1_repr'][k_snap], dummy_rel_repr), dim=1)
                    snaps['person1_hidden'][k_snap] = self.act_fn(self.repr_to_hidden(combined_repr))
                    
                    snaps['relation_repr'][k_snap] = self.act_fn(self.rel_to_repr(rels_input))
                    dummy_person1_repr = self.act_fn(self.person1_to_repr(people_dummy))
                    combined_repr = torch.cat((dummy_person1_repr, snaps['relation_repr'][k_snap]), dim=1)
                    snaps['relation_hidden'][k_snap] = self.act_fn(self.repr_to_hidden(combined_repr))
                    
            
            order = train_inds[torch.randperm(n_inputs_train, device='cpu')]
            loss, acc_each, wacc_each, perfect_each, mean_out, std_out = self.train_epoch(
                order, optimizer, train_params['batch_size'])
            
            # report progress
            if epoch % train_params['report_freq'] == 0:
                k_report = epoch // train_params['report_freq']
                
                with torch.no_grad():
                    mean_loss = loss.item() / n_inputs_train
                    mean_acc = torch.nanmean(acc_each).item()
                    mean_wacc = torch.nanmean(wacc_each).item()
                    frac_perf = torch.nanmean(perfect_each).item()
                    
                    # testing
                    test_outputs = self(self.person1_mat[holdout_inds], self.rel_mat[holdout_inds])
                    # following the paper, use a more lenient threshold (0.5) for test items
                    test_outputs_correct = self.b_outputs_correct(test_outputs, holdout_inds, threshold=0.5)
                    test_acc = torch.mean(test_outputs_correct.to(self.torchfp)).item()
                    test_wacc = torch.mean(self.weighted_acc(test_outputs, holdout_inds, threshold=0.5))
                    test_frac_perf = torch.mean(torch.all(test_outputs_correct, dim=1).to(self.torchfp), dim=0)
                
                print(f'Epoch {epoch:{epoch_digits}d} end:',
                      f'loss = {mean_loss:7.3f},',
                      # f'accuracy = {mean_acc:.3f},',
                      f'weighted acc = {mean_wacc:.3f},',
                      f'{frac_perf*100:3.0f}% perfect (train),',
                      f'{test_frac_perf*100:3.0f}% perfect (test)')
                
                reports['loss'][k_report] = mean_loss
                reports['accuracy'][k_report] = mean_acc
                reports['weighted_acc'][k_report] = mean_wacc
                reports['frac_perfect'][k_report] = frac_perf
                reports['test_accuracy'][k_report] = test_acc
                reports['test_weighted_acc'][k_report] = test_wacc
                reports['test_frac_perfect'][k_report] = test_frac_perf
                reports['mean_output'][k_report] = mean_out
                reports['std_output'][k_report] = std_out
                
        snaps_cpu = {stype: s.cpu().numpy() for stype, s in snaps.items()}
        return {'reports': reports, 'snaps': snaps_cpu}

    
def train_n_fam_nets(n=36, run_type='', net_params=None, train_params=None):
    """Do a series of runs and save results"""
    
    combined_net_params = net_defaults.copy()
    if net_params is not None:
        for key, val in net_params.items():
            if key not in combined_net_params:
                raise KeyError(f'Unrecognized net param {key}')
            combined_net_params[key] = val
            
    combined_train_params = train_defaults.copy()
    if train_params is not None:
        for key, val in train_params.items():
            if key not in combined_train_params:
                raise KeyError(f'Unrecognized train param {key}')
            combined_train_params[key] = val
    
    snaps_all = []
    reports_all = []
    seeds_all = []

    net = None
    for i in range(n):
        print(f'Training iteration {i+1}')
        print('----------------------')
        
        net = FamilyTreeNet(**combined_net_params)
        res = net.do_training(**combined_train_params)
        
        seeds_all.append(net.seed)
        snaps_all.append(res['snaps'])
        reports_all.append(res['reports'])

        print('')
        
    # concatenate across runs
    snaps = {}
    for snap_type in snaps_all[0].keys():
        snaps[snap_type] = np.stack([snaps_one[snap_type] for snaps_one in snaps_all])
        
    reports = {}
    for report_type in reports_all[0].keys():
        reports[report_type] = np.stack([reports_one[report_type] for reports_one in reports_all])
    
    if run_type != '':
        run_type += '_'
    
    save_name = f'data/familytree/{run_type}res_{dt.now():%Y-%m-%d_%H-%M-%S}.npz'
    np.savez(save_name, snapshots=snaps, reports=reports, net_params=combined_net_params,
             train_params=combined_train_params, seeds=seeds_all)
    
    return save_name, net
