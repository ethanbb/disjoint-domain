import torch
import torch.nn as nn
import numpy as np
from datetime import datetime as dt
from copy import deepcopy
from functools import cached_property

import familytree
import util


net_defaults = {
    'trees': ['english', 'italian'],
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
    'include_cross_tree_loss': True,
    'seed': None,
    'device': None,
    'torchfp': None
}

train_defaults = {
    'num_epochs': 5000,
    'lr': 0.003,
    'momentum': 0,
    'weight_decay': 0,
    'report_freq': 50,
    'snap_freq': 50,
    'batch_size': 0,
    'do_combo_testing': True,
    'n_holdout': 4,
    'do_tree_holdout': False,
    'domains_to_hold_out': 0,
    'reports_per_test': 4,
    'test_thresh': 0.85,
    'test_max_epochs': 15000,
    'include_final_eval': True,
    'reset_optim_params_during_holdout': True
}


class FamilyTreeNet(nn.Module):
    """
    Network for learning and testing Hinton's Family Tree domain (replication).
    In this version, for simplicity, all tests are on the "English" tree.
    - trees: List of names of trees to train (and test) on ('english', 'italian', etc.)
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
        self.n_trees = len(self.trees)
        assert self.n_trees > 0, 'Must learn at least one tree'
        self.person1_mat = self.zeros_fn((0, 0))
        self.person2_mat = self.zeros_fn((0, 0))
        self.p2_tree_mask = self.zeros_fn((0, 0))
        self.rel_mat = self.zeros_fn((0, 12 if self.share_rel_units else 0))
        self.full_tree = familytree.FamilyTree([], [])
        self.each_tree = []
        
        for i, tree_name in enumerate(self.trees):
            this_tree = familytree.get_tree(name=tree_name)
            self.full_tree += this_tree
            self.each_tree.append(this_tree)
            this_p1, this_rels, this_p2 = this_tree.get_io_mats(zeros_fn=self.zeros_fn, cat_fn=torch.cat) 
            
            self.person1_mat = torch.block_diag(self.person1_mat, this_p1)
            self.person2_mat = torch.block_diag(self.person2_mat, this_p2)
            self.p2_tree_mask = torch.block_diag(self.p2_tree_mask,
                                                   torch.ones_like(this_p2))
            if self.share_rel_units:
                self.rel_mat = torch.cat((self.rel_mat, this_rels), 0)
            else:
                self.rel_mat = torch.block_diag(self.rel_mat, this_rels)
            
            if i == 0:
                self.n_inputs_first, self.n_people_first = this_p1.shape
        
        self.n_inputs, self.person1_units = self.person1_mat.shape
        self.rel_units = self.rel_mat.shape[1]
        self.person2_units = self.person2_mat.shape[1] 
                    
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
        self.train_x_inds = None  # to be set when training occurs
        
    @cached_property
    def train_p1_inds(self):
        if self.train_x_inds is None:
            return None
        with torch.no_grad():
            return np.flatnonzero(self.person1_mat[self.train_x_inds].any(dim=0).cpu().numpy())
        
    @cached_property
    def train_rel_inds(self):
        if self.train_x_inds is None:
            return None
        with torch.no_grad():
            return np.flatnonzero(self.rel_mat[self.train_x_inds].any(dim=0).cpu().numpy())
        
    # --- feedforward computation methods --- #
    def calc_p1_repr_preact(self, person1):
        return self.person1_to_repr(person1)
    
    def calc_rel_repr_preact(self, rel):
        return self.rel_to_repr(rel)
    
    def calc_hidden_preact(self, person1, rel):
        p1_repr = self.act_fn(self.calc_p1_repr_preact(person1))
        rel_repr = self.act_fn(self.calc_rel_repr_preact(rel))
        repr_full = torch.cat((p1_repr, rel_repr), dim=1)
        return self.repr_to_hidden(repr_full)
    
    def calc_output_preact(self, person1, rel):
        hidden_act = self.act_fn(self.calc_hidden_preact(person1, rel))

        if self.use_preoutput:
            preoutput_act = self.act_fn(self.hidden_to_preoutput(hidden_act))
            return self.preoutput_to_person2(preoutput_act)
        else:
            return self.hidden_to_person2(hidden_act)
        
    def forward(self, person1, rel):
        return self.act_fn(self.calc_output_preact(person1, rel))
        
    # --- training and evaluation methods --- #
    
    def b_outputs_correct(self, outputs, batch_inds, threshold=0.2, tree_mask=False):
        """
        Element-wise function to find which outputs are correct for a batch.
        threshold: the maximum distance from 0 or 1 to be considered correct
        """
        if tree_mask:
            outputs = outputs * self.p2_tree_mask[batch_inds]
            targets = self.person2_mat[batch_inds] * self.p2_tree_mask[batch_inds]
        else:
            targets = self.person2_mat[batch_inds]
            
        return torch.lt(torch.abs(outputs - targets), threshold)
    
    def weighted_acc(self, outputs, batch_inds, threshold=0.2, tree_mask=False):
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
        b_correct = self.b_outputs_correct(outputs, batch_inds,
                                           threshold=threshold, tree_mask=tree_mask)
        return torch.sum(weights * b_correct.to(self.torchfp), dim=1)
    
    def evaluate_input_set(self, input_inds, threshold=0.2, all_masked=False):
        """Get the loss, accuracy, weighted accuracy, etc. on a set of inputs"""
        self.eval()
        with torch.no_grad():
            outputs = self(self.person1_mat[input_inds], self.rel_mat[input_inds])
            targets = self.person2_train_target[input_inds]
            if self.include_cross_tree_loss:
                loss = self.criterion(outputs, targets) / len(input_inds)
            else:
                masked_outputs = outputs * self.p2_tree_mask[input_inds]
                masked_targets = targets * self.p2_tree_mask[input_inds]
                loss = self.criterion(masked_outputs, masked_targets) / len(input_inds)
                
            outputs_correct = self.b_outputs_correct(outputs, input_inds, threshold=threshold, tree_mask=all_masked)
            acc = torch.mean(outputs_correct.to(self.torchfp)).item()
            wacc = torch.mean(self.weighted_acc(outputs, input_inds,
                                                threshold=threshold, tree_mask=all_masked)).item()
            wacc_loose = torch.mean(self.weighted_acc(outputs, input_inds,
                                                      threshold=0.5, tree_mask=all_masked)).item()
            wacc_loose_masked = torch.mean(self.weighted_acc(outputs, input_inds,
                                                             threshold=0.5,
                                                             tree_mask=True)).item()
            perfect = torch.mean(torch.all(outputs_correct, dim=1).to(self.torchfp)).item()

        return loss, acc, wacc, wacc_loose, wacc_loose_masked, perfect
            
    def train_epoch(self, order, optimizer, batch_size=0):
        """Do training on batches of given size of the examples indexed by order."""
        if type(order) != torch.Tensor:
            order = torch.tensor(order, device='cpu', dtype=torch.long)
        
        self.train()
        for batch_inds in torch.split(order, batch_size) if batch_size > 0 else [order]:
            optimizer.zero_grad()
            outputs = self(self.person1_mat[batch_inds], self.rel_mat[batch_inds])
            
            if not self.include_cross_tree_loss:
                outputs = outputs * self.p2_tree_mask[batch_inds]
            
            loss = self.criterion(outputs, self.person2_train_target[batch_inds])
            loss.backward()
            optimizer.step()
    
    def prepare_tree_holdout(self, n=1):
        """Prepare for holding out the last tree and testing how many epochs it takes to generalize"""
        n_train_trees = self.n_trees - n
        assert n_train_trees > 0, f"Can't hold out {n} trees if there are less than {n+1} trees total"
        tree_names = [f'{chr(ord("A") + n)} ({tree_type})' for n, tree_type in zip(range(self.n_trees), self.trees)]
        held_out_trees = tree_names[n_train_trees:]
        print(f'Holding out tree(s): {", ".join(held_out_trees)}')
        
        tree_sizes = torch.tensor([tree.size for tree in self.each_tree])
        cum_tree_sizes = torch.cumsum(tree_sizes, dim=0)
        p1_inds = torch.nonzero(self.person1_mat, as_tuple=True)[1]
        k_tree = torch.sum(p1_inds[np.newaxis, :] >= cum_tree_sizes[:, np.newaxis], dim=0)
        k_tree_rel = k_tree - n_train_trees
        train_inds = torch.squeeze(torch.nonzero(k_tree_rel < 0)).cpu().numpy()
        test_inds = {tname: torch.squeeze(torch.nonzero(k_tree_rel == kt)).cpu().numpy()
                     for kt, tname in enumerate(held_out_trees)}
        
        return train_inds, test_inds

    def prepare_holdout(self, n_holdout):
        """
        Pick n_holdout person1/relationship combinations to test on, in the English tree.
        To avoid any issues with a person or a relationship not being present at all,
        each person and relationship can only be held out at most twice.
        """        
        holdout_inds = torch.empty(n_holdout, device='cpu', dtype=torch.long)
        n_good_inds = 0
        person1_heldout = [0 for _ in range(self.n_people_first)]
        rel_heldout = [0 for _ in range(self.rel_mat.shape[1])]
        person2_heldout = [0 for _ in range(self.n_people_first)]
        
        print('Holding out:')
        while n_good_inds < n_holdout:
            ind = torch.randint(self.n_inputs_first, (1,), device='cpu')
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
            
            person1 = self.each_tree[0].members[person1_ind]
            rel = person1.relationship_fns[rel_ind]
            print(f'{person1.name}/{rel.name}')
            
            holdout_inds[n_good_inds] = ind
            n_good_inds += 1
        
        print()
    
        train_inds = np.setdiff1d(range(self.n_inputs), holdout_inds)
        return train_inds, holdout_inds
    
    def generalize_test(self, batch_size, included_inds, targets, max_epochs, thresh,
                        change_epochs, optim_args):
        """
        See how long it takes the network to reach accuracy threshold on target inputs,
        when training on items specified by included_inds. Then restore the parameters.
        
        'targets' can be an array of indices or a logical mask into the full set of inputs.
        """
        self.train()
        
        # Save original state of network to restore later
        net_state_dict = deepcopy(self.state_dict())
        optimizer = torch.optim.SGD(self.parameters(), **optim_args[0])

        epochs = 0
        while epochs < max_epochs:
            if epochs in change_epochs:
                k_change = change_epochs.index(epochs)
                for key, val in optim_args[k_change + 1].items():
                    optimizer.param_groups[0][key] = val
            
            order = util.permute(included_inds)
            self.train_epoch(order, optimizer, batch_size)
            perfect = self.evaluate_input_set(targets, threshold=0.5, all_masked=not self.include_cross_tree_loss)[3]

            if perfect >= thresh:
                break

            epochs += 1

        # Restore old state of network
        self.load_state_dict(net_state_dict)

        etg_string = '= ' + str(epochs + 1) if epochs < max_epochs else '> ' + str(max_epochs)
        return epochs, etg_string
    
    def take_snapshots(self):
        """
        Return a dict of the activations at each layer, before and after nonlinearity, both unreduced over all inputs
        and averaged over all inputs with each person1 and relationship present.
        """
        def get_snap_means(unreduced_snapshot):
            p1_mean = torch.full((self.person1_units, unreduced_snapshot.shape[1]), np.nan)
            rel_mean = torch.full((self.rel_units, unreduced_snapshot.shape[1]), np.nan)
            
            for k_p1 in self.train_p1_inds:
                p1_mean[k_p1] = torch.mean(unreduced_snapshot[self.person1_mat[:, k_p1] > 0], dim=0)
            
            for k_rel in self.train_rel_inds:
                rel_mean[k_rel] = torch.mean(unreduced_snapshot[self.rel_mat[:, k_rel] > 0], dim=0)
            
            return p1_mean, rel_mean
        
        repr_snaps = {}  # those that don't need to be averaged over p1 and/or rels.
        snaps = {}
        people_input = torch.eye(self.person1_units, device=self.device)[self.train_p1_inds]
        rels_input = torch.eye(self.rel_units, device=self.device)[self.train_rel_inds]
        
        self.eval()
        with torch.no_grad():
            repr_snaps['person1_repr_preact'] = torch.full((self.person1_units, self.person1_repr_units), np.nan)
            repr_snaps['person1_repr'] = repr_snaps['person1_repr_preact'].clone()
            p1_preact = self.calc_p1_repr_preact(people_input[self.train_p1_inds])
            repr_snaps['person1_repr_preact'][self.train_p1_inds] = p1_preact
            repr_snaps['person1_repr'][self.train_p1_inds] = self.act_fn(p1_preact)
                
            repr_snaps['relation_repr_preact'] = torch.full((self.rel_units, self.rel_repr_units), np.nan)
            repr_snaps['relation_repr'] = repr_snaps['relation_repr_preact'].clone()
            rel_preact = self.calc_rel_repr_preact(rels_input[self.train_rel_inds])
            repr_snaps['relation_repr_preact'][self.train_rel_inds] = rel_preact
            repr_snaps['relation_repr'][self.train_rel_inds] = self.act_fn(rel_preact)
            
            # get the rest of the layers for all inputs
            snaps['hidden_preact'] = torch.full((self.n_inputs, self.hidden_units), np.nan)
            snaps['hidden'] = snaps['hidden_preact'].clone()
            hidden_preact = self.calc_hidden_preact(self.person1_mat[self.train_x_inds], self.rel_mat[self.train_x_inds])
            snaps['hidden_preact'][self.train_x_inds] = hidden_preact
            snaps['hidden'][self.train_x_inds] = self.act_fn(hidden_preact)
            
            if self.use_preoutput:
                snaps['preoutput_preact'] = torch.full((self.n_inputs, self.preoutput_units), np.nan)
                snaps['preoutput'] = snaps['preoutput_preact'].clone()
                preoutput_preact = self.hidden_to_preoutput(self.act_fn(hidden_preact))
                snaps['preoutput_preact'][self.train_x_inds] = preoutput_preact
                snaps['preoutput'][self.train_x_inds] = self.act_fn(preoutput_preact)
            
            snaps['output_preact'] = torch.full((self.n_inputs, self.person2_units), np.nan)
            snaps['output'] = snaps['output_preact'].clone()
            output_preact = self.calc_output_preact(self.person1_mat[self.train_x_inds], self.rel_mat[self.train_x_inds])
            snaps['output_preact'][self.train_x_inds] = output_preact
            snaps['output'][self.train_x_inds] = self.act_fn(output_preact)
            
            # get the versions averaged over p1 and relations
            mean_snaps = {}
            for key, snap in snaps.items():
                mean_snaps['person1_' + key], mean_snaps['relation_' + key] = get_snap_means(snap)
            snaps.update(mean_snaps)
            
        snaps.update(repr_snaps)
        return snaps           
    
    
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
        p = {**train_defaults, **train_params}
            
        num_epochs = p['num_epochs']
        if isinstance(num_epochs, tuple):
            n_stages = len(num_epochs)
        else:
            n_stages = 1
            num_epochs = (num_epochs,)

        lr = p['lr']
        if isinstance(lr, tuple):
            assert len(lr) == n_stages, 'Wrong number of lr values for number of stages'
        else:
            lr = tuple(lr for _ in range(n_stages))

        momentum = p['momentum']
        if isinstance(momentum, tuple):
            assert len(momentum) == n_stages, 'Wrong number of momentum values for number of stages'
        else:
            momentum = tuple(momentum for _ in range(n_stages))

        weight_decay = p['weight_decay']
        if isinstance(weight_decay, tuple):
            assert len(weight_decay) == n_stages, 'Wrong number of weight decay values for number of stages'
        else:
            weight_decay = tuple(weight_decay for _ in range(n_stages))
        
        #  weight_decay is Hinton's version which is the fraction to reduce the weights by after an update
        #  (independent of the learning rate). To convert to what pytorch means by weight decay, have to 
        #  divide by the learning rate.
        wd_torch = tuple(wd / rate for wd, rate in zip(weight_decay, lr))
        
        train_optim_args = [{'lr': this_lr, 'momentum': this_mm, 'weight_decay': this_wd}
                            for this_lr, this_mm, this_wd in zip(lr, momentum, wd_torch)]

        test_optim_args = [{**optim_args, 'weight_decay': 0} for optim_args in train_optim_args]
        # test_optim_args = train_optim_args

        optimizer = torch.optim.SGD(self.parameters(), **train_optim_args[0])
        
        # Make tree/domain holdout options consistent
        if p['domains_to_hold_out'] > 0:
            p['do_tree_holdout'] = True
        elif p['do_tree_holdout']:
            if 'domains_to_hold_out' in train_params:  # case where domains_to_hold_out = 0 was explicitly specified
                raise ValueError('Must hold out > 0 trees if doing tree holdout')
            p['domains_to_hold_out'] = 1
        
        # Train and testing input indices
        if p['do_combo_testing']:
            assert not p['do_tree_holdout'], "Can't do both combo testing and tree holdout"
            self.train_x_inds, holdout_inds = self.prepare_holdout(p['n_holdout'])
        elif p['do_tree_holdout']:
            self.train_x_inds, holdout_ind_dict = self.prepare_tree_holdout(n=p['domains_to_hold_out'])
        else:
            self.train_x_inds = np.arange(self.n_inputs)
            holdout_inds = np.array([])
            
        # Reset cached properties, if necessary
        for attr_name in ['train_p1_inds', 'train_rel_inds']:
            if hasattr(self, attr_name):
                delattr(self, attr_name)

        total_epochs = sum(num_epochs)
        change_epochs = list(np.cumsum(num_epochs))[:-1]
        epoch_digits = len(str(total_epochs-1))
        etg_digits = len(str(p['test_max_epochs'])) + 2
        n_report = total_epochs // p['report_freq'] + 1
        n_etg = (n_report - 1) // p['reports_per_test'] + 1
        reports = {rtype: np.zeros(n_report)
                   for rtype in ['loss', 'accuracy', 'weighted_acc', 'weighted_acc_loose',
                                 'weighted_acc_loose_indomain', 'frac_perfect']}
        
        # Prepare snapshots
        snap_epochs = util.calc_snap_epochs(p['snap_freq'], total_epochs, 'lin', p['include_final_eval'])
        epoch_digits = len(str(snap_epochs[-1]))
        snaps = []
        
        
        if p['do_combo_testing']:
            for rtype in ['test_accuracy', 'test_weighted_acc',
                          'test_weighted_acc_indomain', 'test_frac_perfect']:
                reports[rtype] = np.zeros(n_report)

        if p['do_tree_holdout']:
            reports['new_tree_etg'] = np.zeros(n_etg, dtype=int)
            for kt in range(1, p['domains_to_hold_out']):
                reports[f'new_tree_etg{kt+1}'] = np.zeros(n_etg, dtype=int)
        
        for epoch in range(total_epochs + (1 if p['include_final_eval'] else 0)):
            if epoch in change_epochs:
                # Move to new stage of learning
                k_change = change_epochs.index(epoch)
                for key, val in train_optim_args[k_change + 1].items():
                    optimizer.param_groups[0][key] = val
                
            if epoch in snap_epochs:
                # collect snapshots
                k_snap = snap_epochs.index(epoch)
                snaps.append(self.take_snapshots())

            
            # report progress
            if epoch % p['report_freq'] == 0:
                k_report = epoch // p['report_freq']
                
                # get current performance
                (mean_loss, mean_acc, mean_wacc, mean_wacc_loose,
                 mean_wacc_loose_masked,  frac_perf) = self.evaluate_input_set(self.train_x_inds)
                
                reports['loss'][k_report] = mean_loss
                reports['accuracy'][k_report] = mean_acc
                reports['weighted_acc'][k_report] = mean_wacc
                reports['weighted_acc_loose'][k_report] = mean_wacc_loose
                reports['weighted_acc_loose_indomain'][k_report] = mean_wacc_loose_masked
                reports['frac_perfect'][k_report] = frac_perf
                    
                report_str = f'Epoch {epoch:{epoch_digits}d}: loss = {mean_loss:7.3f}'
                if self.include_cross_tree_loss:
                    report_str += f', weighted acc = {mean_wacc_loose:.3f}, {frac_perf*100:3.0f}% perfect (train)'
                else:
                    report_str += f', in-domain weighted acc = {mean_wacc_loose_masked:.3f}'
    
                # testing
                if p['do_tree_holdout'] and k_report % p['reports_per_test'] == 0:
                    k_ho = k_report // p['reports_per_test']
                    
                    # offset change epochs if we don't want to reset training parameters during holdout
                    if not p['reset_optim_params_during_holdout']:
                        test_change_epochs = [e - epoch for e in change_epochs]
                        n_changes_passed = sum(e <= 0 for e in test_change_epochs)
                        this_test_optim_args = test_optim_args[n_changes_passed:]
                        test_change_epochs = test_change_epochs[n_changes_passed:]
                    else:
                        test_change_epochs = change_epochs
                        this_test_optim_args = test_optim_args
                    
                    for kt, (tname, this_test_inds) in enumerate(holdout_ind_dict.items()):
                        included_inds = np.concatenate((self.train_x_inds, this_test_inds))
                        
                        etg, etg_string = self.generalize_test(
                            p['batch_size'], included_inds, this_test_inds,
                            p['test_max_epochs'], p['test_thresh'],
                            test_change_epochs, this_test_optim_args
                        )
                        report_str += f'\n\t Epochs to learn tree {tname} {etg_string:>{etg_digits}}'
                        report_type = 'new_tree_etg' + (str(kt+1) if kt > 0 else '')
                        reports[report_type][k_ho] = etg
                
                if p['do_combo_testing']:
                    # following the paper, use a more lenient threshold (0.5) for test items
                    _, test_acc, test_wacc, _, test_wacc_masked, test_frac_perf = \
                    self.evaluate_input_set(holdout_inds, threshold=0.5)

                    reports['test_accuracy'][k_report] = test_acc
                    reports['test_weighted_acc'][k_report] = test_wacc
                    reports['test_weighted_acc_indomain'][k_report] = test_wacc_masked
                    reports['test_frac_perfect'][k_report] = test_frac_perf

                    report_str += f', {test_frac_perf*100:3.0f}% perfect (test)'
                
                print(report_str)
            
            # do training
            if epoch < total_epochs:
                order = util.permute(self.train_x_inds)
                self.train_epoch(order, optimizer, p['batch_size'])
         
        # concatenate snapshots and move to cpu
        if len(snaps) > 0:
            snaps_cpu = {stype: np.stack([s[stype].cpu().numpy() for s in snaps])
                         for stype in snaps[0]}
        else:
            snaps_cpu = {}
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
    ys = []

    net = None
    for i in range(n):
        print(f'Training iteration {i+1}')
        print('----------------------')
        
        net = FamilyTreeNet(**combined_net_params)
        res = net.do_training(**combined_train_params)
        
        seeds_all.append(net.seed)
        snaps_all.append(res['snaps'])
        reports_all.append(res['reports'])
        ys.append(net.person2_mat.cpu().numpy())

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
    np.savez(save_name, snapshots=snaps, reports=reports, ys=ys, net_params=combined_net_params,
             train_params=combined_train_params, seeds=seeds_all)
    
    return save_name, net
