import torch
import torch.nn as nn
import numpy as np

import disjoint_domain as dd


class DisjointDomainNet(nn.Module):
    """
    Network for disjoint domain learning as depicted in Figure R4.
    Contains separate item representation and context representation layers,
    unless "merged" is True, in which case there is a common representation layer.
    """

    def gen_training_tensors(self):
        """Make PyTorch x and y tensors for training DisjointDomainNet"""

        item_mat, context_mat, attr_mat = dd.make_io_mats(
            ctx_per_domain=self.ctx_per_domain, attrs_per_context=self.attrs_per_context,
            n_domains=self.n_domains)

        x_item = torch.tensor(item_mat, dtype=self.torchfp, device=self.device)
        x_context = torch.tensor(context_mat, dtype=self.torchfp, device=self.device)
        y = torch.tensor(attr_mat, dtype=self.torchfp, device=self.device)

        return x_item, x_context, y

    def __init__(self, ctx_per_domain, attrs_per_context, n_domains,
                 rng_seed=None, torchfp=None, device=None, merged=False):
        super(DisjointDomainNet, self).__init__()

        self.ctx_per_domain = ctx_per_domain
        self.attrs_per_context = attrs_per_context
        self.n_domains = n_domains
        self.n_items = dd.ITEMS_PER_DOMAIN * n_domains
        self.n_contexts = ctx_per_domain * n_domains
        self.n_attributes = attrs_per_context * self.n_contexts
        self.merged = merged

        if rng_seed is None:
            torch.seed()
        else:
            torch.manual_seed(rng_seed)

        self.device, self.torchfp = dd.init_torch(device, torchfp)

        item_rep_size = self.n_items // 2
        ctx_rep_size = item_rep_size
        self.rep_size = item_rep_size + ctx_rep_size # item and context representations combined
        self.hidden_size = item_rep_size * 2

        # define layers
        if self.merged:
            self.x_to_rep = nn.Linear(self.n_items + self.n_contexts, self.rep_size).to(device)
        else:
            self.item_to_irep = nn.Linear(self.n_items, item_rep_size).to(device)
            self.ctx_to_crep = nn.Linear(self.n_contexts, ctx_rep_size).to(device)

        self.rep_to_hidden = nn.Linear(self.rep_size, self.hidden_size).to(device)
        self.hidden_to_attr = nn.Linear(self.hidden_size, self.n_attributes).to(device)

        # make weights start small
        with torch.no_grad():
            for p in self.parameters():
                nn.init.normal_(p.data, std=0.01)
                # nn.init.uniform_(p.data, a=-0.01, b=0.01)

        # make some data
        self.x_item, self.x_context, self.y = self.gen_training_tensors()
        self.n_inputs = len(self.y)

        # individual item/context tensors for evaluating the network
        self.items, self.item_names = dd.get_items(n_domains=n_domains)
        self.contexts, self.context_names = dd.get_contexts(n_domains=n_domains, ctx_per_domain=ctx_per_domain)

        self.criterion = nn.BCELoss(reduction='sum')

    def calc_rep(self, item, context):
        if self.merged:
            x = torch.cat((item, context), dim=1)
            return torch.sigmoid(self.x_to_rep(x))
        else:
            irep = torch.sigmoid(self.item_to_irep(item))
            crep = torch.sigmoid(self.ctx_to_crep(context))
            return torch.cat((irep, crep), dim=1)

    def calc_hidden(self, rep):
        return torch.sigmoid(self.rep_to_hidden(rep))

    def calc_attr(self, hidden):
        return torch.sigmoid(self.hidden_to_attr(hidden))

    def forward(self, item, context):
        return self.calc_attr(self.calc_hidden(self.calc_rep(item, context)))

    def train_epoch(self, order, batch_size, optimizer):
        """
        Do training on batches of given size of the examples indexed by order.
        Return the total loss and output accuracy for each example (in index order).
        Accuracy for any example that is not used will be nan.
        """
        total_loss = torch.tensor(0.0)
        acc_each = torch.full((self.n_inputs,), np.nan)

        for batch_inds in torch.split(order, batch_size):
            optimizer.zero_grad()
            outputs = self(self.x_item[batch_inds], self.x_context[batch_inds])
            loss = self.criterion(outputs, self.y[batch_inds])
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                total_loss += loss
                units_correct = torch.lt(torch.abs(outputs - self.y[batch_inds]), 0.1).to(self.torchfp)
                acc_each[batch_inds] = torch.mean(units_correct, dim=1)

        return total_loss, acc_each

    def prepare_holdout(self):
        """
        Pick an item and context to hold out during regular training. Then, at each epoch,
        the number of additional epochs needed to reach a threshold of accuracy on the held-out
        items is recorded.

        Returns vectors of indices into items, contexts, and x/y that will still be used.
        """
        ho_item_domain, ho_ctx_domain = dd.choose_k_inds(self.n_domains, 2)
        ho_item_ind = ho_item_domain * dd.ITEMS_PER_DOMAIN + dd.choose_k_inds(dd.ITEMS_PER_DOMAIN, 1)
        print(f'Holding out item: {self.item_names[ho_item_ind]}')
        ho_ctx_ind = ho_ctx_domain * self.ctx_per_domain + dd.choose_k_inds(self.ctx_per_domain, 1)
        print(f'Holding out context: {self.context_names[ho_ctx_ind]}')

        items_used_inds = np.setdiff1d(range(self.n_items), ho_item_ind)
        ctx_used_inds = np.setdiff1d(range(self.n_contexts), ho_ctx_ind)

        # figure out which inputs are held out (item/context combinations)
        ho_item = self.items[ho_item_ind]
        ho_context = self.contexts[ho_ctx_ind]
        b_holdout_x = self.x_item.eq(ho_item).all(axis=1) | self.x_context.eq(ho_context).all(axis=1)
        x_used_inds = torch.flatten(torch.nonzero(~b_holdout_x))

        assert len(x_used_inds) == len(self.y) - dd.ITEMS_PER_DOMAIN - self.ctx_per_domain, "Something's fishy"

        return items_used_inds, ctx_used_inds, x_used_inds

    def prepare_snapshots(self, snap_freq, snap_freq_scale, num_epochs, do_holdout_testing):
        """Make tensors to hold representation snapshots and return some relevant info"""

        # Find exactly which epochs to take snapshots (could be on log scale)
        snap_epochs = dd.calc_snap_epochs(snap_freq, snap_freq_scale, num_epochs)

        epoch_digits = len(str(snap_epochs[-1]))
        n_snaps = len(snap_epochs)
        ho_num = 1 if do_holdout_testing else 0

        irep_snaps = torch.empty((n_snaps, self.n_items - ho_num, self.rep_size))
        crep_snaps = torch.empty((n_snaps, self.n_contexts - ho_num, self.rep_size))

        # Dummy inputs for testing item & context inputs individually
        zero_items = torch.zeros((self.n_contexts - ho_num, self.items.shape[1]))
        zero_contexts = torch.zeros((self.n_items - ho_num, self.contexts.shape[1]))

        return snap_epochs, epoch_digits, irep_snaps, crep_snaps, zero_items, zero_contexts


    def generalize_test(self, batch_size, optimizer, target_inds, max_epochs=2000, thresh=0.99):
        """
        See how long it takes the network to reach 95% accuracy on held out item and context,
        when training on all items. Then restore the parameters.
        """

        # Save original state of network to restore later
        state_dict = self.state_dict()

        epochs = 0
        while epochs < max_epochs:
            order = dd.choose_k_inds(self.n_inputs, self.n_inputs)
            _, acc_each = self.train_epoch(order, batch_size, optimizer)
            acc_targets = torch.mean(acc_each[target_inds])
            if acc_targets >= thresh:
                break

            epochs += 1

        # Restore old state of network
        self.load_state_dict(state_dict)

        etg_string = str(epochs + 1) if epochs < max_epochs else '>' + str(max_epochs)
        return epochs, etg_string


    def do_training(self, optimizer, num_epochs, batch_size, report_freq,
                    snap_freq, snap_freq_scale='lin', scheduler=None, do_holdout_testing=True):
        """
        Train the network for the specified number of epochs, etc.
        Return representation snapshots, training reports, and snapshot/report epochs.
        """
        if do_holdout_testing:
            items_used_inds, ctx_used_inds, x_used_inds = self.prepare_holdout()
        else:
            items_used_inds = np.arange(self.n_items)
            ctx_used_inds = np.arange(self.n_contexts)
            x_used_inds = np.arange(self.n_inputs)

        n_used = len(x_used_inds)
        gentest_target_inds = np.setdiff1d(range(self.n_inputs), x_used_inds.cpu())

        snap_epochs, epoch_digits, irep_snaps, crep_snaps, zero_items, zero_contexts = self.prepare_snapshots(
            snap_freq, snap_freq_scale, num_epochs, do_holdout_testing
        )
        n_snaps = len(snap_epochs)

        n_report = (num_epochs-1) // report_freq + 1
        reports = dict()
        reports['loss'] = np.zeros(n_report)
        reports['accuracy'] = np.zeros(n_report)
        if do_holdout_testing:
            reports['etg'] = np.zeros(n_report, dtype=int) # "epochs to generalize"

        for epoch in range(num_epochs):

            # collect snapshot
            k_snap = np.nonzero(snap_epochs == epoch)
            if len(k_snap) == 1:
                k_snap = k_snap[0]

                with torch.no_grad():
                    irep_snaps[k_snap] = self.calc_rep(self.items[items_used_inds], zero_contexts)
                    crep_snaps[k_snap] = self.calc_rep(zero_items, self.contexts[ctx_used_inds])

            # do training
            order = dd.choose_k(x_used_inds, n_used)
            loss, acc_each = self.train_epoch(order, batch_size, optimizer)
            if scheduler is not None:
                scheduler.step()

            # report progress
            if epoch % report_freq == 0:
                k_report = epoch // report_freq
                
                with torch.no_grad():
                    mean_loss = loss.item() / n_used
                    mean_acc = torch.nansum(acc_each).item() / n_used

                report_str = f'Epoch {epoch:{epoch_digits}d} end: loss = {mean_loss:7.3f}, acc = {mean_acc:.3f}'

                reports['loss'][k_report] = mean_loss
                reports['accuracy'][k_report] = mean_acc

                if do_holdout_testing:
                    etg, etg_string = self.generalize_test(batch_size, optimizer, gentest_target_inds)
                    report_str += f', epochs to gen. = {etg_string:5}'
                    reports['etg'][k_report] = etg
                    
                print(report_str)

        # Bring snapshots to CPU and fill held out item and context entries with nans
        snaps = dict()
        snaps['item'] = np.full((n_snaps, self.n_items, self.rep_size), np.nan)
        snaps['item'][:, items_used_inds, :] = irep_snaps.cpu().numpy()
        snaps['context'] = np.full((n_snaps, self.n_contexts, self.rep_size), np.nan)
        snaps['context'][:, ctx_used_inds, :] = crep_snaps.cpu().numpy()

        return snaps, snap_epochs, reports, report_freq
