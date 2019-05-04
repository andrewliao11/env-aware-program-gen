import random
import itertools
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from helper import fc_block, Constant, LayerNormGRUCell


def _initialize_node_embedding_with_graph(
        n_nodes,
        node_name_list,
        node_category_list,
        batch_node_states,
        object_embedding,
        category_embedding,
        object_dict,
        node_state2_hidden,
        to_cuda):

    N = max(n_nodes)
    B = len(n_nodes)

    # extract object name from node
    batch_object_list = np.zeros([B, N]).astype(np.int64)
    for i, node_names in enumerate(node_name_list):
        for j, node in enumerate(node_names):
            name = node.split('.')[0]
            idx = int(object_dict.word2idx[name])
            batch_object_list[i, j] = idx

    batch_object_list = torch.tensor(batch_object_list, dtype=torch.int64)
    if to_cuda:
        batch_object_list = batch_object_list.cuda()

    batch_object_emb = object_embedding(batch_object_list.view(-1))
    batch_object_emb = batch_object_emb.view(B, N, -1)

    batch_category_list = torch.stack(node_category_list, 0)
    batch_category_emb = category_embedding(batch_category_list.view(-1).long())
    batch_category_emb = batch_category_emb.view(B, N, -1)

    batch_node_states = torch.stack(batch_node_states,0).float()        # b, n, n_states
    batch_node_states_emb = node_state2_hidden(batch_node_states.view(B * N, -1))
    batch_node_states_emb = batch_node_states_emb.view(B, N, -1)

    batch_node_emb = torch.cat([batch_category_emb, batch_object_emb, batch_node_states_emb], 2)
    return batch_node_emb, batch_object_list


def _calculate_accuracy(
        action_correct,
        object1_correct,
        object2_correct,
        batch_length,
        info):
    '''
        action_correct: torch tensor with shape (B, T)
    '''

    action_valid_correct = [sum(action_correct[i, :(l - 1)])
                            for i, l in enumerate(batch_length)]
    object1_valid_correct = [sum(object1_correct[i, :(l - 1)])
                             for i, l in enumerate(batch_length)]
    object2_valid_correct = [sum(object2_correct[i, :(l - 1)])
                             for i, l in enumerate(batch_length)]

    action_accuracy = sum(action_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
    object1_accuracy = sum(object1_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))
    object2_accuracy = sum(object2_valid_correct).float() / (sum(batch_length) - 1. * len(batch_length))

    info.update({'action_accuracy': action_accuracy.cpu().item()})
    info.update({'object1_accuracy': object1_accuracy.cpu().item()})
    info.update({'object2_accuracy': object2_accuracy.cpu().item()})


def _calculate_loss(
        action_loss,
        object1_loss,
        object2_loss,
        batch_length,
        info):
    '''
        action_loss: torch tensor with shape (B, T)
    '''

    action_valid_loss = [sum(action_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]
    object1_valid_loss = [sum(object1_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]
    object2_valid_loss = [sum(object2_loss[i, :(l - 1)]) for i, l in enumerate(batch_length)]

    info.update({"action_loss_per_program": \
            [loss.cpu().item() / (l - 1) for loss, l in zip(action_valid_loss, batch_length)]})
    info.update({"object1_loss_per_program": \
            [loss.cpu().item() / (l - 1) for loss, l in zip(object1_valid_loss, batch_length)]})
    info.update({"object2_loss_per_program": \
            [loss.cpu().item() / (l - 1) for loss, l in zip(object2_valid_loss, batch_length)]})

    action_valid_loss = sum(action_valid_loss) / (sum(batch_length) - 1. * len(batch_length))
    object1_valid_loss = sum(object1_valid_loss) / (sum(batch_length) - 1. * len(batch_length))
    object2_valid_loss = sum(object2_valid_loss) / (sum(batch_length) - 1. * len(batch_length))

    loss = action_valid_loss + object1_valid_loss + object2_valid_loss

    info.update({"action_loss": action_valid_loss.cpu().item()})
    info.update({"object1_loss": object1_valid_loss.cpu().item()})
    info.update({"object2_loss": object2_valid_loss.cpu().item()})
    info.update({'total_loss': loss.cpu().item()})

    return loss


class ProgramGraphClassifier(nn.Module):

    def __init__(
            self,
            dset,
            env_graph_encoder,
            env_graph_helper,
            sketch_embedding_dim,
            embedding_dim,
            hidden,
            max_words,
            graph_evolve,
            interaction_grpah_helper=None,
            interaction_graph_encoder=None):

        super(ProgramGraphClassifier, self).__init__()

        if graph_evolve:
            self.interaction_grpah_helper = interaction_grpah_helper
            self.interaction_graph_encoder = interaction_graph_encoder

            env_graph2interactoin_graph = nn.Sequential()
            env_graph2interactoin_graph.add_module(
                'fc1',
                fc_block(
                    env_graph_encoder.graph_hidden,
                    interaction_graph_encoder.graph_hidden,
                    False,
                    nn.Tanh))
            self.env_graph2interactoin_graph = env_graph2interactoin_graph
            graph_state_dim = interaction_graph_encoder.graph_hidden
        else:
            graph_state_dim = env_graph_encoder.graph_hidden

        program_classifier = ProgramClassifier(
            dset, sketch_embedding_dim, embedding_dim, hidden, max_words, graph_state_dim)

        category_embedding = nn.Embedding(
            env_graph_helper.n_node_category, embedding_dim)
        node_state2_hidden = fc_block(
            env_graph_helper.n_node_state,
            embedding_dim,
            False,
            nn.Tanh)

        self.hidden = hidden
        self.action_object_constraints = dset.compatibility_matrix
        self.object_dict = dset.object_dict
        self.env_graph_helper = env_graph_helper
        self.env_graph_encoder = env_graph_encoder
        self.program_classifier = program_classifier
        self.category_embedding = category_embedding
        self.node_state2_hidden = node_state2_hidden
        self.graph_evolve = graph_evolve

        graph_feat2hx = nn.Sequential()
        graph_feat2hx.add_module(
            'fc1',
            fc_block(
                graph_state_dim,
                hidden,
                False,
                nn.Tanh))
        graph_feat2hx.add_module(
            'fc2', fc_block(
                hidden, hidden, False, nn.Tanh))
        self.graph_feat2hx = graph_feat2hx

    def _pad_input(self, list_of_tensor):

        pad_tensor = []
        for tensor in list_of_tensor:
            tensor = pad_sequence(tensor, batch_first=True)   # B, T
            pad_tensor.append(tensor)

        return pad_tensor

    def _create_mask2d(self, B, N, length, value, to_cuda):
        mask = torch.zeros([B, N])
        if to_cuda:
            mask = mask.cuda()
        for i, l in enumerate(length):
            mask[i, l:] = value

        return mask

    def _decode_object(
            self,
            hidden,
            batch_node_mask,
            batch_node_state,
            hidden2logits):

        _, N = batch_node_mask.size()

        hidden = hidden.unsqueeze(1).repeat(1, N, 1)
        logits = hidden2logits(
            torch.cat([hidden, batch_node_state], 2)).squeeze()      # B, N
        logits = logits + batch_node_mask   # mask out invalid values

        pred = torch.argmax(logits, 1)

        return logits, pred

    def _decode_action(self, batch_action_mask,
                       node_predict_emb, hidden):

        # decode the action
        obj1_pred_emb_list = torch.cat(
            [hidden, node_predict_emb, ], 1)   # b, h+g_h
        logits = self.program_classifier.hidden2action_logits(
            obj1_pred_emb_list)
        logits = logits + batch_action_mask
        pred = torch.argmax(logits, 1)

        return logits, pred

    def _calculate_object_loss(
            self,
            B,
            T,
            object1_logits_list,
            object2_logits_list,
            batch_object1,
            batch_object2):

        object1_loss = F.cross_entropy(object1_logits_list.view(
            B * T, -1), batch_object1.contiguous().view(-1), reduction='none').view(B, T)
        object2_loss = F.cross_entropy(object2_logits_list.view(
            B * T, -1), batch_object2.contiguous().view(-1), reduction='none').view(B, T)

        return object1_loss, object2_loss

    def _calcualte_action_loss(self, B, T, action_logits_list, batch_action):

        action_loss = F.cross_entropy(action_logits_list.view(
            B * T, -1), batch_action.contiguous().view(-1), reduction='none')
        action_loss = action_loss.view(B, T)

        return action_loss

    def _initial_input(self, B, initial_program, to_cuda):

        action, object1, object2 = initial_program
        action = torch.tensor([action for i in range(B)])
        object1 = torch.tensor([object1 for i in range(B)])
        object2 = torch.tensor([object2 for i in range(B)])
        if to_cuda:
            action = action.cuda()
            object1 = object1.cuda()
            object2 = object2.cuda()

        return action, object1, object2

    def forward(self, inference, data, sketch_emb, graph, use_constraint_mask, **kwargs):

        '''
            E: number of edge tpyes
            B: batch size
            N: maximum number of nodes 
            H: hidden size
            EMB: embedding size
            A: action space
            O: object space
            T: maximum time steps
        '''

        ### Unpack variables
        batch_length, batch_action, batch_object1, batch_object2 = data
        batch_adjacency_matrix, batch_node_name_list, batch_node_category_list, batch_node_states, _, _ = graph

        batch_action, batch_object1, batch_object2 = self._pad_input([batch_action, batch_object1, batch_object2, ])
        sketch_emb = self.program_classifier.sketch_emb_encoder(sketch_emb)

        float_tensor_type = sketch_emb.dtype
        long_tensor_type = batch_action.dtype
        tensor_device = sketch_emb.device
        to_cuda = 'cuda' in batch_action[0].device.type
        batch_n_nodes = [len(node_name) for node_name in batch_node_name_list]
        B = len(batch_n_nodes)
        N = max(batch_n_nodes)

        ### Initialize variables
        action_logits_list, object1_logits_list, object2_logits_list = [], [], []
        action_correct_list, object1_correct_list, object2_correct_list = [], [], []
        action_predict_list, object1_predict_list, object2_predict_list = [], [], []

        action_object_constraints = [torch.tensor(x, dtype=float_tensor_type, device=tensor_device) \
                            for x in self.action_object_constraints]
        if to_cuda:
            action_object_constraints = [x.cuda() for x in action_object_constraints]

        # zero initialization
        # hx: B, H
        hx = torch.zeros([B, self.program_classifier.decode_gru_cell.hidden_size], 
                        dtype=float_tensor_type, device=tensor_device)

        ### Graph preparation
        # batch_adjacency_matrix: E, B, N, N
        batch_adjacency_matrix = torch.stack(batch_adjacency_matrix, 1)
        # batch_node_mask: B, N (used to mask out invalid nodes)
        batch_node_mask = self._create_mask2d(B, N, batch_n_nodes, -np.inf, to_cuda)
        # batch_node_states: B, N, 3*EMB
        batch_node_states, batch_object_list = _initialize_node_embedding_with_graph(
            batch_n_nodes,
            batch_node_name_list,
            batch_node_category_list,
            batch_node_states,
            self.program_classifier.object_embedding,
            self.category_embedding,
            self.object_dict,
            self.node_state2_hidden,
            to_cuda)

        # batch_node_states: B, N, H
        batch_node_states = self.env_graph_encoder.node_init2hidden(batch_node_states.view(B * N, -1))
        batch_node_states = batch_node_states.view(B, N, -1)
        batch_node_states = self.env_graph_encoder(batch_adjacency_matrix, batch_node_states)

        
        if self.graph_evolve:
            ### Apply action embedding to change the node states. We call this graph as `Interaction Graph`
            ### the `interaction_graph_helper` help us find which nodes should be changed based on the previous prediction
            ### if the previous action is `grab cup`, the `interaction_graph_helper` need to recognize that 
            ### the node states of `character` and `cup` should be changed

            # batch_node_states: B, N, H
            batch_node_states = self.env_graph2interactoin_graph(batch_node_states)
            self.interaction_grpah_helper.set_batch_node_name_list(batch_node_name_list)
            # batch_interaction_adjacency_matrix: E, B, N, N
            batch_interaction_adjacency_matrix = self.interaction_grpah_helper.batch_init_adjacency_matrix(B, N)
            batch_interaction_adjacency_matrix = torch.tensor(batch_interaction_adjacency_matrix)
            if to_cuda:
                batch_interaction_adjacency_matrix = batch_interaction_adjacency_matrix.cuda()

        if inference:
            action, _, _ = self._initial_input(B, kwargs["initial_program"], to_cuda)
            p = Constant(0.)
        else:
            p = kwargs['p']


        ### Main loop
        for i in range(batch_action.size(1) - 1):

            ### Check if we want to use the previous generated intruction or ground truth as input
            is_gt = random.random() > (1 - p.v)
            if i == 0 or is_gt:
                action_input = action if inference else batch_action[:, i]
                action_emb = self.program_classifier.action_embedding(action_input)
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object1[:, i]
                object1_emb = torch.index_select(batch_node_states.view(B * N, -1), 0, idx)
                idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + batch_object2[:, i]
                object2_emb = torch.index_select(batch_node_states.view(B * N, -1), 0, idx)
            else:
                action_input = action_predict
                action_emb = self.program_classifier.action_embedding(action_input)
                object1_emb = object1_predict_emb
                object2_emb = object2_predict_emb

            # atomic_action: B, 3*EMB
            atomic_action = torch.cat([action_emb, object1_emb, object2_emb], 1)
            atomic_action_emb = self.program_classifier.atomic_action2emb(
                atomic_action)

            if self.graph_evolve and i > 0:
                ### Update the interaction graph based on the previous generated/ground truth instruction
                if is_gt:
                    batch_interaction_adjacency_matrix, batch_touch_idx, batch_touch_mask, _ = self.interaction_grpah_helper.batch_graph_evolve(
                        action_input, batch_object1[:, i], batch_object2[:, i], batch_interaction_adjacency_matrix, to_cuda)
                else:
                    batch_interaction_adjacency_matrix, batch_touch_idx, batch_touch_mask, _ = self.interaction_grpah_helper.batch_graph_evolve(
                        action_input, object1_node_predict, object2_node_predict, batch_interaction_adjacency_matrix, to_cuda)

                batch_node_states = self.interaction_graph_encoder.action_applier(
                    action_emb, batch_touch_idx, batch_node_states, batch_touch_mask)


            ### Recurrent Nets
            gru_input = torch.cat([atomic_action_emb, sketch_emb], 1)
            hx = self.program_classifier.decode_gru_cell(gru_input, hx)
            hidden = self.program_classifier.gru2hidden(hx)

            
            ### Decode object1, action, object2 respectively
            # Object 1
            logits_object1_t, object1_node_predict = self._decode_object(
                        hidden, batch_node_mask, batch_node_states, self.program_classifier.hidden2object1_logits)
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object1_node_predict
            object1_predict_emb = torch.index_select(batch_node_states.view(B * N, -1), 0, idx)
            object1_predict_index = torch.index_select(batch_object_list.view(B * N, -1), 0, idx)

        
            # Action
            # batch_action_mask_constraints: A, B
            batch_action_mask_constraints = 1 - torch.index_select(action_object_constraints[0], 1, object1_predict_index.squeeze())
            batch_action_mask_constraints = batch_action_mask_constraints.transpose(1, 0)
            if use_constraint_mask:
                batch_action_mask_constraints[batch_action_mask_constraints == 1.] += -np.inf
            else:
                batch_action_mask_constraints = batch_action_mask_constraints * 0.

            action_logits_t, action_predict = self._decode_action(batch_action_mask_constraints, object1_predict_emb, hidden)
            action_predict_emb = self.program_classifier.action_embedding(action_predict)
            hidden_w_prev_action_object1 = torch.cat([hidden, action_predict_emb, object1_predict_emb], 1)


            # Object2
            if use_constraint_mask:
                # B, O
                batch_action_mask_constraints = torch.index_select(action_object_constraints[1], 0, action_predict.squeeze())
                num_objects = self.object_dict.n_words
                idx_obj = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * num_objects
                idx_obj = idx_obj[:, None] + batch_object_list
                idx_obj = idx_obj.view(-1)

                batch_action_mask_constraints = 1 - torch.index_select(batch_action_mask_constraints.view(-1), 0, idx_obj).view(B, N)
                batch_action_mask_constraints = batch_action_mask_constraints.float()
                batch_action_mask_constraints[batch_action_mask_constraints == 1.] += -np.inf
                batch_action_mask_constraints = torch.min(batch_node_mask, batch_action_mask_constraints)
            else:
                batch_action_mask_constraints = batch_node_mask

            logits_object2_t, object2_node_predict = self._decode_object(hidden_w_prev_action_object1, 
                        batch_action_mask_constraints, batch_node_states, self.program_classifier.hidden2object2_logits)
            idx = torch.arange(B, dtype=long_tensor_type, device=tensor_device) * N + object2_node_predict
            object2_predict_emb = torch.index_select(batch_node_states.view(B * N, -1), 0, idx)


            ### Append tensors
            action_logits_list.append(action_logits_t)
            object1_logits_list.append(logits_object1_t)
            object2_logits_list.append(logits_object2_t)

            action_predict_list.append(action_predict)
            object1_predict_list.append(object1_node_predict)
            object2_predict_list.append(object2_node_predict)

            action_correct = (action_predict == batch_action[:, i + 1])
            object1_correct = (object1_node_predict == batch_object1[:, i + 1])
            object2_correct = (object2_node_predict == batch_object2[:, i + 1])

            action_correct_list.append(action_correct)
            object1_correct_list.append(object1_correct)
            object2_correct_list.append(object2_correct)


        # B, T
        action_predict = torch.stack(action_predict_list, 1) 
        object1_predict = torch.stack(object1_predict_list, 1) 
        object2_predict = torch.stack(object2_predict_list, 1)

        info = {}
        info.update({"action_predict": action_predict.cpu()})
        info.update({"object1_predict": object1_predict.cpu()})
        info.update({"object2_predict": object2_predict.cpu()})
        info.update({"batch_node_name_list": batch_node_name_list})

        # B, T
        action_correct_list = torch.stack(action_correct_list, 1).float()
        object1_correct_list = torch.stack(object1_correct_list, 1).float()
        object2_correct_list = torch.stack(object2_correct_list, 1).float()
        _calculate_accuracy(
            action_correct_list,
            object1_correct_list,
            object2_correct_list,
            batch_length,
            info)


        # action_logits_list: B, T, A
        # object1_logits_list: B, T, N
        T = batch_object1.size(1)
        action_logits_list = torch.stack(action_logits_list, 1)
        object1_logits_list = torch.stack(object1_logits_list, 1)
        object2_logits_list = torch.stack(object2_logits_list, 1)
        
        object1_loss, object2_loss = self._calculate_object_loss(
            B, T - 1, object1_logits_list, object2_logits_list, batch_object1[:, 1:], batch_object2[:, 1:])
        action_loss = self._calcualte_action_loss(B, T - 1, action_logits_list, batch_action[:, 1:])

        loss = _calculate_loss(
            action_loss,
            object1_loss,
            object2_loss,
            batch_length,
            info)

        return loss, info


class ProgramClassifier(nn.Module):

    def __init__(
            self,
            dset,
            sketch_embedding_dim,
            embedding_dim,
            hidden,
            max_words,
            graph_state_dim):

        super(ProgramClassifier, self).__init__()

        layernorm = True
        num_actions = dset.num_actions
        num_objects = dset.num_objects

        # Encode sketch embedding
        sketch_emb_encoder = nn.Sequential()
        sketch_emb_encoder.add_module(
            'fc_block1', fc_block(sketch_embedding_dim, hidden, False, nn.Tanh))

        # Encode input words
        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)

        atomic_action2emb = nn.Sequential()
        atomic_action2emb.add_module(
            'fc_block1', fc_block(embedding_dim + 2 * graph_state_dim, hidden, False, nn.Tanh))
        atomic_action2emb.add_module(
            'fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        if layernorm:
            gru_cell = LayerNormGRUCell
        else:
            gru_cell = nn.GRUCell

        # gru
        decode_gru_cell = gru_cell(2 * hidden, hidden)

        # decode words
        gru2hidden = nn.Sequential()
        gru2hidden.add_module(
            'fc_block1', fc_block(hidden, hidden, False, nn.Tanh))
        gru2hidden.add_module(
            'fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        hidden2object1_logits = nn.Sequential()
        hidden2object1_logits.add_module(
            'fc_block1', fc_block(hidden + graph_state_dim, hidden, False, nn.Tanh))
        hidden2object1_logits.add_module(
            'fc_block2', fc_block(hidden, 1, False, None))

        hidden2action_logits = nn.Sequential()
        hidden2action_logits.add_module(
            'fc_block1', fc_block(hidden + graph_state_dim, hidden, False, nn.Tanh))
        hidden2action_logits.add_module(
            'fc_block2', fc_block(hidden, num_actions, False, None))

        hidden2object2_logits = nn.Sequential()
        hidden2object2_logits.add_module(
            'fc_block1', fc_block(hidden + 2 * graph_state_dim + embedding_dim, hidden, False, nn.Tanh))
        hidden2object2_logits.add_module(
            'fc_block2', fc_block(hidden, 1, False, None))

        self.embedding_dim = embedding_dim
        self.sketch_emb_encoder = sketch_emb_encoder
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding
        self.atomic_action2emb = atomic_action2emb
        self.decode_gru_cell = decode_gru_cell
        self.gru2hidden = gru2hidden
        self.hidden2object1_logits = hidden2object1_logits
        self.hidden2action_logits = hidden2action_logits
        self.hidden2object2_logits = hidden2object2_logits
        self.num_actions = num_actions
        self.max_words = max_words


# To encode the sketch or program
class ProgramEncoder(nn.Module):

    def __init__(self, dset, embedding_dim, hidden):
        super(ProgramEncoder, self).__init__()

        num_actions = dset.num_actions
        num_objects = dset.num_objects
        num_indexes = dset.num_indexes

        # Encode sketch
        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)
        index_embedding = nn.Embedding(num_indexes, embedding_dim)

        object_index2instance = nn.Sequential()
        object_index2instance.add_module(
            'fc1', fc_block(2 * embedding_dim, embedding_dim, False, nn.Tanh))
        object_index2instance.add_module(
            'fc2', fc_block(embedding_dim, embedding_dim, False, nn.Tanh))
        self.object_index2instance = object_index2instance

        atomic_action2emb = nn.Sequential()
        atomic_action2emb.add_module(
            'fc_block1', fc_block(embedding_dim + 2 * embedding_dim, hidden, False, nn.Tanh))
        atomic_action2emb.add_module(
            'fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        self.atomic_action2emb = atomic_action2emb

        sketch_encoding = nn.GRU(hidden, hidden, 2, batch_first=True)

        self.object_dict = dset.object_dict
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding
        self.index_embedding = index_embedding
        self.sketch_encoding = sketch_encoding

    def forward(self, data):

        '''
            B: batch size
            H: hidden size
            EMB: embedding size
            T: maximum time steps
        '''

        ### Unpack variables
        batch_length, batch_action, batch_object1, batch_object2, batch_index1, batch_index2 = data

        ### Initialize variables
        embeddings = []
        tensor_device = batch_action[0].device

        noise_size = self.index_embedding.weight.size()
        noise_dist = torch.distributions.normal.Normal(
            torch.zeros(noise_size, device=tensor_device), 
            torch.ones(noise_size, device=tensor_device))
        noise_vec = noise_dist.sample()


        ### Main loop
        for action, object1, object2, index1, index2 in zip(
                batch_action, batch_object1, batch_object2, batch_index1, batch_index2):

            action_emb = self.action_embedding(action)
            batch_object1 = self.object_embedding(object1)
            batch_object2 = self.object_embedding(object2)

            # use noise to differentiate different object instances
            batch_instance1 = batch_object1 + torch.index_select(noise_vec, 0, index1)
            batch_instance2 = batch_object2 + torch.index_select(noise_vec, 0, index2)

            atomic_action = torch.cat([action_emb, batch_instance1, batch_instance2], 1)
            atomic_action_emb = self.atomic_action2emb(atomic_action)

            embeddings.append(atomic_action_emb)

        # rnn_input: B, T, H
        rnn_input = pad_sequence(embeddings, batch_first=True)
        sketch_emb, _ = self.sketch_encoding(rnn_input)
        # [***] take the i^th output (assume we include sos, and eos)
        sketch_emb = torch.stack([sketch_emb[i, length - 2] for i, length in enumerate(batch_length)])

        return sketch_emb, embeddings


class ProgramDecoder(nn.Module):

    def __init__(
            self,
            dset,
            desc_hidden,
            embedding_dim,
            hidden,
            max_words):
        super(ProgramDecoder, self).__init__()

        layernorm = True
        num_actions = dset.num_actions
        num_objects = dset.num_objects

        # Encode sketch embedding
        input_emb_encoder = nn.Sequential()
        input_emb_encoder.add_module(
            'fc_block1', fc_block(desc_hidden, hidden, False, nn.Tanh))
        self.input_emb_encoder = input_emb_encoder

        # Encode input words
        action_embedding = nn.Embedding(num_actions, embedding_dim)
        object_embedding = nn.Embedding(num_objects, embedding_dim)
        self.action_embedding = action_embedding
        self.object_embedding = object_embedding

        atomic_action2emb = nn.Sequential()
        atomic_action2emb.add_module(
            'fc_block1', fc_block(3 * embedding_dim, hidden, False, nn.Tanh))
        atomic_action2emb.add_module(
            'fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        self.atomic_action2emb = atomic_action2emb

        if layernorm:
            gru_cell = LayerNormGRUCell
        else:
            gru_cell = nn.GRUCell
        decode_gru_cell = gru_cell(2 * hidden, hidden)

        gru2hidden = nn.Sequential()
        gru2hidden.add_module(
            'fc_block1', fc_block(hidden, hidden, False, nn.Tanh))
        gru2hidden.add_module(
            'fc_block2', fc_block(hidden, hidden, False, nn.Tanh))

        hidden2action = nn.Sequential()
        hidden2action.add_module(
            'fc_block1', fc_block(hidden, hidden, False, nn.Tanh))
        hidden2action.add_module(
            'fc_block2', fc_block(hidden, num_actions, False, None))

        hidden2object1 = nn.Sequential()
        hidden2object1.add_module(
            'fc_block1', fc_block(hidden + embedding_dim, hidden, False, nn.Tanh))
        hidden2object1.add_module(
            'fc_block2', fc_block(hidden, num_objects, False, None))

        hidden2object2 = nn.Sequential()
        hidden2object2.add_module(
            'fc_block1', fc_block(hidden + 2 * embedding_dim, hidden, False, nn.Tanh))
        hidden2object2.add_module(
            'fc_block2', fc_block(hidden, num_objects, False, None))

        self.embedding_dim = embedding_dim
        self.decode_gru_cell = decode_gru_cell
        self.gru2hidden = gru2hidden
        self.hidden2action = hidden2action
        self.hidden2object1 = hidden2object1
        self.hidden2object2 = hidden2object2
        self.max_words = max_words

    def _pad_input(self, list_of_tensor):

        pad_tensor = []
        for tensor in list_of_tensor:
            tensor = pad_sequence(tensor, batch_first=True)   # B, L
            pad_tensor.append(tensor)

        return pad_tensor

    def _initial_input(self, B, initial_program, to_cuda):

        action, object1, object2 = initial_program
        action = torch.tensor([action for i in range(B)])
        object1 = torch.tensor([object1 for i in range(B)])
        object2 = torch.tensor([object2 for i in range(B)])
        if to_cuda:
            action = action.cuda()
            object1 = object1.cuda()
            object2 = object2.cuda()

        return action, object1, object2

    def forward(self, inference, data, input_emb, **kwargs):

        batch_length, batch_action, batch_object1, batch_object2, _, _ = data
        batch_action, batch_object1, batch_object2 = self._pad_input(
            [batch_action, batch_object1, batch_object2])  # B, L

        input_emb = self.input_emb_encoder(input_emb)
        B = input_emb.size(0)

        if inference:
            to_cuda = 'cuda' in input_emb.device.type
            action, object1, object2 = self._initial_input(
                B, kwargs['initial_sketch'], to_cuda)
            p = Constant(0.)
        else:
            p = kwargs['p']

        action_loss_list, object1_loss_list, object2_loss_list = [], [], []
        action_correct_list, object1_correct_list, object2_correct_list = [], [], []
        action_predict_list, object1_predict_list, object2_predict_list = [], [], []

        hx = torch.zeros([B, self.decode_gru_cell.hidden_size],
                dtype=input_emb.dtype, device=input_emb.device)

        for i in range(batch_action.size(1) - 1):


            is_gt = random.random() > (1 - p.v)
            if i == 0:
                if not inference:
                    action = batch_action[:, i]
                    object1 = batch_object1[:, i]
                    object2 = batch_object2[:, i]
                    
                action_emb = self.action_embedding(action)
                object1_emb = self.object_embedding(object1)
                object2_emb = self.object_embedding(object2)

            elif is_gt:
                action = batch_action[:, i]
                object1 = batch_object1[:, i]
                object2 = batch_object2[:, i]

                action_emb = self.action_embedding(action)
                object1_emb = self.object_embedding(object1)
                object2_emb = self.object_embedding(object2)

            else:
                action_emb = self.action_embedding(action_predict)
                object1_emb = self.object_embedding(object1_predict)
                object2_emb = self.object_embedding(object2_predict)

            atomic_action = torch.cat([action_emb, object1_emb, object2_emb], 1)
            atomic_action_emb = self.atomic_action2emb(atomic_action)


            gru_input = torch.cat([atomic_action_emb, input_emb], 1)
            hx = self.decode_gru_cell(gru_input, hx)
            hidden = self.gru2hidden(hx)

            action_logits = self.hidden2action(hidden)
            action_predict = torch.argmax(action_logits, 1)
            action_predict_list.append(action_predict)

            object1_logits = self.hidden2object1(
                torch.cat([self.action_embedding(action_predict), hidden], 1))
            object1_predict = torch.argmax(object1_logits, 1)
            object1_predict_list.append(object1_predict)

            object2_logits = self.hidden2object2(torch.cat(
                [self.action_embedding(action_predict), self.object_embedding(object1_predict), hidden], 1))
            object2_predict = torch.argmax(object2_logits, 1)
            object2_predict_list.append(object2_predict)


            action_correct = (action_predict == batch_action[:, i + 1])
            object1_correct = (object1_predict == batch_object1[:, i + 1])
            object2_correct = (object2_predict == batch_object2[:, i + 1])

            action_correct_list.append(action_correct)
            object1_correct_list.append(object1_correct)
            object2_correct_list.append(object2_correct)

            action_loss = F.cross_entropy(action_logits, batch_action[:, i + 1].long(), 
                                            reduction='none')
            object1_loss = F.cross_entropy(object1_logits, batch_object1[:, i + 1].long(), 
                                            reduction='none')
            object2_loss = F.cross_entropy(object2_logits, batch_object2[:, i + 1].long(), 
                                            reduction='none')

            action_loss_list.append(action_loss)
            object1_loss_list.append(object1_loss)
            object2_loss_list.append(object2_loss)


        info = {}
        action_predict_list = torch.stack(action_predict_list, 1)
        object1_predict_list = torch.stack(object1_predict_list, 1)
        object2_predict_list = torch.stack(object2_predict_list, 1)
        info.update({"action_predict": action_predict_list.cpu()})
        info.update({"object1_predict": object1_predict_list.cpu()})
        info.update({"object2_predict": object2_predict_list.cpu()})


        action_correct_list = torch.stack(action_correct_list, 1).float()   # B, T
        object1_correct_list = torch.stack(object1_correct_list, 1).float() # B, T
        object2_correct_list = torch.stack(object2_correct_list, 1).float() # B, T
        # calculate accuracy
        _calculate_accuracy(
            action_correct_list,
            object1_correct_list,
            object2_correct_list,
            batch_length,
            info)

        # calculate loss
        action_loss_list = torch.stack(action_loss_list, 1)     # B, T
        object1_loss_list = torch.stack(object1_loss_list, 1)   # B, T
        object2_loss_list = torch.stack(object2_loss_list, 1)   # B, T
        loss = _calculate_loss(
                action_loss_list,
                object1_loss_list,
                object2_loss_list,
                batch_length,
                info)

        return loss, info


class WordEncoder(nn.Module):

    def __init__(self, dset, embedding_dim, desc_hidden, word_embedding):
    
        super(WordEncoder, self).__init__()

        emb2hidden = nn.Sequential()
        emb2hidden.add_module('fc_block1', fc_block(embedding_dim, desc_hidden, True, nn.Tanh))
        emb2hidden.add_module('fc_block2', fc_block(desc_hidden, desc_hidden, True, nn.Tanh))
        self.emb2hidden = emb2hidden

        word_encoding = nn.GRU(desc_hidden, desc_hidden, 2)

        self.object_dict = dset.object_dict
        self.word_embedding = word_embedding
        self.word_encoding = word_encoding


    def forward(self, batch_data):

        batch_length, batch_words = batch_data
        embeddings = []
        for words in batch_words:

            word_emb = self.word_embedding(words)
            word_emb = self.emb2hidden(word_emb)
            embeddings.append(word_emb)

        # rnn_input: B, T, H
        rnn_input = pad_sequence(embeddings, batch_first=True)
        word_emb, _ = self.word_encoding(rnn_input)
        # [***] take the i^th output (assume we include sos, and eos)
        word_emb = torch.stack([word_emb[i, length-2]  for i, length in enumerate(batch_length)])

        return word_emb, embeddings


class DemoEncoder(nn.Module):

    def __init__(self, dset, demo_hidden, pooling):

        super(DemoEncoder, self).__init__()

        # Image encoder
        import torchvision
        resnet = torchvision.models.resnet18(pretrained=True)
        resnet = list(resnet.children())[:-1]

        self.resnet = nn.Sequential(*resnet)

        self.seg_map = SegmapCNN()
        self.seg_feat2hidden = nn.Sequential()
        self.seg_feat2hidden.add_module(
            'fc_block1', fc_block(
                4160, demo_hidden, False, nn.ReLU))

        feat2hidden = nn.Sequential()
        feat2hidden.add_module(
            'fc_block1',
            fc_block(512, demo_hidden, False, nn.ReLU))
        self.feat2hidden = feat2hidden

        self.combine = nn.Sequential()
        self.combine.add_module(
            'fc_block1',
            fc_block(
                2 * demo_hidden,
                demo_hidden,
                False,
                nn.ReLU))

        self.pooling = pooling

    def forward(self, batch_length, batch_demo_rgb, batch_demo_seg):

        # do cnn forwarding once
        stacked_demo = [torch.stack(demo, 0) for demo in batch_demo_rgb]
        stacked_demo = torch.cat(stacked_demo, 0)
        stacked_demo_feat = self.resnet(stacked_demo)
        B, C, H, W = stacked_demo_feat.size()
        stacked_demo_feat = stacked_demo_feat.view(B, -1)

        stacked_demo_feat = self.feat2hidden(stacked_demo_feat)

        batch_demo_rgb_feat = []
        start = 0
        for length in batch_length:
            feat = stacked_demo_feat[start:(start + length), :]
            if len(feat.size()) == 3:
                feat = feat.unsqueeze(0)
            start += length
            if self.pooling == 'max':
                feat = torch.max(feat, 0)[0]
            elif self.pooling == 'avg':
                feat = torch.mean(feat, 0)
            else:
                raise ValueError

            batch_demo_rgb_feat.append(feat)

        # do cnn forwarding once
        stacked_demo = [torch.stack(demo, 0) for demo in batch_demo_seg]
        stacked_demo = torch.cat(stacked_demo, 0)

        stacked_demo_feat = self.seg_map(stacked_demo)
        stacked_demo_feat = self.seg_feat2hidden(stacked_demo_feat)
        batch_demo_seg_feat = []
        start = 0
        for length in batch_length:
            feat = stacked_demo_feat[start:(start + length), :]
            if len(feat.size()) == 3:
                feat = feat.unsqueeze(0)
            start += length
            if self.pooling == 'max':
                feat = torch.max(feat, 0)[0]
            elif self.pooling == 'avg':
                feat = torch.mean(feat, 0)
            batch_demo_seg_feat.append(feat)

        batch_demo_feat = []
        for rgb, seg in zip(batch_demo_rgb_feat, batch_demo_seg_feat):
            batch_demo_feat.append(self.combine(torch.cat([rgb, seg])))

        demo_emb = torch.stack(batch_demo_feat, 0)

        return demo_emb, batch_demo_feat


class SegmapCNN(nn.Module):

    def __init__(self):

        super(SegmapCNN, self).__init__()

        self.embeddings = nn.Embedding(190, 100)
        self.layer1 = nn.Sequential(
            nn.Conv2d(100, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=3, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x):

        # x: B, 1, H, W
        out = self.embeddings(x.long()).squeeze()
        # out: B, H, W, C
        out = self.layer1(out.permute(0, 3, 1, 2))
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        return out

