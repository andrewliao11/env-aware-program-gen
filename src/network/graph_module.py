import torch
import torch.nn as nn

from program.graph_utils import *
from helper import fc_block, LayerNormGRUCell


# helper class for GraphEncoder
class AttrProxy(object):
    """
    Translates index lookups into attribute lookups.
    To implement some trick which able to use list of nn.Module in a nn.Module
    see https://discuss.pytorch.org/t/list-of-nn-module-in-a-nn-module/219/2
    """

    def __init__(self, module, prefix):
        self.module = module
        self.prefix = prefix

    def __getitem__(self, i):
        return getattr(self.module, self.prefix + str(i))


class VanillaGraphEncoder(nn.Module):

    def __init__(
            self,
            n_timesteps,
            n_edge_types,
            graph_hidden,
            embedding_dim,
            hidden):

        super(VanillaGraphEncoder, self).__init__()

        layernorm = True
        self.n_timesteps = n_timesteps
        self.n_edge_types = n_edge_types
        self.embedding_dim = embedding_dim
        self.input_dim = n_edge_types + embedding_dim
        self.graph_hidden = graph_hidden

        node_init2hidden = nn.Sequential()
        node_init2hidden.add_module(
            'fc1',
            fc_block(
                3 * embedding_dim,
                graph_hidden,
                False,
                nn.Tanh))
        node_init2hidden.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))

        for i in range(n_edge_types):
            hidden2message_in = fc_block(
                graph_hidden, graph_hidden, False, nn.Tanh)
            self.add_module(
                "hidden2message_in_{}".format(i),
                hidden2message_in)

            hidden2message_out = fc_block(
                graph_hidden, graph_hidden, False, nn.Tanh)
            self.add_module(
                "hidden2message_out_{}".format(i),
                hidden2message_out)

        if layernorm:
            self.gru_cell = LayerNormGRUCell
        else:
            self.gru_cell = nn.GRUCell

        propagator = self.gru_cell(
            input_size=2 * n_edge_types * graph_hidden,
            hidden_size=graph_hidden)

        self.node_init2hidden = node_init2hidden
        self.hidden2message_in = AttrProxy(self, "hidden2message_in_")
        self.hidden2message_out = AttrProxy(self, "hidden2message_out_")
        self.propagator = propagator

    def forward(
            self,
            edge_adjacency_matrix,
            node_state_prev,
            related_mask=None):
        """edge_adjacency_matrix: e, b, v, v
            object_state_arry: b, v, p
            state: b, v, h
        """

        B, V, H = node_state_prev.size()
        node_state_prev = node_state_prev.view(B * V, -1)
        node_state = node_state_prev

        edge_adjacency_matrix = edge_adjacency_matrix.float()
        edge_adjacency_matrix_out = edge_adjacency_matrix
        # convert the outgoing edges to incoming edges
        edge_adjacency_matrix_in = edge_adjacency_matrix.permute(0, 1, 3, 2)

        for i in range(self.n_timesteps):
            message_out = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_out[j](
                    node_state)    # b*v, h
                node_state_hidden = node_state_hidden.view(B, V, -1)
                message_out.append(
                    torch.bmm(
                        edge_adjacency_matrix_out[j],
                        node_state_hidden))  # b, v, h

            # concatenate the message from each edges
            message_out = torch.stack(message_out, 2)       # b, v, e, h
            message_out = message_out.view(B * V, -1)         # b, v, e*h

            message_in = []
            for j in range(self.n_edge_types):
                node_state_hidden = self.hidden2message_in[j](
                    node_state)    # b*v, h
                node_state_hidden = node_state_hidden.view(B, V, -1)
                message_in.append(
                    torch.bmm(
                        edge_adjacency_matrix_in[j],
                        node_state_hidden))

            # concatenate the message from each edges
            message_in = torch.stack(message_in, 2)         # b, v, e, h
            message_in = message_in.view(B * V, -1)           # b, v, e*h

            message = torch.cat([message_out, message_in], 1)
            node_state = self.propagator(message, node_state)

        if related_mask is not None:
            # mask out un-related changes
            related_mask_expand = related_mask.unsqueeze(
                2).repeat(1, 1, self.graph_hidden).float()
            related_mask_expand = related_mask_expand.view(B * V, -1)
            node_state = node_state * related_mask_expand + \
                node_state_prev * (-related_mask_expand + 1)

        node_state = node_state.view(B, V, -1)
        return node_state


class ResidualActionGraphEncoder(VanillaGraphEncoder):

    def __init__(
            self,
            n_edge_types,
            n_touch,
            graph_hidden,
            embedding_dim,
            hidden):

        super(
            ResidualActionGraphEncoder,
            self).__init__(
            0,
            n_edge_types,
            graph_hidden,
            embedding_dim,
            hidden)

        self.n_touch = n_touch

        action2hidden = nn.Sequential()
        action2hidden.add_module(
            'fc1',
            fc_block(
                embedding_dim + n_touch,
                graph_hidden,
                False,
                nn.Tanh))
        action2hidden.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))

        compute_residual = nn.Sequential()
        compute_residual.add_module(
            'fc1',
            fc_block(
                2 * graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))
        compute_residual.add_module(
            'fc2',
            fc_block(
                graph_hidden,
                graph_hidden,
                False,
                nn.Tanh))
        self.compute_residual = compute_residual

        self.action2hidden = action2hidden

    def action_applier(
            self,
            action_embedding,
            batch_touch_idx,
            batch_node_state_prev,
            batch_touch_mask):
        """
            action_embedding: b, emb
            batch_touch_idx: b, n, touch_type,
            batch_node_state_prev: b, n, h
            batch_touch_mask: b, n
        """

        B, N, _ = batch_touch_idx.size()

        action_embedding = action_embedding.unsqueeze(1).repeat(1, N, 1)
        graph_input = torch.cat([action_embedding, batch_touch_idx], 2)
        graph_input = self.action2hidden(graph_input)
        graph_input = graph_input.view(B * N, -1)

        batch_node_state_prev = batch_node_state_prev.view(B * N, -1)
        residual = self.compute_residual(
            torch.cat([graph_input, batch_node_state_prev], 1))

        batch_touch_mask = batch_touch_mask.unsqueeze(
            2).repeat(1, 1, self.graph_hidden)
        batch_touch_mask = batch_touch_mask.view(B * N, -1)

        batch_node_state = batch_node_state_prev + residual * batch_touch_mask
        batch_node_state = batch_node_state.view(B, N, -1)

        return batch_node_state


class FCActionGraphEncoder(VanillaGraphEncoder):

    def __init__(
            self, 
            n_edge_types, 
            n_touch, 
            graph_hidden, 
            embedding_dim, 
            hidden):

        super(FCActionGraphEncoder, 
            self).__init__(
            0, 
            n_edge_types, 
            graph_hidden, 
            embedding_dim, 
            hidden)

        self.n_touch = n_touch

        action2hidden = nn.Sequential()
        action2hidden.add_module('fc1', fc_block(embedding_dim + n_touch, graph_hidden, False, nn.Tanh))
        action2hidden.add_module('fc2', fc_block(graph_hidden, graph_hidden, False, nn.Tanh))

        compute_residual = nn.Sequential()
        compute_residual.add_module('fc1', fc_block(2*graph_hidden, graph_hidden, False, nn.Tanh))
        compute_residual.add_module('fc2', fc_block(graph_hidden, graph_hidden, False, nn.Tanh))
        self.compute_residual = compute_residual

        self.action2hidden = action2hidden

        
    def action_applier(
            self, 
            action_embedding,
            batch_touch_idx, 
            batch_node_state_prev, 
            batch_touch_mask):
        """
            action_embedding: b, emb
            batch_touch_idx: b, n, touch_type, 
            batch_node_state_prev: b, n, h
            batch_touch_mask: b, n
        """

        B, N, _ = batch_touch_idx.size()

        action_embedding = action_embedding.unsqueeze(1).repeat(1, N, 1)
        graph_input = torch.cat([action_embedding, batch_touch_idx], 2)
        graph_input = self.action2hidden(graph_input)
        graph_input = graph_input.view(B * N, -1)

        batch_node_state_prev = batch_node_state_prev.view(B * N, -1)
        batch_node_state = self.compute_residual(torch.cat([graph_input, batch_node_state_prev], 1))

        batch_touch_mask = batch_touch_mask.unsqueeze(2).repeat(1, 1, self.graph_hidden)
        batch_touch_mask = batch_touch_mask.view(B * N, -1)

        batch_node_state = batch_node_state * batch_touch_mask + batch_node_state_prev * (-batch_touch_mask + 1)
        batch_node_state = batch_node_state.view(B, N, -1)
 
        return batch_node_state


class GRUActionGraphEncoder(VanillaGraphEncoder):

    def __init__(
            self, 
            n_edge_types, 
            n_touch, 
            graph_hidden, 
            embedding_dim, 
            hidden):

        super(
            GRUActionGraphEncoder, 
            self).__init__(
            0, 
            n_edge_types, 
            graph_hidden, 
            embedding_dim, 
            hidden)
            
        self.n_touch = n_touch

        action2hidden = nn.Sequential()
        action2hidden.add_module('fc1', fc_block(embedding_dim + n_touch, graph_hidden, False, nn.Tanh))
        action2hidden.add_module('fc2', fc_block(graph_hidden, graph_hidden, False, nn.Tanh))

        temporal_propagator = self.gru_cell(input_size=graph_hidden, hidden_size=graph_hidden)

        self.temporal_propagator = temporal_propagator

        self.action2hidden = action2hidden

        
    def action_applier(
            self, 
            action_embedding, 
            batch_touch_idx, 
            batch_node_state_prev, 
            batch_touch_mask):
        """
            action_embedding: b, emb
            batch_touch_idx: b, n, touch_type, 
            batch_node_state_prev: b, n, h
            batch_touch_mask: b, n
        """

        B, N, _ = batch_touch_idx.size()

        action_embedding = action_embedding.unsqueeze(1).repeat(1, N, 1)
        graph_input = torch.cat([action_embedding, batch_touch_idx], 2)
        graph_input = self.action2hidden(graph_input)
        graph_input = graph_input.view(B * N, -1)

        batch_node_state_prev = batch_node_state_prev.view(B * N, -1)
        batch_node_state = self.temporal_propagator(graph_input, batch_node_state_prev)

        batch_touch_mask = batch_touch_mask.unsqueeze(2).repeat(1, 1, self.graph_hidden)
        batch_touch_mask = batch_touch_mask.view(B * N, -1)
        batch_node_state = batch_node_state * batch_touch_mask + batch_node_state_prev * (-batch_touch_mask + 1)

        batch_node_state = batch_node_state.view(B, N, -1)
        
        return batch_node_state
