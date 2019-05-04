import copy
import numpy as np
from termcolor import colored

import torch
import torch.nn as nn
import torch.nn.functional as F


def _align_tensor_index(reference_index, tensor_index):

    where_in_tensor = []
    for i in reference_index:
        where = np.where(i == tensor_index)[0][0]
        where_in_tensor.append(where)
    return np.array(where_in_tensor)


def _sort_by_length(list_of_tensor, batch_length, return_idx=False):
    idx = np.argsort(np.array(copy.copy(batch_length)))[::-1]
    for i, tensor in enumerate(list_of_tensor):
        list_of_tensor[i] = [tensor[j] for j in idx]
    if return_idx:
        return list_of_tensor, idx
    else:
        return list_of_tensor


def _sort_by_index(list_of_tensor, idx):
    for i, tensor in enumerate(list_of_tensor):
        list_of_tensor[i] = [tensor[j] for j in idx]
    return list_of_tensor


class Sketch2Program(nn.Module):

    summary_keys = ['action_loss', 'object1_loss', 'object2_loss', 'total_loss',
                    'action_accuracy', 'object1_accuracy', 'object2_accuracy',
                    'attribute_precision', 'relation_precision', 'total_precision',
                    'attribute_recall', 'relation_recall', 'total_recall',
                    'attribute_f1', 'relation_f1', 'total_f1',
                    'lcs_score', 'parsibility', 'executability', 'schedule_sampling_p']

    def __init__(self, dset, **kwargs):

        from network.module import ProgramGraphClassifier, ProgramEncoder
        from network.graph_module import VanillaGraphEncoder
        from program.graph_utils import EnvGraphHelper
        super(Sketch2Program, self).__init__()

        embedding_dim = kwargs["embedding_dim"]
        sketch_hidden = kwargs["sketch_hidden"]
        program_hidden = kwargs["program_hidden"]
        max_words = kwargs["max_words"]
        graph_hidden = kwargs["graph_hidden"]
        graph_evolve = kwargs["graph_evolve"]
        n_timesteps = kwargs["n_timesteps"]
        model_type = kwargs["model_type"]

        env_graph_helper = EnvGraphHelper()
        env_graph_encoder = VanillaGraphEncoder(
            n_timesteps,
            env_graph_helper.n_edge,
            graph_hidden,
            embedding_dim,
            sketch_hidden)

        # init the submodules
        sketch_encoder = ProgramEncoder(
            dset, embedding_dim, sketch_hidden)

        kwargs = {}
        if graph_evolve:

            from program.graph_utils import InteractionGraphHelper
            if model_type == 'fc':
                from network.graph_module import FCActionGraphEncoder as ActionGraphEncoder
            elif model_type == 'gru':
                from network.graph_module import GRUActionGraphEncoder as ActionGraphEncoder
            elif model_type == 'residual':
                from network.graph_module import ResidualActionGraphEncoder as ActionGraphEncoder
            else:
                raise ValueError

            interaction_grpah_helper = InteractionGraphHelper(dset.action_dict)
            interaction_graph_encoder = ActionGraphEncoder(
                interaction_grpah_helper.n_edge,
                interaction_grpah_helper.n_touch,
                graph_hidden // 2,
                embedding_dim,
                sketch_hidden)
            kwargs.update({"interaction_grpah_helper": interaction_grpah_helper})
            kwargs.update({"interaction_graph_encoder": interaction_graph_encoder})

        program_decoder = ProgramGraphClassifier(
            dset,
            env_graph_encoder,
            env_graph_helper,
            sketch_hidden,
            embedding_dim,
            program_hidden,
            max_words,
            graph_evolve,
            **kwargs)

        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('sketch_encoder', sketch_encoder)
        all_modules.add_module('program_decoder', program_decoder)

        self.initial_program = dset.initial_program
        self.sketch_encoder = sketch_encoder
        self.program_decoder = program_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None

    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def forward(self, data, inference, use_constraint_mask=False, **kwargs):
        '''
            Note: The order of the `data` won't change in this function
        '''

        if inference:
            print(colored("Inference...", "cyan"))

        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)

        batch_sketch_length = data[1][1]

        # sort according to the length of sketch
        program = _sort_by_length(list(data[0]), batch_sketch_length)
        sketch = _sort_by_length(list(data[1]), batch_sketch_length)
        graph = _sort_by_length(list(data[2]), batch_sketch_length)

        # sketch encoder
        batch_data = sketch[1:]
        sketch_emb, _ = self.sketch_encoder(batch_data)

        # program decoder
        batch_program_length = program[1]
        batch_data, sort_idx = _sort_by_length(
            program, batch_program_length, return_idx=True)
        sketch_emb = torch.stack(_sort_by_index([sketch_emb], sort_idx)[0])
        batch_program_index = np.array(batch_data[0])
        batch_data = batch_data[1:]

        graph = _sort_by_index(graph, sort_idx)
        kwargs.update({"initial_program": self.initial_program})
        kwargs.update({"graph": graph})
        kwargs.update({"sketch_emb": sketch_emb})
        kwargs.update({"data": batch_data})

        loss, info = self.program_decoder(
            inference=inference, use_constraint_mask=use_constraint_mask, **kwargs)

        # align the output with the input `data`
        where_in_tensor = _align_tensor_index(data[0][0], batch_program_index)
        for k in info.keys():
            if k in ['batch_node_name_list', 'action_predict', 'object1_predict', 'object2_predict',
                     'action_loss_per_program', 'object1_loss_per_program', 'object2_loss_per_program',
                     'batch_node_state']:
                info[k] = [info[k][i] for i in where_in_tensor]

        program_pred = [info['action_predict'], info['object1_predict'],
                        info['object2_predict'], info['batch_node_name_list']]

        return loss, _, program_pred, info

    def write_summary(self, writer, info, postfix):

        model_name = 'Sketch2Program-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):

        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage,
                loc: storage))


class Desc2Sketch(nn.Module):

    summary_keys = ['action_loss', 'object1_loss', 'object2_loss', 'total_loss',
                    'action_accuracy', 'object1_accuracy', 'object2_accuracy',
                    'lcs_score', 'schedule_sampling_p']
     
    def __init__(self, dset, **kwargs):

        from network.module import ProgramDecoder, WordEncoder
        super(Desc2Sketch, self).__init__()
        self.use_title = True

        embedding_dim = 300
        sketch_hidden = kwargs["sketch_hidden"]
        desc_hidden = kwargs["desc_hidden"]

        # init the submodules
        from helper import CombinedEmbedding
        num_words = dset.num_words
        num_pretrained_words = dset.glove_embedding_vector.shape[0]
        pretrained_word_embedding = nn.Embedding.from_pretrained(torch.tensor(dset.glove_embedding_vector), freeze=True)
        word_embedding = nn.Embedding(num_words - num_pretrained_words, embedding_dim)
        word_embedding = CombinedEmbedding(pretrained_word_embedding, word_embedding)

        desc_encoder = WordEncoder(dset, 300, desc_hidden, word_embedding)
        input_dim = desc_hidden
        if self.use_title:
            title_encoder = WordEncoder(dset, 300, desc_hidden, word_embedding)
            self.title_encoder = title_encoder
            input_dim += desc_hidden

        sketch_decoder = ProgramDecoder(
            dset,
            input_dim,
            embedding_dim,
            sketch_hidden,
            kwargs["max_words"])

        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('desc_encoder', desc_encoder)
        all_modules.add_module('sketch_decoder', sketch_decoder)
        if self.use_title:
            all_modules.add_module('title_encoder', title_encoder)

        self.initial_sketch = dset.initial_sketch
        self.desc_encoder = desc_encoder
        self.sketch_decoder = sketch_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None

    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def forward(self, data, inference, **kwargs):

        '''
            Note: The order of the `data` won't change within this function
        '''
        if inference:
            print(colored("Inference...", "cyan"))

        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)

        ###  Description
        # sort according to the length of description
        batch_desc_length = data[0][1]

        desc = _sort_by_length(list(data[0]), batch_desc_length)
        title = _sort_by_length(list(data[2]), batch_desc_length)
        sketch = _sort_by_length(list(data[1]), batch_desc_length)

        # desc encoder
        batch_data = desc[1:]
        desc_emb, _ = self.desc_encoder(batch_data)
        input_emb = desc_emb

        if self.use_title:
            ### Title
            # sort according to the length of title
            batch_title_length = title[1]
            title, sort_idx = _sort_by_length(title, batch_title_length, return_idx=True)
            sketch = _sort_by_index(sketch, sort_idx)
            input_emb = torch.stack(_sort_by_index([input_emb], sort_idx)[0])

            # title encoder
            batch_data = title[1:]
            title_emb, _ = self.title_encoder(batch_data)
            input_emb = torch.cat([input_emb, title_emb], 1)

        # sketch decoder
        batch_sketch_length = sketch[1]
        batch_data, sort_idx = _sort_by_length(sketch, batch_sketch_length, return_idx=True)
        
        input_emb = torch.stack(_sort_by_index([input_emb], sort_idx)[0])
        batch_sketch_index = np.array(batch_data[0])
        batch_data = batch_data[1:]

        kwargs.update({'input_emb': input_emb})
        kwargs.update({'data': batch_data})
        kwargs.update({'initial_sketch': self.initial_sketch})
        loss, info = self.sketch_decoder(inference=inference, **kwargs)

        # align the output with the input `data`
        where_in_tensor = _align_tensor_index(data[1][0], batch_sketch_index)

        for k in info.keys():
            if k in ['action_predict', 'object1_predict', 'object2_predict', 
                'action_loss_per_program', 'object1_loss_per_program', 'object2_loss_per_program']:
                info[k] = [info[k][i] for i in where_in_tensor]

        sketch_pred = [info['action_predict'], info['object1_predict'],
                        info['object2_predict']]

        return loss, sketch_pred, info

    def write_summary(self, writer, info, postfix):

        model_name = 'Desc2Sketch-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):
        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage,
                loc: storage))


class Demo2Sketch(nn.Module):

    summary_keys = ['action_loss', 'object1_loss', 'object2_loss', 'total_loss',
                    'action_accuracy', 'object1_accuracy', 'object2_accuracy',
                    'lcs_score', 'schedule_sampling_p']
                    
    def __init__(self, dset, **kwargs):

        from network.module import ProgramDecoder
        super(Demo2Sketch, self).__init__()

        embedding_dim = kwargs["embedding_dim"]
        sketch_hidden = kwargs["sketch_hidden"]
        demo_hidden = kwargs["demo_hidden"]
        model_type = kwargs["model_type"]

        # init the submodules
        if model_type.lower() == 'max':
            from network.module import DemoEncoder
            demo_encoder = DemoEncoder(dset, demo_hidden, 'max')
        elif model_type.lower() == 'avg':
            from network.module import DemoEncoder
            demo_encoder = DemoEncoder(dset, demo_hidden, 'avg')
        else:
            raise ValueError
        demo_encoder = torch.nn.DataParallel(demo_encoder)

        sketch_decoder = ProgramDecoder(
            dset,
            demo_hidden,
            embedding_dim,
            sketch_hidden,
            kwargs["max_words"])

        # for quick save and load
        all_modules = nn.Sequential()
        all_modules.add_module('demo_encoder', demo_encoder)
        all_modules.add_module('sketch_decoder', sketch_decoder)

        self.initial_sketch = dset.initial_sketch
        self.demo_encoder = demo_encoder
        self.sketch_decoder = sketch_decoder
        self.all_modules = all_modules
        self.to_cuda_fn = None

    def set_to_cuda_fn(self, to_cuda_fn):
        self.to_cuda_fn = to_cuda_fn

    def forward(self, data, inference, **kwargs):
        '''
            Note: The order of the `data` won't change in this function
        '''
        if inference:
            print(colored("Inference...", "cyan"))

        if self.to_cuda_fn:
            data = self.to_cuda_fn(data)

        # demonstration
        # sort according to the length of description
        batch_demo_length = data[0][1]

        demo = _sort_by_length(list(data[0]), batch_demo_length)
        sketch = _sort_by_length(list(data[1]), batch_demo_length)

        # demonstration encoder
        batch_length, batch_demo_rgb, batch_demo_seg = demo[1:]
        demo_emb, _ = self.demo_encoder(batch_length, batch_demo_rgb, batch_demo_seg)

        # sketch decoder
        batch_sketch_length = sketch[1]
        batch_data, sort_idx = _sort_by_length(
            sketch, batch_sketch_length, return_idx=True)
        demo_emb = torch.stack(_sort_by_index([demo_emb], sort_idx)[0])

        batch_sketch_index = np.array(batch_data[0])
        batch_data = batch_data[1:]

        kwargs.update({'input_emb': demo_emb})
        kwargs.update({'data': batch_data})
        kwargs.update({'initial_sketch': self.initial_sketch})
        loss, info = self.sketch_decoder(inference=inference, **kwargs)

        # align the output with the input `data`
        where_in_tensor = _align_tensor_index(data[1][0], batch_sketch_index)

        for k in info.keys():
            if k in ['action_predict', 'object1_predict', 'object2_predict', 
                'action_loss_per_program', 'object1_loss_per_program', 'object2_loss_per_program']:
                info[k] = [info[k][i] for i in where_in_tensor]

        sketch_pred = [info['action_predict'], info['object1_predict'],
                        info['object2_predict']]

        return loss, sketch_pred, info
                    
    def write_summary(self, writer, info, postfix):

        model_name = 'Demo2Sketch-{}/'.format(postfix)
        for k in self.summary_keys:
            if k in info.keys():
                writer.scalar_summary(model_name + k, info[k])

    def save(self, path, verbose=False):
        
        if verbose:
            print(colored('[*] Save model at {}'.format(path), 'magenta'))
        torch.save(self.all_modules.state_dict(), path)

    def load(self, path, verbose=False):

        if verbose:
            print(colored('[*] Load model at {}'.format(path), 'magenta'))
        self.all_modules.load_state_dict(
            torch.load(
                path,
                map_location=lambda storage,
                loc: storage))
