import os
import re
import json
import copy
import random
import string
import numpy as np
from glob import glob
from tqdm import tqdm


patt_action = r'\[(.+?)\]'
patt_object = r'\<(.+?)\>'
patt_number = r'\((.+?)\)'
graph_dir_name = ['TrimmedTestScene{}_graph'.format(i + 1) for i in range(7)]


def add_ground_truth(p, node_name_list):
    """ Add ground truth node for each time step. Note that ground truth doesnt the start of string token and end of string token
    """

    object1_gt = []
    object2_gt = []

    object1_str_list = p["object1_str_list"]
    instance1_str_list = p["instance1_str_list"]
    object2_str_list = p["object2_str_list"]
    instance2_str_list = p["instance2_str_list"]

    for object1_str, instance1_str in zip(object1_str_list, instance1_str_list):
        node_name = '{}.{}'.format(object1_str, instance1_str)
        idx = node_name_list.index(node_name)
        object1_gt.append(idx)

    for object2_str, instance2_str in zip(
            object2_str_list, instance2_str_list):
        node_name = '{}.{}'.format(object2_str, instance2_str)
        idx = node_name_list.index(node_name)
        object2_gt.append(idx)

    return object1_gt, object2_gt


def convert2idx_program(data, object_merged, action_dict, object_dict):

    detail2abstract = load_detail2abstract(object_merged)

    _data = []
    for case in data:
        action_str_list = []
        object1_str_list = []
        object2_str_list = []
        instance1_str_list = []
        instance2_str_list = []

        action_list = []
        object1_list = []
        object2_list = []

        program_list = case['program_list']
        for atomic_program in program_list:

            action = re.search(
                patt_action,
                atomic_program).group(1).lower().replace(
                ' ',
                '_')
            action_str_list.append(action)
            action_idx = int(action_dict.word2idx[action])
            action_list.append(action_idx)

            try:
                object = re.search(
                    patt_object, atomic_program).group(1).lower().replace(' ', '_')
                if object in detail2abstract.keys():
                    object = detail2abstract[object]

                object1_str_list.append(object)
                object1_idx = int(object_dict.word2idx[object])
                object1_list.append(object1_idx)
                if len(atomic_program.split('<')) > 2:
                    object = atomic_program.split(') ')[1]
                    object = re.search(
                        patt_object, object).group(1).lower().replace(' ', '_')
                    if object in detail2abstract.keys():
                        object = detail2abstract[object]

                    object2_str_list.append(object)
                    object2_idx = int(object_dict.word2idx[object])
                    object2_list.append(object2_idx)
                else:
                    object2_str_list.append('<none>')
                    object2_idx = int(object_dict.word2idx['<none>'])
                    object2_list.append(object2_idx)

            except AttributeError:
                object1_str_list.append('<none>')
                object1_idx = int(object_dict.word2idx['<none>'])
                object1_list.append(object1_idx)

                object2_str_list.append('<none>')
                object2_idx = int(object_dict.word2idx['<none>'])
                object2_list.append(object2_idx)

            try:
                number = re.search(
                    patt_number, atomic_program).group(1).lower().replace(' ', '_')
                instance1 = number.split('.')[1]
                instance1_str_list.append(instance1)

                number = number.split('.')[0]

                if len(atomic_program.split('<')) > 2:
                    number = atomic_program.split(') ')[1]
                    number = re.search(
                        patt_number, number).group(1).lower().replace(' ', '_')
                    instance2 = number.split('.')[1]
                    instance2_str_list.append(instance2)
                else:
                    instance2_str_list.append('<none>')

            except AttributeError:
                instance1_str_list.append('<none>')
                instance2_str_list.append('<none>')

        # index
        case.update({'action_str_list': action_str_list})
        case.update({'object1_str_list': object1_str_list})
        case.update({'object2_str_list': object2_str_list})
        case.update({'action_list': action_list})
        case.update({'object1_list': object1_list})
        case.update({'object2_list': object2_list})
        case.update({'instance1_str_list': instance1_str_list})
        case.update({'instance2_str_list': instance2_str_list})


def expand_dictionary_with_resources(object_dict, object_merged, resource_dir):

    detail2abstract = load_detail2abstract(object_merged)

    # base graphs
    base_graph_paths = glob(os.path.join(resource_dir, 'TrimmedTestScene*_graph.json'))
    for path in base_graph_paths:
        graph = json.load(open(path, 'r'))
        for node in graph["nodes"]:
            name = node["class_name"].lower().replace(' ', '_')
            object_dict.add_items(detail2abstract[name] if name in detail2abstract else name)

    # random objects
    object_placing = json.load(open(os.path.join(resource_dir, 'object_script_placing.json'), 'r'))
    for k, vs in object_placing.items():
        name = k.lower().replace(' ', '_')
        object_dict.add_items(detail2abstract[name] if name in detail2abstract else name)
        for v in vs:
            name = v["destination"].lower().replace(' ', '_')
            object_dict.add_items(detail2abstract[name] if name in detail2abstract else name)


def load_detail2abstract(object_merged):
    """ load the knowledge base containing the object alias
    """

    abstrct2detail = json.load(open(object_merged, 'r'))
    detail2abstract = {}
    for k, v in abstrct2detail.items():
        for i in v:
            detail2abstract.update({i: k})
    return detail2abstract


def get_program_dictionary(data, object_merged, verbose=False):

    detail2abstract = load_detail2abstract(object_merged)

    action_dict = dictionary()
    object_dict = dictionary()
    for case in data:
        program_list = case['program_list']
        for atomic_program in program_list:
            action = re.search(
                patt_action,
                atomic_program).group(1).lower().replace(
                ' ',
                '_')
            action_dict.add_items(action)

            try:
                object = re.search(
                    patt_object, atomic_program).group(1).lower().replace(' ', '_')
                if object in detail2abstract.keys():
                    object = detail2abstract[object]

                object_dict.add_items(object)
                if len(atomic_program.split('<')) > 2:
                    object = atomic_program.split(') ')[1]
                    object = re.search(
                        patt_object, object).group(1).lower().replace(' ', '_')
                    if object in detail2abstract.keys():
                        object = detail2abstract[object]
                    object_dict.add_items(object)
            except AttributeError:
                # not object in this atomic program
                continue

    return action_dict, object_dict


def convert_to_graph_path(program_path):

    program_path = os.path.abspath(program_path)
    graph_path = program_path.replace('txt','json')
    graph_path = graph_path.replace('executable_programs','init_and_final_graphs')
    if not os.path.exists(graph_path):
        for dir_name in graph_dir_name:
            if dir_name in graph_path:
                graph_path = graph_path.replace('/{}'.format(dir_name), '')
                break

    return graph_path


def load_programs(
        program_dir,
        object_merged=None,
        check_graph=False,
        debug=False):

    if isinstance(program_dir, list):
        program_txt_list = []
        for dir in program_dir:
            program_txt_list += glob(dir)
    else:
        program_txt_list = glob(program_dir)

    if debug:
        random.shuffle(program_txt_list)
        program_txt_list = program_txt_list[:1000]

    data_list = []
    print("Load the txt files")
    for program_txt in tqdm(program_txt_list):
        if check_graph:
            graph_path = convert_to_graph_path(program_txt)
            if not os.path.exists(graph_path):
                continue

        with open(program_txt, 'r') as f:
            program = f.read()

        
        program = program.split('\n')

        title = program[0]
        description = program[1]
        program_list = program[4:-1]

        data_dict = {
            "title": title,
            "description": description,
            "program_list": program_list,
            "file_path": program_txt
        }

        data_list.append(data_dict)

    return data_list


class dictionary():

    def __init__(self, word_list=None, no_special_words=False, init_dict=None):
        # special token: <EOS>, <NONE>, <SOS>
        self.word2idx = {}
        self.idx2word = {}
        self.pointer = 0
        if init_dict is not None:
            self.word2idx = init_dict.copy()
            self.idx2word = {v: k for k, v in self.word2idx.items()}
            self.pointer = len(self.word2idx.keys())
            self._update_metadata()
        else:
            if word_list is not None:
                for word in word_list:
                    self.add_items(word=word)

        if not no_special_words:
            self.add_special_tokens()
            self.eos = int(self.word2idx['<eos>'])
            self.sos = int(self.word2idx['<sos>'])
            self.none = int(self.word2idx['<none>'])

    def sample_words(self):

        words = copy.copy(self.words)
        words.remove('<eos>')
        words.remove('<sos>')
        words.remove('<none>')
        return random.sample(words, 1)

    def _update_metadata(self):
        self.words = list(self._get_words())
        self.n_words = len(self._get_words())

    def _get_words(self):
        return self.word2idx.keys()

    def add_special_tokens(self):
        special_tokens = ['<EOS>', '<NONE>', '<SOS>']
        for token in special_tokens:
            self.add_items(token)

    def check_pointer(self):
        idx_list = [int(i) for i in self.idx2word.keys()]
        idx_list = np.sort(idx_list)
        while self.pointer in idx_list:
            self.pointer += 1

    def add_items(self, word, idx=None):
        word = word.lower().replace(' ', '_')
        if word not in self.word2idx.keys():
            if idx is not None:
                assert idx not in self.word2idx.values()
                self.word2idx[str(word)] = str(idx)
                self.idx2word[str(idx)] = str(word)
            else:
                self.check_pointer()
                idx = self.pointer
                self.word2idx[str(word)] = str(idx)
                self.idx2word[str(idx)] = str(word)
                self.pointer += 1
        self._update_metadata()

    def visualize(self):
        idx_list = [int(i) for i in self.idx2word.keys()]
        idx_list = np.sort(idx_list)
        for idx in idx_list:
            print('{}: {}'.format(self.idx2word[str(idx)], idx))

    def convert_word2idx(self, l):
        temp = []
        for i in l:
            try:
                temp.append(int(self.word2idx[i]))
            except KeyError:
                print("Unknown words")
        return temp

    def convert_idx2word(self, l):
        temp = []
        for i in l:
            try:
                temp.append(self.idx2word[str(i)])
            except KeyError:
                print("Unknown words")
        return temp


def get_desc_tokens(programs):

    import nltk
    # remove punctuation
    for program in programs:
        for char in string.punctuation:
            if char not in [',', '.', '!', '?']:
                program["description"] = program["description"].replace(char, '')


    for program in programs:
        desc = program["description"]
        patt_match = re.search('[a-z]\.[a-z]', desc)
        while patt_match:
            match = patt_match.group(0)
            desc = desc.replace(match, '{} . {}'.format(match[0], match[2]))
            patt_match = re.search('[a-z]\.[a-z]', desc)

        patt_match = re.search('[a-z]\.[A-Z]', desc)
        while patt_match:
            match = patt_match.group(0)
            desc = desc.replace(match, '{} . {}'.format(match[0], match[2]))
            patt_match = re.search('[a-z]\.[A-Z]', desc)

        words = nltk.word_tokenize(desc)
        
        words = [word.lower() for word in filter(lambda v: v != '', words)]
        # replace number
        words = ['#' if word.isdigit() else word for word in words]
        program.update({'desc_words': words})


def get_title_tokens(programs):

    import nltk
    # remove punctuation
    for program in programs:
        for char in string.punctuation:
            if char not in [',', '.', '!', '?']:
                program["title"] = program["title"].replace(char, '')


    for program in programs:
        title = program["title"]
        patt_match = re.search('[a-z]\.[a-z]', title)
        while patt_match:
            match = patt_match.group(0)
            title = title.replace(match, '{} . {}'.format(match[0], match[2]))
            patt_match = re.search('[a-z]\.[a-z]', title)

        patt_match = re.search('[a-z]\.[A-Z]', title)
        while patt_match:
            match = patt_match.group(0)
            title = title.replace(match, '{} . {}'.format(match[0], match[2]))
            patt_match = re.search('[a-z]\.[A-Z]', title)

        words = nltk.word_tokenize(title)
        
        words = [word.lower() for word in filter(lambda v: v != '', words)]
        # replace number
        words = ['#' if word.isdigit() else word for word in words]
        program.update({'title_words': words})


def get_words(programs, existing_words=None):

    words = set()
    for program in programs:
        for word in program["desc_words"]:
            words.add(word)

    for program in programs:
        for word in program["title_words"]:
            words.add(word)

    return list(words)


def convert2idx_title(programs, dict):

    for program in programs:
        title_words = program["title_words"]
        title_words_idx = [int(dict.word2idx[word]) if word.lower() in dict.word2idx else int(dict.word2idx["<unk>"]) for word in title_words]
        program.update({"title_words_idx": title_words_idx})


def convert2idx_desc(programs, dict):

    for program in programs:
        desc_words = program["desc_words"]
        desc_words_idx = [int(dict.word2idx[word]) if word.lower() in dict.word2idx else int(dict.word2idx["<unk>"]) for word in desc_words]
        program.update({"desc_words_idx": desc_words_idx})
        
