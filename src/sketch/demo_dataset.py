import os
import random
import copy
import json
import re
import imageio
import numpy as np
from termcolor import colored
from glob import glob
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from dataset_utils import load_programs, get_program_dictionary, \
        expand_dictionary_with_resources, convert2idx_program, dictionary


cur_dir = os.path.dirname(os.path.realpath(__file__))
dataset_dir = os.path.join(cur_dir, '../../dataset/VirtualHome-Env/')
resource_dir = os.path.join(cur_dir, '../../dataset/VirtualHome-Env/resources/')

program_path = dataset_dir + 'original_programs/executable_programs/*/*/*txt'
augment_program_path = dataset_dir + 'augment_programs/*/executable_programs/*/*/*/*txt'
sketch_path = dataset_dir + 'sketch_annotation.json'
class2rgb_path = resource_dir + 'class2rgb.txt'

object_merged = resource_dir + 'object_merged.json'
train_patt = dataset_dir + 'split/train_progs_path.txt'
test_patt = dataset_dir + 'split/test_progs_path.txt'


# General function
def add_end(tensor, value):
    return tensor + [value]


def add_front(tensor, value):
    return [value] + tensor


################################
# Demonstration
################################

def get_dataset(args, train):

    programs = load_programs([augment_program_path,
                              program_path],
                             object_merged,
                             check_graph=True,
                             debug=args.debug)

    print("Total available of programs:", len(programs))
    programs = sorted(programs, key=lambda k: k['title'])

    if train:
        action_dict, object_dict = get_program_dictionary(
            programs, object_merged, verbose=args.verbose)
        expand_dictionary_with_resources(
            object_dict,
            object_merged,
            resource_dir=resource_dir)
    else:
        # use the dictionary saved during training
        _dict_path = os.path.dirname(args.checkpoint)
        _dict_path = os.path.join(_dict_path, 'dict.json')
        fp = open(_dict_path, 'r')
        _dict = json.load(fp)
        action_dict = dictionary(init_dict=_dict['action_dict'])
        object_dict = dictionary(init_dict=_dict['object_dict'])
        fp.close()

    convert2idx_program(programs, object_merged, action_dict, object_dict)

    train_dset = demo_dset(
        programs,
        action_dict,
        object_dict,
        is_train=True,
        max_length=args.max_words)
    test_dset = demo_dset(
        programs,
        action_dict,
        object_dict,
        is_train=False,
        max_length=args.max_words)

    return train_dset, test_dset


def collate_fn(data_list):

    demo_data = [data[0] for data in data_list]
    sketch_data = [data[1] for data in data_list]
    path_data = [data[2] for data in data_list]

    batch_sketch_length = [data[0] for data in sketch_data]
    batch_sketch_action = [data[1] for data in sketch_data]
    batch_sketch_object1 = [data[2] for data in sketch_data]
    batch_sketch_object2 = [data[3] for data in sketch_data]
    batch_sketch_index1 = [data[4] for data in sketch_data]
    batch_sketch_index2 = [data[5] for data in sketch_data]
    batch_sketch_index = np.arange(len(batch_sketch_length))

    batch_demo_length = [data[0] for data in demo_data]
    batch_demo_imgs = [data[1] for data in demo_data]
    batch_demo_seg_imgs = [data[2] for data in demo_data]
    batch_demo_index = np.arange(len(batch_demo_length))

    batch_file_path = [data[0] for data in path_data]
    batch_img_path = [data[1] for data in path_data]

    batch_demo_data = (
        batch_demo_index,
        batch_demo_length,
        batch_demo_imgs,
        batch_demo_seg_imgs)
    batch_sketch_data = (
        batch_sketch_index,
        batch_sketch_length,
        batch_sketch_action,
        batch_sketch_object1,
        batch_sketch_object2,
        batch_sketch_index1,
        batch_sketch_index2)
    batch_path_data = (batch_file_path, batch_img_path)
    return [batch_demo_data, batch_sketch_data, batch_path_data]


def to_cuda_fn(data):

    batch_demo_data, batch_sketch_data, batch_path_data = data
    batch_sketch_index, batch_sketch_length, batch_sketch_action, batch_sketch_object1, batch_sketch_object2, batch_sketch_index1, batch_sketch_index2 = batch_sketch_data
    batch_title_index, batch_demo_length, batch_demo_imgs, batch_demo_seg_imgs = batch_demo_data

    # to cuda
    batch_sketch_action = [i.cuda() for i in batch_sketch_action]
    batch_sketch_object1 = [i.cuda() for i in batch_sketch_object1]
    batch_sketch_object2 = [i.cuda() for i in batch_sketch_object2]
    batch_sketch_index1 = [i.cuda() for i in batch_sketch_index1]
    batch_sketch_index2 = [i.cuda() for i in batch_sketch_index2]

    batch_demo_imgs = [[j.cuda() for j in i] for i in batch_demo_imgs]
    batch_demo_seg_imgs = [[j.cuda() for j in i] for i in batch_demo_seg_imgs]

    batch_demo_data = (
        batch_title_index,
        batch_demo_length,
        batch_demo_imgs,
        batch_demo_seg_imgs)
    batch_sketch_data = (
        batch_sketch_index,
        batch_sketch_length,
        batch_sketch_action,
        batch_sketch_object1,
        batch_sketch_object2,
        batch_sketch_index1,
        batch_sketch_index2)

    return [batch_demo_data, batch_sketch_data, batch_path_data]


class demo_dset(Dataset):

    sketch_keys = [
        'sketch_action_list',
        'sketch_object1_list',
        'sketch_object2_list',
        'sketch_instance1_str_list',
        'sketch_instance2_str_list']

    def __init__(
            self,
            programs,
            action_dict,
            object_dict,
            is_train,
            max_length=30):

        # trim the programs with images
        programs = self._trim_programs_without_images(programs)
        programs = self._trim_programs_with_too_many_steps(programs, max_length)

        programs_w_sketch = self._add_sketch_from_real_sketch(programs)
        train_programs, test_programs = self._split_set(programs_w_sketch)

        print("-" * 30)
        print("Train set size: {}, Test set size: {}".format(
            len(train_programs), len(test_programs)))

        if is_train:
            self.programs = train_programs
        else:
            self.programs = test_programs

        self.max_indexes = 10
        self.index_dict = self._get_index_dict()
        self.num_indexes = self.index_dict.n_words
        self.action_dict = action_dict
        self.object_dict = object_dict
        self.num_actions = action_dict.n_words
        self.num_objects = object_dict.n_words
        self.initial_sketch = (int(action_dict.word2idx['<sos>']),
                               int(object_dict.word2idx['<sos>']),
                               int(object_dict.word2idx['<sos>']))

        self.rgb_image_transform = transforms.Compose([
            transforms.Resize(size=(240, 320)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

        self.seg_image_transform = transforms.Compose([
            transforms.Resize(size=(240, 320)),
            transforms.ToTensor()])

    def _trim_programs_without_images(self, programs):

        # get image pattern
        image_path = glob(dataset_dir + 'demonstration/images/*/*/*/normal.png') + \
                        glob(dataset_dir + 'demonstration/images_augment/*/*/*/*/*/normal.png')

        print("-" * 30)
        print("Total available images: {}".format(len(image_path)))

        image_programs = []
        for path in image_path:
            
            prefix_i = path.split('_graph/')[-1]
            prefix_i = prefix_i.replace('/normal.png', '')
            prefix_i = prefix_i.replace('.txt', '')
            
            for program in programs:
                prefix_p = program["file_path"].split('_graph/')[-1]
                prefix_p = prefix_p.replace('.txt', '')

                if prefix_i == prefix_p:
                    img_program = {"image_path": path}
                    for k, v in program.items():
                        img_program[k] = copy.deepcopy(v)
                    image_programs.append(img_program)
                    break

        print("Total demo-program pairs: {}".format(len(image_programs)))
        return image_programs

    def _trim_programs_with_too_many_steps(self, programs, max_length):

        valid_program = []
        for program in programs:
            if len(program['action_str_list']) >= 3 and len(
                    program['action_str_list']) <= max_length:
                valid_program.append(program)

        return valid_program

    def _get_index_dict(self):
        index_dict = dictionary()
        for i in range(self.max_indexes):
            index_dict.add_items(str(i))
        return index_dict

    def _add_sketch_from_real_sketch(self, programs):

        sketch_annotation_data = json.load(open(sketch_path))

        program_with_collected_sketch = {}
        for answer, input in zip(
                sketch_annotation_data['answer'], sketch_annotation_data['input']):
            
            prefix = input["prefix"]
            valid = sum(answer) != 0
            if valid:
                program_with_collected_sketch[prefix] = answer

        # create dictionary with key: results_intentions_march-13-18/file463_2 (parent program name)
        #                      value: [?, ?], [?, ?], [?, ?], [?, ?], [?, ?]
        sketch_annotation = {}
        for program in programs:
            if 'original_programs' in program["file_path"]:
                # original programs
                path_split = program["file_path"].split('/')
                prefix = '/'.join(path_split[-2:])
                prefix = prefix.replace('.txt', '')
                if prefix in program_with_collected_sketch and prefix not in sketch_annotation:
                    answer = program_with_collected_sketch[prefix]
                    answer_idx = np.where(answer)[0]
                    assert len(answer) == len(program['action_list']), print(
                        "The sketch is not aligned with the program")

                    sketch_annotation[prefix] = []
                    for key in self.sketch_keys:
                        value_list = [program[key.replace(
                            'sketch_', '')][idx] for idx in answer_idx]
                        sketch_annotation[prefix].append(value_list)

        programs_w_sketch = []
        programs_wo_sketch = 0
        for program in programs:
            path_split = program["file_path"].split('/')
            j = path_split.index('executable_programs')
            prefix = '/'.join(path_split[(j + 2):] \
                        if len(path_split[(j + 2):]) == 2 else path_split[(j + 2):-1])
            prefix = prefix.replace('.txt', '')

            if prefix in sketch_annotation:
                # with sketh annotated
                for n, key in enumerate(self.sketch_keys):
                    program[key] = sketch_annotation[prefix][n]
                programs_w_sketch.append(program)
            else:
                programs_wo_sketch += 1

        print("-" * 30)
        print("Total programs with sketch annotation: {}".format(
                len(programs_w_sketch)))

        return programs_w_sketch

    def _split_set(self, programs):

        # read the split
        f = open(train_patt)
        train_split_patt = f.read()
        train_split_patt = [
            i for i in filter(
                lambda v: v != '',
                train_split_patt.split("\n"))]
        f.close()

        f = open(test_patt)
        test_split_patt = f.read()
        test_split_patt = [
            i for i in filter(
                lambda v: v != '',
                test_split_patt.split("\n"))]
        f.close()

        # split by scenes and the unique programs
        train_programs, test_programs = [], []
        for program in programs:
            path_split = program["image_path"].split('_graph/')[-1]
            prefix = '/'.join(path_split.split('/')[:2]).replace('.txt', '')
            if prefix in train_split_patt and "TrimmedTestScene7_graph" not in program[
                    "image_path"]:
                train_programs.append(program)
            elif prefix in test_split_patt and "TrimmedTestScene7_graph" in program["image_path"]:
                test_programs.append(program)

        return train_programs, test_programs

    def _trim_programs(self, programs):

        valid_programs = []
        for program in programs:
            if len(
                    program['action_list']) >= 3 and len(
                    program['action_list']) <= 15:
                valid_programs.append(program)
        print("Trim out {:.2f}%  of program".format(
            100. * (len(programs) - len(valid_programs)) / len(programs)))
        return valid_programs

    def __getitem__(self, index):
        program = self.programs[index]
        data = self._preprocess_one_program(program)
        return data

    def __len__(self):
        return len(self.programs)

    def _is_terminate(self, action, object1, object2):
        # terminate when action is eos
        if action == '<eos>':
            return True
        else:
            return False

    def interpret_sketch(
            self,
            batch_action,
            batch_object1,
            batch_object2,
            with_sos=True):

        start_idx = 1 if with_sos else 0
        sketchs_str = []
        for sketch_action, sketch_object1, sketch_object2 in zip(
                batch_action, batch_object1, batch_object2):
            sketch_action = self.action_dict.convert_idx2word(
                np.array(sketch_action))
            sketch_object1 = self.object_dict.convert_idx2word(
                np.array(sketch_object1))
            sketch_object2 = self.object_dict.convert_idx2word(
                np.array(sketch_object2))

            sketch_str = []
            for action, object1, object2 in zip(
                    sketch_action[start_idx:], sketch_object1[start_idx:], sketch_object2[start_idx:]):

                sketch = '[{}] <{}> <{}>'.format(action, object1, object2)

                if self._is_terminate(action, object1, object2):
                    break
                sketch_str.append(sketch)

            sketch_str = ', '.join(sketch_str)
            sketchs_str.append(sketch_str)

        return sketchs_str

    def _preprocess_one_program(self, program):

        _, sketch_action, sketch_fake_node1_name, sketch_fake_node2_name = self._sample_one_program_w_annotation(
            program)

        eos_idx = int(self.action_dict.word2idx["<eos>"])
        sos_idx = int(self.action_dict.word2idx["<sos>"])
        sketch_action = add_end(add_front(sketch_action, sos_idx), eos_idx)
        sketch_length = len(sketch_action)

        # get the sketch object and index within the program
        sketch_instance1, sketch_object1 = self._convert_fake_node_idx_to_node_name(
            sketch_fake_node1_name)
        sketch_index1 = self._get_index_within_program(
            sketch_instance1, sketch_object1)
        sketch_instance2, sketch_object2 = self._convert_fake_node_idx_to_node_name(
            sketch_fake_node2_name)
        sketch_index2 = self._get_index_within_program(
            sketch_instance2, sketch_object2)

        sketch_object1 = add_end(
            add_front(sketch_object1, int(self.object_dict.word2idx['<sos>'])), 
                int(self.object_dict.word2idx['<eos>']))
        sketch_object2 = add_end(
            add_front(
                sketch_object2, int(self.object_dict.word2idx['<sos>'])), 
                int(self.object_dict.word2idx['<eos>']))
        sketch_index1 = add_end(
            add_front(
                sketch_index1, int(self.index_dict.word2idx['<sos>'])), 
                int(self.index_dict.word2idx['<eos>']))
        sketch_index2 = add_end(
            add_front(
                sketch_index2, int(self.index_dict.word2idx['<sos>'])), 
                int(self.index_dict.word2idx['<eos>']))

        # to tensor
        sketch_action = torch.tensor(sketch_action, dtype=torch.int64)
        sketch_object1 = torch.tensor(sketch_object1, dtype=torch.int64)
        sketch_object2 = torch.tensor(sketch_object2, dtype=torch.int64)
        sketch_index1 = torch.tensor(sketch_index1, dtype=torch.int64)
        sketch_index2 = torch.tensor(sketch_index2, dtype=torch.int64)

        sketch_data = (
            sketch_length,
            sketch_action,
            sketch_object1,
            sketch_object2,
            sketch_index1,
            sketch_index2)
        file_path = program['file_path']

        # load the images
        img_path = program["image_path"]
        rgb_img = self._process_one_img(img_path, mode='rgb')
        rgb_img = [self.rgb_image_transform(Image.fromarray(img)) for img in rgb_img]

        seg_path = img_path.replace("normal.png", "seg_class.png")
        seg_img = self._process_one_img(seg_path, mode='segmap')
        seg_img = [self.seg_image_transform(Image.fromarray(img)) for img in seg_img]

        image_length = len(rgb_img)

        image_data = (image_length, rgb_img, seg_img)
        path = (file_path, program['image_path'])

        data = [image_data, sketch_data, path]
        return data

    def _rgb2id(self, img, bins=9):
        expansion = int(255 / (bins - 1))
        return img[:,:,2].astype(np.int32)/expansion + \
            bins*img[:,:,1].astype(np.int32)/expansion + \
            bins*bins*img[:,:,0].astype(np.int32)/expansion

    def _process_one_img(self, path, mode='rgb'):

        image = imageio.imread(path)
        # image size: 480, 640, 3
        H, _, _ = image.shape
        top_view_images = image[:, :640, :]
        top_view_images = np.array(top_view_images)
        if mode == 'segmap':
            top_view_images = self._rgb2id(top_view_images)

        top_view_images_list = np.array_split(top_view_images, H / 480, 0)
        return top_view_images_list

    def _convert_fake_node_idx_to_node_name(self, fake_node_names):

        object_list = []
        instance_list = []
        for node_name in fake_node_names:
            
            name, id = node_name.split('.')
            name = int(self.object_dict.word2idx[name])
            object_list.append(name)
            instance_list.append(node_name)

        return instance_list, object_list

    def _get_index_within_program(self, instance_list, object_list):

        objs_id2index = {i: {"available_index": np.arange(
            self.max_indexes)} for i in object_list}
        index_list = []
        for obj_instance in instance_list:
            name, id = obj_instance.split('.')
            if '<none>' in name:
                index_list.append(int(self.index_dict.word2idx["<none>"]))
            else:
                name = int(self.object_dict.word2idx[name])
                id2index = objs_id2index[name]

                if id not in id2index:
                    index = np.random.choice(id2index["available_index"])
                    i_to_removed = np.where(
                        id2index["available_index"] == index)[0][0]
                    id2index["available_index"] = np.delete(
                        id2index["available_index"], i_to_removed)
                    id2index[id] = index
                else:
                    index = id2index[id]

                index = int(self.index_dict.word2idx[str(index)])
                index_list.append(index)

        return index_list

    def _sample_one_program_w_annotation(self, program):

        sample_action = program['sketch_action_list']
        sample_object1_str = [self.object_dict.idx2word[str(
            i)] for i in program['sketch_object1_list']]
        sample_object2_str = [self.object_dict.idx2word[str(
            i)] for i in program['sketch_object2_list']]
        sample_instance1_str = program['sketch_instance1_str_list']
        sample_instance2_str = program['sketch_instance2_str_list']

        sample_fake_node_name1 = [
            '{}.{}'.format(
                object, instance) for object, instance in zip(
                sample_object1_str, sample_instance1_str)]
        sample_fake_node_name2 = [
            '{}.{}'.format(
                object, instance) for object, instance in zip(
                sample_object2_str, sample_instance2_str)]
        sample_length = len(sample_action)

        return sample_length, sample_action, sample_fake_node_name1, sample_fake_node_name2
