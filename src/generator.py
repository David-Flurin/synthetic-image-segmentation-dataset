import os
from os import mkdir
from os.path import join
import shutil
import matplotlib.pyplot as plt
from random import randint, sample, choice
from tqdm import tqdm
from pathlib import Path
import xml.etree.ElementTree as ET
import hashlib
from .shapes import *
import numpy as np
import math


def get_i(idx, list):
    return [list[i] for i in idx]

def string_to_rgb(s):
    h = int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % 10 ** 12
    h_str = str(h)
    r = int(h_str[0:4]) % 256
    g = int(h_str[4:8]) % 256
    b = int(h_str[8:12]) % 256
    return np.array([r,g,b])

class Generator:

    shapes = [Circle, Triangle, Square, Pentagon, Hexagon, Heart, Cross, Star]

    def __init__(self, f_textures, b_textures, img_size=(224,244)):

        self.img_size = img_size
        self.min_radius = int(0.134*min(img_size))
        self.max_radius = int(0.357*min(img_size))
        
        with open(f_textures, 'r') as f:
            f_textures_files = f.read().splitlines()
        with open(b_textures, 'r') as f:
            b_textures_files = f.read().splitlines()

        base_dir = '/'.join(__file__.split('/')[0:-1])

        self.f_texture_names = [s.split('.')[0].lower() for s in f_textures_files]
        self.b_texture_names = [s.split('.')[0].lower() for s in b_textures_files]
        self.shape_names = []
        
        self.f_textures=[]
        for f in f_textures_files:
            self.f_textures.append(plt.imread(join(base_dir, 'textures', f)))
        
        self.b_textures = []
        for f in b_textures_files:
            self.b_textures.append(plt.imread(join(base_dir, 'textures', f)))

        self.num_f_tex = len(self.f_textures)
        self.num_b_tex = len(self.b_textures)
            
        
    def generate_sample(self, num_objects):
        if num_objects > 2:
            raise ValueError(f'Currently not more objects than 2 are supported.')

        self.max_radius = max(self.max_radius, self.min_radius)
        idx = sample(range(0, len(self.shapes)), num_objects)
        return self.__generate(get_i(idx, self.shapes))

    
    def create_set(self, number_per_shape, proportions, multiclass=False):
        '''Create a set of samples.
        Parameters:
            number_per_shape (int)       -- Number of samples generates for each shape
            proportions (List of float)  -- Proportions of samples for train, validation and test set 
            multilabel (bool)            -- Whether to have multilabel samples in the dataset or not
        '''

        proportions = [float(x) for x in proportions]
        if len(proportions) != 3 or sum(proportions) != 1.0:
            raise ValueError('\'Proportions\' must have three elements (Train, Val, Test) and must sum to 1')

        self.__directories()
        samples_shape = {}
        samples_texture = {}
        with tqdm(total=number_per_shape*len(self.shapes)) as pbar:
            num_length = math.ceil(math.log(number_per_shape*len(self.shapes), 10))+1
            i = 1
            for _ in range(number_per_shape):
                for shape in self.shapes:
                    gen_shapes = [shape]
                    if multiclass:
                        r = randint(1, 2)
                        if r == 2:
                            shape_2 = choice(self.shapes)
                            gen_shapes.append(shape_2)
                    filename = f'{i:0{num_length}d}'
                    s = self.__generate(gen_shapes)
                    self.__save(s, filename)
                    for o in s['objects']:
                        try:
                            samples_shape[o[0]] += [filename]
                        except KeyError:
                            samples_shape[o[0]] = [filename]

                        try:
                            samples_texture[o[1]] += [filename]
                        except KeyError:
                            samples_texture [o[1]] = [filename]
                    pbar.update()
                    i += 1

        train = []
        val = []
        test = []
        for k,v in samples_shape.items():
            v.sort(key=int)
            indices = range(0, len(v))
            total = len(indices)
            num_train = int(proportions[0] * total)
            num_val = int(proportions[1] * total)
            num_test = int(proportions[2] * total)
            num_train += (total - num_train - num_val - num_test)
            train_idx = sample(indices, num_train)
            indices = [i for i in indices if i not in train_idx]
            val_idx = sample(indices, num_val)
            test_idx = [i for i in indices if i not in val_idx]

            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_train.txt'), get_i(train_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_val.txt'), get_i(val_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', f'{k}_test.txt'), get_i(test_idx, v))
            
            train += (get_i(train_idx, v))
            val += (get_i(val_idx, v))
            test += (get_i(test_idx, v))
        
        if len(train) > 0:
            train = sorted(list(set(train)))
        if len(val) > 0:
            val = sorted(list(set(val)))
        if len(test) > 0:
            test = sorted(list(set(test)))

        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'train.txt'), train)
        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'val.txt'), val)
        self.__write_sample_list(Path(self.base, 'imagesets', 'shapes', 'test.txt'), test)

        train = []
        val = []
        test = []
        for k,v in samples_texture.items():
            v.sort(key=int)
            indices = range(0, len(v))
            total = len(indices)
            num_train = int(proportions[0] * total)
            num_val = int(proportions[1] * total)
            num_test = int(proportions[2] * total)
            num_train += (total - num_train - num_val - num_test)
            train_idx = sample(indices, num_train)
            indices = [i for i in indices if i not in train_idx]
            val_idx = sample(indices, num_val)
            test_idx = [i for i in indices if i not in val_idx]

            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_train.txt'), get_i(train_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_val.txt'), get_i(val_idx, v))
            self.__write_sample_list(Path(self.base, 'imagesets', 'textures', f'{k}_test.txt'), get_i(test_idx, v))
            
            
            train += (get_i(train_idx, v))
            val += (get_i(val_idx, v))
            test += (get_i(test_idx, v))

        if len(train) > 0:
            train = sorted(list(set(train)))
        if len(val) > 0:
            val = sorted(list(set(val)))
        if len(test) > 0:
            test = sorted(list(set(test)))
        self.__write_sample_list(Path(self.base, 'imagesets', 'textures', 'train.txt'), train)
        self.__write_sample_list(Path(self.base, 'imagesets', 'textures', 'val.txt'), val)
        self.__write_sample_list(Path(self.base, 'imagesets', 'textures', 'test.txt'), test)

                    
    def __write_sample_list(self, filename, list):
        with open(filename, 'w') as f:
            for sample in list:
                f.write(f'{sample}\n')

    def __generate(self, shapes):
        shape_obj = []
        specs = self.__randomize(self.img_size[0], len(shapes))
        for i, shape in enumerate(shapes):
            radius, center_x, center_y, rotation, f_tex_idx, b_tex_idx = specs[i]
            s = shape(radius, (center_x, center_y), self.img_size, rotation)
            shape_obj.append((s, f_tex_idx, radius))
        img, seg_tex, seg_shape = self.__assemble(shape_obj, self.b_textures[b_tex_idx])

        objects = []
        for shape in shape_obj:
            shape_name = (str(shape[0].__class__).split('.')[-1]).split('\'')[0].lower()
            objects.append((shape_name, self.f_texture_names[shape[1]], shape[2]))
        return {'image': img, 'seg_tex': seg_tex, 'seg_shape': seg_shape, 'objects': objects, 'background': self.b_texture_names[b_tex_idx]}


    def __save(self, sample, filename):

        annotation = ET.Element('annotation')
        xml_filename = ET.SubElement(annotation, 'filename')
        xml_filename.text = filename
        xml_objects = ET.SubElement(annotation, 'objects')
        
        for i, obj in enumerate(sample['objects']):
            xml_obj = ET.SubElement(xml_objects, f'object{i}')
            s = ET.SubElement(xml_obj, 'shape')
            s.text = obj[0]
            t = ET.SubElement(xml_obj, 'texture')
            t.text = obj[1]
        background = ET.SubElement(annotation, 'background')
        background.text = sample['background']

        plt.imsave(join(self.base, 'images', f'{filename}.png'), sample['image'], format='png')
        plt.imsave(join(self.base, 'segmentations', 'shapes', f'{filename}.png'), sample['seg_shape']/255., format='png')
        plt.imsave(join(self.base, 'segmentations', 'textures', f'{filename}.png'), sample['seg_tex']/255., format='png')

        tree = ET.ElementTree(annotation)
        tree.write(join(self.base, 'annotations', f'{filename}.xml'))

    def __randomize(self, size, objects):
        '''Generate objects with randomized radius, location and textures. 
            If objects==2, two objects are generated, which are not overlapping.
        '''
        if objects == 1:
            radius =  randint(self.min_radius, min((size-30)/2, self.max_radius))
            center_x = randint(20 + radius, size - 20 - radius)
            center_y = randint(20 + radius, size - 20 - radius)
            rotation = randint(0, 360)
            f_tex_idx = randint(0, self.num_f_tex -1)
            b_tex_idx = randint(0, self.num_b_tex -1)
            
            return [[radius, center_x, center_y, rotation, f_tex_idx, b_tex_idx]]
        if objects == 2:
            max_radius = min(((size -20 - 2*self.min_radius) // 2), self.max_radius)
            radius_1 = randint(self.min_radius, max_radius)
            max_radius = min(((size - 20 - 2*radius_1) // 2), self.max_radius) 
            radius_2 = randint(self.min_radius, max_radius)
            free = size - 20 - 2*(radius_1 + radius_2)

            offset_x_1 = 10 if randint(0,1) == 0 else (size - 10 - 2*radius_1 - free//2)
            offset_y_1 = 10 if randint(0,1) == 0 else (size - 10 - 2*radius_1 - free//2)
            offset_x_2 = 10 if offset_x_1 > 10 else (size - 10 - 2*radius_2 - free//2)
            offset_y_2 = 10 if offset_y_1 > 10 else (size - 10 - 2*radius_2 - free//2)
            center_x_1 = randint(offset_x_1 + radius_1, offset_x_1 + radius_1 + free//2)
            center_y_1 = randint(offset_y_1 + radius_1, offset_y_1 + radius_1 + free//2)
            center_x_2 = randint(offset_x_2 + radius_2, offset_x_2 + radius_2 + free//2)
            center_y_2 = randint(offset_y_2 + radius_2, offset_y_2 + radius_2 + free//2)
            rotation_1 = randint(0, 360)
            rotation_2 = randint(0, 360)
            f_tex_idx_1 = randint(0, self.num_f_tex -1)
            b_tex_idx_1 = randint(0, self.num_b_tex -1)
            f_tex_idx_2 = randint(0, self.num_f_tex -1)
            b_tex_idx_2 = randint(0, self.num_b_tex -1)
            return [
               [radius_1, center_x_1, center_y_1, rotation_1, f_tex_idx_1, b_tex_idx_1],
               [radius_2, center_x_2, center_y_2, rotation_2, f_tex_idx_2, b_tex_idx_2] 
            ]

    def __directories(self):
        if os.path.exists(self.base) and os.path.isdir(self.base):
            shutil.rmtree(self.base)
        mkdir(self.base)
        mkdir(Path(self.base, 'images'))
        mkdir(Path(self.base, 'segmentations'))
        mkdir(Path(self.base, 'segmentations', 'shapes'))
        mkdir(Path(self.base, 'segmentations', 'textures'))

        mkdir(Path(self.base, 'annotations'))
        mkdir(Path(self.base, 'imagesets'))

        mkdir(Path(self.base, 'imagesets', 'shapes'))
        mkdir(Path(self.base, 'imagesets', 'textures'))
        


    def __assemble(self, shapes, b_tex):
        b_tex = self.__crop_texture(b_tex, (224, 224))
        img = b_tex.copy()
        seg_shape = np.zeros((self.img_size[0], self.img_size[1], 3))
        seg_tex = np.zeros((self.img_size[0], self.img_size[1], 3))
        for s, f_tex_idx,_ in shapes:
            f_tex = self.__crop_texture(self.f_textures[f_tex_idx], (224, 224))
            img[np.where(s.mask)] = f_tex[np.where(s.mask)]
            seg_tex[np.where(s.mask)] = string_to_rgb(self.f_texture_names[f_tex_idx])
            seg_shape[np.where(s.mask)] = string_to_rgb(str(s.__class__).split('.')[-1].split('\'')[0].lower())
        
        return img, seg_tex, seg_shape


    def __crop_texture(self, tex, img_size):
        t_h, t_w, _ = tex.shape
        assert(t_h > img_size[0] and t_w > img_size[1])

        x = randint(0, t_h - img_size[0] -1)
        y = randint(0, t_w - img_size[1] -1)
        cropped = tex[x:x+img_size[0], y:y+img_size[1]]
        return cropped



