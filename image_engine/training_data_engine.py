import sys
sys.path.append('./image_engine/alphageometry')
import numericals as nm
import problem as pr
import graph as gh
from image_engine.question_engine import *
import random
import string
import copy
import argparse
import json
import os
import time
from tqdm import tqdm
import multiprocessing
import numpy as np
import itertools
from image_engine.produce_shape import produce_shape_datas


def get_angle(g, points):
    a = g._name2node[points[0]]
    b = g._name2node[points[1]]
    c = g._name2node[points[2]]
    ba = np.array([a.num.x - b.num.x, a.num.y - b.num.y])
    bc = np.array([c.num.x - b.num.x, c.num.y - b.num.y])
    cos_theta = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    return np.arccos(cos_theta) * 180 / np.pi

def get_distance(g, points):
    a = g._name2node[points[0]]
    b = g._name2node[points[1]]
    return np.linalg.norm(np.array([a.num.x - b.num.x, a.num.y - b.num.y]))

def power_set(lst):
    """Generate all subsets of a list using itertools, excluding empty and full sets."""
    return list(itertools.chain.from_iterable(itertools.combinations(lst, r) for r in range(1, len(lst))))

# function to replace the letters in the orcle_text
def generate_txt(letters, shape_data):
    letter_map = dict(zip(list(string.ascii_uppercase), letters))
    return shape_data['orcle_text'].translate(str.maketrans(letter_map))

def make_connection(letters, shape_data, g):
    all_nodes = g.type2nodes[gh.Line]
    for node in all_nodes:
        if node.name in g._name2node.keys():
            g._name2node.pop(node.name)
    g.type2nodes[gh.Line] = []
    letter_map = dict(zip(list(string.ascii_uppercase), letters))
    connections = shape_data['connection_list'].translate(str.maketrans(letter_map))
    if connections != '':
        connection_list = [i.strip() for i in connections.split(',')]
        for connection in connection_list:
            g.get_new_line_thru_pair(g._name2node[connection[0]], g._name2node[connection[1]])

def remove_points_from_graph(letters, shape_data, g):
    if 'remove_set' in shape_data:
        letter_map = dict(zip(list(string.ascii_uppercase), letters))
        remove_set = shape_data['remove_set'].translate(str.maketrans(letter_map))
        points = [i.strip() for i in remove_set.split(',')]
        g.remove([g._name2node[point] for point in points])

def draw_g(g, save_to=None, highlights=None, equals=None):
    if save_to is None:
        img = nm.draw(
            g.type2nodes[gh.Point],
            g.type2nodes[gh.Line],
            g.type2nodes[gh.Circle],
            g.type2nodes[gh.Segment],
            theme='light',
            highlights=highlights,
            equals=equals
        )
        return img
    else:
        nm.draw(
            g.type2nodes[gh.Point],
            g.type2nodes[gh.Line],
            g.type2nodes[gh.Circle],
            g.type2nodes[gh.Segment],
            theme='light',
            save_to=save_to,
            highlights=highlights,
            equals=equals
        )
        return save_to

def make_highlights(shape_data, letter_map, g):
    highlights = []
    if 'highlights' in shape_data:
        for item in shape_data['highlights']:
            parts = item.split(';')
            if len(parts) == 2:
                hl_type = parts[0].strip()
                points_str = parts[1].strip()
                points_names = [pt.strip() for pt in points_str.split(',')]
                points_names = [letter_map.get(pt, pt) for pt in points_names]
                nodes = [g._name2node[pt] for pt in points_names]
                highlights.append((hl_type, nodes))
    return highlights

def all_angles(connections):
    angles = []
    for line_a, line_b in itertools.combinations(connections, 2):
        if len(set(line_a) | set(line_b)) != 3:
            continue
        else:
            vertex = ''.join(set(line_a) & set(line_b))
            point1 = ''.join(set(line_a) - set(vertex))
            point2 = ''.join(set(line_b) - set(vertex))
            angles.append(f'{point1}{vertex}{point2}')
    return angles

def add_annotation(shape_data):
    if 'equals' in shape_data:
        return shape_data
    else:
        if random.random() < 0.7:
            return shape_data
        shape_data['equals'] = []
        if random.random() < 0.5:
            all_connections = [i.strip() for i in shape_data['connection_list'].split(',')]
            num_connections = len(all_connections)
            num_annotations = random.randint(1, num_connections)
            annotation_connections = random.sample(all_connections, num_annotations)
            for connection in annotation_connections:
                shape_data['equals'].append(f'segments_value;{connection}=annotation')
        else:
            all_connections = [i.strip() for i in shape_data['connection_list'].split(',')]
            angles = all_angles(all_connections)
            if len(angles) == 0:
                return shape_data
            num_annotations = random.randint(1, min(3, len(angles)))
            annotation_angles = random.sample(angles, num_annotations)
            for angle in annotation_angles:
                shape_data['equals'].append(f'angles_value;{angle}=annotation')
        return shape_data

def get_actual_annotation(shape_data):
    segname_to_value = {}
    letters = 'abcdefghijklmnopqrstuvwxyz'
    digits = '123456789'
    operations = '+-'
    if 'equals' in shape_data:
        for i, item in enumerate(shape_data['equals']):
            if 'annotation' in item:
                all_annotations = []
                all_annotations.append(random.choice(letters))
                all_annotations.append(random.choice(digits))
                all_annotations.append(random.choice([f'{a}{o}{b}' for a in letters for b in digits for o in operations]))
                value = random.choice(all_annotations)
                shape_data['equals'][i] = item.replace('annotation', value)
                segname_to_value[item.split(';')[1].split('=')[0]] = value
    
        for i, item in enumerate(shape_data['points_set']):
            if 'annotation' in item:
                seg_name = item.split(';')[1].split('=')[0]
                if seg_name in segname_to_value:
                    shape_data['points_set'][i] = item.replace('annotation', segname_to_value[seg_name])
    return shape_data

def get_actual_angle(shape_data, g, letter_map):
    if 'equals' in shape_data:
        for i, item in enumerate(shape_data['equals']):
            if 'actual' in item:
                angle_letter = item.split(';')[1].split('=')[0]
                angle_letter = [letter_map.get(pt, pt) for pt in angle_letter]
                angle_num = str(int(get_angle(g, list(angle_letter))))
                shape_data['equals'][i] = shape_data['equals'][i].replace('actual', angle_num)
    for i, item in enumerate(shape_data['points_set']):
        if 'actual' in item:
            angle_letter = item.split(';')[1].split('=')[0]
            angle_letter = [letter_map.get(pt, pt) for pt in angle_letter]
            angle_num = str(int(get_angle(g, list(angle_letter))))
            shape_data['points_set'][i] = shape_data['points_set'][i].replace('actual', angle_num)
    return shape_data

def make_equal(shape_data, letter_map, g):
    equals = {}
    if 'equals' in shape_data:
        for item in shape_data['equals']:
            parts = item.split(';')
            if len(parts) >= 2:
                eq_type = parts[0].strip()
                rest = ';'.join(parts[1:]).strip()
                if eq_type == 'segments':
                    segments = rest.split('=')
                    if len(segments) >= 2:
                        seg_list = []
                        for seg in segments:
                            pt_names = list(seg.strip())
                            pt_names = [letter_map.get(pt, pt) for pt in pt_names]
                            a = g._name2node[pt_names[0]]
                            b = g._name2node[pt_names[1]]
                            seg_list.append((a.num, b.num))
                        equals.setdefault('segments', []).append(seg_list)
                elif eq_type == 'angles':
                    angle_parts = rest.split('=')
                    angle_list = []
                    angle_value = None
                    for part in angle_parts:
                        part = part.strip()
                        pt_names = list(part.strip())
                        pt_names = [letter_map.get(pt, pt) for pt in pt_names]
                        if len(pt_names) == 3:
                            a = g._name2node[pt_names[0]]
                            b = g._name2node[pt_names[1]]
                            c = g._name2node[pt_names[2]]
                            angle_list.append((a.num, b.num, c.num, b.num))
                    if angle_value is not None:
                        angle_list.append(angle_value)
                    equals.setdefault('angles', []).append(angle_list)
                elif eq_type == 'segments_value':
                    seg_parts = rest.split('=')
                    if len(seg_parts) == 2:
                        seg_str = seg_parts[0].strip()
                        value = seg_parts[1].strip()
                        pt_names = list(seg_str)
                        pt_names = [letter_map.get(pt, pt) for pt in pt_names]
                        a = g._name2node[pt_names[0]]
                        b = g._name2node[pt_names[1]]
                        equals.setdefault('segments_value', []).append([(a.num, b.num), value])
                elif eq_type == 'angles_value':
                    angle_parts = rest.split('=')
                    if len(angle_parts) == 2:
                        angle_str = angle_parts[0].strip()
                        angle_str = [letter_map.get(pt, pt) for pt in angle_str]
                        if len(angle_str) == 3:
                            a = g._name2node[angle_str[0]]
                            b = g._name2node[angle_str[1]]
                            c = g._name2node[angle_str[2]]
                        value = angle_parts[1].strip()
                        equals.setdefault('angles_value', []).append([(a.num, b.num, c.num, b.num), value])
    return equals

def get_graph_from_shape(shape_data, tol=0.3):
    letters = shape_data['letters'] if 'letters' in shape_data else 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    letters = list(letters)
    random.shuffle(letters)
    letters = ''.join(letters)
    letter_map = dict(zip(list(string.ascii_uppercase), letters))
    orcle_text = shape_data['orcle_text'].translate(str.maketrans(letter_map))
    defs = pr.Definition.from_txt_file('./image_engine/alphageometry/defs.txt', to_dict=True)
    p = pr.Problem.from_txt(orcle_text, translate=False)
    g, _ = gh.Graph.build_problem(p, defs, tol=tol)
    shape_data = get_actual_angle(shape_data, g, letter_map)
    make_connection(letters, shape_data, g)
    remove_points_from_graph(letters, shape_data, g)
    highlights = make_highlights(shape_data, letter_map, g)
    equals = make_equal(shape_data, letter_map, g)
    return g, letter_map, highlights, equals, shape_data

def worker_init():
    random.seed(os.getpid())
    np.random.seed(os.getpid())

class Euclid_DataEngine_LLaVA:
    def __init__(self, euclid_data_engine):
        self.euclid_data_engine = euclid_data_engine
        self.tasks = self.euclid_data_engine.tasks
        self.stages = self.euclid_data_engine.stages
        if os.path.exists('./playground/llava_data/llava_v1_5_mix665k.json'):
            with open('./playground/llava_data/llava_v1_5_mix665k.json', 'r') as f:
                self.llava_datas = json.load(f)
        else:
            self.llava_datas = []
        random.shuffle(self.llava_datas)
        self.cur_llava_index = 0
        
    def generate_datas(self, num_sample: int):
        euclid_data_num = int(num_sample * 1)
        llava_data_num = num_sample - euclid_data_num
        euclid_datas = self.euclid_data_engine.generate_datas(euclid_data_num)
        llava_datas = self.llava_datas[self.cur_llava_index: self.cur_llava_index + llava_data_num]
        self.cur_llava_index += llava_data_num
        if self.cur_llava_index >= len(self.llava_datas):
            random.shuffle(self.llava_datas)
            self.cur_llava_index = 0
        for llava_data in llava_datas:
            if 'image' in llava_data:
                llava_data['image'] = os.path.join('playground/llava_data', llava_data['image'])
        total_datas = euclid_datas + llava_datas
        random.shuffle(total_datas)
        return total_datas

    def update_training_status(self, eval_results):
        return self.euclid_data_engine.update_training_status(eval_results)


class Euclid_DataEngine:
    def __init__(self, tasks, stages, attenuation_rate=None, image_path=None, tol=0.1):

        self.tasks = tasks
        self.stages = stages
        self.accuracy_threshold = 0.99
        self.attenuation_rate = attenuation_rate if attenuation_rate != None else 1.5
        self.image_path = image_path
        self.tol = tol

        self.shape_datas = produce_shape_datas()

        self.training_status = {t: {'stage': 1} for t in self.tasks}

    def update_training_status(self, eval_results):
        self.shape_datas = produce_shape_datas()
        if_updated = False
        for task, stage in self.training_status.items():
            stage = stage['stage']
            if f'{task}_{stage}' not in eval_results:
                continue
            accuracy = eval_results[f'{task}_{stage}']
            if accuracy > self.accuracy_threshold:
                self.training_status[task]['stage'] += 1
                if_updated = True
        cur_status_str = ''
        for task, stage in self.training_status.items():
            cur_status_str += f'{task:>20}: {stage["stage"]:>2}|'
        print(cur_status_str)
        return if_updated
    
    def generate_datas_for_task_stage_one_chuck(self, task: str, stage: int, id_chuck: list):
        shape_datas = self.shape_datas[task][f'level_{stage}']
        if task == 'AngleClassification' or task == 'AngleClassification_empirical':
            angle_flag = random.choice(['acute', 'obtuse'])
        datas = []
        for i in id_chuck:
            shape_data = copy.deepcopy(random.choice(shape_datas))
            shape_data = add_annotation(shape_data)
            shape_data = get_actual_annotation(shape_data)
            while True:
                g, letter_map, highlights, equals, shape_data = get_graph_from_shape(shape_data, self.tol)
                if task == 'AngleClassification' or task == 'AngleClassification_empirical':
                    angles = [points.translate(str.maketrans(letter_map)) for points in shape_data['points_set']]
                    angle_value_and_names = [(get_angle(g, angle), angle) for angle in angles]
                    if angle_flag == 'acute':
                        angle_value_and_names = [angle for angle in angle_value_and_names if angle[0] < 80 and angle[0] > 10]
                    elif angle_flag == 'obtuse':
                        angle_value_and_names = [angle for angle in angle_value_and_names if angle[0] > 100 and angle[0] < 170]
                    if len(angle_value_and_names) != 0:
                        if angle_flag == 'acute':
                            angle_flag = 'obtuse'
                        else:
                            angle_flag = 'acute'
                        angle_value_and_name = random.choice(angle_value_and_names)
                        break
                elif task == 'LineComparison' or task == 'LineComparison_empirical':
                    lines = [line.translate(str.maketrans(letter_map)) for line in shape_data['points_set']]
                    lengths_and_names = [(get_distance(g, line), line) for line in lines]
                    line_combinations = list(itertools.combinations(lengths_and_names, 2))
                    valid_line_pairs = []
                    for j in range(len(line_combinations)):
                        line_pair = line_combinations[j]
                        max_length = max(line_pair[0][0], line_pair[1][0])
                        min_length = min(line_pair[0][0], line_pair[1][0])
                        if (max_length - min_length) / min_length > 0.5:
                            valid_line_pairs.append(line_pair)
                    if len(valid_line_pairs) != 0:
                        line_pair = random.choice(valid_line_pairs)
                        break
                else:
                    break
            if self.image_path == None:
                img = draw_g(g, highlights=highlights, equals=equals)
            else:
                img = draw_g(g, save_to=os.path.join(self.image_path, f'{task}_{stage}_{i}.png'), highlights=highlights, equals=equals)
            if task == 'AngleClassification' or task == 'AngleClassification_empirical':
                data_info = angle_value_and_name
            elif task == 'LineComparison' or task == 'LineComparison_empirical':
                data_info = line_pair
            elif task == 'PointLiesOnLine' or task == 'PointLiesOnLine_empirical':
                data_info = [line.translate(str.maketrans(letter_map)) for line in shape_data["points_set"]]
            elif task == 'PointLiesOnCircle':
                data_info = {key.translate(str.maketrans(letter_map)): value.translate(str.maketrans(letter_map)) for key, value in shape_data["points_set"].items()}
            elif task == 'Parallel':
                data_info = [[l.translate(str.maketrans(letter_map)) for l in line] for line in shape_data["points_set"]]
            elif task == 'Perpendicular':
                data_info = {key.translate(str.maketrans(letter_map)): [i.translate(str.maketrans(letter_map)) for i in value] for key, value in shape_data["points_set"].items()}
            elif task == 'Connection':
                data_info = {key.translate(str.maketrans(letter_map)): value.translate(str.maketrans(letter_map)) for key, value in shape_data["points_set"].items()}
            elif task == 'Equal':
                data_info = [value.translate(str.maketrans(letter_map)) for value in shape_data["points_set"]]
            data = random.choice(generate_questions(data_info, task))
            datas.append({
                'conversations': [
                    {
                        'from': 'human',
                        'value': f"<image>\n{data['question']}"
                    },
                    {
                        'from': 'gpt',
                        'value': data['answer']
                    }
                ],
                'task': task,
                'stage': stage,
                'gt': data['gt'],
                'image': img
            })
        return datas

    def generate_datas_for_task_stage(self, task: str, stage: int, num_sample: list):
        id_list = list(range(num_sample))
        random.shuffle(id_list)
        num_chunks = min(40, len(id_list))
        chunk_size = max(len(id_list) // num_chunks, 1)
        id_chunks = [id_list[i:i + chunk_size] for i in range(0, len(id_list), chunk_size)]
        with multiprocessing.Pool(processes=num_chunks, initializer=worker_init) as pool:
            results = pool.starmap(self.generate_datas_for_task_stage_one_chuck, [(task, stage, chunk) for chunk in id_chunks])

        return [item for sublist in results for item in sublist]


    def generate_datas_for_task(self, task: str, num_sample: int):
        datas = []
        cur_stage = self.training_status[task]['stage']
        if cur_stage > len(self.stages):
            ratios = [1 for _ in self.stages]
        else:
            ratios = []
            for stage in self.stages:
                distance = np.abs(stage - cur_stage)
                cur_sample_ratio = np.exp(-distance * self.attenuation_rate)
                ratios.append(cur_sample_ratio)
        sum_ratio = sum(ratios)

        for stage, ratio in zip(self.stages, ratios):
            num_cur_sample  = int(ratio / sum_ratio * num_sample) + 1
            datas.extend(self.generate_datas_for_task_stage(task, stage, num_cur_sample))
        return datas

    def generate_datas(self, num_sample: int):
        start_time = time.time()
        datas = []
        cur_stages = [self.training_status[task]['stage'] for task in self.tasks]
        task_weights = [1 if stage < len(self.stages) + 1 else 0.4 for stage in cur_stages]
        task_weights = [weight / sum(task_weights) for weight in task_weights]
        for task, weight in zip(self.tasks, task_weights):
            print(f'Generating {task} datas')
            datas.extend(self.generate_datas_for_task(task, int(num_sample * weight)))
        random.shuffle(datas)
        datas = datas[:num_sample]
        end_time = time.time()
        print(f'Generating datas cost {end_time - start_time} seconds')
        return datas