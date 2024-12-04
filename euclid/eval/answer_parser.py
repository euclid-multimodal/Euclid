import json
from copy import deepcopy
import difflib

def string_similarity(s1, s2):
    return difflib.SequenceMatcher(None, s1, s2).ratio()

def prediction_parsing(prediction):
    prediction = prediction.split('is ')[1] if 'is ' in prediction else prediction
    prediction = prediction.split('are ')[1] if 'are ' in prediction else prediction
    prediction = prediction.split(': ')[1] if ': ' in prediction else prediction
    prediction = prediction.split('.')[0]
    prediction = prediction.split('\n')[0]
    prediction = prediction.split(', ')
    for i in range(len(prediction)):
        if '°' in prediction[i]:
            prediction[i] = prediction[i].replace('°', '')
        prediction[i] = prediction[i].replace(' ', '').replace('(', '').replace(')', '').replace('angle', '').replace('"', '')
    return prediction

def process_gt(gt):
    return [i.replace(' ', '').replace('angle', '') for i in gt.split(', ')]

def match(prediction, gt):
    if len(gt) == 1 and gt[0] in ['acute', 'obtuse']:
        similarity = [string_similarity(prediction[0], i) for i in ['acute', 'obtuse']]
        prediction = ['acute', 'obtuse'][similarity.index(max(similarity))]
        if prediction == gt[0]:
            return 1
        else:
            return 0
        
    if set(prediction) - set(gt):
        return 0
    else:
        prediction_length = len(set([''.join(set(i)) for i in prediction]))
        gt_length = len(set([''.join(set(i)) for i in gt]))
        return prediction_length / gt_length


def compute_accuracy(prediction_list):
    accuracy = {}
    for i in prediction_list:
        answer = i['answer']
        prediction = i['prediction']
        
        gt = process_gt(answer)
        prediction = prediction_parsing(prediction)
        if i['predicate'] == 'PointLiesOnLine':
            new_prediction = []
            question_points = [i['question'][-2], i['question'][-3]]
            for p in prediction:
                if p not in question_points:
                    new_prediction.append(p)
            prediction = new_prediction
        

        if i['predicate'] not in accuracy:
            accuracy[i['predicate']] = []
        accuracy[i['predicate']].append(match(prediction, gt))
    
    return accuracy