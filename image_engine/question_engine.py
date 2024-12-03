from itertools import permutations
import random

def generate_questions(data_info, task):
    datas = []
    if task == 'PointLiesOnLine' or task == 'PointLiesOnLine_empirical':
        for points_set in data_info:
            for A, B in permutations(points_set, 2):
                all_rest_points = [p for p in points_set if p not in [A, B]]
                for rest_points in permutations(all_rest_points):
                    verb_agreement = 'is' if len(rest_points) == 1 else 'are'
                    rest_points = [f"{p}" for p in rest_points]
                    rest_points = sorted(rest_points)
                    data = {
                        "question": f"What is the point lying on line {A}{B}?",
                        "answer": f'The point lying on line {A}{B} {verb_agreement} {", ".join(rest_points)}',
                        "task": task,
                        "gt": ', '.join(rest_points)
                    }
                    datas.append(data)
    elif task == 'PointLiesOnCircle' or task == 'PointLiesOnCircle_empirical':
        point_set = random.choice(list(data_info.items()))
        center_point = point_set[0]
        target_points = point_set[1]
        target_points = sorted(target_points)
        data = {
            "question": f"What are the point lying on circle {center_point}?",
            "answer": f'The point lying on circle {center_point} are {", ".join(target_points)}',
            "task": task,
            "gt": ', '.join(target_points)
        }
        datas.append(data)
    elif task == 'Parallel' or task == 'Parallel_empirical':
        for points_set in data_info:
            # every all parallel lines
            for line_points in points_set:
                for A, B in permutations(line_points, 2):
                    all_rest_lines = [p for p in points_set if p != line_points]
                    gts = [''.join(f'{p}' for line in all_rest_lines for p in line)]
                    rest_point_pairs = []
                    for rest_line in all_rest_lines:
                        C, D = random.sample(rest_line, 2)
                        rest_point_pairs.append([C, D])
                    all_possible_answer = ', '.join([f'{C}{D}' for C, D in rest_point_pairs])
                    verb_agreement = 'is' if len(rest_point_pairs) == 1 else 'are'
                    data = {
                        "question": f"What is the line parallel to line {A}{B}?",
                        "answer": f'According to the diagram, the line parallel to {A}{B} {verb_agreement} {all_possible_answer}',
                        "task": task,
                        "gt": ', '.join(gts)
                    }
                    datas.append(data)
    elif task == 'Perpendicular' or task == 'Perpendicular_empirical':
        for source_line, target_lines in data_info.items():
            all_possible_answer = []
            gts = target_lines
            # randomly choose two points from each target line
            for target_line in target_lines:
                C, D = random.sample(target_line, 2)
                all_possible_answer.append(f'{C}{D}')
            verb_agreement = 'is' if len(all_possible_answer) == 1 else 'are'
            for A, B in permutations(source_line, 2):
                data = {
                    "question": f"What is the line perpendicular to line {A}{B}?",
                    "answer": f'According to the diagram, the line perpendicular to {A}{B} {verb_agreement} {", ".join(all_possible_answer)}',
                    "task": task,
                    "gt": ', '.join(gts)
                }
                datas.append(data)
    elif task == 'AngleClassification' or task == 'AngleClassification_empirical':
        angle = data_info
        angle_letter = random.choice([f'{angle[1][0]}{angle[1][1]}{angle[1][2]}', f'{angle[1][2]}{angle[1][1]}{angle[1][0]}'])
        angle_class = 'acute' if angle[0] < 90 else 'obtuse'
        data = {
            "question": f"Is angle {angle_letter} acute or obtuse?",
            "answer": f'According to the diagram, angle {angle_letter} is {angle_class}',
            "task": task,
            "gt": angle_class
        }
        datas.append(data)
    elif task == 'LineComparison' or task == 'LineComparison_empirical':
        names = [data_info[0][1], data_info[1][1]]
        lengths = [data_info[0][0], data_info[1][0]]
        if lengths[0] > lengths[1]:
            longer_name, shorter_name = names[0], names[1]
        else:
            longer_name, shorter_name = names[1], names[0]
        if task == 'LineComparison':
            data = [{
                        "question": f"Which line is longer, {longer_name} or {shorter_name}?",
                        "answer": f'According to the diagram, the longer line is {longer_name}',
                        "task": task,
                        "gt": longer_name
                    },
                    {
                        "question": f"Which line is longer, {shorter_name} or {longer_name}?",
                        "answer": f'According to the diagram, the longer line is {longer_name}',
                        "task": task,
                        "gt": longer_name
                    }]
        else:
            data = [{
                        "question": f"Which line is longer, {longer_name} or {shorter_name}?",
                        "answer": f'The longer line is {longer_name}',
                        "task": task,
                        "gt": longer_name
                    },
                    {
                        "question": f"Which line is longer, {shorter_name} or {longer_name}?",
                        "answer": f'The longer line is {longer_name}',
                        "task": task,
                        "gt": longer_name
                    }]
        datas.append(random.choice(data))
    elif task == 'Connection' or task == 'Connection_empirical':
        connected = random.choice(data_info['connected'].split(', '))
        disconnected = random.choice(data_info['disconnected'].split(', '))
        connected = random.choice([connected, f'{connected[1]}{connected[0]}'])
        disconnected = random.choice([disconnected, f'{disconnected[1]}{disconnected[0]}'])
        data = [{
                    "question": f"Is point {connected[0]} connected with point {connected[1]}?",
                    "answer": f'Point {connected[0]} and point {connected[1]} are connected',
                    "task": task,
                    "gt": 'connected'
                },
                {
                    "question": f"Is point {disconnected[0]} connected with point {disconnected[1]}?",
                    "answer": f'Point {disconnected[0]} and point {disconnected[1]} are not connected',
                    "task": task,
                    "gt": 'not connected'
                }]
        datas.append(random.choice(data))
    elif task == 'Equal' or task == 'Equal_empirical':
        for points_set in data_info:
            statement, content = points_set.split(';')
            if statement == 'angles_value':
                angle_letter, angle_measure = content.split('=')
                angle_letter = random.choice([angle_letter, angle_letter[::-1]])
                data = {
                    "question": f"What is the measure of angle {angle_letter} as annotated?",
                    "answer": f'Angle {angle_letter} is annotated as {angle_measure}',
                    "task": task,   
                    "gt": angle_measure
                }
            elif statement == 'segments_value':
                segment_letter, segment_length = content.split('=')
                segment_letter = random.choice([segment_letter, segment_letter[::-1]])
                data = {
                    "question": f"What is the length of line {segment_letter} as annotated?",
                    "answer": f'Line {segment_letter} is annotated as {segment_length}',
                    "task": task,
                    "gt": segment_length
                }
            elif statement == 'angles':
                angle1, angle2 = content.split('=')
                angle1 = random.choice([angle1, angle1[::-1]])
                angle2 = random.choice([angle2, angle2[::-1]])
                query_angle = random.choice([angle1, angle2])
                answer_angle = angle2 if query_angle == angle1 else angle1
                data = {
                    "question": f"What is the angle in the diagram that is equal to angle {query_angle}?",
                    "answer": f'Angle {query_angle} is equal to angle {answer_angle}',
                    "task": task,
                    "gt": answer_angle
                }
            elif statement == 'segments':
                segment1, segment2 = content.split('=')
                segment1 = random.choice([segment1, segment1[::-1]])
                segment2 = random.choice([segment2, segment2[::-1]])
                query_segment = random.choice([segment1, segment2])
                answer_segment = segment2 if query_segment == segment1 else segment1
                data = {
                    "question": f"What is the segment in the diagram that is equal to segment {query_segment}?",
                    "answer": f'Segment {query_segment} is equal to segment {answer_segment}',
                    "task": task,   
                    "gt": answer_segment
                }
            datas.append(data)
    return datas

def answer_match(task, pred, gt):
    pred = [i.strip() for i in pred.split(',')]
    gt = [i.strip() for i in gt.split(',')]
    pred_set = set(pred)
    gt_set = set(gt)

    if task == 'PointLiesOnLine' or task == 'PointLiesOnCircle' or task == 'PointLiesOnLine_empirical' or task == 'PointLiesOnCircle_empirical':
        if pred_set.issubset(gt_set):
            return len(pred_set)/len(gt_set)
        else:
            return 0
    elif task == 'AngleClassification' or task == 'LineComparison' or task == 'Connection' or task == 'AngleClassification_empirical' or task == 'LineComparison_empirical' or task == 'Connection_empirical':
        return int(set(pred) == set(gt))
    elif task == 'Parallel' or task == 'Perpendicular' or task == 'Parallel_empirical' or task == 'Perpendicular_empirical':
        covered_gt = []
        correct_pred = []
        for p in pred_set:
            for g in gt_set:
                if set(p).issubset(set(g)):
                    covered_gt.append(g)
                    correct_pred.append(p)
                    break
        # if there is at least one incorrect prediction, return 0
        if len(correct_pred) < len(pred_set):
            return 0
        else:
            return len(covered_gt)/len(gt_set)
    elif task == 'Equal' or task == 'Equal_empirical':
        pred = pred[0]
        gt = gt[0]
        if all(g.isupper() for g in gt):
            return int(set(pred) == set(gt))
        else:
            return int(pred == gt)