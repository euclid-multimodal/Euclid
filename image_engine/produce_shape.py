# preparing all the dependencies
from IPython.display import clear_output
import sys
sys.path.append('./image_engine/alphageometry')
import numericals as nm
import problem as pr
import graph as gh
import random
import string
from itertools import combinations
import itertools
import numpy as np

def produce_shape_datas():
    shape_datas = {
        'PointLiesOnLine': {},
        'PointLiesOnCircle': {},
        'AngleClassification': {},
        'LineComparison': {},
        'Parallel': {},
        'Perpendicular': {},
        'Equal': {},
        'PointLiesOnLine_empirical': {},
        'AngleClassification_empirical': {},
        'LineComparison_empirical': {},
    }

    levels = ['level_1', 'level_2', 'level_3', 'level_4']

    for level in levels:
        shape_datas['PointLiesOnLine'][level] = []
        shape_datas['PointLiesOnCircle'][level] = []
        shape_datas['AngleClassification'][level] = []
        shape_datas['LineComparison'][level] = []
        shape_datas['Equal'][level] = []
        shape_datas['Parallel'][level] = []
        shape_datas['Perpendicular'][level] = []
        shape_datas['PointLiesOnLine_empirical'][level] = []
        shape_datas['AngleClassification_empirical'][level] = []
        shape_datas['LineComparison_empirical'][level] = []

    # point lies on line
    shape_datas['PointLiesOnLine']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, AD, BC',
        'points_set': ['BCD']
    })
    shape_datas['PointLiesOnLine']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; O = circle O A B C',
        'connection_list': 'AB, AC, AD, BC',
        'remove_set': 'O',
        'points_set': ['BCD']
    })

    shape_datas['PointLiesOnLine']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['ABD', 'ACE']
    })
    shape_datas['PointLiesOnLine']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C; O = circle O A B C',
        'connection_list': 'AB, AC, BC, DE',
        'remove_set': 'O',
        'points_set': ['ABD', 'ACE']
    })

    shape_datas['PointLiesOnLine']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E',
        'connection_list': 'AB, AC, BC, DA, BE',
        'points_set': ['AFD', 'ACE', 'BFE', 'BDC']
    })
    shape_datas['PointLiesOnLine']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E; O = circle O A B C',
        'connection_list': 'AB, AC, BC, DA, BE',
        'remove_set': 'O',
        'points_set': ['AFD', 'ACE', 'BFE', 'BDC']
    })

    # point lies on circle
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B',
        'connection_list': 'AB',
        'remove_set': 'C',
        'points_set': {'A': 'B'}
    })
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B',
        'connection_list': 'AB, AC',
        'points_set': {'A': 'BC'}
    })
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = on_circle D A B',
        'connection_list': 'AB, AC, AD',
        'points_set': {'A': 'BCD'}
    })
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B',
        'connection_list': 'AB, AC, AD, AE',
        'points_set': {'A': 'BCDE'}
    })
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B; F = on_circle F A B',
        'connection_list': 'AB, AC, AD, AE, AF',
        'points_set': {'A': 'BCDEF'}
    })
    shape_datas['PointLiesOnCircle']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = on_circle D A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B',
        'connection_list': 'AB, AC, AD, AE, AF, AG',
        'points_set': {'A': 'BCDEFG'}
    })


    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B',
        'connection_list': 'AB',
        'remove_set': 'C',
        'points_set': {'A': 'B'}
    })
    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B',
        'connection_list': 'AB, AC',
        'points_set': {'A': 'BC'}
    })
    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B',
        'connection_list': 'AB, AC, AE',
        'points_set': {'A': 'BCE'}
    })
    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B',
        'connection_list': 'AB, AC, AE, AF',
        'points_set': {'A': 'BCEF'}
    })
    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B',
        'connection_list': 'AB, AC, AE, AF, AG',
        'points_set': {'A': 'BCEFG'}
    })
    shape_datas['PointLiesOnCircle']['level_2'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H = on_circle H A B',
        'connection_list': 'AB, AC, AE, AF, AG, AH',
        'points_set': {'A': 'BCEFGH'}
    })

    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC',
        'points_set': {'A': 'BC'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B',
        'connection_list': 'AB, AC, AF',
        'points_set': {'A': 'BCF'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B',
        'connection_list': 'AB, AC, AF, AG',
        'points_set': {'A': 'BCFG'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B; H = on_circle H A B',
        'connection_list': 'AB, AC, AF, AG, AH',
        'points_set': {'A': 'BCFGH'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint A C; F = on_circle F A B; G = on_circle G A B; H = on_circle H A B; I = on_circle I A B',
        'connection_list': 'AB, AC, AF, AG, AH, AI',
        'points_set': {'A': 'BCFGHI'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H = on_circle H A B; I = midpoint B C',
        'connection_list': 'AB, AC, AE, AF, AG, AH, BC, IA',
        'points_set': {'A': 'BCEFGH'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = midpoint B C',
        'connection_list': 'AB, AC, BC, AE',
        'points_set': {'A': 'BC'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = lc_tangent E C A',
        'connection_list': 'AB, AC, CE, EA',
        'points_set': {'A': 'BC'}
    })
    shape_datas['PointLiesOnCircle']['level_3'].append({
        'orcle_text': 'A B = segment A B; C = on_circle C A B; D = midpoint A B; E = on_circle E A B; F = on_circle F A B; G = on_circle G A B; H = lc_tangent H C A',
        'connection_list': 'AB, AC, AE, AF, AG, AH, CH',
        'points_set': {'A': 'BCEFG'}
    })

    # parallel
    shape_datas['Parallel']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': [['DE', 'BC']],
        'highlights':['para; D, E, B, C']
    })
    shape_datas['Parallel']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': [['DE', 'BC']],
        'highlights':['para; D, E, B, C']
    })
    shape_datas['Parallel']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': [['DE', 'BC']],
        'highlights':['para; D, E, B, C'],
        'equals':['segments_value;DE=BC']
    })
    shape_datas['Parallel']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = parallelogram A B C D',
        'connection_list': 'AB, AC, BC, DA, DC',
        'points_set': [['AD', 'BC'], ['AB', 'DC']],
        'highlights':['para; A, D, B, C', 'para; A, B, D, C']
    })
    shape_datas['Parallel']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C; F = midpoint B C',
        'connection_list': 'AB, AC, BC, DE, DF, FE',
        'points_set': [['ABD', 'EF'], ['AEC', 'DF'], ['BFC', 'DE']],
        'highlights':['para; A, D, E, F', 'para; A, E, D, F', 'para; B, F, D, E']
    })

    # perpendicular
    shape_datas['Perpendicular']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = foot A B C',
        'connection_list': 'AB, AC, AD, CD, BD',
        'points_set': {'AD': ['BCD'], 'BCD': ['AD']},
        'highlights':['perp; D, A, D, C']
    })
    shape_datas['Perpendicular']['level_1'].append({
        'orcle_text': 'A B C = r_triangle A B C',
        'connection_list': 'AB, AC, BC',
        'points_set': {'AC': ['AB'], 'AB': ['AC']},
        'highlights':['perp; A, C, A, B']
    })
    shape_datas['Perpendicular']['level_1'].append({
        'orcle_text': 'A B = segment A B; C = eq_triangle C A B; D = eq_triangle D A B; E = on_circle E A B',
        'connection_list': 'DC, AB',
        'remove_set': 'E',
        'points_set': {'AB': ['CD'], 'CD': ['AB']},
        'highlights':['perp; A, B, C, D']
    })

    shape_datas['Perpendicular']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = foot A B C; E = foot C A B',
        'connection_list': 'AB, AC, BC, AD, BD, CD, AE, CE, BE',
        'points_set': {'AD': ['BCD'], 'BCD': ['AD'], 'CE': ['BAE'], 'BAE': ['CE']},
        'highlights': ['perp; D, A, D, B', 'perp; E, C, E, B']
    })
    shape_datas['Perpendicular']['level_2'].append({
        'orcle_text': 'A B C = r_triangle A B C; D = foot A B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': {'AD': ['BCD'], 'BCD': ['AD'], 'AB': ['AC'], 'AC': ['AB']},
        'highlights': ['perp; A, C, A, B', 'perp; D, A, D, C']
    })
    shape_datas['Perpendicular']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = foot O A B; E = foot O C A',
        'connection_list': 'AB, AC, OD, OE',
        'points_set': {'OE': ['CEA'], 'ECA': ['OE'], 'ADB': ['OD'], 'OD': ['ADB']},
        'highlights': ['perp; E, O, E, A', 'perp; D, O, D, B']
    })

    shape_datas['Perpendicular']['level_3'].append({
        'orcle_text': 'A B C D = rectangle A B C D; E = intersection_ll A C B D',
        'connection_list': 'AB, AC, BC, DA, DC, BD',
        'points_set': {'AD': ['AB', 'DC'], 'AB': ['AD', 'BC'], 'BC': ['AB', 'DC'], 'DC': ['AD', 'BC']},
        'highlights':['perp; B, A, B, C', 'perp; A, B, A, D', 'perp; D, A, D, C', 'perp; C, D, C, B']
    })
    shape_datas['Perpendicular']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = incenter A B C; D = foot O A C; E = foot O B C; F = foot O A B',
        'connection_list': 'AB, AC, BC, OD, OE, OF',
        'points_set': {'OD': ['ADC'], 'ADC': ['OD'], 'OE': ['BEC'], 'BEC': ['OE'], 'OF': ['AFB'], 'AFB': ['OF']},
        'highlights': ['perp; F, O, F, A', 'perp; D, O, D, C', 'perp; E, O, E, C']
    })
    shape_datas['Perpendicular']['level_3'].append({
        'orcle_text': 'A B C = r_triangle A B C; D = foot A B C; E = foot D A B',
        'connection_list': 'AB, AC, BC, AD, DE',
        'points_set': {'AD': ['BCD'], 'BCD': ['AD'], 'AB': ['AC', "DE"], 'AC': ['AB'], 'DE': ['AB']},
        'highlights': ['perp; A, C, A, B', 'perp; D, A, D, C', 'perp; E, D, E, A']
    })
    shape_datas['Perpendicular']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = foot A B C; E = foot C A B; F = foot B A C',
        'connection_list': 'AB, AC, BC, AD, BD, CD, AE, CE, BE, AF, CF, BF',
        'points_set': {'AD': ['BCD'], 'BCD': ['AD'], 'CE': ['BAE'], 'BAE': ['CE'], 'BF': ['CAF'], 'CAF': ['BF']},
        'highlights': ['perp; D, A, D, B', 'perp; F, B, F, A', 'perp; E, C, E, B']
    })
    
    # equal

    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['segments;DB=DC'],
        'equals': ['segments;DB=DC']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B; O = circle O A B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['segments;DB=DC'],
        'remove_set': 'O',
        'equals': ['segments;DB=DC']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['segments_value;DB=annotation'],
        'equals': ['segments_value;DB=annotation', f'segments_value;DC=annotation']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['segments_value;AC=annotation'],
        'equals': ['segments_value;AC=annotation']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': [f'segments_value;AC=annotation'],
        'equals': [f'segments_value;AC=annotation', f'segments_value;AB=annotation']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['angles_value;ABC=actual'],
        'equals': ['angles_value;ABC=actual']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['angles_value;DAC=actual'],
        'equals': ['angles_value;DAC=actual']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['angles_value;ABC=annotation'],
        'equals': ['angles_value;ABC=annotation']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['angles_value;DAC=annotation'],
        'equals': ['angles_value;DAC=annotation']
    })
    shape_datas['Equal']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = angle_bisector B A C, on_line D C B',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['angles;BAD=CAD'],
        'equals': ['angles;BAD=CAD']
    })

    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['segments;AD=DB'],
        'equals': ['segments;AD=DB']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C; O = circle O A B C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['segments;AD=DB'],
        'remove_set': 'O',
        'equals': ['segments;AD=DB']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['segments;AE=EC'],
        'equals': ['segments;AE=EC']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': [f'segments_value;DE=annotation'],
        'equals': [f'segments_value;DE=annotation', f'segments_value;BE=annotation']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': [f'segments_value;BC=annotation'],
        'equals': [f'segments_value;BC=annotation', f'segments_value;AD=annotation']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['angles;DEB=EBC'],
        'equals': ['angles;DEB=EBC']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['angles_value;BAC=actual'],
        'equals': ['angles_value;BAC=actual']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['angles_value;BCA=actual'],
        'equals': ['angles_value;BCA=actual']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['angles_value;BAC=annotation'],
        'equals': ['angles_value;BAC=annotation']
    })
    shape_datas['Equal']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE, BE',
        'points_set': ['angles_value;BCA=annotation'],
        'equals': ['angles_value;BCA=annotation']
    })

    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': ['segments;DC=DB'],
        'equals': ['segments;DC=DB']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': [f'segments_value;AC=annotation'],
        'equals': [f'segments_value;AC=annotation']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': [f'segments_value;AC=annotation'],
        'equals': [f'segments_value;AC=annotation', f'segments_value;AB=annotation']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': ['angles_value;BCA=actual'],
        'equals': ['angles_value;BCA=actual']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': ['angles_value;BCA=annotation'],
        'equals': ['angles_value;BCA=annotation']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': ['angles;BCD=CBD'],
        'equals': ['angles;BCD=CBD']
    })
    shape_datas['Equal']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; O = circle A B C; D = on_circle D O C, angle_bisector C A B',
        'connection_list': 'AB, AC, BC, CD, DB, AD',
        'remove_set': 'O',
        'points_set': ['angles;CAD=BAD'],
        'equals': ['angles;CAD=BAD']
    })
    

    # angle classification
    shape_datas['AngleClassification']['level_1'].append({
        'orcle_text': f"A B C = triangle A B C",
        'connection_list': 'AB, AC',
        'points_set': ['BAC'],
    })
    shape_datas['AngleClassification']['level_2'].append({
        'orcle_text': f"A B C = triangle A B C",
        'connection_list': 'AB, AC, BC',
        'points_set': ['ABC', 'BCA', 'CAB'],
        'letters': 'ABC'
    })
    shape_datas['AngleClassification']['level_2'].append({
        'orcle_text': f"A B C = triangle A B C",
        'connection_list': 'AB, AC, BC',
        'points_set': ['ABC', 'BCA', 'CAB'],
    })
    shape_datas['AngleClassification']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, AD, BC',
        'points_set': ['BAD', 'ABD', 'BDA', 'DAC', 'ACD', 'ADC', 'BAC']
    })
    shape_datas['AngleClassification']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll F A D B E',
        'connection_list': f'AB, AC, BC, AD, BE',
        'points_set': ['BAF', 'BAD', 'BAE', 'BAC', 'FAE', 'FAC', 'DAE', 'DAC', 'ABE', 'ABF', 'ABD', 'ABC', 'FBD', 'FBC', 'EBD', 'EBC', 'AFB', 'BFD', 'DFE', 'AFE', 'AEF', 'AEB', 'BDA', 'BDF', 'ADC', 'FDC', 'BEC', 'FEC', 'ACB', 'ECB', 'ACD', 'ECD']
    })


    ## LineComparison
    shape_datas['LineComparison']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C',
        'connection_list': 'AB, AC',
        'points_set': ['AB', 'AC'],
        'letters': 'ABCDEFGHIJK'
    })

    shape_datas['LineComparison']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C',
        'connection_list': 'AB, AC',
        'points_set': ['AB', 'AC'],
    })

    shape_datas['LineComparison']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C',
        'connection_list': 'AB, AC, BC',
        'points_set': ['AB', 'AC', 'BC']
    })

    shape_datas['LineComparison']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'BD', 'CD']
    })

    shape_datas['LineComparison']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'AE', 'DE', 'BD', 'CE']
    })


    shape_datas['PointLiesOnLine_empirical']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, AD, BC',
        'points_set': ['BCD']
    })
    shape_datas['PointLiesOnLine_empirical']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['ABD', 'ACE']
    })
    shape_datas['PointLiesOnLine_empirical']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E',
        'connection_list': 'AB, AC, BC, DA, BE',
        'points_set': ['AFD', 'ACE', 'BFE', 'BDC']
    })

    shape_datas['AngleClassification_empirical']['level_1'].append({
        'orcle_text': f"A B C = triangle A B C",
        'connection_list': 'AB, AC',
        'points_set': ['BAC'],
        'letters': 'ABCDEF'
    })
    shape_datas['AngleClassification_empirical']['level_2'].append({
        'orcle_text': f"A B C = triangle A B C",
        'connection_list': 'AB, AC, BC',
        'points_set': ['ABC', 'BCA', 'CAB'],
        'letters': 'ABCDEF'
    })
    shape_datas['AngleClassification_empirical']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, AD',
        'points_set': ['BAD', 'ABD', 'BDA', 'DAC', 'ACD', 'ADC', 'BAC'],
        'letters': 'ABCDEF'
    })


    shape_datas['LineComparison_empirical']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'BD', 'CD'],
        'letters': 'ABCDEF'
    })
    shape_datas['LineComparison_empirical']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'AE', 'DE', 'BD', 'CE'],
        'letters': 'ABCDEF'
    })
    shape_datas['LineComparison_empirical']['level_1'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E',
        'connection_list': 'AB, AC, BC, DA, BE',
        'points_set': ['AB', 'AE', 'EC', 'AC', 'AF', 'FD', 'AD', 'BF', 'FE', 'BE', 'BD', 'DC', 'BC'],
        'letters': 'ABCDEF'
    })

    shape_datas['LineComparison_empirical']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'BD', 'CD'],
        'letters': 'ABCDEFGHIJKLM'
    })
    shape_datas['LineComparison_empirical']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'AE', 'DE', 'BD', 'CE'],
        'letters': 'ABCDEFGHIJKLM'
    })
    shape_datas['LineComparison_empirical']['level_2'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E',
        'connection_list': 'AB, AC, BC, DA, BE',
        'points_set': ['AB', 'AE', 'EC', 'AC', 'AF', 'FD', 'AD', 'BF', 'FE', 'BE', 'BD', 'DC', 'BC'],
        'letters': 'ABCDEFGHIJKLM'
    })

    shape_datas['LineComparison_empirical']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C',
        'connection_list': 'AB, AC, BC, AD',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'BD', 'CD'],
    })
    shape_datas['LineComparison_empirical']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint A B; E = midpoint A C',
        'connection_list': 'AB, AC, BC, DE',
        'points_set': ['AB', 'AC', 'BC', 'AD', 'AE', 'DE', 'BD', 'CE'],
    })
    shape_datas['LineComparison_empirical']['level_3'].append({
        'orcle_text': 'A B C = triangle A B C; D = midpoint B C; E = midpoint A C; F = intersection_ll A D B E',
        'connection_list': 'AB, AC, BC, DA, BE',
        'points_set': ['AB', 'AE', 'EC', 'AC', 'AF', 'FD', 'AD', 'BF', 'FE', 'BE', 'BD', 'DC', 'BC'],
    })

    return shape_datas


if __name__ == '__main__':
    shape_datas = produce_shape_datas()
    for predicate, datas in shape_datas.items():
        print(predicate)
        for level, data in datas.items():
            print(level, end=' ')
            for d in data:
                try:
                    check_data_valid(d)
                except:
                    print(d)
        print()