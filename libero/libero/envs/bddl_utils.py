from bddl.parsing import *

import itertools
import numpy as np
import random
import copy
import numpy as np
import re

pi = np.pi


def get_regions(t, regions, group):
    group.pop(0)
    while group:
        region = group.pop(0)
        region_name = region[0]
        target_name = None
        region_dict = {
            "target": None,
            "ranges": [],
            "extra": [],
            "yaw_rotation": [0, 0],
            "rgba": [0, 0, 1, 0],
        }
        for attribute in region[1:]:
            if attribute[0] == ":target":
                assert len(attribute) == 2
                region_dict["target"] = attribute[1]
                target_name = attribute[1]
            elif attribute[0] == ":ranges":
                for rect_range in attribute[1]:
                    assert (
                        len(rect_range) == 4
                    ), f"Dimension of rectangular range mismatched!!, supposed to be 4, only found {len(rect_range)}"
                    region_dict["ranges"].append([float(x) for x in rect_range])
            elif attribute[0] == ":yaw_rotation":
                # print(attribute[1])
                for value in attribute[1]:
                    region_dict["yaw_rotation"] = [eval(x) for x in value]
            elif attribute[0] == ":rgba":
                assert (
                    len(attribute[1]) == 4
                ), f"Missing specification for rgba color, supposed to be 4 dimension, but only got  {attribute[1]}"
                region_dict["rgba"] = [float(x) for x in attribute[1]]
            else:
                raise NotImplementedError
        regions[target_name + "_" + region_name] = region_dict


def get_scenes(t, scene_properties, group):
    group.pop(0)
    while group:
        scene_property = group.pop(0)
        scene_properties_dict = {}
        for attribute in region[1:]:
            if attribute[0] == ":floor":
                assert len(attribute) == 2
                scene_properties_dict["floor_style"] = attribute[1]
            elif attribute[0] == ":wall":
                assert len(attribute) == 2
                scene_properties_dict["wall_style"] = attribute[1]
            else:
                raise NotImplementedError


def get_problem_info(problem_filename):
    domain_name = "unknown"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                domain_name = "robosuite"
            elif t == ":language":
                group.pop(0)
                language_instruction = group
    return {
        "problem_name": problem_name,
        "domain_name": domain_name,
        "language_instruction": " ".join(language_instruction),
    }


def robosuite_parse_problem(problem_filename):
    domain_name = "robosuite"
    problem_filename = problem_filename
    tokens = scan_tokens(filename=problem_filename)
    if isinstance(tokens, list) and tokens.pop(0) == "define":
        problem_name = "unknown"
        objects = {}
        obj_of_interest = []
        initial_state = []
        goal_state = []
        fixtures = {}
        regions = {}
        scene_properties = {}
        language_instruction = ""
        while tokens:
            group = tokens.pop()
            t = group[0]
            if t == "problem":
                problem_name = group[-1]
            elif t == ":domain":
                if domain_name != group[-1]:
                    raise Exception("Different domain specified in problem file")
            elif t == ":requirements":
                pass
            elif t == ":objects":
                group.pop(0)
                object_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        objects[group.pop(0)] = object_list
                        object_list = []
                    else:
                        object_list.append(group.pop(0))
                if object_list:
                    if not "object" in objects:
                        objects["object"] = []
                    objects["object"] += object_list
            elif t == ":obj_of_interest":
                group.pop(0)
                while group:
                    obj_of_interest.append(group.pop(0))
            elif t == ":fixtures":
                group.pop(0)
                fixture_list = []
                while group:
                    if group[0] == "-":
                        group.pop(0)
                        fixtures[group.pop(0)] = fixture_list
                        fixture_list = []
                    else:
                        fixture_list.append(group.pop(0))
                if fixture_list:
                    if not "fixture" in fixtures:
                        fixtures["fixture"] = []
                    fixtures["fixture"] += fixture_list
            elif t == ":regions":
                get_regions(t, regions, group)
            elif t == ":scene_properties":
                get_scenes(t, scene_properties, group)
            elif t == ":language":
                group.pop(0)
                language_instruction = group

            elif t == ":init":
                group.pop(0)
                initial_state = group
            elif t == ":goal":
                package_predicates(group[1], goal_state, "", "goals")
            else:
                print("%s is not recognized in problem" % t)
        config = {
            "problem_name": problem_name,
            "fixtures": fixtures,
            "regions": regions,
            "objects": objects,
            "scene_properties": scene_properties,
            "initial_state": initial_state,
            "goal_state": goal_state,
            "language_instruction": language_instruction,
            "obj_of_interest": obj_of_interest,
        }
        print("config", config)
        return config
    else:
        raise Exception(
            f"Problem {behavior_activity} {activity_definition} does not match problem pattern"
        )


def modify_initial_state(bddl_config, offset=0.03, protected_objects=None):
    """
    修改BDDL配置，为物品的初始位置添加随机偏移，保持区域大小不变
    
    Args:
        bddl_config: 完整的BDDL配置字典
        protected_objects: 不需要修改位置的物品列表
        offset: 随机偏移的最大值
    
    Returns:
        修改后的BDDL配置字典
    """
    if protected_objects is None:
        protected_objects = []
    
    # 深拷贝配置以避免修改原始数据
    config = copy.deepcopy(bddl_config)
    
    # 修改regions中的位置信息
    for region_name, region_info in config['regions'].items():
        # 跳过不包含ranges的区域或特殊区域
        if not region_info['ranges'] or region_name.endswith('_contain_region'):
            continue
            
        # 检查是否有物体在此区域且在protected_objects中
        skip_region = False
        for state in config['initial_state']:
            if isinstance(state, list) and len(state) >= 3:
                if state[0].lower() == 'on' and state[2] == region_name and state[1] in protected_objects:
                    skip_region = True
                    break
                
        if skip_region:
            continue
            
        # 为区域的每个范围添加随机偏移，保持区域大小不变
        for i, range_coords in enumerate(region_info['ranges']):
            # 确保range_coords是列表而不是浮点数
            if not isinstance(range_coords, list):
                continue
                
            # 确保有4个坐标值
            if len(range_coords) != 4:
                continue
                
            # 假设格式为 [x_min, y_min, x_max, y_max]
            x_min, y_min, x_max, y_max = range_coords
            
            # 计算区域宽度和高度
            width = x_max - x_min
            height = y_max - y_min
            
            # 生成x和y方向的随机偏移
            x_offset = random.uniform(-offset, offset)
            y_offset = random.uniform(-offset, offset)
            
            # 应用偏移，保持区域大小不变
            new_range = [
                x_min + x_offset,  # 新的x_min
                y_min + y_offset,  # 新的y_min
                x_max + x_offset,  # 新的x_max
                y_max + y_offset   # 新的y_max
            ]
            
            region_info['ranges'][i] = new_range
    
    return config


