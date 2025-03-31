import numpy as np
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import os
from pylab import mpl
import torch
import os.path as osp

# 设置中文显示字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_object" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# 创建保存路径
save_dir = "/home/ljr/embodied/LIBERO/libero/libero/init_files/libero_object"
save_filename = f"{task.bddl_file.split('.')[0]}.test_init"
save_path = osp.join(save_dir, save_filename)

# 确保目录存在
os.makedirs(save_dir, exist_ok=True)

# 收集50个不同seed的状态
num_seeds = 50
all_states = []

for seed in range(num_seeds):
    # 为每个种子重新创建环境
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": 256,  # 增加相机分辨率
        "camera_widths": 256,   # 增加相机分辨率
        "scene_properties": {
            "floor_style": "light-gray",  # 自定义地板样式
            "wall_style": "flower",  # 自定义墙壁样式
        },
        "modify": True,
        "offset": 0.03,
    }
    env = OffScreenRenderEnv(**env_args)
    
    # # 设置种子并重置环境
    # env.seed(seed)
    # env.reset()
    
    # 获取当前状态
    current_state = env.sim.get_state().flatten()
    all_states.append(current_state)
    
    print(f"已收集seed {seed} 的状态，维度: {current_state.shape}")
    
    # 关闭环境
    env.close()

# 将所有状态转换为numpy数组，然后转为torch tensor
all_states_np = np.array(all_states)
all_states_torch = torch.from_numpy(all_states_np)

# 保存到指定文件
torch.save(all_states_torch, save_path)
print(f"已保存{num_seeds}个初始状态到文件: {save_path}")
print(f"状态数组形状: {all_states_np.shape}")