import asyncio
import websockets
import numpy as np
import json
import pathlib
import os
import logging
import math
import imageio
import random

from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
os.environ["MUJOCO_GL"] = "osmes"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [0.0]

EVO_TEST_ROOT   = os.environ["EVO_TEST_ROOT"]
EVO_CKPT_TMPL   = os.environ["EVO_CKPT_TMPL"]
EVO_SERVER_URL  = os.environ["EVO_SERVER_URL"]
EVO_TASK_SUITES = os.environ["EVO_TASK_SUITES"].split(",")   # 转 list
###################################### 
class Args():
    horizon = 8
    max_steps = [660, 840, 900, 1560]   # 步数与 suite 顺序仍保持一一对应
    SERVER_URL = EVO_SERVER_URL
    filter_category = "background"
    task_suites = EVO_TASK_SUITES       # 现在由环境变量控制
    num_episodes = 1
    SEED = 0

    ckpt_name   = EVO_CKPT_TMPL.format(filter_category=filter_category, horizon=horizon)
    log_file    = f"{EVO_TEST_ROOT}/{ckpt_name}.txt"
    video_log_dir = f"{EVO_TEST_ROOT}/{ckpt_name}/"
    
    
    

args = Args()

log_dir = os.path.dirname(args.log_file)
os.makedirs(log_dir, exist_ok=True)

# 2. 创建视频日志目录
os.makedirs(args.video_log_dir, exist_ok=True)

def fliter(filter_category,task_suite_name,json_path="/home/dell/code/ljt/LIBERO-plus/libero/libero/benchmark/task_classification.json", ):
    with open(json_path, 'r', encoding='utf-8') as file:
        task_json = json.load(file)
    if filter_category != "":
        task_lists = task_json[task_suite_name]
        filter_ids = [t['id'] - 1 for t in task_lists if any(cat in t['category'].lower() for cat in filter_category.split(','))]
        #log_message(f'Evaluating {len(filter_ids)} tasks in perturbation category [{filter_category}]", log_file)
        return filter_ids
    else:
        task_lists = task_json[task_suite_name]
        all_ids = [t['id'] - 1 for t in task_lists]
        return all_ids
    
    
########################################
# Evo1_700M_libero_4suite_stage2_v2/TEST2_65000_h10
# ========= Logging 配置 =========
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        
        logging.FileHandler(args.log_file, mode='a'),
        logging.StreamHandler()
    ]

)
log = logging.getLogger(__name__)

# ========= 图像转为 list[list[list[int]]] =========
def encode_image_array(img_array: np.ndarray):
    return img_array.astype(np.uint8).tolist()

# ========= 四元数转轴角 =========
def quat2axisangle(quat):
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

# ========= Observation 转为 JSON-compatible dict =========
def obs_to_json_dict(obs, prompt, resize_size=448):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    dummy_proc = np.zeros((resize_size, resize_size, 3), dtype=np.uint8)

    data = {
        "image": [
            encode_image_array(img),
            encode_image_array(wrist_img),
            encode_image_array(dummy_proc)
        ],
        "state": np.concatenate((
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )).tolist(),
        "prompt": prompt,
        "image_mask": [1, 1, 0],
        "action_mask": [1] * 7 + [0] * 17,
    }
    return data

# ========= 获取 LIBERO 环境 =========
def get_libero_env(task, resolution=448, seed=args.SEED):
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description

# ========= 保存视频 =========
def save_video(frames, filename="simulation.mp4", fps=20, save_dir="videos_2"):
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)

    if len(frames) > 0:
        imageio.mimsave(filepath, frames, fps=fps)
        print(f" 已保存视频: {filepath} ({len(frames)} 帧)")
    else:
        log.warning(f"⚠️ 无帧数据，未生成视频: {filepath}")






# ========= 主交互逻辑 =========
async def run(SERVER_URL: str, max_steps: int = None, num_episodes: int = None, horizon = None, task_suite_name = None):
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[task_suite_name]()

    print(f"Numbers of tasks: {num_tasks_in_suite}")

    total_success = 0
    total_episodes = 0
    total_steps = 0
    # assert 1==2
    async with websockets.connect(SERVER_URL) as ws:
        # assert 1==2
        log.info(f"===========================Start task suite {task_suite_name}========================")

        for idx, task_id in enumerate(filter_ids):

            print(f"task_id{task_id}")
            #if task_id+1 not in [1,5,7,9] :
             #   continue

            task = task_suite.get_task(task_id)
            initial_states = task_suite.get_task_init_states(task_id)
            env, task_description = get_libero_env(task, resolution=448, seed=args.SEED)

            #log.info(f"\n========= 开始任务 {task_id+1}: {task_description} =========")

            task_success = 0
            task_episodes = min(num_episodes, len(initial_states))

            for ep in range(task_episodes):
                #print(f"\n===== Task {task_id} | Episode {ep+1} =====")

                env.reset()


                obs = env.set_init_state(initial_states[ep])
                t = 0
                while t < 10:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        

                prompt = str(task_description)
                #print(prompt)
                episode_done = False
                max_step = 0
                frames = []

                for step in range(max_steps//args.horizon):
                    max_step += args.horizon

                    send_data = obs_to_json_dict(obs, prompt)
                    await ws.send(json.dumps(send_data))
                    #print(f"[Step {step}] 发送 observation")

                    result = await ws.recv()
                    try:
                        action_list = json.loads(result)
                        actions = np.array(action_list)
                        #print(f"[Step {step}] 收到动作 (shape={actions[0][6]})")
                    except Exception as e:
                        print(f"❌ 动作解析失败: {e}, 内容: {result}")
                        break

                    # 执行动作序列
                    for i in range(horizon):
                        action = actions[i].tolist()
                        #print(action[:7])
                        if action[6]>0.5:
                            action[6] = -1
                        else:
                            action[6] = 1
                        
                        # action[6] = abs(1.0 - action[6])
                        
                        #print(f"gripper action", action[6])
                        try:
                            obs, reward, done, info = env.step(action[:7])
                        except ValueError as ve:
                            print(f"❌ 环境执行动作失败: {ve}")
                            episode_done = False
                            break

                        # 保存渲染帧 (拼接主视角 + 手眼相机)
                        frame = np.hstack([
                            np.rot90(obs["agentview_image"], 2),
                            np.rot90(obs["robot0_eye_in_hand_image"], 2)
                        ])
                        frames.append(frame)

                        #print(f"[Step {step}] reward={reward:.2f}, done={done}")
                        if done:
                            print(" 任务完成。")
                            episode_done = True
                            task_success += 1
                            total_success += 1
                            total_steps += max_step
                            break
                    if episode_done:
                        break

                # 保存视频（文件名带 task_id）
                save_video(frames, f"task{task_id}_episode{ep+1}.mp4", fps=30, save_dir = f"{args.video_log_dir}/{task_suite_name}/{args.filter_category}")

                if episode_done:
                    log.info(f"Task {task_id} | {idx}task | Episode {ep+1}: ✅ Success")
                else:
                    log.info(f"Task {task_id} | Episode {ep+1}: ❌ Fail")

                # exit(0)

            #log.info(f"========= 任务 {task_id} 总结: {task_success}/{task_episodes} 成功 =========")
            total_episodes += task_episodes

        # ======= 全部总结 =======
        log.info("\n========= 全部任务总结 =========")
        log.info(f"✅ 总成功 episode 数量: {total_success}/{total_episodes} | 成功率: {total_success / total_episodes * 100:.2f}%")
        if total_episodes > 0:
            log.info(f" 平均步数: {total_steps / total_episodes:.2f}")

# ========= 启动入口 =========
if __name__ == "__main__":
     # 全局随机种子
    np.random.seed(args.SEED)
    random.seed(args.SEED)
    
    for name, max_steps in zip(args.task_suites, args.max_steps):
        filter_ids = fliter(filter_category=args.filter_category, task_suite_name=name)
        num_tasks_in_suite = len(filter_ids)
        asyncio.run(run(SERVER_URL = args.SERVER_URL,
                        max_steps=max_steps, 
                        num_episodes=args.num_episodes,
                        horizon=args.horizon,
                        task_suite_name=name))