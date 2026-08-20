"""'
Refer to:   lerobot/lerobot/scripts/eval.py
            lerobot/lerobot/scripts/econtrol_robot.py
            lerobot/robot_devices/control_utils.py
"""

import time
import torch
import logging

import numpy as np
from pprint import pformat
from dataclasses import asdict
from torch import nn
from contextlib import nullcontext
from typing import Any
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.utils import (
    get_safe_torch_device,
    init_logging,
)
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.pretrained import PreTrainedPolicy
from multiprocessing.sharedctypes import SynchronizedArray
from lerobot.processor.rename_processor import rename_stats
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
)
from unitree_lerobot.eval_robot.make_robot import (
    setup_image_client,
    setup_robot_interface,
    process_images_and_observations,
)
from unitree_lerobot.eval_robot.utils.utils import (
    cleanup_resources,
    predict_action,
    to_list,
    to_scalar,
    EvalRealConfig,
)
from unitree_lerobot.eval_robot.utils.rerun_visualizer import RerunLogger, visualization_data

import logging_mp

logger_mp = logging_mp.getLogger(__name__)
logger_mp.setLevel(logging_mp.INFO)


def eval_policy(
    cfg: EvalRealConfig,
    dataset: LeRobotDataset,
    policy: PreTrainedPolicy | None = None,
    preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
    postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
):
    assert isinstance(policy, nn.Module), "Policy must be a PyTorch nn module."

    logger_mp.info(f"Arguments: {cfg}")

    if cfg.visualization:
        rerun_logger = RerunLogger()

    # Reset policy and processor if they are provided
    if policy is not None and preprocessor is not None and postprocessor is not None:
        policy.reset()
        preprocessor.reset()
        postprocessor.reset()

    image_info = None
    try:
        # --- Setup Phase ---
        image_info = setup_image_client(cfg)
        robot_interface = setup_robot_interface(cfg)

        # Unpack interfaces for convenience
        arm_ctrl, arm_ik, ee_shared_mem, arm_dof, ee_dof = (
            robot_interface[key] for key in ["arm_ctrl", "arm_ik", "ee_shared_mem", "arm_dof", "ee_dof"]
        )
        # local patch: setup_image_client returns (client, config); the dict of
        # shared-memory arrays this expected no longer exists upstream.
        img_client, camera_config = image_info

        # Get initial pose from the first step of the dataset
        from_idx = dataset.meta.episodes["dataset_from_index"][0]
        step = dataset[from_idx]
        init_arm_pose = step["observation.state"][:arm_dof].cpu().numpy()

        user_input = input("Enter 's' to initialize the robot and start the evaluation: ")
        idx = 0
        print(f"user_input: {user_input}")
        full_state = None
        if user_input.lower() == "s":
            # "The initial positions of the robot's arm and fingers take the initial positions during data recording."
            logger_mp.info("Initializing robot to starting pose...")
            tau = robot_interface["arm_ik"].solve_tau(init_arm_pose)
            robot_interface["arm_ctrl"].ctrl_dual_arm(init_arm_pose, tau)
            time.sleep(1.0)  # Give time for the robot to move

            # local patch: asynchronous inference. A worker thread owns
            # sensing + policy inference and keeps a small buffer of ready
            # actions; the main loop only executes from the buffer at the
            # control rate. The policy's expensive chunk inference (e.g.
            # diffusion denoising) then overlaps with execution of the
            # previous chunk instead of freezing the robot.
            if getattr(cfg, "async_inference", False):
                import threading
                from collections import deque

                logger_mp.info(f"Starting ASYNC evaluation loop at {cfg.frequency} Hz.")
                if cfg.visualization:
                    logger_mp.warning("visualization is not supported in async mode; ignoring.")
                action_buffer = deque()
                buffer_lock = threading.Lock()
                stop_event = threading.Event()
                # keep at most one policy chunk of actions ahead: bounds
                # observation staleness to ~n_action_steps control ticks
                buffer_target = int(getattr(policy.config, "n_action_steps", 8))
                device = get_safe_torch_device(policy.config.device)

                def sense_and_infer():
                    while not stop_event.is_set():
                        with buffer_lock:
                            depth = len(action_buffer)
                        if depth >= buffer_target:
                            time.sleep(0.002)
                            continue
                        obs, arm_q = process_images_and_observations(img_client, camera_config, arm_ctrl)
                        if arm_q is None or not any(
                            k.startswith("observation.images.") and v is not None for k, v in obs.items()
                        ):
                            time.sleep(0.01)
                            continue
                        l_ee = r_ee = np.array([])
                        if cfg.ee:
                            with ee_shared_mem["lock"]:
                                fs = np.array(ee_shared_mem["state"][:])
                                l_ee, r_ee = fs[:ee_dof], fs[ee_dof:]
                        obs["observation.state"] = torch.from_numpy(
                            np.concatenate((arm_q, l_ee, r_ee), axis=0)
                        ).float()
                        act = predict_action(
                            obs, policy, device, preprocessor, postprocessor,
                            policy.config.use_amp, step["task"],
                            use_dataset=cfg.use_dataset, robot_type=None,
                        )
                        with buffer_lock:
                            action_buffer.append(act.cpu().numpy())

                worker = threading.Thread(target=sense_and_infer, daemon=True, name="sense_and_infer")
                worker.start()
                try:
                    while True:
                        loop_start_time = time.perf_counter()
                        action_np = None
                        with buffer_lock:
                            if action_buffer:
                                action_np = action_buffer.popleft()
                        if action_np is None:
                            # buffer starved (e.g. very first chunk still
                            # computing): hold pose for one tick
                            time.sleep(1.0 / cfg.frequency)
                            continue
                        arm_action = action_np[:arm_dof]
                        tau = arm_ik.solve_tau(arm_action)
                        arm_ctrl.ctrl_dual_arm(arm_action, tau)
                        if cfg.ee:
                            ee_start = arm_dof
                            l_act = action_np[ee_start : ee_start + ee_dof]
                            r_act = action_np[ee_start + ee_dof : ee_start + 2 * ee_dof]
                            if isinstance(ee_shared_mem["left"], SynchronizedArray):
                                ee_shared_mem["left"][:] = to_list(l_act)
                                ee_shared_mem["right"][:] = to_list(r_act)
                            elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                                ee_shared_mem["left"].value = to_scalar(l_act)
                                ee_shared_mem["right"].value = to_scalar(r_act)
                        idx += 1
                        time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
                finally:
                    stop_event.set()
                    worker.join(timeout=2.0)

            # --- Run Main Loop ---
            logger_mp.info(f"Starting evaluation loop at {cfg.frequency} Hz.")
            while True:
                loop_start_time = time.perf_counter()
                # 1. Get Observations
                # local patch: current upstream signature
                observation, current_arm_q = process_images_and_observations(img_client, camera_config, arm_ctrl)
                # local patch: skip the tick instead of crashing when a camera
                # frame or arm state is momentarily unavailable (e.g. image
                # client still warming up); the arm holds its last target.
                if current_arm_q is None or not any(k.startswith("observation.images.") and v is not None for k, v in observation.items()):
                    logger_mp.warning("Incomplete observation (no image/arm state); skipping tick.")
                    time.sleep(1.0 / cfg.frequency)
                    continue
                left_ee_state = right_ee_state = np.array([])
                if cfg.ee:
                    with ee_shared_mem["lock"]:
                        full_state = np.array(ee_shared_mem["state"][:])
                        left_ee_state = full_state[:ee_dof]
                        right_ee_state = full_state[ee_dof:]
                state_tensor = torch.from_numpy(
                    np.concatenate((current_arm_q, left_ee_state, right_ee_state), axis=0)
                ).float()
                observation["observation.state"] = state_tensor
                # 2. Get Action from Policy
                action = predict_action(
                    observation,
                    policy,
                    get_safe_torch_device(policy.config.device),
                    preprocessor,
                    postprocessor,
                    policy.config.use_amp,
                    step["task"],
                    use_dataset=cfg.use_dataset,
                    robot_type=None,
                )
                action_np = action.cpu().numpy()
                # 3. Execute Action
                arm_action = action_np[:arm_dof]
                tau = arm_ik.solve_tau(arm_action)
                arm_ctrl.ctrl_dual_arm(arm_action, tau)

                if cfg.ee:
                    ee_action_start_idx = arm_dof
                    left_ee_action = action_np[ee_action_start_idx : ee_action_start_idx + ee_dof]
                    right_ee_action = action_np[ee_action_start_idx + ee_dof : ee_action_start_idx + 2 * ee_dof]
                    # logger_mp.info(f"EE Action: left {left_ee_action}, right {right_ee_action}")

                    if isinstance(ee_shared_mem["left"], SynchronizedArray):
                        ee_shared_mem["left"][:] = to_list(left_ee_action)
                        ee_shared_mem["right"][:] = to_list(right_ee_action)
                    elif hasattr(ee_shared_mem["left"], "value") and hasattr(ee_shared_mem["right"], "value"):
                        ee_shared_mem["left"].value = to_scalar(left_ee_action)
                        ee_shared_mem["right"].value = to_scalar(right_ee_action)

                if cfg.visualization:
                    visualization_data(idx, observation, state_tensor.numpy(), action_np, rerun_logger)
                idx += 1
                # Maintain frequency
                time.sleep(max(0, (1.0 / cfg.frequency) - (time.perf_counter() - loop_start_time)))
    except Exception as e:
        logger_mp.info(f"An error occurred: {e}")
    finally:
        if isinstance(image_info, dict):  # local patch: shm dict no longer exists upstream
            cleanup_resources(image_info)


@parser.wrap()
def eval_main(cfg: EvalRealConfig):
    logging.info(pformat(asdict(cfg)))

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Making policy.")

    dataset = LeRobotDataset(repo_id=cfg.repo_id)

    policy = make_policy(cfg=cfg.policy, ds_meta=dataset.meta)
    policy.eval()

    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        dataset_stats=rename_stats(dataset.meta.stats, cfg.rename_map),
        preprocessor_overrides={
            "device_processor": {"device": cfg.policy.device},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )

    with torch.no_grad(), torch.autocast(device_type=device.type) if cfg.policy.use_amp else nullcontext():
        eval_policy(cfg, dataset, policy, preprocessor, postprocessor)

    logging.info("End of eval")


if __name__ == "__main__":
    init_logging()
    eval_main()
