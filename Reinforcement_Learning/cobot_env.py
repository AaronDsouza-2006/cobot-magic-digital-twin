import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import time
import random
import mujoco
import mujoco.viewer

class CobotEnv(gym.Env):
    def __init__(self, render_mode = None):

        xml_path= "/home/aaron-dsouza/programming/digital_twin/robots/mobile_aloha_sim/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
        self.render_mode = render_mode
        self.model = mujoco.MjModel.from_xml_path(xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

        self.steps = 0
        max_delta_const = 0.1
        self.max_delta = np.array([2, 0.7, 0.7, 1, 1, 1, 0.2]) * max_delta_const
        self.max_speed = 8.0
        self.max_steps = 500
        self.viewer = None

        self.cube_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "cube")
        self.left_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link7_pad")
        self.right_gripper_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "fr_link8_pad")

        self.motor_qpos_low = np.array([-3.14158, 0, -3.14158, -2, -1.5708, -3.14158, 0])
        self.motor_qpos_high = np.array([3.14158, 3.14158, 0, 2, 1.5708, 3.14158, 0.095])

        self.motor_qvel_low = np.full(7, -8.0)
        self.motor_qvel_high = np.full(7, 8.0)
        
        self.cube_pos_low = np.array([0.45, -0.35, 0.7])
        self.cube_pos_high = np.array([0.7, 0.05, 1.5])
        
        self.disk_pos_low = np.array([-0.1, -0.55, 0.7])
        self.disk_pos_high = np.array([0.15, -0.1, 0.8])

        self.gripper_pos_low = np.array([0.3, -0.6, 0.75])
        self.gripper_pos_high = np.array([0.95, 0.2, 1.5])

        all_obs_low = np.concatenate((self.motor_qpos_low, self.motor_qvel_low, self.cube_pos_low, 
                                      self.disk_pos_low, self.gripper_pos_low), dtype=np.float32)
        all_obs_high = np.concatenate((self.motor_qpos_high, self.motor_qvel_high, self.cube_pos_high, 
                                       self.disk_pos_high, self.gripper_pos_high), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=all_obs_low, high=all_obs_high, dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-self.max_delta, high=self.max_delta, shape=(7,), dtype=np.float32
        )

    def get_observation(self):
        motor_qpos = self.sim2real(self.data.qpos[-8:])
        motor_qvel = self.sim2real(self.data.qvel[-8:])
        cube_pos = self.data.xpos[1]
        disk_pos = self.data.xpos[2]
        gripper_pos = self.data.site_xpos[1]
        all_observations = np.concatenate((motor_qpos, motor_qvel, cube_pos, disk_pos, gripper_pos), dtype=np.float32)
        
        return all_observations

    def apply_action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        desired_pos = self.sim2real(self.data.qpos[-8:]) + action
        desired_pos = np.clip(desired_pos, self.motor_qpos_low, self.motor_qpos_high)
        
        self.data.ctrl[-8:] = self.real2sim(desired_pos)
        for _ in range(20):
            mujoco.mj_step(self.model, self.data)

    def compute_rewards(self):
        raise NotImplementedError

    def check_bounds(self, position, min_pos, max_pos):
        within_low = min_pos <= position
        within_high =position <= max_pos 
        is_inside = np.all(within_low) and np.all(within_high)

        return is_inside

    def reset(self):
        mujoco.mj_resetData(self.model, self.data)

        # while(True):
        cube_x = np.random.uniform(self.cube_pos_low[0], self.cube_pos_high[0])
        cube_y = np.random.uniform(self.cube_pos_low[1], self.cube_pos_high[1])
        self.data.qpos[:2] = np.array([cube_x, cube_y]) #(0.65,-0.2)

        disk_x = np.random.uniform(self.disk_pos_low[0], self.disk_pos_high[0])
        disk_y = np.random.uniform(self.disk_pos_low[1], self.disk_pos_high[1])
        self.data.qpos[7:9] = np.array([disk_x, disk_y])

            # if np.linalg.norm( self.data.xpos[1]- self.data.xpos[2]) > 0.012:
            #     break

        mujoco.mj_forward(self.model, self.data)
        self.steps = 0
        return self.get_observation(), {}
        
    def step(self, action):
        self.apply_action(action)
        next_state = self.get_observation()
        reward, info = self.compute_rewards()
        self.steps +=1

        cube_inside = self.check_bounds(self.data.qpos[:3], self.cube_pos_low - 0.1, self.cube_pos_high + 0.1)
        robot_inside = self.check_bounds(self.data.qpos[-8], -1.0, 2.0)
        
        terminated = (
            np.max(np.abs(self.data.qacc[-8:])) > 1e6 or
            info["success"] or 
            not cube_inside or 
            not robot_inside
        )
        truncated = self.steps >= self.max_steps

        if self.render_mode == 'human':
            self.render()
        
        #next_state, reward, terminated, truncated, info =
        return next_state, reward, terminated, truncated, info

    def render(self):
        if self.viewer == None:
            self.viewer = mujoco.viewer.launch_passive(self.model, self.data,
                                                       show_left_ui=False, show_right_ui=False)

        self.viewer.sync()

        if not self.viewer.is_running():
            self.viewer.close()
            self.viewer = None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    #functions to convert between real and simulated robot values
    def sim2real(self, state):
        x = state.copy()
        x[2] *= -1
        x[3], x[4] = x[4], x[3]
        x[6] *=2
        x = x[:-1]
        return x

    def real2sim(self, state):
        x = state.copy()
        x[2] *= -1
        x[3], x[4] = x[4], x[3]
        x[6] /= 2
        x = np.append(x, x[6])
        return x