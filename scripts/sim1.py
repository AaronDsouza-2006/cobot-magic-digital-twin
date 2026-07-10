import mujoco
import numpy as np
import mujoco.viewer

# Load model and data
model = mujoco.MjModel.from_xml_path("../models/pole.xml")
data = mujoco.MjData(model)

# Get actuator index (or just know it's 0 since there's only one)
act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "hinge_ctrl")


with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        #target_angle = input()
        # Set control
        data.ctrl[act_id] = np.sin(data.time)
        # Step simulation for 1 second
        for _ in range(60):
            mujoco.mj_step(model, data)
            # optionally render here
        viewer.sync()