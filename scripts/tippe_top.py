import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path("/home/aaron-dsouza/programming/mujoco/models/tippe_top.xml")
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)

# CRITICAL: Apply the keyframe
#mujoco.mj_resetDataKeyframe(model, data, 0)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)