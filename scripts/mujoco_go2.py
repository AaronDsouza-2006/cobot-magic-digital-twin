import mujoco
import mujoco.viewer
import time

model = mujoco.MjModel.from_xml_path(
    "/home/aaron-dsouza/programming/mujoco/unitree_mujoco/unitree_robots/go2/scene.xml"
)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(0.001)