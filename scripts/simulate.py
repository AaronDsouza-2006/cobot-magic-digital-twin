import mujoco
import mujoco.viewer
import sys

model_path = sys.argv[1]
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        for _ in range(10):
            mujoco.mj_step(model, data)
        viewer.sync()