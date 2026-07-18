import mujoco
import mujoco.viewer
import sys

if len(sys.argv) == 2:
    model_path = sys.argv[1]
else:
    model_path = "robots/mobile_aloha_sim/aloha_mujoco/aloha/meshes_mujoco/aloha_v1.xml"
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for _ in range(10):
                mujoco.mj_step(model, data)
            viewer.sync()

except KeyboardInterrupt:
    sys.exit(0)