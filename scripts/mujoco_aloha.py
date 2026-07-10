import mujoco
import os

# Get absolute paths
base_dir = os.path.dirname(os.path.abspath(__file__))  # ~/programming/mujoco
repo_dir = os.path.join(base_dir, "mobile_aloha_sim", "aloha_new_description")
urdf_path = os.path.join(repo_dir, "urdf", "aloha_new.urdf")

# Read original URDF
with open(urdf_path, 'r') as f:
    urdf_content = f.read()

# Replace package:// with relative path from URDF location to mesh folder
# URDF is in: mobile_aloha_sim/aloha_new_description/urdf/
# Meshes are in: mobile_aloha_sim/aloha_new_description/meshes/
# So from URDF's perspective, meshes are at: ../meshes/
urdf_content = urdf_content.replace(
    "package://aloha_new_description/",
    "../"  # relative to urdf/ folder
)

# Save fixed URDF in the SAME directory as original (so relative paths work)
fixed_urdf = os.path.join(repo_dir, "urdf", "aloha_new_fixed.urdf")
with open(fixed_urdf, 'w') as f:
    f.write(urdf_content)

# Load it — MuJoCo resolves mesh paths relative to the XML file's location
model = mujoco.MjModel.from_xml_path(fixed_urdf)
data = mujoco.MjData(model)

print(f"Loaded! Bodies: {model.nbody}, DOFs: {model.nv}, Joints: {model.njnt}")