# OBB corner extraction instead of Convex Hull
import os
import trimesh
import numpy as np
import open3d as o3d

# -------------------------
# 設定路徑
# -------------------------
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)

obj_path = os.path.join(project_root, "obj_files", "Cabinet_12_07.obj")
xyz_path = os.path.join(project_root, "Construct_test")
output_xyz = os.path.join(xyz_path, "S_cabinet_12_07.xyz")

mesh = trimesh.load(obj_path)
vertices = np.array(mesh.vertices)

# ===========================================
# Step 1：取 Oriented Bounding Box（OBB）
# ===========================================
obb = mesh.bounding_box_oriented
obb_points = np.array(obb.vertices)   # (8,3)

print("使用 OBB 角點作為控制點:", obb_points.shape)

# ===========================================
# Step 2：可視化（用 open3d 顯示 OBB 角點）
# ===========================================
mesh_o3d = o3d.geometry.TriangleMesh()
mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
mesh_o3d.triangles = o3d.utility.Vector3iVector(mesh.faces)
mesh_o3d.compute_vertex_normals()
mesh_o3d.paint_uniform_color([0.7, 0.7, 0.7])

bbox = mesh.bounding_box_oriented
diag_len = np.linalg.norm(bbox.extents)
radius = diag_len * 0.003

spheres = []
for p in obb_points:
    sp = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
    sp.translate(p)
    sp.paint_uniform_color([1, 0, 0])   # Red spheres for OBB corners
    spheres.append(sp)

o3d.visualization.draw_geometries([mesh_o3d] + spheres)

# ===========================================
# Step 3：旋轉並輸出 XYZ（與你原本一致）
# ===========================================
theta = np.pi / 2
R_x = np.array([
    [1, 0, 0],
    [0, np.cos(theta), -np.sin(theta)],
    [0, np.sin(theta),  np.cos(theta)]
])

obb_rot = obb_points @ R_x.T
np.savetxt(output_xyz, obb_rot, fmt="%.6f")

print(f"已輸出 {len(obb_rot)} 個控制點至：{output_xyz}")
print(f"當前工作路徑: {os.getcwd()}")