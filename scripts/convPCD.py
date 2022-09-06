import open3d as o3d
pcd = o3d.io.read_point_cloud(r"E:\Users\kgsel\Downloads\001_chips_can_berkeley_meshes\001_chips_can\tsdf\nontextured.ply")
o3d.io.write_point_cloud(r"E:\Users\kgsel\Downloads\001_chips_can_berkeley_meshes\001_chips_can\tsdf\sink_pointcloud.pcd", pcd)