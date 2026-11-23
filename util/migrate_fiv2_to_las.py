import os
import time
import glob
import laspy
import yaml
import argparse

import numpy as np

from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.spatial import ConvexHull
from pc_io import read_pc, PlyData, _add_las_field_if_not_exist


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_path", help="Path to original FOR-InstanceV2 dataset.")
    parser.add_argument(
        "--dest_path", help="Path where las version of FOR-InstanceV2 should be saved."
    )
    parser.add_argument(
        "--keep_names", action="store_true", help="If True, keep original names."
    )
    parser.add_argument(
        "--crown_ctg",
        action="store_true",
        help="If True, compute crown categories for each plot and save them in a YAML file.",
    )
    args = parser.parse_args()
    if args.keep_names:
        # Crown categories cannot be computed because the original plot names are incompatible with the format
        assert not args.crown_ctg, (
            "Cannot compute and store crown categories when '--keep_names' is set!"
        )

    return args


def migrate_fiv2_to_las(
    src_path: str,
    dest_path: str,
    keep_names: bool,
    crown_ctg: bool,
    training_dir: str = "train_val_data",
    test_dir: str = "test_data",
    ply_semantic_gt_field: str = "semantic_seg",
    ply_instance_gt_field: str = "treeID",
    las_semantic_gt_field: str = "classification",
    las_instance_gt_field: str = "tree_index",
) -> None:
    train_file_list = glob.glob(os.path.join(src_path, training_dir, "*.ply"))
    train_file_list.sort()
    test_file_list = glob.glob(os.path.join(src_path, test_dir, "*.ply"))
    test_file_list.sort()
    if not keep_names:
        train_filename_mapping = _generate_filename_mapping(train_file_list)
        n_train_files = len(train_filename_mapping)
        test_filename_mapping = _generate_filename_mapping(
            test_file_list, start_ind=n_train_files + 1
        )
    else:
        train_filename_mapping = None
        test_filename_mapping = None
    # Migrate files in training set
    train_dest_path = os.path.join(dest_path, training_dir)
    if not os.path.exists(train_dest_path):
        os.makedirs(train_dest_path)
    print("==> Migrating FOR-InstanceV2 training split to las")
    train_start_time = time.time()
    train_plot_data = _migrate_files_to_las(
        train_file_list,
        train_dest_path,
        train_filename_mapping,
        crown_ctg,
        ply_semantic_gt_field,
        ply_instance_gt_field,
        las_semantic_gt_field,
        las_instance_gt_field,
    )
    train_end_time = time.time()
    print(
        f"==> Finish migrating FOR-InstanceV2 training split in {(train_end_time - train_start_time):.3f} s!"
    )
    # Migrate files in test set
    test_dest_path = os.path.join(dest_path, test_dir)
    if not os.path.exists(test_dest_path):
        os.makedirs(test_dest_path)
    print("==> Migrating FOR-InstanceV2 test set to las")
    test_start_time = time.time()
    test_plot_data = _migrate_files_to_las(
        test_file_list,
        test_dest_path,
        test_filename_mapping,
        crown_ctg,
        ply_semantic_gt_field,
        ply_instance_gt_field,
        las_semantic_gt_field,
        las_instance_gt_field,
    )
    test_end_time = time.time()
    print(
        f"==> Finish migrating FOR-InstanceV2 test split in {(test_end_time - test_start_time):.3f} s!"
    )
    if crown_ctg:
        print("==> Saving plot data in YAML format")
        yaml_start_time = time.time()
        # Merge plot data from train and test sets into a single dictionary
        plot_data = {}
        for key, val in train_plot_data.items():
            val["test"] = False
            plot_data[key] = val
        for key, val in test_plot_data.items():
            val["test"] = True
            plot_data[key] = val
        _write_to_yaml(plot_data, os.path.join(dest_path, "plot_data.yaml"))
        # Construct a dictionary for mapping each file back to the original filename
        filename_mapping_inv = {}
        for key, val in train_filename_mapping.items():
            filename_mapping_inv[val] = key
        for key, val in test_filename_mapping.items():
            filename_mapping_inv[val] = key
        _write_to_yaml(
            filename_mapping_inv, os.path.join(dest_path, "filename_mapping_inv.yaml")
        )
        yaml_end_time = time.time()
        print(
            f"==> Finish saving plot data in {(yaml_end_time - yaml_start_time):.3f} s!"
        )


def _generate_filename_mapping(
    file_list: List[str],
    start_ind: int = 1,
    files_to_split: List[str] = ["BlueCat_RN_merged_trees_train"],
    split_size: Dict[str, Tuple[int, int]] = {"BlueCat_RN_merged_trees_train": (3, 2)},
) -> Dict[str, str]:
    filename_mapping = {}
    plot_id_counter = start_ind
    for file in file_list:
        filename = Path(file).name[:-4]
        if filename in files_to_split:
            file_split_size = split_size[filename]
            n_tiles = file_split_size[0] * file_split_size[1]
            for i in range(1, n_tiles + 1):
                tile_filename = f"{filename}_tile_{i}"
                filename_mapping[tile_filename] = f"plot_{plot_id_counter}"
                plot_id_counter += 1
        else:
            filename_mapping[filename] = f"plot_{plot_id_counter}"
            plot_id_counter += 1

    return filename_mapping


def _migrate_files_to_las(
    file_list: List[str],
    dest_path: str,
    filename_mapping: Optional[Dict[str, str]],
    crown_ctg: bool,
    ply_semantic_gt_field: str = "semantic_seg",
    ply_instance_gt_field: str = "treeID",
    las_semantic_gt_field: str = "classification",
    las_instance_gt_field: str = "tree_index",
    files_to_split: List[str] = ["BlueCat_RN_merged_trees_train"],
    split_size: Dict[str, Tuple[int, int]] = {"BlueCat_RN_merged_trees_train": (3, 2)},
) -> Optional[Dict]:
    if crown_ctg:
        plot_data = {}
    else:
        plot_data = None
    n_files = len(file_list)
    for i, file in enumerate(file_list, start=1):
        filename = Path(file).name[:-4]
        status = f"==> Migrating '{filename}' [{i}/{n_files}]"
        print(f"\r{status.ljust(100)}", end="", flush=True)
        ply_pc = read_pc(
            file,
            "ply",
            instance_gt_field=ply_instance_gt_field,
            semantic_gt_field=ply_semantic_gt_field,
        )
        if filename in files_to_split:
            nx = split_size[filename][0]
            ny = split_size[filename][1]
            # The BlueCat training file is so larger that we split it into 6 tiles (of similar size)
            tile_masks = _split_pointcloud_into_tiles(ply_pc.xyz, nx, ny)
            # Extract instance corresponding to the tile from the point cloud and save the point cloud in las
            # (The function ensures that 1. each original instance is only saved once, 2. instances are not split at tile boundaries)
            _process_tiles(
                ply_pc,
                tile_masks,
                dest_path,
                filename,
                filename_mapping,
                crown_ctg,
                plot_data,
                ply_semantic_gt_field,
                ply_instance_gt_field,
                las_semantic_gt_field,
                las_instance_gt_field,
            )
        else:
            if filename_mapping is not None:
                out_filename = filename_mapping[filename]
            else:
                out_filename = filename
            # Write ply data into a las file
            out_path = os.path.join(dest_path, f"{out_filename}.las")
            # Center point cloud around (0, 0)
            x, y, z = ply_pc.x, ply_pc.y, ply_pc.z
            x_center = (x.min() + x.max()) / 2
            y_center = (y.min() + y.max()) / 2
            z_min = z.min()
            x_centered = x - x_center
            y_centered = y - y_center
            z_normalized = z - z_min
            xyz = np.column_stack((x_centered, y_centered, z_normalized))
            # Map instance ids to a continuous range
            instance_gt = ply_pc[ply_instance_gt_field]
            instance_mask = instance_gt > 0
            instance_gt[instance_mask] = (
                np.unique(instance_gt[instance_mask], return_inverse=True)[1] + 1
            )
            _write_las(
                out_path,
                xyz,
                ply_pc[ply_semantic_gt_field],
                ply_pc[ply_instance_gt_field],
                las_semantic_gt_field,
                las_instance_gt_field,
            )
            # Compute plot information (tree location, height and crown category), and save it in the plot_data dictionary (modified in-place)
            if crown_ctg:
                _compute_tree_loc_and_height(
                    out_path, las_semantic_gt_field, las_instance_gt_field, plot_data
                )
    print()

    return plot_data


def _split_pointcloud_into_tiles(
    points: np.ndarray, nx: int = 3, ny: int = 2, sample_size: int = 200_000
) -> List[np.ndarray]:
    N = points.shape[0]
    # Subsample for faster PCA
    ind = np.random.choice(N, min(sample_size, N), replace=False)
    sample = points[ind]
    c = sample.mean(axis=0)
    cov = np.cov((sample - c).T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    R = eigvecs[:, order]
    proj = (points - c) @ R
    x = proj[:, 0]
    y = proj[:, 1]
    # Compute grid edges
    x_edges = np.linspace(x.min(), x.max(), nx + 1)
    y_edges = np.linspace(y.min(), y.max(), ny + 1)
    # Bin assignment
    xi = np.clip(np.digitize(x, x_edges) - 1, 0, nx - 1)
    yi = np.clip(np.digitize(y, y_edges) - 1, 0, ny - 1)
    # Boolean masks for each tile
    tile_ids = yi * nx + xi
    masks = [(tile_ids == i) for i in range(nx * ny)]

    return masks


def _process_tiles(
    pc: PlyData,
    tile_masks: List[np.ndarray],
    dest_path: str,
    filename: str,
    filename_mapping: Optional[Dict[str, str]],
    crown_ctg: bool,
    plot_data: Optional[Dict],
    ply_semantic_gt_field: str = "semantic_seg",
    ply_instance_gt_field: str = "treeID",
    las_semantic_gt_field: str = "classification",
    las_instance_gt_field: str = "tree_index",
) -> None:
    used_instance_ids = []
    for i, mask in enumerate(tile_masks):
        tile_instance_gt = pc[ply_instance_gt_field][mask]
        unq_instance_ids = np.unique(tile_instance_gt)
        valid_instance_ids = ~np.isin(unq_instance_ids, used_instance_ids)
        valid_unq_instances = unq_instance_ids[valid_instance_ids]
        tile_points_mask = np.isin(pc[ply_instance_gt_field], valid_unq_instances)
        # Mask of non-instance points within tile boundaries
        tile_non_instance_points_mask = mask & (pc[ply_instance_gt_field] <= 0)
        # Add non-instance points within boundaries to current tile mask
        tile_points_mask = tile_points_mask | tile_non_instance_points_mask
        # Extract tile data
        tile_instance_gt = pc[ply_instance_gt_field][tile_points_mask]
        tile_semantic_gt = pc[ply_semantic_gt_field][tile_points_mask]
        tile_xyz = pc.xyz[tile_points_mask]
        # Center point cloud around (0, 0)
        x, y, z = tile_xyz[:, 0], tile_xyz[:, 1], tile_xyz[:, 2]
        x_center = (x.min() + x.max()) / 2
        y_center = (y.min() + y.max()) / 2
        z_min = z.min()
        x_centered = x - x_center
        y_centered = y - y_center
        z_normalized = z - z_min
        tile_xyz = np.column_stack((x_centered, y_centered, z_normalized))
        # Mark instances as used
        used_instance_ids.extend(valid_unq_instances.tolist())
        # Map instances to continuous range
        instance_mask = tile_instance_gt > 0
        tile_instance_gt[instance_mask] = (
            np.unique(tile_instance_gt[instance_mask], return_inverse=True)[1] + 1
        )
        tile_filename = f"{filename}_tile_{i + 1}"
        if filename_mapping is not None:
            tile_filename = filename_mapping[tile_filename]
        out_path = os.path.join(dest_path, f"{tile_filename}.las")
        _write_las(
            out_path,
            tile_xyz,
            tile_semantic_gt,
            tile_instance_gt,
            las_semantic_gt_field,
            las_instance_gt_field,
        )
        if crown_ctg:
            _compute_tree_loc_and_height(
                out_path, las_semantic_gt_field, las_instance_gt_field, plot_data
            )


def _write_las(
    out_path: str,
    xyz: np.ndarray,
    semantic_gt: np.ndarray,
    instance_gt: np.ndarray,
    las_semantic_gt_field: str,
    las_instance_gt_field: str,
) -> None:
    header = laspy.LasHeader(point_format=7, version="1.4")
    offsets = np.array([0.0, 0.0, 0.0])
    scales = np.array([0.0001, 0.0001, 0.0001])
    header.scales = scales
    header.offsets = offsets
    # Create a new LasData object
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    _add_las_field_if_not_exist(
        las,
        [
            las_semantic_gt_field,
            las_instance_gt_field,
        ],
        [np.uint16, np.uint16],
    )
    las[las_semantic_gt_field] = semantic_gt
    las[las_instance_gt_field] = instance_gt
    las.write(out_path)


def _compute_tree_loc_and_height(
    las_path: str,
    las_semantic_gt_field: str,
    las_instance_gt_field: str,
    plot_data: Dict,
    ground_index: int = 0,
    crown_subsample_n_points: int = 10_000,
    area_subsample_n_points: int = 1_000_000,
    z_outlier_dist: float = 0.25,
    ground_circle_r: float = 0.5,
    crown_only: bool = True,
    crown_h: float = 3,
    neighbor_d: float = 3,
    next_to_d: float = 1.5,
    dominant_h: float = 2,
) -> None:
    """Compute location, height and crown category of each tree within a forest plot. Append this information to the plot_data dictionary in-place.

    Args:
        las_path (str): path to las file.
        las_semantic_gt_field (str): name of field containing semantic ground truth.
        las_instance_gt_field (str): name of field containing instance ground truth.
        plot_data (Dict): plot data dictionary. Data of current plot is added to this.
        ground_index (int, optional): id of ground points in the semantic field. Defaults to 0.
        crown_subsample_n_points (int, optional): number of points used when computing tree location based on centroid of crown convex hull. Defaults to 1_000.
        area_subsample_n_points (int, optional): number of points to use when computing plot area based on convex hull fitted around the points projected into the xy-plane. Defaults to 1_000_000.
        z_outlier_dist (float, optional): Maximum distance from the highest/lowest point to the second highest/lowest before the point is considered an outlier. Defaults to 0.25.
        ground_circle_r (float, optional): Radius of the circle from which the height of the ground below the tree is computed. Defaults to 0.5.
        crown_only (bool, optional): If True, only use the crown points for convex hull. Defaults to True.
        crown_h (float, optional): Size of crown (points <crown_h> meters downwards from the top of the tree). Defaults to 3.
        neighbor_d (float, optional): Distance at which trees are considered neighbors. Defaults to 3.
        next_to_d (float, optional): Distance at which trees are considered 'next to' each other (closer then just neighbors). Defaults to 1.5.
        dominant_h (float, optional): Height difference at which a tree is considered dominant. Defaults to 2.
    """
    rng = np.random.default_rng(seed=33)
    filename = Path(las_path).name[:-4]
    plot_id = int(filename.split("_")[-1])
    # Read plot las file and extract ground points
    las = laspy.read(las_path)
    n_points = len(las.x)
    tree_ind = np.unique(las[las_instance_gt_field][las[las_instance_gt_field] != 0])
    n_trees = len(tree_ind)
    ground_points = las[las[las_semantic_gt_field] == ground_index]
    if len(ground_points) < 1:
        # Some plots contain only tree points
        ground_points = None
    # Compute area and tree density of the plot
    # Area is computed from subsampled point cloud (approximately same result, much faster for large point clouds)
    area_subsample_n_points = np.min([area_subsample_n_points, n_points])
    sampled_indices = rng.choice(
        np.arange(0, n_points), area_subsample_n_points, replace=False
    )
    las_xy = las.xyz[sampled_indices, :2]
    ch = ConvexHull(las_xy)
    # When points are 2D, the volume property returns the area when using scipy ConvexHull
    plot_area = ch.volume
    # Convert to hectares
    plot_area = plot_area * 0.0001
    # Forest density (tree/ha)
    density = n_trees / plot_area
    # Dictionary for plot data
    current_plot_data = {
        "n_trees": n_trees,
        "area": float(np.round(plot_area, 3)),
        "density": float(np.round(density, 3)),
        "test": None,
        "trees": {},
    }

    # Iterate through all trees and compute their location and height
    for ind in tree_ind:
        tree_points = las[las[las_instance_gt_field] == ind]
        z_sorted = np.sort(tree_points.z)
        z_max, z_2max = z_sorted[-1], z_sorted[-2]
        z_diff = np.abs(z_max - z_2max)
        if z_diff > z_outlier_dist:
            z_max = z_2max
        if crown_only:
            # Only consider crown points, when computing tree location
            crown_mask = tree_points.z >= (z_max - crown_h)
            n_points_crown = np.sum(crown_mask)
            crown_h_new = crown_h
            while n_points_crown < 3:
                crown_h_new += 0.1
                # If not enough crown points to generate convex hull, increase crown height until there are
                crown_mask = tree_points.z >= (z_max - crown_h_new)
                n_points_crown = np.sum(crown_mask)
            tree_points = tree_points[crown_mask]
        # Project tree points to xy-plane and compute centroid of convex hull
        convex_hull = ConvexHull(tree_points.xyz[:, :2], qhull_options="QJ")
        # Centroid computation from: https://stackoverflow.com/questions/31562534/scipy-centroid-of-convex-hull
        loc_x = np.mean(convex_hull.points[convex_hull.vertices, 0])
        loc_y = np.mean(convex_hull.points[convex_hull.vertices, 1])
        if ground_points is not None:
            # Distance from each ground point to tree center
            dist_to_tree = np.linalg.norm(
                ground_points.xyz[:, :2] - [loc_x, loc_y], axis=1
            )
            ground_circle_points = ground_points[dist_to_tree <= ground_circle_r]
        else:
            ground_circle_points = []
        # Determine tree height
        if len(ground_circle_points) > 1:
            z_sorted_ground = np.sort(ground_circle_points.z)
            z_min, z_2min = z_sorted_ground[0], z_sorted_ground[1]
            z_diff = np.abs(z_max - z_2max)
            if z_diff > z_outlier_dist:
                z_min = z_2min
        else:
            z_min = z_sorted[0]
        # Minimum z_coordinate cannot be higher than lowest coordinate in given tree
        z_min = np.min([z_sorted[0], z_min])
        tree_height = z_max - z_min
        # Save tree location and height in dict
        current_plot_data["trees"][int(ind)] = {
            "x": float(np.round(loc_x, 3)),
            "y": float(np.round(loc_y, 3)),
            "h": float(np.round(tree_height, 3)),
        }
    # Iterate through all trees and compute their crown categories
    # Extract x, y and height of trees into separate numpy arrays
    tree_x = np.array([current_plot_data["trees"][i]["x"] for i in tree_ind])
    tree_y = np.array([current_plot_data["trees"][i]["y"] for i in tree_ind])
    tree_h = np.array([current_plot_data["trees"][i]["h"] for i in tree_ind])
    # Matrix of distances from each tree to all other trees
    dist = np.sqrt(
        np.power(tree_x - tree_x.T[:, np.newaxis], 2)
        + np.power(tree_y - tree_y.T[:, np.newaxis], 2)
    )
    # Matrix of height differences between each tree
    h_diff = tree_h.T[:, np.newaxis] - tree_h

    # Iterate through all trees
    for i, ind in enumerate(tree_ind):
        dist_i = dist[i, :]  # Distance from this tree to all other trees
        h_diff_i = h_diff[
            i, :
        ]  # Height difference of this tree compared to all other trees
        # Remove current tree from list of distances and differences, as distance and difference in height
        # relative to the tree itself is always 0
        dist_i = np.delete(dist_i, i)
        h_diff_i = np.delete(h_diff_i, i)
        # Extract trees that are neighbors or next to the current tree
        neighbor_mask = dist_i < neighbor_d
        next_to_mask = dist_i < next_to_d
        neighbor_dist, neighbor_h_diff = dist_i[neighbor_mask], h_diff_i[neighbor_mask]
        next_to_dist, next_to_h_diff = dist_i[next_to_mask], h_diff_i[next_to_mask]
        tree_i_ctg = None
        # Classify the current tree
        if neighbor_dist.size == 0 or np.all(neighbor_h_diff >= dominant_h):
            tree_i_ctg = "A"
        # All below must have at least one neighbor (neighbor_dist.size == 0 check failed)
        elif np.all(neighbor_h_diff > -dominant_h):
            tree_i_ctg = "B"
        elif np.any(neighbor_h_diff <= -dominant_h) and not np.any(
            next_to_h_diff <= -dominant_h
        ):
            tree_i_ctg = "C"
        elif next_to_dist.size > 0 and np.any(next_to_h_diff <= -dominant_h):
            tree_i_ctg = "D"
        else:
            # If a tree was given class 'X' there is an issue with the data (or this script :D)
            print(f"Class 'X' assigned to tree with id {ind}!")
            tree_i_ctg = "X"
        current_plot_data["trees"][ind]["c"] = tree_i_ctg
    # Having determined a class for each tree in the current plot, compute their numbers and add to data
    class_list = ["A", "B", "C", "D"]
    tree_c = [current_plot_data["trees"][i]["c"] for i in tree_ind]
    n_trees = current_plot_data["n_trees"]
    current_plot_data["n_trees"] = {"all": n_trees}
    for c in class_list:
        n_c = tree_c.count(c)
        current_plot_data["n_trees"][c] = n_c

    # Store tree locations, heights and categories in the plot_data dictionary
    plot_data[plot_id] = current_plot_data


def _write_to_yaml(data: dict, path: str):
    with open(path, "w+") as f:
        yaml.dump(data, f)


if __name__ == "__main__":
    args = _parse_args()
    migrate_fiv2_to_las(args.src_path, args.dest_path, args.keep_names, args.crown_ctg)
