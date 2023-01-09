
import os
import numpy as np

from typing import Callable
from datetime import datetime
from multiprocessing.pool import Pool

from imlib.IO.cells import save_cells
from imlib.general.system import get_num_processes
from cellfinder_core.detect.filters.plane import TileProcessor
from cellfinder_core.detect.filters.setup_filters import setup_tile_filtering
from cellfinder_core.detect.filters.volume.volume_filter import VolumeFilter


def calculate_parameters_in_pixels(
    voxel_sizes,
    soma_diameter_um,
    max_cluster_size_um3,
    ball_xy_size_um,
    ball_z_size_um,
):
    """
    Convert the command-line arguments from real (um) units to pixels
    """

    mean_in_plane_pixel_size = 0.5 * (
        float(voxel_sizes[2]) + float(voxel_sizes[1])
    )
    voxel_volume = (
        float(voxel_sizes[2]) * float(voxel_sizes[1]) * float(voxel_sizes[0])
    )
    soma_diameter = int(round(soma_diameter_um / mean_in_plane_pixel_size))
    max_cluster_size = int(round(max_cluster_size_um3 / voxel_volume))
    ball_xy_size = int(round(ball_xy_size_um / mean_in_plane_pixel_size))
    ball_z_size = int(round(ball_z_size_um / float(voxel_sizes[0])))

    return soma_diameter, max_cluster_size, ball_xy_size, ball_z_size


def main(
    signal_array,
    start_plane,
    end_plane,
    save_path,
    chunk_size,
    voxel_sizes,
    soma_diameter,
    max_cluster_size,
    ball_xy_size,
    ball_z_size,
    ball_overlap_fraction,
    soma_spread_factor,
    n_free_cpus,
    log_sigma_size,
    n_sds_above_mean_thresh,
    outlier_keep=False,
    artifact_keep=False,
    save_planes=False,
    plane_directory=None,
    *,
    callback: Callable[[int], None] = None,
):
    """
    Parameters
    ----------
    callback : Callable[int], optional
        A callback function that is called every time a plane has finished
        being processed. Called with the plane number that has finished.
    """
    n_processes = get_num_processes(min_free_cpu_cores=n_free_cpus)
    start_time = datetime.now()

    (
        soma_diameter,
        max_cluster_size,
        ball_xy_size,
        ball_z_size,
    ) = calculate_parameters_in_pixels(
        voxel_sizes,
        soma_diameter,
        max_cluster_size,
        ball_xy_size,
        ball_z_size,
    )

    if end_plane == -1:
        end_plane = len(signal_array)
    signal_array = signal_array[start_plane:end_plane]

    callback = callback or (lambda *args, **kwargs: None)

    if signal_array.ndim != 3:
        raise IOError("Input data must be 3D")

    setup_params = [
        signal_array[0, :, :],
        soma_diameter,
        ball_xy_size,
        ball_z_size,
        ball_overlap_fraction,
        start_plane,
    ]

    # Create 3D analysis filter
    mp_3d_filter = VolumeFilter(
        soma_diameter=soma_diameter,
        setup_params=setup_params,
        soma_size_spread_factor=soma_spread_factor,
        planes_paths_range=signal_array,
        save_planes=save_planes,
        plane_directory=plane_directory,
        start_plane=start_plane,
        max_cluster_size=max_cluster_size,
        outlier_keep=outlier_keep,
        artifact_keep=artifact_keep,
    )

    clipping_val, threshold_value = setup_tile_filtering(signal_array[0, :, :])
    # Create 2D analysis filter
    mp_tile_processor = TileProcessor(
        clipping_val,
        threshold_value,
        soma_diameter,
        log_sigma_size,
        n_sds_above_mean_thresh,
    )
    
    worker_pool = Pool(n_processes)

    # Start 2D filter
    # Submits each plane to the worker pool, and sets up a list of
    # asyncronous results
    block = 1
    async_results = []

    print("Start Modified Loop")
    for id, plane in enumerate(signal_array):
        res = worker_pool.apply_async(
            mp_tile_processor.get_tile_mask, args=(np.array(plane),)
        )
        async_results.append(res)

        if len(async_results) % chunk_size == 0 or id == signal_array.shape[0] - 1:
            
            print('Offloading data on plane {0}'.format(id))

            # Start 3D filter
            # This runs in the main thread
            cells = mp_3d_filter.process(
                async_results, signal_array, callback=callback
                )
            async_results = []
            
            # save the blocks 
            fname = 'cells_block_' + str(block) + '.xml'
            save_cells(cells, os.path.join(save_path, fname))
            block += 1

    print(
        "Detection complete - all planes done in : {}".format(
            datetime.now() - start_time
        )
    )
    
    cells = []
    return cells
