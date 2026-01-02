"""
Lazy deskewing function
"""

import logging
from typing import Optional, Tuple

import numpy as np
from dask_image.ndinterp import affine_transform as dask_affine_transform

from .._shared.types import ArrayLike


def ceil_to_mulitple(x, base: int = 4):
    """rounds up to the nearest integer multiple of base

    Parameters
    ----------
    x : scalar or np.array
        value/s to round up from
    base : int, optional
        round up to multiples of base (the default is 4)
    Returns
    -------
    scalar or np.array:
        rounded up value/s

    """

    return (np.int32(base) * np.ceil(np.array(x).astype(np.float32) / base)).astype(np.int32)


def get_transformed_corners(aff: np.ndarray, vol_or_shape, zeroindex: bool = True):
    """Input
    aff: an affine transformation matrix
    vol_or_shape: a numpy volume or shape of a volume.

    This function will return the positions of the corner points of the volume (or volume with
    provided shape) after applying the affine transform.
    """
    # get the dimensions of the array.
    # see whether we got a volume
    if np.array(vol_or_shape).ndim == 3:
        d0, d1, d2 = np.array(vol_or_shape).shape
    elif np.array(vol_or_shape).ndim == 1:
        d0, d1, d2 = vol_or_shape
    else:
        raise ValueError
    # By default we calculate where the corner points in
    # zero-indexed (numpy) arrays will be transformed to.
    # set zeroindex to False if you want to perform the calculation
    # for Matlab-style arrays.
    if zeroindex:
        d0 -= 1
        d1 -= 1
        d2 -= 1
    # all corners of the input volume (maybe there is
    # a more concise way to express this with itertools?)
    corners_in = [
        (0, 0, 0, 1),
        (d0, 0, 0, 1),
        (0, d1, 0, 1),
        (0, 0, d2, 1),
        (d0, d1, 0, 1),
        (d0, 0, d2, 1),
        (0, d1, d2, 1),
        (d0, d1, d2, 1),
    ]
    corners_out = list(map(lambda c: aff @ np.array(c), corners_in))
    corner_array = np.concatenate(corners_out).reshape((-1, 4))
    # print(corner_array)
    return corner_array


def get_output_dimensions(aff: np.ndarray, vol_or_shape):
    """given an 4x4 affine transformation matrix aff and
    a 3d input volume (numpy array) or volumen shape (iterable with 3 elements)
    this function returns the output dimensions required for the array after the
    transform. Rounds up to create an integer result.
    """
    corners = get_transformed_corners(aff, vol_or_shape, zeroindex=True)
    # +1 to avoid fencepost error
    dims = np.max(corners, axis=0) - np.min(corners, axis=0) + 1
    dims = ceil_to_mulitple(dims, 2)
    return dims[:3].astype(np.int32)


# def plot_all(imlist, backend: str = "matplotlib"):
#     """ given an iterable of 2d numpy arrays (images),
#         plots all of them in order.
#         Will add different backends (Bokeh) later """
#     if backend == "matplotlib":
#         for im in imlist:
#             plt.imshow(im)
#             plt.show()
#     else:
#         pass


def get_projection_montage(vol: np.ndarray, gap: int = 10, proj_function=np.max) -> np.ndarray:
    """given a volume vol, creates a montage with all three projections (orthogonal views)

    Parameters
    ----------
    vol : np.ndarray
        input volume
    gap : int, optional
        gap between projections in montage (the default is 10 pixels)
    proj_function : Callable, optional
        function to create the projection (the default is np.max, which performs maximum projection)

    Returns
    -------
    np.ndarray
        the montage of all projections
    """

    assert len(vol.shape) == 3, "only implemented for 3D-volumes"
    nz, ny, nx = vol.shape
    m = np.zeros((ny + nz + gap, nx + nz + gap), dtype=vol.dtype)
    m[:ny, :nx] = proj_function(vol, axis=0)
    m[ny + gap :, :nx] = np.max(vol, axis=1)
    m[:ny, nx + gap :] = np.max(vol, axis=2).transpose()
    return m


def get_dispim_config():
    """
    Returns some dispim microscope parameters
    """
    return {
        "resolution": {"x": 0.298, "y": 0.298, "z": 0.176},
        "angle_degrees": 45,  # with respect to xy
    }


def shear_angle_to_shear_factor(angle_in_degrees):
    """
    Converts a shearing angle into a shearing factor
    Parameters
    ----------
    angle_in_degrees: float
    Returns
    -------
    float
    """
    return 1.0 / np.tan((90 - angle_in_degrees) * np.pi / 180)


def shear_factor_to_shear_angle(shear_factor):
    """
    Converts a shearing angle into a shearing factor
    Parameters
    ----------
    shear_factor: float
    Returns
    -------
    float
    """
    return -np.atan(1.0 / shear_factor) * 180 / np.pi + 90


def create_translation_in_centre(image_shape: tuple, orientation: int = -1) -> np.matrix:
    """
    Creates a translation from the center

    Parameters
    ----------
    image_shape: tuple
        Image shape

    orientation: int
        Orientation. -1 to move from the center,
        1 to come back to original center

    Returns
    -------
    np.matrix
        Matrix with the image transformation
    """
    centre = np.array(image_shape) / 2
    shift_transformation = np.eye(4, dtype=np.float32)
    shift_transformation[:3, 3] = orientation * centre

    return shift_transformation


def rot_around_y(angle_deg: float) -> np.ndarray:
    """create affine matrix for rotation around y axis

    Parameters
    ----------
    angle_deg : float
        rotation angle in degrees

    Returns
    -------
    np.ndarray
        4x4 affine rotation matrix
    """
    arad = angle_deg * np.pi / 180.0
    roty = np.array(
        [
            [np.cos(arad), 0, np.sin(arad), 0],
            [0, 1, 0, 0],
            [-np.sin(arad), 0, np.cos(arad), 0],
            [0, 0, 0, 1],
        ]
    )
    return roty


def create_rotation_transformation(angle_radians: float) -> np.matrix:
    """
    Rotation in Y

    Parameters
    ----------
    angle_radians: float
        Angle in radians for the rotation

    Returns
    -------
    np.matrix
        Matrix with the rotation transformation
        around y
    """
    rotation_transformation = np.eye(4, dtype=np.float32)
    rotation_transformation[0][0] = np.cos(angle_radians)
    rotation_transformation[0][2] = np.sin(angle_radians)
    rotation_transformation[2][0] = -rotation_transformation[0][2]
    rotation_transformation[2][2] = rotation_transformation[0][0]

    return rotation_transformation


def create_dispim_config(multiscale: int, camera: int) -> dict:
    """
    Creates the dispim configuration dictionary
    for deskewing the data

    Parameters
    ----------
    multiscale: int
        Multiscale we want to use to deskew

    camera: int
        Camera the data was acquired with. The dispim
        has two cameras in an angle of 45 and -45 degrees

    Returns
    -------
    dict
        Dictionary with the information for deskewing the data
    """
    config = get_dispim_config()
    config["resolution"]["x"] = config["resolution"]["x"] * (2**multiscale)
    config["resolution"]["y"] = config["resolution"]["y"] * (2**multiscale)
    config["resolution"]["z"] = config["resolution"]["z"] * (2**multiscale)

    shift = 1

    if camera:
        shift = -1
        print("Changing shift: ", shift)

    # Shifting
    config["shift"] = shift

    # Angle in radians
    config["angle_radians"] = np.deg2rad(config["angle_degrees"])

    # Z stage movement in um
    config["zstage"] = (config["resolution"]["z"]) / np.sin(config["angle_radians"])

    # XY pixel size ratio in um
    config["xy_pixel_size"] = config["resolution"]["x"] / config["resolution"]["y"]

    # ZY pixel size in um
    config["yz_pixel_size"] = config["resolution"]["y"] / config["resolution"]["z"]

    return config


def create_dispim_transform(
    image_data_shape: Tuple[int, ...],
    config: dict,
    scale: bool,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.matrix, Tuple[int, ...]]:
    """
    Creates the dispim transformation following
    the provided parameters in the configuration

    Parameters
    ----------
    image_data_shape: Tuple[int, ...]
        Image data shape

    config: dict
        Configuration dictionary

    scale: bool
        If true, we make the data isotropy scaling
        Z to XY

    Returns
    -------
    Tuple[np.matrix, Tuple[int, ...]]
        Matrix with the affine transformation
        and the output shape for the transformed
        image
    """
    shear_factor = shear_angle_to_shear_factor(config["shift"] * config["angle_degrees"])

    # num_imaged_images = image_data.shape[0] * config["zy_pixel_size"]

    shear_matrix = np.array([[1, 0, shear_factor, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

    # new_dz = np.sin(config["angle_degrees"] * np.pi / 180.0) * config["resolution"]["z"]
    # scale_factor_z = (new_dz / config["resolution"]["x"]) * 1.0

    scale_matrix = np.eye(4, dtype=np.float32)

    if scale:
        scale_matrix[0, 0] = config["resolution"]["y"] / config["resolution"]["z"]  # scale_factor_z
        # scale_matrix[1, 1] = config["resolution"]["y"] / config["resolution"]["z"] #scale_factor_z
        # scale_matrix[2, 2] = config["resolution"]["x"] / config["resolution"]["z"] #scale_factor_z

    # Rotation to coverslip
    translation_transformation_center = create_translation_in_centre(
        image_shape=image_data_shape, orientation=-1
    )

    # rotation_transformation = create_rotation_transformation(config=config["angle_radians"])
    # print("Rotation matrix: ", rotation_transformation)

    # Axis order X:0, Y:1, Z:2 - Using yaw-pitch-roll transform ZYX
    # rotation_matrix = R_yaw @ R_pitch @ R_roll -- Here we're only
    # applying pitch and identities for yaw and roll
    # new_axis_order = np.argsort(rotation_transformation[0, :3])

    shift_shear_rot = (
        # rotation_transformation
        scale_matrix
        @ shear_matrix
        @ translation_transformation_center
    )

    output_shape_after_rot = get_output_dimensions(shift_shear_rot, image_data_shape)

    back_from_translation = create_translation_in_centre(output_shape_after_rot, orientation=1)

    final_transform = back_from_translation @ shift_shear_rot

    if logger is not None:
        logger.info(f"Scale matrix: {scale_matrix}")
        logger.info(f"Shear matrix: {shear_matrix}")
        logger.info(f"Translation matrix: {translation_transformation_center}")
        logger.info(f"Back from translation: {back_from_translation}")
        logger.info(f"Affine transformation: {final_transform}")

    return final_transform, output_shape_after_rot


def lazy_deskewing(
    lazy_data: ArrayLike,
    multiscale: int,
    camera: int,
    make_isotropy_voxels: bool,
    logger: Optional[logging.Logger] = None,
):
    """
    Applies an affine transformation
    to a zarr image dataset.

    Parameters
    ----------
    dataset_path: str
        Path where the data is stored in the
        cloud

    multiscale: int
        Multiscale we will load

    camera: int
        Camera the dataset was acquired with

    make_isotropy_voxels: bool
        Makes the voxel size isotropic
    """
    multiscale = int(multiscale)
    camera = int(camera)

    # Creates diSPIM config
    config = create_dispim_config(multiscale=multiscale, camera=camera)

    # Creates the image transformation
    affine_transformation, output_shape = create_dispim_transform(
        image_data_shape=lazy_data.shape,
        config=config,
        scale=make_isotropy_voxels,
        logger=logger,
    )

    # Lazy affine transformation
    transformed_image = dask_affine_transform(
        image=lazy_data,
        matrix=np.linalg.inv(affine_transformation.copy()),
        output_shape=output_shape,
        output_chunks=lazy_data.chunksize,
        order=1,  # Bilinear interpolation -> balance between speed/quality
    )

    return transformed_image
