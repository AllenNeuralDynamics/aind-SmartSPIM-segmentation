"""
Compatibility helpers to read acquisition metadata written with
aind-data-schema v1 (schema 1.x) or v2 (schema 2.x).

v1 acquisitions expose voxel scale in ``tiles[].coordinate_transformations``
(ordered X, Y, Z) and orientation in a top-level ``axes`` list that carries
an explicit ``dimension`` per axis. v2 acquisitions expose the scale in
``data_streams[].configurations[].images[].image_to_acquisition_transform``
(ordered following the coordinate system axes) and the axes inside the
imaging configuration's ``coordinate_system`` (list order = dimension).

These helpers normalize both layouts to the v1 shapes the pipeline
consumes, so downstream orientation/resolution logic stays unchanged.
"""

from typing import Dict, List, Optional, Tuple


def _get_imaging_config(acquisition_config: Dict) -> Optional[Dict]:
    """
    Returns the first imaging configuration of a v2 acquisition,
    or None if there is not one.
    """
    for data_stream in acquisition_config.get("data_streams", []):
        for configuration in data_stream.get("configurations", []):
            if configuration.get("object_type") == "Imaging config":
                return configuration

    return None


def get_acquisition_axes(acquisition_config: Dict) -> List[Dict]:
    """
    Extracts the acquisition axes from a v1 or v2 acquisition dict.

    Parameters
    ----------
    acquisition_config: Dict
        Parsed acquisition.json. It can also be a dict that already
        contains v1-shaped ``axes`` (e.g., a processing manifest's
        prelim acquisition block).

    Returns
    -------
    List[Dict]
        Axes in the v1 shape: [{"name", "dimension", "direction"}, ...]
        where ``dimension`` is the image array axis.
    """
    axes = acquisition_config.get("axes")

    if axes:
        return axes

    coordinate_system = None
    imaging_config = _get_imaging_config(acquisition_config)

    if imaging_config is not None:
        coordinate_system = imaging_config.get("coordinate_system")

    if coordinate_system is None:
        coordinate_system = acquisition_config.get("coordinate_system")

    if not coordinate_system or not coordinate_system.get("axes"):
        raise ValueError(
            "No axes found in the acquisition metadata. "
            f"Provided keys: {list(acquisition_config.keys())}"
        )

    return [
        {
            "name": axis["name"],
            "dimension": dimension,
            "direction": axis["direction"],
        }
        for dimension, axis in enumerate(coordinate_system["axes"])
    ]


def get_voxel_resolution(acquisition_config: Dict) -> Tuple[float, float, float]:
    """
    Extracts the voxel resolution from a v1 or v2 acquisition dict.
    We assume all the dataset was acquired with the same resolution.

    Parameters
    ----------
    acquisition_config: Dict
        Parsed acquisition.json.

    Returns
    -------
    Tuple[float, float, float]
        Voxel resolution in (x, y, z) order.
    """
    if "tiles" in acquisition_config:
        # v1: scale is ordered X, Y, Z
        transforms = acquisition_config["tiles"][0]["coordinate_transformations"]
        scale = [t["scale"] for t in transforms if t.get("type") == "scale"][0]
        return float(scale[0]), float(scale[1]), float(scale[2])

    # v2: scale is ordered following the coordinate system axes (e.g. Z, Y, X)
    imaging_config = _get_imaging_config(acquisition_config)

    if imaging_config is None or not imaging_config.get("images"):
        raise ValueError(
            "No tiles or imaging configuration images found in the "
            "acquisition metadata to get the voxel resolution"
        )

    axes = get_acquisition_axes(acquisition_config)
    transforms = imaging_config["images"][0]["image_to_acquisition_transform"]
    scale = [t["scale"] for t in transforms if t.get("object_type") == "Scale"][0]

    resolution = {
        axis["name"].upper(): float(value) for axis, value in zip(axes, scale)
    }

    return resolution["X"], resolution["Y"], resolution["Z"]


def normalize_orientation(acquisition_config: Dict) -> Dict:
    """
    Returns a dict with v1-shaped ``axes`` so it can be passed to
    code that expects a v1 acquisition orientation (e.g., the
    neuroglancer link generation).
    """
    return {"axes": get_acquisition_axes(acquisition_config)}
