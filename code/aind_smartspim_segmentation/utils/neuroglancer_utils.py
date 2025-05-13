#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  6 17:25:57 2025

@author: nicholas.lusk
"""

import os
import json
import time
import struct
import logging
import inspect
import multiprocessing

import numpy as np
import pandas as pd
import dask.array as da

from multiprocessing.managers import BaseManager, NamespaceProxy

from ..utils import utils
from .._shared.types import PathLike

def volume_orientation(acquisition_params: dict):
    """
    Uses the acquisition orientation to set the cross-section
    orientation in the neuroglancer links

    Parameters
    ----------
    acquisition_params : dict
        acquisition paramenters from the processing manifest

    Raises
    ------
    ValueError
        if a brain is aquired in a way other than those predifined here

    Returns
    -------
    orientation : list
        orientation values for the neuroglancer link

    """

    acquired = ["", "", ""]

    for axis in acquisition_params["axes"]:
        acquired[axis["dimension"]] = axis["direction"][0]

    acquired = "".join(acquired)

    if acquired in ["SPR", "SPL"]:
        orientation = [0.5, 0.5, 0.5, -0.5]
    elif acquired == "SAL":
        orientation = [0.5, 0.5, -0.5, 0.5]
    elif acquired == "IAR":
        orientation = [0.5, -0.5, 0.5, 0.5]
    elif acquired == "RAS":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]
    elif acquired == "RPI":
        orientation = [np.cos(np.pi / 4), 0.0, 0.0, -np.cos(np.pi / 4)]
    elif acquired == "LAI":
        orientation = [0.0, np.cos(np.pi / 4), -np.cos(np.pi / 4), 0.0]
    else:
        raise ValueError(
            "Acquisition orientation: {acquired} has unknown NG parameters"
        )

    return orientation

def calculate_dynamic_range(image_path: PathLike, percentile: 99, level: 3):
    """
    Calculates the default dynamic range for teh neuroglancer link
    using a defined percentile from the downsampled zarr

    Parameters
    ----------
    image_path : PathLike
        location of the zarr used for classification
    percentile : 99
        The top percentile value for setting the dynamic range
    level : 3
        level of zarr to use for calculating percentile

    Returns
    -------
    dynamic_ranges : list
        The dynamic range and window range values for zarr

    """

    img = da.from_zarr(image_path, str(level)).squeeze()
    range_max = da.percentile(img.flatten(), percentile).compute()[0]
    window_max = int(range_max * 1.5)
    dynamic_ranges = [int(range_max), window_max]

    return dynamic_ranges

class ObjProxy(NamespaceProxy):
    """Returns a proxy instance for any user defined data-type. The proxy instance will have the namespace and
    functions of the data-type (except private/protected callables/attributes). Furthermore, the proxy will be
    pickable and can its state can be shared among different processes."""

    @classmethod
    def populate_obj_attributes(cls, real_cls):
        """
        Populates attributes of the proxy object
        """
        DISALLOWED = set(dir(cls))
        ALLOWED = [
            "__sizeof__",
            "__eq__",
            "__ne__",
            "__le__",
            "__repr__",
            "__dict__",
            "__lt__",
            "__gt__",
        ]
        DISALLOWED.add("__class__")
        new_dict = {}
        for attr, value in inspect.getmembers(real_cls, callable):
            if attr not in DISALLOWED or attr in ALLOWED:
                new_dict[attr] = cls._proxy_wrap(attr)
        return new_dict

    @staticmethod
    def _proxy_wrap(attr):
        """
        This method creates function that calls the proxified object's method.
        """

        def f(self, *args, **kwargs):
            """
            Function that calls the proxified object's method.
            """
            return self._callmethod(attr, args, kwargs)

        return f


def buf_builder(x, y, z, buf_):
    """builds the buffer"""
    pt_buf = struct.pack("<3f", x, y, z)
    buf_.extend(pt_buf)


attributes = ObjProxy.populate_obj_attributes(bytearray)
bytearrayProxy = type("bytearrayProxy", (ObjProxy,), attributes)


def generate_precomputed_cells(cells, precompute_path, configs):
    """
    Function for saving precomputed annotation layer

    Parameters
    -----------------

    cells: dict
        output of the xmltodict function for importing cell locations
    precomputed_path: str
        path to where you want to save the precomputed files
    comfigs: dict
        data on the space that the data will be viewed

    """

    BaseManager.register(
        "bytearray",
        bytearray,
        bytearrayProxy,
        exposed=tuple(dir(bytearrayProxy)),
    )
    manager = BaseManager()
    manager.start()

    buf = manager.bytearray()

    cell_list = []
    for idx, cell in cells.iterrows():
        cell_list.append([int(cell["Z"]), int(cell["Y"]), int(cell["X"])])

    l_bounds = np.min(cell_list, axis=0)
    u_bounds = np.max(cell_list, axis=0)

    output_path = os.path.join(precompute_path, "spatial0")
    utils.create_folder(output_path)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": dict((key, configs['dimensions'][key]) for key in ('z', 'y', 'x')),
        "lower_bound": [float(x) for x in l_bounds],
        "upper_bound": [float(x) for x in u_bounds],
        "annotation_type": "point",
        "properties": [],
        "relationships": [],
        "by_id": {"key": "by_id",},
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1] * configs['rank'],
                "chunk_size": [max(1, float(x)) for x in u_bounds - l_bounds],
                "limit": len(cell_list),
            },
        ],
    }

    with open(os.path.join(precompute_path, "info"), "w") as f:
        f.write(json.dumps(metadata))

    with open(os.path.join(output_path, "0_0_0"), "wb") as outfile:
        start_t = time.time()

        total_count = len(cell_list)  # coordinates is a list of tuples (x,y,z)

        print("Running multiprocessing")

        if not isinstance(buf, type(None)):
            buf.extend(struct.pack("<Q", total_count))

            with multiprocessing.Pool(processes=os.cpu_count()) as p:
                p.starmap(
                    buf_builder, [(x, y, z, buf) for (x, y, z) in cell_list]
                )

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf.extend(id_buf)
        else:
            buf = struct.pack("<Q", total_count)

            for x, y, z in cell_list:
                pt_buf = struct.pack("<3f", x, y, z)
                buf += pt_buf

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack(
                "<%sQ" % len(cell_list), *range(len(cell_list))
            )
            buf += id_buf

        print(
            "Building file took {0} minutes".format(
                (time.time() - start_t) / 60
            )
        )

        outfile.write(bytes(buf))

def generate_neuroglancer_link(
    cells_df: pd.DataFrame,
    ng_configs: dict,
    smartspim_config: dict,
    dynamic_range: list,
    logger: logging.Logger,
    bucket="aind-open-data",
):
    """
    Creates the json state dictionary for the neuroglancer link

    Parameters
    ----------
    cells_df: pd.DataFrame
        the location of all the cells from proposal phase
    ng_configs : dict
        Parameters for creating neuroglancer link defined in run_capsule.py
    smartspim_config : dict
        Dataset specific parameters from processing_manifest
    dynamic_range : list
        The intensity range calculated from the zarr
    logger: logging.Logger
    bucket: str
        Location on AWS where the data lives

    Returns
    -------
    json_state : dict
        fully configured JSON for neuroglancer visualization
    """

    output_precomputed = os.path.join(
        smartspim_config["output_folder"], "visualization/detected_precomputed"
    )
    utils.create_folder(output_precomputed)
    print(f"Output cells precomputed: {output_precomputed}")

    generate_precomputed_cells(
        cells_df, precompute_path=output_precomputed, configs=ng_configs
    )

    ng_path = f"s3://{bucket}/{smartspim_config['name']}/image_cell_segmentation/{smartspim_config['channel']}/proposals/visualization/neuroglancer_config.json"

    if isinstance(ng_configs["orientation"], dict):
        crossSectionOrientation = volume_orientation(ng_configs["orientation"])
    else:
        crossSectionOrientation = [np.cos(np.pi / 4), 0.0, 0.0, np.cos(np.pi / 4)]

    json_state = {
        "ng_link": f"{ng_configs['base_url']}{ng_path}",
        "title": smartspim_config["channel"],
        "dimensions": ng_configs["dimensions"],
        "crossSectionOrientation": crossSectionOrientation,
        "crossSectionScale": ng_configs["crossSectionScale"],
        "projectionScale": ng_configs["projectionScale"],
        "layers": [
            {
                "source": f"zarr://s3://{bucket}/{smartspim_config['name']}/image_tile_fusing/OMEZarr/{smartspim_config['channel']}.zarr",
                "type": "image",
                "tab": "rendering",
                "shader": '#uicontrol vec3 color color(default="#ffffff")\n#uicontrol invlerp normalized\nvoid main() {\nemitRGB(color * normalized());\n}',
                "shaderControls": {
                    "normalized": {
                        "range": [0, dynamic_range[0]],
                        "window": [0, dynamic_range[1]],
                    },
                },
                "name": f"Channel: {smartspim_config['channel']}",
            },
            {
                "source": f"precomputed://s3://{bucket}/{smartspim_config['name']}/image_cell_segmentation/{smartspim_config['channel']}/proposals/visualization/detected_precomputed",
                "type": "annotation",
                "tool": "annotatePoint",
                "tab": "annotations",
                "crossSectionAnnotationSpacing": 1.0,
                "name": "Proposed Cells",
            },
        ],
        "gpuMemoryLimit": ng_configs["gpuMemoryLimit"],
        "selectedLayer": {
            "visible": True,
            "layer": f"Channel: {smartspim_config['channel']}",
        },
        "layout": "4panel",
    }

    logger.info(f"Visualization link: {json_state['ng_link']}")
    output_path = os.path.join(
        smartspim_config["output_folder"], "visualization/neuroglancer_config.json"
    )

    with open(output_path, "w") as outfile:
        json.dump(json_state, outfile, indent=2)