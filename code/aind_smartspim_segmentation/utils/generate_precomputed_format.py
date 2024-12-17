"""
Generated the points in a precomputed
format to be able to visualize them
in neuroglancer
"""

import inspect
import json
import multiprocessing
import os
import struct
import time
from multiprocessing.managers import BaseManager, NamespaceProxy
from pathlib import Path
from typing import Dict, List, Union

import neuroglancer
import numpy as np

from . import utils

# IO types
PathLike = Union[str, Path]
SourceLike = Union[PathLike, List[Dict]]


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


def generate_precomputed_spots(spots, path, res):
    """
    Function for saving precomputed annotation layer

    Parameters
    -----------------

    spots: List[int]
        List with the ZYX locations of the identified spots
    path: str
        path to where you want to save the precomputed files
    res: neuroglancer.CoordinateSpace()
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

    spot_list = []
    for spot in spots:
        spot_list.append([int(spot[0]), int(spot[1]), int(spot[2])])

    l_bounds = np.min(spot_list, axis=0)
    u_bounds = np.max(spot_list, axis=0)

    output_path = os.path.join(path, "spatial0")
    utils.create_folder(output_path)

    metadata = {
        "@type": "neuroglancer_annotations_v1",
        "dimensions": res.to_json(),
        "lower_bound": [float(x) for x in l_bounds],
        "upper_bound": [float(x) for x in u_bounds],
        "annotation_type": "point",
        "properties": [],
        "relationships": [],
        "by_id": {
            "key": "by_id",
        },
        "spatial": [
            {
                "key": "spatial0",
                "grid_shape": [1] * res.rank,
                "chunk_size": [max(1, float(x)) for x in u_bounds - l_bounds],
                "limit": len(spot_list),
            },
        ],
    }

    with open(os.path.join(path, "info"), "w") as f:
        f.write(json.dumps(metadata))

    with open(os.path.join(output_path, "0_0_0"), "wb") as outfile:
        start_t = time.time()

        total_count = len(spot_list)  # coordinates is a list of tuples (x,y,z)

        print("Running multiprocessing")

        if not isinstance(buf, type(None)):
            buf.extend(struct.pack("<Q", total_count))

            with multiprocessing.Pool(processes=os.cpu_count()) as p:
                p.starmap(buf_builder, [(x, y, z, buf) for (x, y, z) in spot_list])

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack("<%sQ" % len(spot_list), *range(len(spot_list)))
            buf.extend(id_buf)
        else:
            buf = struct.pack("<Q", total_count)

            for x, y, z in spot_list:
                pt_buf = struct.pack("<3f", x, y, z)
                buf += pt_buf

            # write the ids at the end of the buffer as increasing integers
            id_buf = struct.pack("<%sQ" % len(spot_list), *range(len(spot_list)))
            buf += id_buf

        print("Building file took {0} minutes".format((time.time() - start_t) / 60))

        outfile.write(bytes(buf))
