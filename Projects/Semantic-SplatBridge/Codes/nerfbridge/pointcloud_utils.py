from typing import Union

"""
PointCloud Utility Functions
"""
import struct
import math
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
import numpy as np


_DATATYPES = {}
_DATATYPES[PointField.INT8] = ("b", 1)
_DATATYPES[PointField.UINT8] = ("B", 1)
_DATATYPES[PointField.INT16] = ("h", 2)
_DATATYPES[PointField.UINT16] = ("H", 2)
_DATATYPES[PointField.INT32] = ("i", 4)
_DATATYPES[PointField.UINT32] = ("I", 4)
_DATATYPES[PointField.FLOAT32] = ("f", 4)
_DATATYPES[PointField.FLOAT64] = ("d", 8)


def read_points(cloud, field_names=None, skip_nans=False, uvs=[]):
    assert isinstance(cloud, PointCloud2), "cloud is not a sensor_msgs.msg.PointCloud2"

    fmt = get_struct_fmt(cloud.is_bigendian, cloud.fields, field_names)
    width, height, point_step, row_step, data, isnan = (
        cloud.width,
        cloud.height,
        cloud.point_step,
        cloud.row_step,
        cloud.data,
        math.isnan,
    )
    unpack_from = struct.Struct(fmt).unpack_from

    if skip_nans:
        if uvs:
            for u, v in uvs:
                p = unpack_from(data, (row_step * v) + (point_step * u))
                has_nan = False
                for pv in p:
                    if isnan(pv):
                        has_nan = True
                        break
                if not has_nan:
                    yield p
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    p = unpack_from(data, offset)
                    has_nan = False
                    for pv in p:
                        if isnan(pv):
                            has_nan = True
                            break
                    if not has_nan:
                        yield p
                    offset += point_step
    else:
        if uvs:
            for u, v in uvs:
                yield unpack_from(data, (row_step * v) + (point_step * u))
        else:
            for v in range(height):
                offset = row_step * v
                for u in range(width):
                    yield unpack_from(data, offset)
                    offset += point_step


def get_struct_fmt(is_bigendian, fields, field_names=None):
    fmt = ">" if is_bigendian else "<"
    offset = 0
    for field in (
        f
        for f in sorted(fields, key=lambda f: f.offset)
        if field_names is None or f.name in field_names
    ):
        if offset < field.offset:
            fmt += "x" * (field.offset - offset)
            offset = field.offset
        if field.datatype in _DATATYPES:
            datatype_fmt, datatype_length = _DATATYPES[field.datatype]
            fmt += field.count * datatype_fmt
            offset += field.count * datatype_length
    return fmt

def make_point_cloud(points, colors, sims, deltas=None, frame_id="map"):
    num_points = len(points)
    if deltas is not None:
        # Define the dtype for a structured NumPy array
        dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgba", np.uint32),
            ("sim", np.float32),
            ("deltas", np.float32)
        ])
    else:
        # Define the dtype for a structured NumPy array
        dtype = np.dtype([
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgba", np.uint32),
            ("sim", np.float32)
        ])

    # Create the structured array
    cloud_array = np.zeros(num_points, dtype=dtype)

    # Assign values efficiently
    cloud_array["x"] = points[:, 0]
    cloud_array["y"] = points[:, 1]
    cloud_array["z"] = points[:, 2]
    
    # Pack colors into RGBA (assuming colors are in [0, 255])
    rgba = (colors[:, 2].astype(np.uint32) << 16) | (colors[:, 1].astype(np.uint32) << 8) | (colors[:, 0].astype(np.uint32)) | (255 << 24)
    cloud_array["rgba"] = rgba
    
    cloud_array["sim"] = sims[:, 0]

    if deltas is not None:
        cloud_array["deltas"] = deltas[:,0]

    # Convert to bytes
    buffer = cloud_array.tobytes()

    # Construct PointCloud2 message
    if deltas is not None:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="sim", offset=16, datatype=PointField.FLOAT32, count=1),
            PointField(name="deltas", offset=20, datatype=PointField.FLOAT32, count=1)
        ]
        pcd_out = PointCloud2(
            header=Header(frame_id=frame_id),
            height=1,
            width=num_points,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=struct.calcsize("<fffI ff"),
            row_step=len(buffer),
            data=buffer,
        )
    else:
        fields = [
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name="rgba", offset=12, datatype=PointField.UINT32, count=1),
            PointField(name="sim", offset=16, datatype=PointField.FLOAT32, count=1),
        ]
        pcd_out = PointCloud2(
            header=Header(frame_id=frame_id),
            height=1,
            width=num_points,
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=struct.calcsize("<fffI f"),
            row_step=len(buffer),
            data=buffer,
        )
    return pcd_out