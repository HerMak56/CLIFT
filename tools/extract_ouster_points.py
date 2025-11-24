#!/usr/bin/env python3
import os
import argparse
import pickle

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def read_ouster_points(bag_dir: str,
                       topic_name: str = "/ouster/points",
                       step: int = 1,
                       max_frames: int | None = None):
    """
    Читает PointCloud2 из rosbag2 и возвращает список numpy-массивов (N, 3) или (N, >=3).

    step: брать каждый step-ый кадр (1 = все).
    max_frames: ограничить количество кадров (или None = без ограничения).
    """
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_dir,
        storage_id='sqlite3',
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr',
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}

    if topic_name not in topic_types:
        raise RuntimeError(f"Topic {topic_name} not found in bag. Available: {list(topic_types.keys())}")

    print(f"[INFO] Found topic {topic_name} with type {topic_types[topic_name]}")

    pc_list = []
    idx = 0
    saved = 0

    while reader.has_next():
        (topic, data, t) = reader.read_next()
        if topic != topic_name:
            continue

        if idx % step != 0:
            idx += 1
            continue

        msg = deserialize_message(data, PointCloud2)

        # Конвертация в numpy: (x, y, z, intensity, ...)
        # skip_nans=True — выброс NaN
        points = np.array(list(point_cloud2.read_points(msg,
                                                        field_names=None,
                                                        skip_nans=True)))
        pc_list.append(points)

        saved += 1
        idx += 1

        if max_frames is not None and saved >= max_frames:
            break

        if saved % 50 == 0:
            print(f"[INFO] Saved {saved} frames...")

    print(f"[INFO] Done. Total saved frames: {saved}")
    return pc_list


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag-dir", required=True,
                        help="Путь к директории rosbag2 (где лежит metadata.yaml)")
    parser.add_argument("--out", required=True,
                        help="Выходной .pkl файл")
    parser.add_argument("--step", type=int, default=1,
                        help="Брать каждый N-ый кадр (1 = все)")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Максимальное количество кадров (None = без лимита)")

    args = parser.parse_args()

    bag_dir = os.path.abspath(args.bag_dir)
    out_path = os.path.abspath(args.out)

    print(f"[INFO] Reading bag from: {bag_dir}")
    pc_list = read_ouster_points(
        bag_dir=bag_dir,
        topic_name="/ouster/points",
        step=args.step,
        max_frames=args.max_frames,
    )

    print(f"[INFO] Saving to: {out_path}")
    with open(out_path, "wb") as f:
        pickle.dump(pc_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    print("[INFO] Done.")


if __name__ == "__main__":
    main()