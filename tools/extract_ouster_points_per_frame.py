#!/usr/bin/env python3
import os
import argparse
import pickle
from pathlib import Path

import numpy as np

import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


def extract_ouster_points_per_frame(
    bag_dir: str,
    out_dir: str,
    topic_name: str = "/ouster/points",
    step: int = 1,
    max_frames: int | None = None,
):
    """
    Читает /ouster/points из rosbag2 и сохраняет каждый кадр в отдельный .pkl.

    out_dir/frame_000000.pkl, frame_000001.pkl, ...
    Внутри файла: {"stamp": float, "frame_idx": int, "points": np.ndarray}
    """
    storage_options = rosbag2_py.StorageOptions(
        uri=bag_dir,
        storage_id="sqlite3",
    )
    converter_options = rosbag2_py.ConverterOptions(
        input_serialization_format="cdr",
        output_serialization_format="cdr",
    )

    reader = rosbag2_py.SequentialReader()
    reader.open(storage_options, converter_options)

    topics_and_types = reader.get_all_topics_and_types()
    topic_types = {t.name: t.type for t in topics_and_types}

    if topic_name not in topic_types:
        raise RuntimeError(
            f"Topic {topic_name} not found. Available: {list(topic_types.keys())}"
        )

    print(f"[INFO] Reading topic {topic_name} ({topic_types[topic_name]})")

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    idx = 0          # индекс всех сообщений топика
    saved = 0        # сколько реально сохранено

    while reader.has_next():
        topic, data, t = reader.read_next()
        if topic != topic_name:
            continue

        if idx % step != 0:
            idx += 1
            continue

        msg = deserialize_message(data, PointCloud2)

        # timestamp в секундах (наносекунды → float)
        stamp_sec = t * 1e-9

        # points: (N, D), поля зависят от лидара (x,y,z,intensity,...)
        points = np.array(
            list(
                point_cloud2.read_points(
                    msg, field_names=None, skip_nans=True
                )
            )
        )

        frame_fname = out_path / f"frame_{saved:06d}.pkl"
        payload = {
            "stamp": stamp_sec,
            "frame_idx": saved,
            "points": points,
        }

        with open(frame_fname, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

        saved += 1
        idx += 1

        if saved % 50 == 0:
            print(f"[INFO] Saved {saved} frames...")

        if max_frames is not None and saved >= max_frames:
            break

    print(f"[INFO] Finished. Total saved frames: {saved}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bag-dir",
        required=True,
        help="Путь к директории rosbag2 (где metadata.yaml)",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        help="Директория, куда писать .pkl по кадрам",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Брать каждый N-й кадр (1 = все)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Максимум кадров для сохранения (None = без лимита)",
    )

    args = parser.parse_args()

    bag_dir = os.path.abspath(args.bag_dir)
    out_dir = os.path.abspath(args.out_dir)

    print(f"[INFO] Bag dir: {bag_dir}")
    print(f"[INFO] Output dir: {out_dir}")

    extract_ouster_points_per_frame(
        bag_dir=bag_dir,
        out_dir=out_dir,
        topic_name="/ouster/points",
        step=args.step,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()