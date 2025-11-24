#!/opt/venv/bin/python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
import sensor_msgs_py.point_cloud2 as pc2

import numpy as np
import torch
import yaml
from pathlib import Path

# Импорты из DCCLA
import sys
sys.path.append('/home/capitan/workspace/DCCLA')
from lidar_det.model import get_model
from lidar_det.dataset.utils import target_to_boxes_torch
import lidar_det.utils.utils_box3d as ub3d
from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize

from tf_transformations import quaternion_from_euler

class DCCLADetectorNode(Node):
    """
    ROS2 нода для 3D детекции пешеходов с использованием DCCLA модели.
    
    Подписывается на: /points (sensor_msgs/PointCloud2)
    Публикует: 
        - /detections/boxes (visualization_msgs/MarkerArray) - визуализация
        - /detections/info (std_msgs/String) - информация о детекциях
    """
    
    def __init__(self):
        super().__init__('dccla_detector_node')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. ПАРАМЕТРЫ НОДЫ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.declare_parameter('config_file', '/home/capitan/workspace/DCCLA/bin/jrdb19.yaml')
        self.declare_parameter('checkpoint_file', '/home/capitan/workspace/DCCLA/ckp/DCCLA_JRDB2019.pth')
        self.declare_parameter('input_topic', '/ouster/points')
        self.declare_parameter('output_topic', '/detections/boxes')
        self.declare_parameter('frame_id', 'os_lidar')
        self.declare_parameter('score_threshold', 0.85)
        self.declare_parameter('nms_threshold', 0.7)
        self.declare_parameter('device', 'cuda')
        
        # Получение параметров
        config_file = self.get_parameter('config_file').value
        ckpt_file = self.get_parameter('checkpoint_file').value
        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value
        self.frame_id = self.get_parameter('frame_id').value
        self.score_threshold = self.get_parameter('score_threshold').value
        self.nms_threshold = self.get_parameter('nms_threshold').value
        self.device = self.get_parameter('device').value
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('DCCLA Detector Node Initialization')
        self.get_logger().info('=' * 60)
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. ЗАГРУЗКА КОНФИГУРАЦИИ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.get_logger().info(f'Loading config from: {config_file}')
        with open(config_file, 'r') as f:
            self.cfg = yaml.safe_load(f)
        
        # Параметры voxelization
        vs = self.cfg['dataset']['voxel_size']
        self.voxel_size = np.array(vs, dtype=np.float32).reshape(3, 1)
        self.voxel_offset = np.array([1e5, 1e5, 1e4], dtype=np.int32).reshape(3, 1)
        self.num_points = self.cfg['dataset']['num_points']
        
        # Параметры модели
        self.num_anchors = self.cfg['model']['kwargs']['num_anchors']
        self.num_ori_bins = self.cfg['model']['kwargs']['num_ori_bins']
        self.ave_lwh = np.array([0.9, 0.5, 1.7])  # JRDB pedestrian
        
        self.get_logger().info(f'  Voxel size: {vs}')
        self.get_logger().info(f'  Max voxels: {self.num_points}')
        self.get_logger().info(f'  Num anchors: {self.num_anchors}')
        self.get_logger().info(f'  Num ori bins: {self.num_ori_bins}')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. ИНИЦИАЛИЗАЦИЯ МОДЕЛИ
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.get_logger().info(f'Initializing model...')
        
        # Настройка конфигурации модели
        self.cfg['model']['kwargs']['num_classes'] = 1
        self.cfg['model']['kwargs']['input_dim'] = 3
        self.cfg['model']['nuscenes'] = False
        
        # Создание модели
        self.model = get_model(self.cfg['model'])
        
        # Загрузка весов
        self.get_logger().info(f'Loading checkpoint from: {ckpt_file}')
        checkpoint = torch.load(ckpt_file, map_location='cpu')
        
        # Извлечение весов модели из checkpoint
        if 'model_state' in checkpoint:
            state_dict = checkpoint['model_state']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        self.model.load_state_dict(state_dict, strict=True)
        
        # Перенос на устройство
        if self.device == 'cuda' and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.get_logger().info(f'  Device: CUDA ({torch.cuda.get_device_name(0)})')
        else:
            self.device = 'cpu'
            self.get_logger().info(f'  Device: CPU')
        
        self.model.eval()
        self.get_logger().info('  Model loaded successfully!')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. ROS2 PUBLISHERS & SUBSCRIBERS
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.get_logger().info('Setting up ROS2 interface...')
        
        # Subscriber для point cloud
        self.pc_sub = self.create_subscription(
            PointCloud2,
            input_topic,
            self.pointcloud_callback,
            10
        )
        
        # Publisher для bounding boxes
        self.boxes_pub = self.create_publisher(
            MarkerArray,
            output_topic,
            10
        )
        
        self.get_logger().info(f'  Subscribed to: {input_topic}')
        self.get_logger().info(f'  Publishing to: {output_topic}')
        
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. СТАТИСТИКА
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.frame_count = 0
        self.total_time = 0.0
        
        self.get_logger().info('=' * 60)
        self.get_logger().info('DCCLA Detector Node Ready!')
        self.get_logger().info('=' * 60)
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ОСНОВНОЙ CALLBACK
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def pointcloud_callback(self, msg):
        """
        Callback для обработки входящих point clouds.
        
        Args:
            msg (sensor_msgs/PointCloud2): Входящее облако точек
        """
        import time
        start_time = time.time()
        
        try:
            # ━━━ Шаг 1: Конвертация PointCloud2 → numpy ━━━
            pc = self.pointcloud2_to_array(msg)
            
            if pc.shape[1] == 0:
                self.get_logger().warn('Received empty point cloud!')
                return
            
            # ━━━ Шаг 2: Voxelization ━━━
            net_input = self.voxelize_pointcloud(pc)
            
            if net_input is None:
                self.get_logger().warn('Voxelization failed!')
                return
            
            # ━━━ Шаг 3: Inference ━━━
            boxes, scores = self.run_inference(net_input)
            
            # ━━━ Шаг 4: Публикация результатов ━━━
            if len(boxes) > 0:
                self.publish_boxes(boxes, scores, msg.header)
            
            # ━━━ Статистика ━━━
            elapsed = time.time() - start_time
            self.total_time += elapsed
            self.frame_count += 1
            
            if self.frame_count % 10 == 0:
                avg_time = self.total_time / self.frame_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                self.get_logger().info(
                    f'Frame {self.frame_count}: '
                    f'{len(boxes)} detections, '
                    f'{elapsed*1000:.1f}ms, '
                    f'Avg FPS: {fps:.1f}'
                )
        
        except Exception as e:
            self.get_logger().error(f'Error in callback: {str(e)}')
            import traceback
            self.get_logger().error(traceback.format_exc())
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PREPROCESSING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def pointcloud2_to_array(self, msg):
        """
        Конвертация ROS2 PointCloud2 в numpy array.
        
        Args:
            msg (sensor_msgs/PointCloud2): ROS2 point cloud
            
        Returns:
            np.ndarray: (3, N) массив с xyz координатами
        """
        # Чтение точек из PointCloud2
        points_list = []
        for point in pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True):
            points_list.append([point[0], point[1], point[2]])
        
        if len(points_list) == 0:
            return np.zeros((3, 0), dtype=np.float32)
        
        # Конвертация в numpy (N, 3) → (3, N)
        pc = np.array(points_list, dtype=np.float32).T
        
        return pc
    
    def voxelize_pointcloud(self, pc):
        """
        Voxelization облака точек (аналогично dataset_det3d.py).
        
        Args:
            pc (np.ndarray): (3, N) облако точек
            
        Returns:
            SparseTensor: Входные данные для модели
        """
        # ━━━ 1. Вычисление вокселных координат ━━━
        pc_voxel = np.round(pc / self.voxel_size) + self.voxel_offset
        pc_voxel = pc_voxel.T.astype(np.int32)  # (N, 3)
        
        # ━━━ 2. Удаление дубликатов (sparse quantization) ━━━
        _, inds, inverse_map = sparse_quantize(
            pc_voxel, return_index=True, return_inverse=True,
        )  
        
        # ━━━ 3. Ограничение количества вокселей ━━━
        if len(inds) > self.num_points:
            kept_inds = np.random.choice(len(inds), self.num_points, replace=False)
            inds = inds[kept_inds]
        
        if len(inds) == 0:
            return None
        
        # ━━━ 4. Создание SparseTensor ━━━
        input_feat = pc.T[inds]  # (M, 3) - xyz координаты
        input_coords = pc_voxel[inds]  # (M, 3) - вокселные координаты
        
        # Добавление batch index (всегда 0 для single inference)
        batch_inds = np.zeros((len(inds), 1), dtype=np.int32)
        input_coords = np.concatenate([input_coords, batch_inds], axis=1)  # (M, 4)
        
        net_input = SparseTensor(
            torch.from_numpy(input_feat).float(),
            torch.from_numpy(input_coords).int()
        )
        
        return net_input
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # INFERENCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def run_inference(self, net_input):
        """
        Запуск inference модели (аналогично model_eval_fn).
        
        Args:
            net_input (SparseTensor): Входные данные
            
        Returns:
            tuple: (boxes, scores)
                boxes: np.ndarray (K, 7) - [x, y, z, l, w, h, theta]
                scores: np.ndarray (K,) - confidence scores
        """
        with torch.no_grad():
            # ━━━ 1. Перенос на GPU ━━━
            if self.device == 'cuda':
                net_input = net_input.cuda()
            
            # ━━━ 2. Forward pass ━━━
            net_pred = self.model(net_input)  # (M, 31)
            
            # ━━━ 3. Преобразование в структурированный формат ━━━
            M = net_pred.shape[0]
            net_pred = net_pred.view(M, self.num_anchors, 1, -1)  # (M, A, S, 31)
            
            cls_pred = net_pred[..., 0]   # (M, A, S)
            reg_pred = net_pred[..., 1:]  # (M, A, S, 30)
            
            # ━━━ 4. Коррекция позиции ━━━
            voxel_center = net_input.C[:, :3].clone().float() + 0.5
            voxel_offset_tensor = torch.from_numpy(self.voxel_offset).float()
            voxel_size_tensor = torch.from_numpy(self.voxel_size).float()
            
            if self.device == 'cuda':
                voxel_offset_tensor = voxel_offset_tensor.cuda()
                voxel_size_tensor = voxel_size_tensor.cuda()
            
            voxel_center = (voxel_center - voxel_offset_tensor.view(1, 3)) * \
                           voxel_size_tensor.view(1, 3)
            voxel_center = voxel_center[:, None, None, :]  # (M, 1, 1, 3)
            
            reg_pred[..., :3] = reg_pred[..., :3] + voxel_center
            
            # ━━━ 5. Sigmoid для classification ━━━
            cls_pred = torch.sigmoid(cls_pred)  # (M, A, S)
            
            # ━━━ 6. Декодирование boxes ━━━
            boxes = target_to_boxes_torch(
                reg_pred[:, :, 0, :],  # (M, A, 30)
                self.ave_lwh,
                self.num_ori_bins
            ).view(-1, 7)  # (M*A, 7)
            
            scores = cls_pred[:, :, 0].view(-1)  # (M*A,)
            
            # ━━━ 7. NMS (подавление дубликатов) ━━━
            nms_inds = ub3d.nms_3d_dist_gpu(
                boxes,
                scores,
                l_ave=self.ave_lwh[0],
                w_ave=self.ave_lwh[1],
                nms_thresh=self.nms_threshold
            )
            
            boxes_nms = boxes[nms_inds].cpu().numpy()
            scores_nms = scores[nms_inds].cpu().numpy()
            
            # ━━━ 8. Фильтрация по threshold ━━━
            valid_mask = scores_nms > self.score_threshold
            boxes_nms = boxes_nms[valid_mask]
            scores_nms = scores_nms[valid_mask]
            
            return boxes_nms, scores_nms
    
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLISHING
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def publish_boxes(self, boxes, scores, header):
        """
        Публикация bounding boxes как MarkerArray для RViz.
        
        Args:
            boxes (np.ndarray): (K, 7) - [x, y, z, l, w, h, theta]
            scores (np.ndarray): (K,) - confidence scores
            header (std_msgs/Header): Header из исходного point cloud
        """
        marker_array = MarkerArray()
        
        for i, (box, score) in enumerate(zip(boxes, scores)):
            # ━━━ Создание маркера для каждого бокса ━━━
            marker = Marker()
            marker.header = header
            marker.header.frame_id = self.frame_id
            marker.ns = "detections"
            marker.id = i
            marker.type = Marker.CUBE
            marker.action = Marker.ADD
            
            # ━━━ Позиция ━━━
            marker.pose.position.x = float(box[0])
            marker.pose.position.y = float(box[1])
            marker.pose.position.z = float(box[2])
            
            # ━━━ Ориентация (конвертация angle → quaternion) ━━━
            q = quaternion_from_euler(0, 0, float(box[6]))
            marker.pose.orientation.x = q[0]
            marker.pose.orientation.y = q[1]
            marker.pose.orientation.z = q[2]
            marker.pose.orientation.w = q[3]
            
            # ━━━ Размеры ━━━
            marker.scale.x = float(box[3])  # length
            marker.scale.y = float(box[4])  # width
            marker.scale.z = float(box[5])  # height
            
            # ━━━ Цвет (зависит от confidence) ━━━
            marker.color.r = 0.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            marker.color.a = float(score)  # прозрачность = confidence
            
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = 200_000_000  # 0.2 секунды
            
            marker_array.markers.append(marker)
            
            # ━━━ Текст с confidence score ━━━
            text_marker = Marker()
            text_marker.header = marker.header
            text_marker.ns = "scores"
            text_marker.id = i + 10000
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            
            text_marker.pose.position.x = float(box[0])
            text_marker.pose.position.y = float(box[1])
            text_marker.pose.position.z = float(box[2]) + float(box[5])/2 + 0.3
            
            text_marker.text = f"{score:.2f}"
            text_marker.scale.z = 0.2
            
            text_marker.color.r = 1.0
            text_marker.color.g = 1.0
            text_marker.color.b = 1.0
            text_marker.color.a = 1.0
            
            text_marker.lifetime = marker.lifetime
            
            marker_array.markers.append(text_marker)
        
        # ━━━ Публикация ━━━
        self.boxes_pub.publish(marker_array)


def main(args=None):
    rclpy.init(args=args)
    node = DCCLADetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()