"""
YOLO 目标检测器
用于检测微信 UI 元素：消息气泡、红点、联系人项、聊天标题

依赖: ultralytics, torch, opencv-python
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from collections import deque
from datetime import datetime

try:
    from ultralytics import YOLO
    import torch
except ImportError:
    YOLO = None
    torch = None

from utils.logger import logger

# 检测类别定义
CLASS_NAMES = [
    "bubble_other",   # 0 - 对方消息气泡
    "bubble_own",     # 1 - 自己消息气泡
    "red_dot",        # 2 - 新消息红点
    "contact_item",   # 3 - 联系人列表项
    "chat_title"      # 4 - 聊天窗口标题
]


@dataclass
class Detection:
    """检测结果数据类"""
    class_id: int
    class_name: str
    bbox: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float

    @property
    def center(self) -> Tuple[float, float]:
        """获取边界框中心点坐标"""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def width(self) -> float:
        """获取边界框宽度"""
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> float:
        """获取边界框高度"""
        return self.bbox[3] - self.bbox[1]

    @property
    def area(self) -> float:
        """获取边界框面积"""
        return self.width * self.height

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": list(self.bbox),
            "confidence": self.confidence
        }


@dataclass
class GPUMemoryStats:
    """GPU 内存统计信息"""
    allocated_mb: float = 0.0
    reserved_mb: float = 0.0
    free_mb: float = 0.0
    total_mb: float = 0.0
    utilization_percent: float = 0.0
    timestamp: Optional[datetime] = None

    @property
    def allocated_gb(self) -> float:
        """已分配内存（GB）"""
        return self.allocated_mb / 1024

    @property
    def reserved_gb(self) -> float:
        """已保留内存（GB）"""
        return self.reserved_mb / 1024

    @property
    def total_gb(self) -> float:
        """总内存（GB）"""
        return self.total_mb / 1024

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "allocated_mb": round(self.allocated_mb, 2),
            "reserved_mb": round(self.reserved_mb, 2),
            "free_mb": round(self.free_mb, 2),
            "total_mb": round(self.total_mb, 2),
            "allocated_gb": round(self.allocated_gb, 2),
            "reserved_gb": round(self.reserved_gb, 2),
            "total_gb": round(self.total_gb, 2),
            "utilization_percent": round(self.utilization_percent, 2),
            "timestamp": self.timestamp.isoformat() if self.timestamp else None
        }


class YoloDetector:
    """
    YOLO 目标检测器

    用于检测微信 UI 元素，支持多种检测模式和过滤条件。
    内置 GPU 内存管理，自动清理缓存防止内存溢出。

    示例:
        detector = YoloDetector(model_path="data/models/wechat_yolov8s.pt")
        detections = detector.detect(image)
        red_dots = detector.detect_red_dots(image)

        # 手动清理 GPU 缓存
        detector.clear_gpu_cache()

        # 获取 GPU 内存使用情况
        stats = detector.get_gpu_memory_stats()
        print(f"GPU 内存使用: {stats.allocated_gb:.2f}GB / {stats.total_gb:.2f}GB")
    """

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.5,
        iou_threshold: float = 0.45,
        use_gpu: bool = True,
        device: Optional[str] = None,
        auto_cleanup_interval: int = 50,
        enable_memory_monitoring: bool = True
    ):
        """
        初始化 YOLO 检测器

        Args:
            model_path: 模型文件路径 (.pt 或 .onnx)
            confidence: 置信度阈值 (0-1)
            iou_threshold: IOU 阈值 (0-1)
            use_gpu: 是否使用 GPU 加速
            device: 指定设备 (如 "cuda:0", "cpu")，None 则自动选择
            auto_cleanup_interval: 自动清理 GPU 缓存的检测次数间隔
            enable_memory_monitoring: 是否启用内存监控
        """
        if YOLO is None:
            raise ImportError("ultralytics 未安装，请运行: pip install ultralytics")

        self.model_path = model_path
        self.confidence = confidence
        self.iou_threshold = iou_threshold

        # GPU 内存管理配置
        self.auto_cleanup_interval = auto_cleanup_interval
        self.enable_memory_monitoring = enable_memory_monitoring
        self._detection_count = 0
        self._last_cleanup_count = 0
        self._memory_history = deque(maxlen=100)  # 保存最近 100 次内存记录

        # 设备选择
        if device:
            self.device = device
        elif use_gpu and torch and torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info(f"使用 GPU 设备: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.info("使用 CPU 设备")
            # CPU 模式下禁用内存监控
            self.enable_memory_monitoring = False

        # 记录初始内存状态
        if self.enable_memory_monitoring:
            initial_stats = self._get_current_memory_stats()
            logger.info(f"初始 GPU 内存: {initial_stats.allocated_gb:.2f}GB / {initial_stats.total_gb:.2f}GB")

        # 加载模型
        try:
            self.model = YOLO(model_path)
            self.model.to(self.device)
            logger.info(f"YOLO 模型加载成功: {model_path}")
        except Exception as e:
            logger.error(f"YOLO 模型加载失败: {e}")
            raise

    def _get_current_memory_stats(self) -> GPUMemoryStats:
        """获取当前 GPU 内存使用情况"""
        if not self.enable_memory_monitoring or not torch or not torch.cuda.is_available():
            return GPUMemoryStats()

        try:
            allocated = torch.cuda.memory_allocated(self.device) / (1024 ** 2)  # MB
            reserved = torch.cuda.memory_reserved(self.device) / (1024 ** 2)  # MB

            # 获取 GPU 总内存和剩余内存
            total = torch.cuda.get_device_properties(self.device).total / (1024 ** 2)  # MB
            free = total - reserved

            utilization = (reserved / total * 100) if total > 0 else 0

            return GPUMemoryStats(
                allocated_mb=allocated,
                reserved_mb=reserved,
                free_mb=free,
                total_mb=total,
                utilization_percent=utilization,
                timestamp=datetime.now()
            )
        except Exception as e:
            logger.debug(f"获取 GPU 内存信息失败: {e}")
            return GPUMemoryStats()

    def _auto_cleanup_if_needed(self):
        """根据检测次数自动清理 GPU 缓存"""
        if not self.enable_memory_monitoring:
            return

        self._detection_count += 1

        # 检查是否需要自动清理
        if self._detection_count - self._last_cleanup_count >= self.auto_cleanup_interval:
            stats = self._get_current_memory_stats()
            self._memory_history.append(stats)

            # 如果 GPU 内存使用超过阈值，进行清理
            if stats.utilization_percent > 80:
                logger.info(
                    f"GPU 内存使用率 {stats.utilization_percent:.1f}% "
                    f"({stats.reserved_gb:.2f}GB / {stats.total_gb:.2f}GB)，"
                    f"执行自动清理"
                )
                self.clear_gpu_cache()
            elif self._detection_count % (self.auto_cleanup_interval * 2) == 0:
                # 定期轻度清理（即使内存使用率不高）
                self.clear_gpu_cache(light=True)

    def clear_gpu_cache(self, light: bool = False) -> GPUMemoryStats:
        """
        清理 GPU 缓存

        Args:
            light: 是否轻度清理（仅清理空缓存，不强制同步）

        Returns:
            清理后的内存统计信息
        """
        if not self.enable_memory_monitoring or not torch or not torch.cuda.is_available():
            return GPUMemoryStats()

        # 获取清理前的内存
        before_stats = self._get_current_memory_stats()

        try:
            if light:
                # 轻度清理：仅释放空缓存
                torch.cuda.empty_cache()
            else:
                # 完整清理：同步、释放缓存、重置统计
                torch.cuda.synchronize(self.device)
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(self.device)

            # 等待清理完成
            torch.cuda.synchronize(self.device)

            self._last_cleanup_count = self._detection_count

            # 获取清理后的内存
            after_stats = self._get_current_memory_stats()
            freed_mb = before_stats.reserved_mb - after_stats.reserved_mb

            logger.info(
                f"GPU 缓存清理完成: "
                f"释放 {freed_mb:.1f}MB, "
                f"当前使用 {after_stats.reserved_gb:.2f}GB / {after_stats.total_gb:.2f}GB"
            )

            return after_stats

        except Exception as e:
            logger.error(f"GPU 缓存清理失败: {e}")
            return before_stats

    def get_gpu_memory_stats(self) -> GPUMemoryStats:
        """
        获取当前 GPU 内存使用情况

        Returns:
            GPU 内存统计信息
        """
        return self._get_current_memory_stats()

    def get_memory_history(self) -> List[GPUMemoryStats]:
        """
        获取内存使用历史记录

        Returns:
            内存统计信息列表
        """
        return list(self._memory_history)

    def get_detection_count(self) -> int:
        """获取累计检测次数"""
        return self._detection_count

    def reset_detection_count(self):
        """重置检测计数器"""
        self._detection_count = 0
        self._last_cleanup_count = 0

    def detect(self, image: np.ndarray) -> List[Detection]:
        """
        检测图像中的所有目标

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表，每个结果包含 class_id, class_name, bbox, confidence
        """
        # 自动清理 GPU 缓存
        self._auto_cleanup_if_needed()

        try:
            results = self.model(
                image,
                conf=self.confidence,
                iou=self.iou_threshold,
                verbose=False,
                device=self.device
            )
        except Exception as e:
            logger.error(f"YOLO 检测失败: {e}")
            # 检测失败时尝试清理 GPU 缓存
            if self.enable_memory_monitoring:
                self.clear_gpu_cache()
            return []

        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                # 获取边界框坐标
                xyxy = box.xyxy[0].cpu().numpy()
                x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

                # 获取类别和置信度
                class_id = int(box.cls.cpu().numpy())
                confidence = float(box.conf.cpu().numpy())

                # 确保类别 ID 在有效范围内
                if 0 <= class_id < len(CLASS_NAMES):
                    class_name = CLASS_NAMES[class_id]
                else:
                    class_name = f"unknown_{class_id}"
                    logger.warning(f"未知类别 ID: {class_id}")

                detections.append(Detection(
                    class_id=class_id,
                    class_name=class_name,
                    bbox=(x1, y1, x2, y2),
                    confidence=confidence
                ))

        return detections

    def detect_by_class(
        self,
        image: np.ndarray,
        class_names: List[str],
        max_detections: Optional[int] = None
    ) -> List[Detection]:
        """
        只检测指定类别的目标

        Args:
            image: 输入图像
            class_names: 类别名称列表 (如 ["red_dot", "contact_item"])
            max_detections: 最大检测数量，None 表示不限制

        Returns:
            指定类别的检测结果
        """
        all_detections = self.detect(image)
        filtered = [d for d in all_detections if d.class_name in class_names]

        if max_detections is not None:
            filtered = filtered[:max_detections]

        return filtered

    def detect_red_dots(self, image: np.ndarray, max_detections: Optional[int] = None) -> List[Detection]:
        """
        检测红点

        Args:
            image: 输入图像
            max_detections: 最大检测数量

        Returns:
            红点检测结果列表
        """
        return self.detect_by_class(image, ["red_dot"], max_detections)

    def detect_contact_items(self, image: np.ndarray, max_detections: Optional[int] = None) -> List[Detection]:
        """
        检测联系人列表项

        Args:
            image: 输入图像
            max_detections: 最大检测数量

        Returns:
            联系人项检测结果列表
        """
        return self.detect_by_class(image, ["contact_item"], max_detections)

    def detect_bubbles(self, image: np.ndarray) -> Dict[str, List[Detection]]:
        """
        检测消息气泡，返回分组结果

        Args:
            image: 输入图像

        Returns:
            字典，包含 "other" 和 "own" 两个键，值为对应的消息气泡列表
        """
        detections = self.detect_by_class(image, ["bubble_other", "bubble_own"])

        return {
            "other": [d for d in detections if d.class_name == "bubble_other"],
            "own": [d for d in detections if d.class_name == "bubble_own"]
        }

    def detect_chat_title(self, image: np.ndarray) -> Optional[Detection]:
        """
        检测聊天标题

        Args:
            image: 输入图像

        Returns:
            标题检测结果或 None
        """
        results = self.detect_by_class(image, ["chat_title"], max_detections=1)
        return results[0] if results else None

    def crop_detection(
        self,
        image: np.ndarray,
        detection: Detection,
        padding: int = 5
    ) -> np.ndarray:
        """
        裁剪检测区域

        Args:
            image: 原始图像
            detection: 检测结果
            padding: 扩展边距（像素）

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection.bbox

        # 添加边距并限制在图像范围内
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        # 确保裁剪区域有效
        if x2 <= x1 or y2 <= y1:
            logger.warning(f"无效的裁剪区域: ({x1}, {y1}, {x2}, {y2})")
            return np.array([])

        return image[y1:y2, x1:x2]

    def match_dots_to_contacts(
        self,
        dots: List[Detection],
        contacts: List[Detection],
        max_distance: float = 100,
        direction: str = "right"
    ) -> List[Dict]:
        """
        将红点关联到对应的联系人项

        Args:
            dots: 红点检测结果
            contacts: 联系人项检测结果
            max_distance: 最大关联距离（像素）
            direction: 关联方向 ("right" 表示红点在联系人右侧)

        Returns:
            带红点的联系人项列表，每项包含 contact, dot, distance
        """
        contacts_with_dots = []

        for contact in contacts:
            contact_center = contact.center
            best_match = None
            best_distance = float('inf')

            for dot in dots:
                dot_center = dot.center

                # 计算距离
                distance = ((contact_center[0] - dot_center[0]) ** 2 +
                           (contact_center[1] - dot_center[1]) ** 2) ** 0.5

                # 检查方向约束
                if direction == "right" and dot_center[0] <= contact_center[0]:
                    continue  # 红点不在右侧
                elif direction == "left" and dot_center[0] >= contact_center[0]:
                    continue  # 红点不在左侧

                # 检查距离约束
                if distance <= max_distance and distance < best_distance:
                    best_match = dot
                    best_distance = distance

            if best_match:
                contacts_with_dots.append({
                    "contact": contact,
                    "dot": best_match,
                    "distance": best_distance
                })

        return contacts_with_dots

    def filter_by_position(
        self,
        detections: List[Detection],
        region: Tuple[float, float, float, float],
        image_shape: Tuple[int, int]
    ) -> List[Detection]:
        """
        根据位置过滤检测结果

        Args:
            detections: 检测结果列表
            region: 区域范围 (top, left, bottom, right)，0-1 之间的相对坐标
            image_shape: 图像尺寸 (height, width)

        Returns:
            在指定区域内的检测结果
        """
        h, w = image_shape[:2]
        top, left, bottom, right = region

        # 转换为绝对坐标
        top = int(top * h)
        left = int(left * w)
        bottom = int(bottom * h)
        right = int(right * w)

        filtered = []
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            center_x, center_y = det.center

            # 检查中心点是否在区域内
            if left <= center_x <= right and top <= center_y <= bottom:
                filtered.append(det)

        return filtered

    def get_highest_confidence(self, detections: List[Detection]) -> Optional[Detection]:
        """
        获取置信度最高的检测结果

        Args:
            detections: 检测结果列表

        Returns:
            置信度最高的检测结果或 None
        """
        if not detections:
            return None
        return max(detections, key=lambda d: d.confidence)

    def sort_by_position(
        self,
        detections: List[Detection],
        by: str = "y",
        reverse: bool = False
    ) -> List[Detection]:
        """
        按位置排序检测结果

        Args:
            detections: 检测结果列表
            by: 排序依据 ("x" 或 "y")
            reverse: 是否反向排序

        Returns:
            排序后的检测结果列表
        """
        key_func = lambda d: d.center[0] if by == "x" else d.center[1]
        return sorted(detections, key=key_func, reverse=reverse)

    def visualize_detections(
        self,
        image: np.ndarray,
        detections: List[Detection],
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        在图像上可视化检测结果

        Args:
            image: 输入图像
            detections: 检测结果列表
            show_confidence: 是否显示置信度

        Returns:
            带有标注的图像
        """
        vis_image = image.copy()

        # 定义每种颜色的 BGR 值
        colors = {
            "bubble_other": (0, 255, 0),    # 绿色
            "bubble_own": (255, 0, 0),      # 蓝色
            "red_dot": (0, 0, 255),         # 红色
            "contact_item": (255, 255, 0),  # 青色
            "chat_title": (255, 0, 255),    # 紫色
        }

        for det in detections:
            x1, y1, x2, y2 = [int(c) for c in det.bbox]
            color = colors.get(det.class_name, (128, 128, 128))

            # 绘制边界框
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

            # 准备标签文本
            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.2f}"

            # 绘制标签背景
            (text_w, text_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )
            cv2.rectangle(
                vis_image,
                (x1, y1 - text_h - 5),
                (x1 + text_w, y1),
                color,
                -1
            )

            # 绘制标签文字
            cv2.putText(
                vis_image,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2
            )

        return vis_image

    @staticmethod
    def get_class_names() -> List[str]:
        """获取所有类别名称"""
        return CLASS_NAMES.copy()

    @staticmethod
    def get_class_id(class_name: str) -> Optional[int]:
        """根据类别名称获取类别 ID"""
        try:
            return CLASS_NAMES.index(class_name)
        except ValueError:
            return None


# 便捷函数
def create_detector(
    model_path: str = "data/models/wechat_yolov8s.pt",
    confidence: float = 0.5,
    use_gpu: bool = True,
    auto_cleanup_interval: int = 50
) -> YoloDetector:
    """
    创建 YOLO 检测器的便捷函数

    Args:
        model_path: 模型文件路径
        confidence: 置信度阈值
        use_gpu: 是否使用 GPU
        auto_cleanup_interval: 自动清理间隔

    Returns:
        YoloDetector 实例
    """
    return YoloDetector(
        model_path=model_path,
        confidence=confidence,
        use_gpu=use_gpu,
        auto_cleanup_interval=auto_cleanup_interval
    )
