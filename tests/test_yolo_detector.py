"""
YOLO 检测器单元测试

运行方式: pytest tests/test_yolo_detector.py -v
"""

import pytest
import numpy as np
import cv2
from pathlib import Path

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# 检查依赖是否可用
try:
    from ultralytics import YOLO
    import torch
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

from core.yolo_detector import YoloDetector, Detection, create_detector, CLASS_NAMES, YOLO


# 需要实例化 YoloDetector 的测试跳过条件
skip_if_no_ultralytics = pytest.mark.skipif(
    not ULTRALYTICS_AVAILABLE,
    reason="ultralytics 未安装，请运行: pip install ultralytics"
)

skip_if_no_yolo_module = pytest.mark.skipif(
    YOLO is None,
    reason="YOLO 模块不可用"
)


class TestDetection:
    """Detection 数据类测试"""

    def test_detection_properties(self):
        """测试 Detection 属性计算"""
        det = Detection(
            class_id=0,
            class_name="bubble_other",
            bbox=(10, 20, 110, 120),
            confidence=0.95
        )

        # 测试中心点
        assert det.center == (60.0, 70.0)

        # 测试宽高
        assert det.width == 100.0
        assert det.height == 100.0

        # 测试面积
        assert det.area == 10000.0

    def test_detection_to_dict(self):
        """测试 Detection 转字典"""
        det = Detection(
            class_id=2,
            class_name="red_dot",
            bbox=(5, 5, 15, 15),
            confidence=0.88
        )

        result = det.to_dict()
        assert result["class_id"] == 2
        assert result["class_name"] == "red_dot"
        assert result["bbox"] == [5, 5, 15, 15]
        assert result["confidence"] == 0.88


class TestYoloDetector:
    """YOLO 检测器测试"""

    @pytest.fixture
    def sample_image(self):
        """创建示例图像 (640x480 RGB)"""
        return np.ones((480, 640, 3), dtype=np.uint8) * 128

    @pytest.fixture
    def detector_with_mock(self, sample_image):
        """
        创建带 mock 模型的检测器
        注意: 这是一个模拟测试，实际测试需要真实模型
        """
        # 检查模型文件是否存在
        model_path = "data/models/wechat_yolov8s.pt"
        if not Path(model_path).exists():
            pytest.skip(f"模型文件不存在: {model_path}")

        return YoloDetector(model_path=model_path, confidence=0.5)

    def test_class_names_constant(self):
        """测试类别名称常量"""
        assert len(CLASS_NAMES) == 5
        assert "bubble_other" in CLASS_NAMES
        assert "red_dot" in CLASS_NAMES
        assert "contact_item" in CLASS_NAMES

    def test_get_class_names(self):
        """测试获取类别名称"""
        names = YoloDetector.get_class_names()
        assert names == CLASS_NAMES

    def test_get_class_id(self):
        """测试获取类别 ID"""
        assert YoloDetector.get_class_id("bubble_other") == 0
        assert YoloDetector.get_class_id("red_dot") == 2
        assert YoloDetector.get_class_id("invalid") is None

    @skip_if_no_ultralytics
    def test_crop_detection_basic(self, sample_image):
        """测试基本的裁剪功能"""
        det = Detection(
            class_id=0,
            class_name="bubble_other",
            bbox=(100, 100, 200, 200),
            confidence=0.9
        )

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        cropped = detector.crop_detection(sample_image, det, padding=0)

        # 验证裁剪尺寸
        assert cropped.shape == (100, 100, 3)

    @skip_if_no_ultralytics
    def test_crop_detection_with_padding(self, sample_image):
        """测试带边距的裁剪"""
        det = Detection(
            class_id=0,
            class_name="bubble_other",
            bbox=(100, 100, 200, 200),
            confidence=0.9
        )

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        cropped = detector.crop_detection(sample_image, det, padding=10)

        # 验证裁剪尺寸 (100 + 10*2 = 120)
        assert cropped.shape == (120, 120, 3)

    @skip_if_no_ultralytics
    def test_crop_detection_boundary(self, sample_image):
        """测试边界情况（超出图像范围）"""
        # 图像尺寸 640x480，裁剪区域超出边界
        det = Detection(
            class_id=0,
            class_name="bubble_other",
            bbox=(600, 400, 700, 500),
            confidence=0.9
        )

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        cropped = detector.crop_detection(sample_image, det, padding=0)

        # 验证裁剪后不会超出图像范围
        assert cropped.shape[0] <= 80  # 480 - 400
        assert cropped.shape[1] <= 40  # 640 - 600

    @skip_if_no_ultralytics
    def test_match_dots_to_contacts(self):
        """测试红点与联系人关联"""
        # 创建测试数据
        dot = Detection(
            class_id=2,
            class_name="red_dot",
            bbox=(250, 50, 260, 60),
            confidence=0.9
        )

        contact = Detection(
            class_id=3,
            class_name="contact_item",
            bbox=(50, 40, 200, 80),
            confidence=0.95
        )

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        # 测试关联
        matches = detector.match_dots_to_contacts(
            dots=[dot],
            contacts=[contact],
            max_distance=100
        )

        # 验证关联结果
        assert len(matches) == 1
        assert matches[0]["contact"] == contact
        assert matches[0]["dot"] == dot

    @skip_if_no_ultralytics
    def test_filter_by_position(self):
        """测试按位置过滤"""
        detections = [
            Detection(0, "bubble_other", (50, 50, 100, 100), 0.9),
            Detection(0, "bubble_other", (300, 300, 400, 400), 0.8),
            Detection(0, "bubble_other", (500, 500, 600, 600), 0.7),
        ]

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        # 过滤出左上区域 (0-0.5, 0-0.5)
        filtered = detector.filter_by_position(
            detections=detections,
            region=(0, 0, 0.5, 0.5),
            image_shape=(1000, 1000)
        )

        # 只有第一个检测在区域内
        assert len(filtered) == 1
        assert filtered[0].bbox == (50, 50, 100, 100)

    @skip_if_no_ultralytics
    def test_sort_by_position(self):
        """测试按位置排序"""
        detections = [
            Detection(0, "bubble_other", (100, 100, 200, 200), 0.9),
            Detection(0, "bubble_other", (100, 50, 200, 150), 0.8),
            Detection(0, "bubble_other", (100, 150, 200, 250), 0.7),
        ]

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        # 按 Y 坐标排序
        sorted_dets = detector.sort_by_position(detections, by="y")

        assert sorted_dets[0].center[1] == 100.0  # 第二个 (50+100)/2
        assert sorted_dets[1].center[1] == 150.0  # 第一个 (100+200)/2
        assert sorted_dets[2].center[1] == 200.0  # 第三个 (150+250)/2

    @skip_if_no_ultralytics
    def test_get_highest_confidence(self):
        """测试获取最高置信度"""
        detections = [
            Detection(0, "bubble_other", (0, 0, 100, 100), 0.7),
            Detection(0, "bubble_other", (0, 0, 100, 100), 0.95),
            Detection(0, "bubble_other", (0, 0, 100, 100), 0.5),
        ]

        detector = YoloDetector(
            model_path="data/models/wechat_yolov8s.pt",
            confidence=0.5
        )

        best = detector.get_highest_confidence(detections)
        assert best.confidence == 0.95


@skip_if_no_ultralytics
def test_create_detector():
    """测试便捷创建函数"""
    detector = create_detector(
        model_path="data/models/wechat_yolov8s.pt",
        confidence=0.6
    )

    assert detector.confidence == 0.6
    assert detector.model_path == "data/models/wechat_yolov8s.pt"


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])
