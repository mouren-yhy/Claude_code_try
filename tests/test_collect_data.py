"""
数据采集脚本单元测试
测试数据采集器、图像增强器和相关功能
"""

import unittest
import tempfile
import shutil
import json
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import cv2
import sys

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.collect_data import (
    CLASS_NAMES,
    Annotation,
    CollectStats,
    DataCollector,
    ImageAugmentor,
    find_wechat_window
)


class TestAnnotation(unittest.TestCase):
    """Annotation 数据类测试"""

    def test_annotation_creation(self):
        """测试创建标注对象"""
        annotation = Annotation(
            image_path="/test/image.jpg",
            class_name="red_dot",
            class_id=2,
            bbox=[10, 20, 50, 60],
            bbox_normalized=[0.1, 0.2, 0.5, 0.6]
        )
        self.assertEqual(annotation.class_name, "red_dot")
        self.assertEqual(annotation.class_id, 2)
        self.assertFalse(annotation.annotated)
        self.assertFalse(annotation.augment)

    def test_to_yolo_format(self):
        """测试 YOLO 格式转换"""
        annotation = Annotation(
            image_path="/test/image.jpg",
            class_name="contact_item",
            class_id=3,
            bbox=[100, 100, 300, 200],
            bbox_normalized=[0.1, 0.1, 0.3, 0.2]
        )

        # 640x480 图像
        yolo_str = annotation.to_yolo_format(640, 480)
        parts = yolo_str.split()

        self.assertEqual(parts[0], "3")  # class_id
        # x_center = (100 + 300) / 2 / 640 = 0.3125
        self.assertAlmostEqual(float(parts[1]), 0.3125, places=4)
        # y_center = (100 + 200) / 2 / 480 = 0.3125
        self.assertAlmostEqual(float(parts[2]), 0.3125, places=4)
        # width = (300 - 100) / 640 = 0.3125
        self.assertAlmostEqual(float(parts[3]), 0.3125, places=4)
        # height = (200 - 100) / 480 = 0.2083
        self.assertAlmostEqual(float(parts[4]), 0.2083, places=3)

    def test_to_dict(self):
        """测试转换为字典"""
        annotation = Annotation(
            image_path="/test/image.jpg",
            class_name="bubble_own",
            class_id=1,
            bbox=[0, 0, 100, 100],
            bbox_normalized=[0, 0, 0.1, 0.1],
            confidence=0.95,
            augment=True
        )
        data = annotation.to_dict()
        self.assertIn("image_path", data)
        self.assertIn("class_name", data)
        self.assertIn("class_id", data)
        self.assertTrue(data["augment"])


class TestCollectStats(unittest.TestCase):
    """CollectStats 统计类测试"""

    def test_stats_initialization(self):
        """测试初始化"""
        stats = CollectStats()
        self.assertEqual(stats.total_images, 0)
        self.assertEqual(stats.augmented_images, 0)
        self.assertIsInstance(stats.images_by_class, dict)

    def test_stats_update(self):
        """测试更新统计"""
        stats = CollectStats()
        stats.update("red_dot")
        stats.update("red_dot")
        stats.update("contact_item", is_augment=True)

        self.assertEqual(stats.total_images, 3)
        self.assertEqual(stats.images_by_class["red_dot"], 2)
        self.assertEqual(stats.images_by_class["contact_item"], 1)
        self.assertEqual(stats.augmented_images, 1)

    def test_to_dict(self):
        """测试转换为字典"""
        stats = CollectStats()
        stats.update("bubble_other")
        stats.end_time = datetime.now().isoformat()

        data = stats.to_dict()
        self.assertIn("total_images", data)
        self.assertIn("images_by_class", data)
        self.assertIn("start_time", data)
        self.assertIn("end_time", data)


class TestImageAugmentor(unittest.TestCase):
    """图像增强器测试"""

    def setUp(self):
        """创建测试图像"""
        # 创建一个简单的测试图像 (100x100 灰色)
        self.test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128
        self.augmentor = ImageAugmentor(seed=42)

    def test_rotate(self):
        """测试旋转增强"""
        rotated = self.augmentor._rotate(self.test_image, angle=45)
        self.assertEqual(rotated.shape, self.test_image.shape)

    def test_flip(self):
        """测试翻转增强"""
        flipped = self.augmentor._flip(self.test_image)
        self.assertEqual(flipped.shape, self.test_image.shape)

    def test_brightness(self):
        """测试亮度调整"""
        bright = self.augmentor._adjust_brightness(self.test_image, factor=1.5)
        self.assertEqual(bright.shape, self.test_image.shape)

    def test_contrast(self):
        """测试对比度调整"""
        contrast = self.augmentor._adjust_contrast(self.test_image, factor=1.2)
        self.assertEqual(contrast.shape, self.test_image.shape)

    def test_blur(self):
        """测试模糊"""
        blurred = self.augmentor._blur(self.test_image, kernel_size=5)
        self.assertEqual(blurred.shape, self.test_image.shape)

    def test_noise(self):
        """测试添加噪声"""
        noisy = self.augmentor._add_noise(self.test_image, intensity=0.05)
        self.assertEqual(noisy.shape, self.test_image.shape)

    def test_augment_method(self):
        """测试通用增强方法"""
        for aug_type in ["rotate", "flip", "brightness", "contrast", "blur", "noise"]:
            result = self.augmentor.augment(self.test_image, aug_type)
            self.assertIsNotNone(result, f"{aug_type} 增强失败")
            self.assertEqual(result.shape, self.test_image.shape)

    def test_invalid_augment_type(self):
        """测试无效增强类型"""
        result = self.augmentor.augment(self.test_image, "invalid_type")
        self.assertIsNone(result)


class TestDataCollector(unittest.TestCase):
    """数据采集器测试"""

    def setUp(self):
        """创建临时目录"""
        self.temp_dir = tempfile.mkdtemp()
        self.collector = DataCollector(
            output_dir=self.temp_dir,
            image_size=(320, 240)
        )

    def tearDown(self):
        """清理临时目录"""
        shutil.rmtree(self.temp_dir)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.collector.image_size, (320, 240))
        self.assertEqual(len(self.collector.classes), 5)  # 默认所有类别

        # 检查目录是否创建
        for class_name in CLASS_NAMES:
            class_dir = Path(self.temp_dir) / class_name
            self.assertTrue(class_dir.exists())

        labels_dir = Path(self.temp_dir) / "labels"
        self.assertTrue(labels_dir.exists())

    def test_save_image(self):
        """测试保存图像"""
        # 创建测试图像
        test_image = np.ones((100, 100, 3), dtype=np.uint8) * 255

        annotation = self.collector.save_image(
            test_image,
            class_name="red_dot"
        )

        self.assertIsNotNone(annotation)
        self.assertEqual(annotation.class_name, "red_dot")
        self.assertEqual(annotation.class_id, 2)
        self.assertFalse(annotation.augment)

        # 检查文件是否存在
        self.assertTrue(Path(annotation.image_path).exists())

        # 检查标注文件
        label_path = Path(self.temp_dir) / "labels" / f"{Path(annotation.image_path).stem}.txt"
        self.assertTrue(label_path.exists())

    def test_save_invalid_class(self):
        """测试保存无效类别"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        annotation = self.collector.save_image(test_image, class_name="invalid_class")
        self.assertIsNone(annotation)

    def test_save_augmented_image(self):
        """测试保存增强图像"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        annotation = self.collector.save_image(
            test_image,
            class_name="contact_item",
            augment=True,
            source_path="/source/image.jpg"
        )

        self.assertIsNotNone(annotation)
        self.assertTrue(annotation.augment)
        self.assertEqual(annotation.source_image, "/source/image.jpg")

    @unittest.skip("需要实际的截图库支持，跳过单元测试")
    def test_capture_full_screen(self):
        """测试全屏截图（跳过：需要实际依赖）"""
        pass

    def test_stats_update_on_save(self):
        """测试保存时更新统计"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8)

        self.collector.save_image(test_image, "bubble_other")
        self.collector.save_image(test_image, "bubble_other")
        self.collector.save_image(test_image, "red_dot", augment=True)

        self.assertEqual(self.collector.stats.total_images, 3)
        self.assertEqual(self.collector.stats.images_by_class["bubble_other"], 2)
        self.assertEqual(self.collector.stats.images_by_class["red_dot"], 1)
        self.assertEqual(self.collector.stats.augmented_images, 1)

    def test_save_stats(self):
        """测试保存统计信息"""
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        self.collector.save_image(test_image, "chat_title")

        self.collector.save_stats()

        stats_path = Path(self.temp_dir) / "metadata" / "collect_stats.json"
        self.assertTrue(stats_path.exists())

        with open(stats_path, "r") as f:
            data = json.load(f)
            self.assertEqual(data["total_images"], 1)
            self.assertIn("chat_title", data["images_by_class"])

    def test_export_yolo_dataset(self):
        """测试导出 YOLO 数据集"""
        # 创建测试图像
        test_image = np.ones((100, 100, 3), dtype=np.uint8)
        self.collector.save_image(test_image, "red_dot")

        export_dir = Path(self.temp_dir) / "yolo_dataset"
        self.collector.export_yolo_dataset(str(export_dir))

        # 检查目录结构
        for split in ["train", "val", "test"]:
            self.assertTrue((export_dir / "images" / split).exists())
            self.assertTrue((export_dir / "labels" / split).exists())

        # 检查配置文件
        yaml_path = export_dir / "data.yaml"
        self.assertTrue(yaml_path.exists())


class TestFindWechatWindow(unittest.TestCase):
    """微信窗口查找测试"""

    @unittest.skip("需要 win32gui 支持，跳过单元测试")
    def test_find_wechat_window_success(self):
        """测试成功查找微信窗口（跳过：需要实际依赖）"""
        pass

    @unittest.skip("需要 win32gui 支持，跳过单元测试")
    def test_find_wechat_window_no_module(self):
        """测试无 win32gui 模块时返回 None（跳过：需要实际依赖）"""
        pass


class TestAnnotationEdgeCases(unittest.TestCase):
    """Annotation 边界情况测试"""

    def test_empty_bbox(self):
        """测试空边界框"""
        annotation = Annotation(
            image_path="/test.jpg",
            class_name="red_dot",
            class_id=2,
            bbox=[0, 0, 0, 0],
            bbox_normalized=[0, 0, 0, 0]
        )
        yolo_str = annotation.to_yolo_format(640, 480)
        # 即使 bbox 为空，也应该生成有效的 YOLO 格式
        self.assertIsNotNone(yolo_str)

    def test_full_image_bbox(self):
        """测试全图边界框"""
        annotation = Annotation(
            image_path="/test.jpg",
            class_name="chat_title",
            class_id=4,
            bbox=[0, 0, 640, 480],
            bbox_normalized=[0, 0, 1, 1]
        )
        yolo_str = annotation.to_yolo_format(640, 480)
        parts = yolo_str.split()
        # 中心点应该在 (0.5, 0.5)
        self.assertAlmostEqual(float(parts[1]), 0.5, places=4)
        self.assertAlmostEqual(float(parts[2]), 0.5, places=4)


if __name__ == "__main__":
    unittest.main()
