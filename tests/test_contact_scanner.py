"""
联系人扫描器单元测试

运行方式: pytest tests/test_contact_scanner.py -v
"""

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, MagicMock

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.contact_scanner import (
    ContactScanner,
    PendingMessage,
    ScanResult,
    RegionConfig,
    OCREngine,
    ScanReason,
    create_scanner
)
from core.yolo_detector import Detection
from core.username_validator import WhitelistConfig, UsernameValidator


class TestPendingMessage:
    """PendingMessage 数据类测试"""

    def test_pending_message_creation(self):
        """测试创建待处理消息"""
        msg = PendingMessage(
            contact_name="张三 [wechat123]",
            contact_bbox=(10, 20, 110, 120),
            dot_bbox=(200, 50, 210, 60),
            confidence=0.95
        )

        assert msg.contact_name == "张三 [wechat123]"
        assert msg.confidence == 0.95

    def test_contact_center(self):
        """测试联系人中心点计算"""
        msg = PendingMessage(
            contact_name="测试",
            contact_bbox=(10, 20, 110, 120),
            dot_bbox=(200, 50, 210, 60)
        )

        center = msg.contact_center
        assert center == (60.0, 70.0)

    def test_dot_center(self):
        """测试红点中心点计算"""
        msg = PendingMessage(
            contact_name="测试",
            contact_bbox=(10, 20, 110, 120),
            dot_bbox=(200, 50, 210, 60)
        )

        center = msg.dot_center
        assert center == (205.0, 55.0)

    def test_to_dict(self):
        """测试转换为字典"""
        msg = PendingMessage(
            contact_name="李四 [wechat456]",
            contact_bbox=(10, 20, 110, 120),
            dot_bbox=(200, 50, 210, 60),
            confidence=0.88
        )

        result = msg.to_dict()
        assert result["contact_name"] == "李四 [wechat456]"
        assert result["contact_bbox"] == [10, 20, 110, 120]
        assert result["confidence"] == 0.88


class TestScanResult:
    """ScanResult 数据类测试"""

    def test_empty_result(self):
        """测试空的扫描结果"""
        result = ScanResult()

        assert len(result.pending_messages) == 0
        assert result.total_contacts == 0
        assert result.total_red_dots == 0
        assert not result.has_messages
        assert result.success

    def test_result_with_messages(self):
        """测试带消息的扫描结果"""
        msg = PendingMessage(
            contact_name="测试",
            contact_bbox=(0, 0, 100, 100),
            dot_bbox=(150, 50, 160, 60)
        )

        result = ScanResult(
            pending_messages=[msg],
            total_contacts=5,
            total_red_dots=2
        )

        assert result.has_messages
        assert result.total_contacts == 5
        assert result.total_red_dots == 2

    def test_result_with_errors(self):
        """测试带错误的扫描结果"""
        result = ScanResult(errors=["错误1", "错误2"])

        assert not result.success
        assert len(result.errors) == 2


class TestRegionConfig:
    """RegionConfig 数据类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = RegionConfig()

        assert config.top == 0.0
        assert config.bottom == 0.6
        assert config.left == 0.0
        assert config.right == 1.0

    def test_to_tuple(self):
        """测试转换为元组"""
        config = RegionConfig(top=0.1, left=0.2, bottom=0.8, right=0.9)

        result = config.to_tuple()
        assert result == (0.1, 0.2, 0.8, 0.9)

    def test_to_dict(self):
        """测试转换为字典"""
        config = RegionConfig(top=0.0, left=0.0, bottom=0.5, right=1.0)

        result = config.to_dict()
        assert result["top"] == 0.0
        assert result["bottom"] == 0.5

    def test_from_dict(self):
        """测试从字典创建"""
        data = {"top": 0.1, "left": 0.0, "bottom": 0.7, "right": 0.8}
        config = RegionConfig.from_dict(data)

        assert config.top == 0.1
        assert config.bottom == 0.7


class TestOCREngine:
    """OCREngine 测试"""

    def test_ocr_with_none_engine(self):
        """测试无引擎时的 OCR"""
        ocr = OCREngine(engine=None)

        result = ocr.ocr(np.ones((100, 100, 3), dtype=np.uint8))
        assert result == ""

    def test_ocr_with_mock_engine(self):
        """测试使用 mock 引擎"""
        mock_engine = Mock()
        # 正确的 PaddleOCR 格式: [[[[x1,y1],[x2,y2],...], (text, confidence)], ...]
        mock_engine.ocr.return_value = [[
            [[(0, 0), (10, 0), (10, 10), (0, 10)], ("测试文本", 0.95)]
        ]]

        ocr = OCREngine(engine=mock_engine)
        result = ocr.ocr(np.ones((100, 100, 3), dtype=np.uint8))

        assert result == "测试文本"

    def test_ocr_with_empty_result(self):
        """测试空 OCR 结果"""
        mock_engine = Mock()
        mock_engine.ocr.return_value = None

        ocr = OCREngine(engine=mock_engine)
        result = ocr.ocr(np.ones((100, 100, 3), dtype=np.uint8))

        assert result == ""


class TestContactScanner:
    """ContactScanner 测试"""

    @pytest.fixture
    def mock_detector(self):
        """创建 mock YOLO 检测器"""
        detector = Mock()
        detector.detect_red_dots = Mock(return_value=[])
        detector.detect_contact_items = Mock(return_value=[])
        detector.match_dots_to_contacts = Mock(return_value=[])
        detector.crop_detection = Mock(return_value=np.ones((50, 100, 3), dtype=np.uint8))
        return detector

    @pytest.fixture
    def mock_validator(self):
        """创建 mock 验证器"""
        validator = Mock()
        validator.validate_contact_name = Mock(
            return_value=Mock(is_valid=False, reason=ScanReason.NOT_IN_WHITELIST.value)
        )
        return validator

    @pytest.fixture
    def mock_ocr_engine(self):
        """创建 mock OCR 引擎"""
        ocr = Mock()
        ocr.ocr = Mock(return_value="")
        return OCREngine(engine=ocr)

    @pytest.fixture
    def sample_image(self):
        """创建示例图像"""
        return np.ones((1080, 1920, 3), dtype=np.uint8) * 128

    def test_scanner_initialization(self, mock_detector, mock_validator, mock_ocr_engine):
        """测试扫描器初始化"""
        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        assert scanner.detector == mock_detector
        assert scanner.validator == mock_validator
        assert scanner.max_distance == 100

    def test_scan_with_no_screenshot(self, mock_detector, mock_validator, mock_ocr_engine):
        """测试无截图时的扫描"""
        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        result = scanner.scan(None)

        assert not result.has_messages
        assert ScanReason.NO_SCREENSHOT.value in result.errors

    def test_scan_with_no_red_dots(
        self, mock_detector, mock_validator, mock_ocr_engine, sample_image
    ):
        """测试无红点时的扫描"""
        mock_detector.detect_red_dots.return_value = []

        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        result = scanner.scan(sample_image)

        assert not result.has_messages
        assert ScanReason.NO_RED_DOTS.value in result.errors

    def test_scan_successful_flow(
        self, mock_detector, mock_validator, mock_ocr_engine, sample_image
    ):
        """测试完整的成功扫描流程"""
        # 设置 mock 返回值
        dot = Detection(
            class_id=2,
            class_name="red_dot",
            bbox=(200, 50, 210, 60),
            confidence=0.9
        )
        contact = Detection(
            class_id=3,
            class_name="contact_item",
            bbox=(50, 40, 200, 80),
            confidence=0.95
        )

        mock_detector.detect_red_dots.return_value = [dot]
        mock_detector.detect_contact_items.return_value = [contact]
        mock_detector.match_dots_to_contacts.return_value = [
            {"contact": contact, "dot": dot, "distance": 50}
        ]

        # 设置 OCR 返回值
        mock_ocr_engine.engine.ocr.return_value = [[
            [[(0, 0), (10, 0), (10, 10), (0, 10)], ("张三 [wechat123]", 0.95)]
        ]]

        # 设置验证器返回值
        mock_validator.validate_contact_name.return_value = Mock(
            is_valid=True,
            matched_name="张三 [wechat123]"
        )

        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        result = scanner.scan(sample_image)

        assert result.has_messages
        assert len(result.pending_messages) == 1
        assert result.pending_messages[0].contact_name == "张三 [wechat123]"
        assert result.total_red_dots == 1
        assert result.total_contacts == 1

    def test_scan_filters_non_whitelist(
        self, mock_detector, mock_validator, mock_ocr_engine, sample_image
    ):
        """测试白名单过滤"""
        dot = Detection(2, "red_dot", (200, 50, 210, 60), 0.9)
        contact = Detection(3, "contact_item", (50, 40, 200, 80), 0.95)

        mock_detector.detect_red_dots.return_value = [dot]
        mock_detector.detect_contact_items.return_value = [contact]
        mock_detector.match_dots_to_contacts.return_value = [
            {"contact": contact, "dot": dot, "distance": 50}
        ]

        # OCR 识别出用户名，但不在白名单
        mock_ocr_engine.engine.ocr.return_value = [[
            [[(0, 0), (10, 0), (10, 10), (0, 10)], ("非白名单用户", 0.9)]
        ]]

        mock_validator.validate_contact_name.return_value = Mock(
            is_valid=False,
            reason=ScanReason.NOT_IN_WHITELIST.value
        )

        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        result = scanner.scan(sample_image)

        assert not result.has_messages
        assert result.total_contacts == 1

    def test_set_scan_region(self, mock_detector, mock_validator, mock_ocr_engine):
        """测试设置扫描区域"""
        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        new_region = RegionConfig(top=0.1, left=0.0, bottom=0.5, right=1.0)
        scanner.set_scan_region(new_region)

        assert scanner.scan_region == new_region

    def test_set_max_distance(self, mock_detector, mock_validator, mock_ocr_engine):
        """测试设置最大关联距离"""
        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine
        )

        scanner.set_max_distance(150)

        assert scanner.max_distance == 150

    def test_get_scan_region_absolute(self, mock_detector, mock_validator, mock_ocr_engine):
        """测试获取绝对扫描区域"""
        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine,
            scan_region=RegionConfig(top=0.0, left=0.0, bottom=0.5, right=1.0)
        )

        region = scanner.get_scan_region_absolute((1080, 1920))

        assert region == (0, 0, 540, 1920)

    def test_messages_sorted_by_y(
        self, mock_detector, mock_validator, mock_ocr_engine, sample_image
    ):
        """测试消息按 Y 坐标排序"""
        # 创建多个联系人
        dots = [
            Detection(2, "red_dot", (200, 150, 210, 160), 0.9),
            Detection(2, "red_dot", (200, 50, 210, 60), 0.9),
            Detection(2, "red_dot", (200, 250, 210, 260), 0.9),
        ]
        contacts = [
            Detection(3, "contact_item", (50, 140, 200, 180), 0.95),
            Detection(3, "contact_item", (50, 40, 200, 80), 0.95),
            Detection(3, "contact_item", (50, 240, 200, 280), 0.95),
        ]

        mock_detector.detect_red_dots.return_value = dots
        mock_detector.detect_contact_items.return_value = contacts
        mock_detector.match_dots_to_contacts.return_value = [
            {"contact": contacts[0], "dot": dots[0], "distance": 50},
            {"contact": contacts[1], "dot": dots[1], "distance": 50},
            {"contact": contacts[2], "dot": dots[2], "distance": 50},
        ]

        # 设置 OCR 和验证器
        mock_ocr_engine.engine.ocr.return_value = [[
            [[(0, 0), (10, 0), (10, 10), (0, 10)], ("测试用户", 0.95)]
        ]]

        mock_validator.validate_contact_name.return_value = Mock(
            is_valid=True,
            matched_name="测试用户"
        )

        scanner = ContactScanner(
            yolo_detector=mock_detector,
            username_validator=mock_validator,
            ocr_engine=mock_ocr_engine,
            sort_by_y=True
        )

        result = scanner.scan(sample_image)

        assert len(result.pending_messages) == 3
        # 验证按 Y 坐标排序
        assert result.pending_messages[0].contact_bbox[1] < result.pending_messages[1].contact_bbox[1]
        assert result.pending_messages[1].contact_bbox[1] < result.pending_messages[2].contact_bbox[1]


def test_create_scanner():
    """测试便捷创建函数"""
    mock_detector = Mock()
    mock_validator = Mock()
    mock_ocr = Mock()

    scanner = create_scanner(
        yolo_detector=mock_detector,
        username_validator=mock_validator,
        ocr_engine=mock_ocr,
        scan_region={"top": 0.1, "bottom": 0.5}
    )

    assert scanner.detector == mock_detector
    assert scanner.validator == mock_validator
    assert scanner.scan_region.top == 0.1


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])
