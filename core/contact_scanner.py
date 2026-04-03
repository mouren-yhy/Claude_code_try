"""
联系人列表扫描器模块
负责扫描联系人列表，找出白名单用户的新消息

核心流程:
1. 截取联系人列表区域
2. YOLO 检测 red_dot 和 contact_item
3. 关联红点和联系人项
4. OCR 识别联系人名字
5. 白名单匹配
6. 返回待处理消息列表
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

from utils.logger import logger


class ScanReason(Enum):
    """扫描失败原因"""
    NO_SCREENSHOT = "无截图输入"
    NO_RED_DOTS = "未检测到红点"
    NO_CONTACT_ITEMS = "未检测到联系人项"
    NO_MATCHES = "红点与联系人项无匹配"
    OCR_FAILED = "OCR 识别失败"
    NOT_IN_WHITELIST = "不在白名单"


@dataclass
class PendingMessage:
    """
    待处理消息数据类

    Attributes:
        contact_name: 联系人名称（格式：昵称 [微信号]）
        contact_bbox: 联系人项边界框 (x1, y1, x2, y2)
        dot_bbox: 红点边界框 (x1, y1, x2, y2)
        confidence: 检测置信度
        ocr_confidence: OCR 置信度
        raw_ocr_result: OCR 原始结果
    """
    contact_name: str
    contact_bbox: Tuple[float, float, float, float]
    dot_bbox: Tuple[float, float, float, float]
    confidence: float = 0.0
    ocr_confidence: float = 0.0
    raw_ocr_result: Optional[str] = None

    @property
    def contact_center(self) -> Tuple[float, float]:
        """获取联系人项中心点"""
        x1, y1, x2, y2 = self.contact_bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def dot_center(self) -> Tuple[float, float]:
        """获取红点中心点"""
        x1, y1, x2, y2 = self.dot_bbox
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "contact_name": self.contact_name,
            "contact_bbox": list(self.contact_bbox),
            "dot_bbox": list(self.dot_bbox),
            "confidence": self.confidence,
            "ocr_confidence": self.ocr_confidence,
            "contact_center": self.contact_center,
            "dot_center": self.dot_center
        }


@dataclass
class ScanResult:
    """
    扫描结果数据类

    Attributes:
        pending_messages: 待处理消息列表（按从上到下排序）
        total_contacts: 检测到的联系人总数
        total_red_dots: 检测到的红点总数
        scan_duration: 扫描耗时（毫秒）
        errors: 扫描过程中的错误列表
    """
    pending_messages: List[PendingMessage] = field(default_factory=list)
    total_contacts: int = 0
    total_red_dots: int = 0
    scan_duration: float = 0.0
    errors: List[str] = field(default_factory=list)

    @property
    def has_messages(self) -> bool:
        """是否有待处理消息"""
        return len(self.pending_messages) > 0

    @property
    def success(self) -> bool:
        """扫描是否成功"""
        return not self.errors


@dataclass
class RegionConfig:
    """
    区域配置

    Attributes:
        top: 顶部相对位置 (0-1)
        left: 左侧相对位置 (0-1)
        bottom: 底部相对位置 (0-1)
        right: 右侧相对位置 (0-1)
    """
    top: float = 0.0
    left: float = 0.0
    bottom: float = 0.6
    right: float = 1.0

    def to_tuple(self) -> Tuple[float, float, float, float]:
        """转换为元组"""
        return (self.top, self.left, self.bottom, self.right)

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "top": self.top,
            "left": self.left,
            "bottom": self.bottom,
            "right": self.right
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "RegionConfig":
        """从字典创建"""
        return cls(
            top=data.get("top", 0.0),
            left=data.get("left", 0.0),
            bottom=data.get("bottom", 0.6),
            right=data.get("right", 1.0)
        )


@dataclass
class OCREngine:
    """
    OCR 引擎抽象类
    实际使用时传入 PaddleOCR 或其他 OCR 引擎的实例
    """
    engine: Any = None

    def ocr(self, image: np.ndarray) -> str:
        """
        对图像进行 OCR 识别

        Args:
            image: 输入图像

        Returns:
            识别的文本字符串
        """
        if self.engine is None:
            return ""

        try:
            result = self.engine.ocr(image)
            if result and result[0]:
                # PaddleOCR 格式: [[[[x1,y1],[x2,y2],...], (text, confidence)], ...]
                texts = []
                for line in result[0]:
                    if len(line) >= 2:
                        text_info = line[1]
                        if isinstance(text_info, (tuple, list)) and len(text_info) >= 2:
                            text = str(text_info[0]) if text_info[0] else ""
                            if text:
                                texts.append(text)
                        elif isinstance(text_info, str):
                            texts.append(text_info)
                        elif isinstance(text_info, (tuple, list)) and len(text_info) == 0:
                            # 空元组，跳过
                            continue
                return " ".join(texts).strip()
        except Exception as e:
            logger.debug(f"OCR 识别失败: {e}")

        return ""


class ContactScanner:
    """
    联系人列表扫描器

    负责扫描微信联系人列表，检测白名单用户的新消息。

    核心流程:
    1. 裁剪联系人列表区域
    2. YOLO 检测 red_dot 和 contact_item
    3. 关联红点到联系人项
    4. OCR 识别联系人名字
    5. 白名单过滤
    6. 返回待处理消息列表

    示例:
        from core.yolo_detector import YoloDetector
        from core.username_validator import UsernameValidator
        from paddleocr import PaddleOCR

        detector = YoloDetector("data/models/wechat_yolov8s.pt")
        validator = UsernameValidator.create_from_users(["张三 [wechat123]"])
        ocr_engine = OCREngine(engine=PaddleOCR(use_angle_cls=True, lang='ch'))

        scanner = ContactScanner(detector, validator, ocr_engine)
        result = scanner.scan(screenshot)
        for msg in result.pending_messages:
            print(f"新消息: {msg.contact_name}")
    """

    def __init__(
        self,
        yolo_detector,
        username_validator,
        ocr_engine: OCREngine,
        scan_region: Optional[RegionConfig] = None,
        max_distance: float = 100,
        sort_by_y: bool = True
    ):
        """
        初始化扫描器

        Args:
            yolo_detector: YoloDetector 实例
            username_validator: UsernameValidator 实例
            ocr_engine: OCREngine 实例
            scan_region: 扫描区域配置
            max_distance: 红点到联系人的最大关联距离（像素）
            sort_by_y: 是否按 Y 坐标排序（从上到下）
        """
        self.detector = yolo_detector
        self.validator = username_validator
        self.ocr = ocr_engine
        self.scan_region = scan_region or RegionConfig()
        self.max_distance = max_distance
        self.sort_by_y = sort_by_y

        logger.info(
            f"联系人扫描器初始化完成，"
            f"扫描区域: {self.scan_region.to_dict()}, "
            f"最大关联距离: {max_distance}px"
        )

    def scan(self, screenshot: Optional[np.ndarray]) -> ScanResult:
        """
        扫描联系人列表，返回白名单用户的新消息

        Args:
            screenshot: 完整截图

        Returns:
            扫描结果，包含待处理消息列表
        """
        import time
        start_time = time.time()

        result = ScanResult()

        # 1. 验证输入
        if screenshot is None:
            result.errors.append(ScanReason.NO_SCREENSHOT.value)
            return result

        try:
            # 2. 裁剪联系人列表区域
            contact_region, region_offset = self._crop_contact_region(screenshot)
            if contact_region is None or contact_region.size == 0:
                result.errors.append("裁剪联系人区域失败")
                return result

            # 3. YOLO 检测
            dots = self.detector.detect_red_dots(contact_region)
            contacts = self.detector.detect_contact_items(contact_region)

            result.total_red_dots = len(dots)
            result.total_contacts = len(contacts)

            if not dots:
                result.errors.append(ScanReason.NO_RED_DOTS.value)
                return result

            if not contacts:
                result.errors.append(ScanReason.NO_CONTACT_ITEMS.value)
                return result

            # 4. 关联红点和联系人项
            contacts_with_dots = self.detector.match_dots_to_contacts(
                dots, contacts, self.max_distance
            )

            if not contacts_with_dots:
                result.errors.append(ScanReason.NO_MATCHES.value)
                return result

            # 5. OCR 识别并白名单匹配
            whitelist_messages = []

            for item in contacts_with_dots:
                contact = item["contact"]
                dot = item["dot"]

                # OCR 识别
                name = self._ocr_contact_name(contact_region, contact)

                if name is None:
                    continue

                # 白名单匹配
                validation = self.validator.validate_contact_name(name)

                if validation.is_valid:
                    # 计算在完整截图中的坐标
                    full_contact_bbox = self._adjust_bbox_to_full_screenshot(
                        contact.bbox, region_offset
                    )
                    full_dot_bbox = self._adjust_bbox_to_full_screenshot(
                        dot.bbox, region_offset
                    )

                    whitelist_messages.append(PendingMessage(
                        contact_name=name,
                        contact_bbox=full_contact_bbox,
                        dot_bbox=full_dot_bbox,
                        confidence=contact.confidence
                    ))

            # 6. 按 Y 坐标排序（从上到下）
            if self.sort_by_y:
                whitelist_messages.sort(key=lambda x: x.contact_bbox[1])

            result.pending_messages = whitelist_messages

            if whitelist_messages:
                logger.info(
                    f"扫描完成，发现 {len(whitelist_messages)} 个白名单用户的新消息"
                )

        except Exception as e:
            logger.error(f"扫描过程异常: {e}", exc_info=True)
            result.errors.append(f"扫描异常: {str(e)}")

        finally:
            result.scan_duration = (time.time() - start_time) * 1000

        return result

    def _crop_contact_region(
        self, screenshot: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Tuple[int, int]]:
        """
        裁剪联系人列表区域

        Args:
            screenshot: 完整截图

        Returns:
            (裁剪后的图像, (x偏移, y偏移))
        """
        try:
            h, w = screenshot.shape[:2]

            top = int(self.scan_region.top * h)
            left = int(self.scan_region.left * w)
            bottom = int(self.scan_region.bottom * h)
            right = int(self.scan_region.right * w)

            # 确保坐标有效
            if top >= bottom or left >= right:
                return None, (0, 0)

            region = screenshot[top:bottom, left:right]
            return region, (left, top)

        except Exception as e:
            logger.error(f"裁剪联系人区域失败: {e}")
            return None, (0, 0)

    def _adjust_bbox_to_full_screenshot(
        self,
        bbox: Tuple[float, float, float, float],
        offset: Tuple[int, int]
    ) -> Tuple[float, float, float, float]:
        """
        将裁剪区域的坐标调整为完整截图的坐标

        Args:
            bbox: 裁剪区域中的边界框
            offset: 区域偏移 (x_offset, y_offset)

        Returns:
            完整截图中的边界框
        """
        x_offset, y_offset = offset
        x1, y1, x2, y2 = bbox
        return (
            x1 + x_offset,
            y1 + y_offset,
            x2 + x_offset,
            y2 + y_offset
        )

    def _ocr_contact_name(
        self,
        region_image: np.ndarray,
        contact
    ) -> Optional[str]:
        """
        OCR 识别联系人名字

        Args:
            region_image: 联系人列表区域图像
            contact: 联系人项检测结果

        Returns:
            识别的用户名或 None
        """
        try:
            # 裁剪联系人项区域
            contact_crop = self.detector.crop_detection(region_image, contact, padding=5)

            if contact_crop.size == 0:
                return None

            # 进一步裁剪昵称区域
            # 微信 UI 布局：头像在左，昵称在头像右侧偏上
            h, w = contact_crop.shape[:2]

            # 假设昵称在中间偏上区域（可根据实际 UI 调整）
            name_crop = contact_crop[:h//2, w//6:w*5//6]

            # OCR 识别
            name = self.ocr.ocr(name_crop)

            if name:
                # 清理识别结果
                name = name.strip()
                # 移除常见的干扰字符
                name = name.replace("...", "").replace("…", "")
                if name:
                    logger.debug(f"OCR 识别用户名: {name}")
                    return name

        except Exception as e:
            logger.debug(f"OCR 识别联系人名字失败: {e}")

        return None

    def set_scan_region(self, region: RegionConfig):
        """
        更新扫描区域

        Args:
            region: 新的区域配置
        """
        self.scan_region = region
        logger.info(f"扫描区域已更新: {region.to_dict()}")

    def set_max_distance(self, distance: float):
        """
        更新最大关联距离

        Args:
            distance: 新的最大距离（像素）
        """
        self.max_distance = distance
        logger.info(f"最大关联距离已更新: {distance}px")

    def get_scan_region_absolute(
        self, image_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        获取扫描区域的绝对坐标

        Args:
            image_shape: 图像尺寸 (height, width)

        Returns:
            (top, left, bottom, right) 绝对坐标
        """
        h, w = image_shape[:2]
        return (
            int(self.scan_region.top * h),
            int(self.scan_region.left * w),
            int(self.scan_region.bottom * h),
            int(self.scan_region.right * w)
        )


# 便捷函数
def create_scanner(
    yolo_detector,
    username_validator,
    ocr_engine: Any,
    scan_region: Optional[Dict] = None
) -> ContactScanner:
    """
    创建联系人扫描器的便捷函数

    Args:
        yolo_detector: YoloDetector 实例
        username_validator: UsernameValidator 实例
        ocr_engine: OCR 引擎实例（如 PaddleOCR）
        scan_region: 扫描区域字典

    Returns:
        ContactScanner 实例
    """
    region = None
    if scan_region:
        region = RegionConfig.from_dict(scan_region)

    ocr_wrapper = OCREngine(engine=ocr_engine)

    return ContactScanner(
        yolo_detector=yolo_detector,
        username_validator=username_validator,
        ocr_engine=ocr_wrapper,
        scan_region=region
    )
