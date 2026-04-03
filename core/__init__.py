# core module
"""
微信代管核心模块

包含:
- YoloDetector: YOLO 目标检测器
- UsernameValidator: 用户名白名单验证器
- ContactScanner: 联系人列表扫描器
"""

from .yolo_detector import (
    YoloDetector,
    Detection,
    create_detector,
    CLASS_NAMES
)

from .username_validator import (
    UsernameValidator,
    WhitelistConfig,
    ValidationResult,
    create_validator,
    load_whitelist_from_file
)

from .contact_scanner import (
    ContactScanner,
    PendingMessage,
    ScanResult,
    RegionConfig,
    OCREngine,
    ScanReason,
    create_scanner
)

__all__ = [
    # YOLO 检测器
    "YoloDetector",
    "Detection",
    "create_detector",
    "CLASS_NAMES",

    # 用户名验证器
    "UsernameValidator",
    "WhitelistConfig",
    "ValidationResult",
    "create_validator",
    "load_whitelist_from_file",

    # 联系人扫描器
    "ContactScanner",
    "PendingMessage",
    "ScanResult",
    "RegionConfig",
    "OCREngine",
    "ScanReason",
    "create_scanner",
]
