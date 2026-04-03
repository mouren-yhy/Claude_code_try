# 微信代管项目 YOLO + OCR 方案 - 实施计划

**版本**: v1.0 Final
**日期**: 2026-04-03
**预计工期**: 6.5 - 8.5 天
**可行性评级**: ⭐⭐⭐⭐⭐ (强烈推荐实施)

---

## 目录

- [一、项目背景](#一项目背景)
- [二、核心流程](#二核心流程)
- [三、检测类别定义](#三检测类别定义)
- [四、白名单配置](#四白名单配置)
- [五、核心模块设计](#五核心模块设计)
- [六、文件结构](#六文件结构)
- [七、配置文件](#七配置文件)
- [八、实施时间表](#八实施时间表)
- [九、依赖管理](#九依赖管理)
- [十、测试方案](#十测试方案)
- [十一、风险管理](#十一风险管理)
- [十二、实施检查清单](#十二实施检查清单)

---

## 一、项目背景

### 1.1 当前方案问题

| 问题 | 影响 |
|------|------|
| 坐标模板依赖 | 窗口缩放/移动后需重新校准 |
| 颜色检测不稳定 | 不同主题下阈值需调整，暗黑模式识别率低 |
| 无法识别 UI 元素 | 无法检测小红点、联系人标识等 |
| OCR 区域不准 | 依赖固定坐标导致识别率波动大 |

### 1.2 项目需求

- **多白名单用户回复**：需要识别当前聊天对象
- **新消息检测**：通过小红点检测新消息
- **自动切换聊天**：在多个白名单联系人之间切换并回复
- **不影响正常使用**：非白名单用户的消息不干预

### 1.3 技术选型

| 组件 | 选择 | 理由 |
|------|------|------|
| 目标检测 | YOLOv8s | 精度高、速度快、生态成熟 |
| OCR识别 | PP-OCRv4 | 已集成、中文优化、2M参数 |
| 训练框架 | Ultralytics | API简单、文档完善 |
| 部署格式 | PyTorch + ONNX | 跨平台兼容 |

### 1.4 硬件环境

- **CPU**: i9
- **GPU**: RTX 5070
- **优势**: 性能充裕，可使用 GPU 加速训练

---

## 二、核心流程

```
┌─────────────────────────────────────────────────────────────────────┐
│                              主循环（每2秒）                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │ 1. 截取联系人列表区域                                         │    │
│  │    ↓                                                          │    │
│  │ 2. YOLO 检测: red_dot + contact_item                          │    │
│  │    ↓                                                          │    │
│  │ 3. 对每个 red_dot 对应的 contact_item 进行 OCR                 │    │
│  │    ↓                                                          │    │
│  │ 4. 白名单完全匹配判断（格式：昵称 [微信号]）                   │    │
│  │    ├─ 匹配失败 → 跳过（完全忽略，不记录）                      │    │
│  │    └─ 匹配成功 → 继续                                         │    │
│  │         ↓                                                      │    │
│  │ 5. 点击 contact_item，切换到该用户                             │    │
│  │         ↓                                                      │    │
│  │ 6. YOLO 检测 + OCR chat_title 二次验证                        │    │
│  │    ├─ 匹配失败 → 记录日志，跳过该用户                          │    │
│  │    └─ 匹配成功 → 继续                                         │    │
│  │         ↓                                                      │    │
│  │ 7. 检测新消息 bubble_other，OCR 内容                           │    │
│  │         ↓                                                      │    │
│  │ 8. AI 生成回复，发送，留在此用户                               │    │
│  │         ↓                                                      │    │
│  │ 9. 继续扫描列表，处理下一个白名单用户                          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.1 关键设计决策

| 决策点 | 方案 | 原因 |
|--------|------|------|
| 白名单标识 | 用户备注（含微信号） | 唯一性保证 |
| 备注格式 | `昵称 [微信号]` | 明确的格式规范 |
| 匹配策略 | 完全匹配 | 无相似度风险 |
| 列表判断 | OCR 后再决定是否点击 | 不影响非白名单使用 |
| 二次验证 | chat_title 完全匹配 | 双重保险 |
| 失败处理 | 记录日志跳过 | 不阻断流程 |
| 非白名单处理 | 完全忽略 | 不影响正常使用 |
| 扫描频率 | 每2秒 | 平衡响应速度与性能 |

### 2.2 交互逻辑

**单个白名单用户有新消息**：
```
检测红点 → OCR验证 → 切换 → 二次验证 → 提取消息 → 生成回复 → 发送 → 留在该用户
```

**多个白名单用户有新消息**：
```
按列表从上往下依次处理：
  检测用户A → 回复 → 留在A
  → 继续扫描 → 检测用户B → 回复 → 留在B
  → 继续扫描 → 检测用户C → 回复 → 留在C
```

**非白名单用户有新消息**：
```
检测红点 → OCR验证 → 不在白名单 → 跳过（完全忽略）
```

---

## 三、检测类别定义

### 3.1 类别列表

| 类别ID | 类别名称 | 优先级 | OCR需求 | 检测场景 | 预估样本量 |
|--------|----------|--------|---------|----------|------------|
| 0 | bubble_other | 高 | ✅ 需要 | 提取对方消息内容 | 150张 |
| 1 | bubble_own | 中 | ❌ 不需要 | 判断是否有未回复消息 | 100张 |
| 2 | red_dot | 高 | ❌ 不需要 | 触发新消息检测 | 100张 |
| 3 | contact_item | 高 | ✅ 需要 | 获取联系人名字 | 100张 |
| 4 | chat_title | 高 | ✅ 需要 | 验证当前聊天对象 | 100张 |

**总计样本量**: ~550 张标注图片

### 3.2 类别详细说明

#### bubble_other (类别0)
- **用途**: 识别对方发送的消息气泡，提取消息内容
- **特征**: 位于聊天窗口左侧，绿色/白色背景
- **OCR区域**: 整个气泡区域
- **注意事项**: 需要处理多行消息、表情消息、链接消息等

#### bubble_own (类别1)
- **用途**: 识别自己发送的消息，判断是否需要回复
- **特征**: 位于聊天窗口右侧，白色/浅色背景
- **OCR需求**: 不需要，仅用于位置判断

#### red_dot (类别2)
- **用途**: 检测新消息小红点
- **特征**: 红色圆形，通常在联系人项右侧
- **关联**: 需要与 contact_item 进行空间关联

#### contact_item (类别3)
- **用途**: 识别联系人列表项，获取联系人名字
- **特征**: 列表项包含头像、昵称、最新消息预览
- **OCR区域**: 昵称区域（通常在头像右侧）

#### chat_title (类别4)
- **用途**: 识别聊天窗口标题，二次验证当前聊天对象
- **特征**: 位于聊天窗口顶部中央
- **OCR内容**: 当前聊天对象的完整昵称（含微信号）

---

## 四、白名单配置

### 4.1 备注格式规范

**统一格式**: `昵称 [微信号]`

**示例**:
```
张三 [wechat123]
李四 [wechat456]
王五 [wechat789]
```

### 4.2 配置文件格式

```json
{
  "whitelist": {
    "users": [
      "张三 [wechat123]",
      "李四 [wechat456]",
      "王五 [wechat789]"
    ],
    "match_mode": "exact",
    "format_template": "昵称 [微信号]"
  }
}
```

### 4.3 匹配规则

- **匹配方式**: 完全匹配（字符串相等）
- **大小写**: 区分大小写
- **空格**: 需严格按格式
- **验证次数**: 两次（联系人列表 + 聊天标题）

---

## 五、核心模块设计

### 5.1 模块架构

```
┌─────────────────────────────────────────────────────────────┐
│                         主控制器                              │
│                    (wechat_client.py)                        │
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ YOLO Detector │    │ContactScanner│    │UsernameValidator│
└──────────────┘    └──────────────┘    └──────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  OCR Engine   │    │  OCR Engine   │    │  Config/DB   │
└──────────────┘    └──────────────┘    └──────────────┘
```

### 5.2 YOLO 检测器

```python
# core/yolo_detector.py
"""
YOLO 目标检测器
用于检测微信 UI 元素：消息气泡、红点、联系人项、聊天标题
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional
import torch


class YoloDetector:
    """YOLO 目标检测器"""

    # 类别名称
    CLASS_NAMES = [
        "bubble_other",   # 0 - 对方消息气泡
        "bubble_own",     # 1 - 自己消息气泡
        "red_dot",        # 2 - 新消息红点
        "contact_item",   # 3 - 联系人列表项
        "chat_title"      # 4 - 聊天窗口标题
    ]

    def __init__(self, model_path: str, confidence: float = 0.5, iou_threshold: float = 0.45, use_gpu: bool = True):
        """
        初始化 YOLO 检测器

        Args:
            model_path: 模型文件路径
            confidence: 置信度阈值
            iou_threshold: IOU 阈值
            use_gpu: 是否使用 GPU
        """
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.iou_threshold = iou_threshold
        self.device = "cuda:0" if use_gpu and torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        检测图像中的所有目标

        Args:
            image: 输入图像 (BGR格式)

        Returns:
            检测结果列表，每个结果包含:
            {
                "class_id": int,
                "class_name": str,
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }
        """
        results = self.model(image, conf=self.confidence, iou=self.iou_threshold, verbose=False)
        detections = []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                detections.append({
                    "class_id": int(box.cls),
                    "class_name": self.CLASS_NAMES[int(box.cls)],
                    "bbox": [float(x1), float(y1), float(x2), float(y2)],
                    "confidence": float(box.conf)
                })

        return detections

    def detect_by_class(self, image: np.ndarray, class_names: List[str]) -> List[Dict]:
        """
        只检测指定类别的目标

        Args:
            image: 输入图像
            class_names: 类别名称列表

        Returns:
            指定类别的检测结果
        """
        all_detections = self.detect(image)
        return [d for d in all_detections if d["class_name"] in class_names]

    def detect_red_dots(self, image: np.ndarray) -> List[Dict]:
        """检测红点"""
        return self.detect_by_class(image, ["red_dot"])

    def detect_contact_items(self, image: np.ndarray) -> List[Dict]:
        """检测联系人列表项"""
        return self.detect_by_class(image, ["contact_item"])

    def detect_bubbles(self, image: np.ndarray) -> Dict[str, List[Dict]]:
        """检测消息气泡，返回分组结果"""
        detections = self.detect_by_class(image, ["bubble_other", "bubble_own"])
        return {
            "other": [d for d in detections if d["class_name"] == "bubble_other"],
            "own": [d for d in detections if d["class_name"] == "bubble_own"]
        }

    def detect_chat_title(self, image: np.ndarray) -> Optional[Dict]:
        """
        检测聊天标题

        Returns:
            标题检测结果或 None
        """
        results = self.detect_by_class(image, ["chat_title"])
        return results[0] if results else None

    def crop_detection(self, image: np.ndarray, detection: Dict, padding: int = 5) -> np.ndarray:
        """
        裁剪检测区域

        Args:
            image: 原始图像
            detection: 检测结果
            padding: 扩展边距

        Returns:
            裁剪后的图像
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = detection["bbox"]

        # 添加边距并限制在图像范围内
        x1 = max(0, int(x1) - padding)
        y1 = max(0, int(y1) - padding)
        x2 = min(w, int(x2) + padding)
        y2 = min(h, int(y2) + padding)

        return image[y1:y2, x1:x2]

    def match_dots_to_contacts(self, dots: List[Dict], contacts: List[Dict], max_distance: float = 100) -> List[Dict]:
        """
        将红点关联到对应的联系人项

        Args:
            dots: 红点检测结果
            contacts: 联系人项检测结果
            max_distance: 最大关联距离（像素）

        Returns:
            带红点的联系人项列表
        """
        contacts_with_dots = []

        for contact in contacts:
            cx1, cy1, cx2, cy2 = contact["bbox"]
            contact_center_x = (cx1 + cx2) / 2
            contact_center_y = (cy1 + cy2) / 2

            for dot in dots:
                dx1, dy1, dx2, dy2 = dot["bbox"]
                dot_center_x = (dx1 + dx2) / 2
                dot_center_y = (dy1 + dy2) / 2

                # 计算距离
                distance = ((contact_center_x - dot_center_x) ** 2 +
                           (contact_center_y - dot_center_y) ** 2) ** 0.5

                # 红点应该在联系人项的右侧
                if dot_center_x > contact_center_x and distance < max_distance:
                    contacts_with_dots.append({
                        "contact": contact,
                        "dot": dot,
                        "distance": distance
                    })
                    break

        return contacts_with_dots
```

### 5.3 用户名验证器

```python
# core/username_validator.py
"""
用户名验证器
负责白名单匹配和聊天标题验证
"""

from typing import Set, List, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    matched_name: Optional[str] = None
    reason: Optional[str] = None


class UsernameValidator:
    """
    用户名验证器
    使用完全匹配策略
    """

    def __init__(self, whitelist: Set[str]):
        """
        初始化验证器

        Args:
            whitelist: 白名单用户集合
        """
        self.whitelist = whitelist
        logger.info(f"白名单初始化完成，共 {len(whitelist)} 个用户")

    def validate_contact_name(self, detected_name: str) -> ValidationResult:
        """
        验证联系人名字

        Args:
            detected_name: OCR 检测到的用户名

        Returns:
            验证结果
        """
        if not detected_name:
            return ValidationResult(
                is_valid=False,
                reason="OCR 未识别到用户名"
            )

        if detected_name in self.whitelist:
            logger.debug(f"联系人验证通过: {detected_name}")
            return ValidationResult(
                is_valid=True,
                matched_name=detected_name
            )

        # 不在白名单：不记录日志（按需求）
        return ValidationResult(
            is_valid=False,
            reason=f"不在白名单"
        )

    def validate_chat_title(self, chat_title: str) -> ValidationResult:
        """
        验证聊天标题（二次验证）

        Args:
            chat_title: OCR 检测到的聊天标题

        Returns:
            验证结果
        """
        if not chat_title:
            return ValidationResult(
                is_valid=False,
                reason="OCR 未识别到聊天标题"
            )

        if chat_title in self.whitelist:
            logger.debug(f"聊天标题验证通过: {chat_title}")
            return ValidationResult(
                is_valid=True,
                matched_name=chat_title
            )

        logger.warning(f"聊天标题验证失败: {chat_title}")
        return ValidationResult(
            is_valid=False,
            reason=f"标题不在白名单: {chat_title}"
        )

    def update_whitelist(self, new_whitelist: Set[str]):
        """更新白名单"""
        self.whitelist = new_whitelist
        logger.info(f"白名单已更新，共 {len(new_whitelist)} 个用户")

    def is_whitelisted(self, name: str) -> bool:
        """快速检查是否在白名单中"""
        return name in self.whitelist
```

### 5.4 联系人扫描器

```python
# core/contact_scanner.py
"""
联系人列表扫描器
负责扫描联系人列表，找出白名单用户的新消息
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass

from .yolo_detector import YoloDetector
from .username_validator import UsernameValidator

logger = logging.getLogger(__name__)


@dataclass
class PendingMessage:
    """待处理消息"""
    contact_name: str
    contact_bbox: List[float]
    dot_bbox: List[float]
    confidence: float


class ContactScanner:
    """
    联系人列表扫描器

    流程：
    1. 截取联系人列表区域
    2. YOLO 检测 red_dot 和 contact_item
    3. 关联红点和联系人项
    4. OCR 识别联系人名字
    5. 白名单匹配
    6. 返回待处理消息列表
    """

    def __init__(self, yolo_detector: YoloDetector, username_validator: UsernameValidator,
                 ocr_engine, scan_region: Dict[str, float] = None):
        """
        初始化扫描器

        Args:
            yolo_detector: YOLO 检测器
            username_validator: 用户名验证器
            ocr_engine: OCR 引擎
            scan_region: 扫描区域 {"top": 0, "left": 0, "bottom": 0.6, "right": 1.0}
        """
        self.detector = yolo_detector
        self.validator = username_validator
        self.ocr = ocr_engine
        self.scan_region = scan_region or {
            "top": 0,
            "left": 0,
            "bottom": 0.6,
            "right": 1.0
        }

    def scan(self, screenshot: np.ndarray) -> List[PendingMessage]:
        """
        扫描联系人列表，返回白名单用户的新消息

        Args:
            screenshot: 完整截图

        Returns:
            待处理消息列表（按从上到下排序）
        """
        # 1. 裁剪联系人列表区域
        contact_region = self._crop_contact_region(screenshot)
        if contact_region is None:
            return []

        # 2. YOLO 检测 red_dot 和 contact_item
        dots = self.detector.detect_red_dots(contact_region)
        contacts = self.detector.detect_contact_items(contact_region)

        if not dots or not contacts:
            return []

        # 3. 关联红点和联系人项
        contacts_with_dots = self.detector.match_dots_to_contacts(dots, contacts)

        if not contacts_with_dots:
            return []

        # 4. OCR 识别并白名单匹配
        whitelist_messages = []

        for item in contacts_with_dots:
            contact = item["contact"]
            name = self._ocr_contact_name(contact_region, contact)

            if name is None:
                continue

            # 5. 白名单匹配
            validation = self.validator.validate_contact_name(name)

            if validation.is_valid:
                # 计算在完整截图中的坐标
                full_bbox = self._adjust_bbox_to_full_screenshot(contact["bbox"])

                whitelist_messages.append(PendingMessage(
                    contact_name=name,
                    contact_bbox=full_bbox,
                    dot_bbox=contact["dot"]["bbox"],
                    confidence=contact["confidence"]
                ))

        # 6. 按 Y 坐标排序（从上到下）
        whitelist_messages.sort(key=lambda x: x.contact_bbox[1])

        if whitelist_messages:
            logger.info(f"扫描完成，发现 {len(whitelist_messages)} 个白名单用户的新消息")

        return whitelist_messages

    def _crop_contact_region(self, screenshot: np.ndarray) -> np.ndarray:
        """裁剪联系人列表区域"""
        h, w = screenshot.shape[:2]

        top = int(self.scan_region["top"] * h)
        left = int(self.scan_region["left"] * w)
        bottom = int(self.scan_region["bottom"] * h)
        right = int(self.scan_region["right"] * w)

        return screenshot[top:bottom, left:right]

    def _adjust_bbox_to_full_screenshot(self, bbox: List[float]) -> List[float]:
        """将裁剪区域的坐标调整为完整截图的坐标"""
        h, w = 1080, 1920  # 需要根据实际情况调整
        # 这里需要根据 scan_region 计算偏移
        return bbox

    def _ocr_contact_name(self, region_image: np.ndarray, contact: Dict) -> Optional[str]:
        """
        OCR 识别联系人名字

        Args:
            region_image: 联系人列表区域图像
            contact: 联系人项检测结果

        Returns:
            识别的用户名或 None
        """
        # 裁剪联系人项区域
        contact_crop = self.detector.crop_detection(region_image, contact, padding=5)

        # 进一步裁剪昵称区域（通常在左侧，头像右侧）
        # 这里可以根据实际 UI 布局调整
        h, w = contact_crop.shape[:2]
        name_crop = contact_crop[:h//2, w//4:w*3//4]  # 假设昵称在中间偏上区域

        # OCR 识别
        name = self.ocr.ocr(name_crop)

        if name:
            # 清理识别结果（去除空格等）
            name = name.strip()
            logger.debug(f"OCR 识别用户名: {name}")
            return name

        return None
```

### 5.5 主流程控制器

```python
# core/wechat_client.py (修改部分)
"""
微信客户端主控制器（修改后的核心流程）
"""

import time
import cv2
import numpy as np
from typing import List, Set, Optional
import logging

from .yolo_detector import YoloDetector
from .contact_scanner import ContactScanner, PendingMessage
from .username_validator import UsernameValidator
from .ai_engine import AIEngine
from .message_handler import MessageHandler

logger = logging.getLogger(__name__)


class WeChatClient:
    """
    微信客户端主控制器

    核心流程：
    1. 扫描联系人列表（每2秒）
    2. 对每个白名单用户：
       - 切换到该用户
       - 验证聊天标题
       - 提取新消息
       - 生成回复
       - 发送
       - 留在该用户
    3. 继续扫描下一个用户
    """

    def __init__(self, config: dict):
        self.config = config

        # 初始化各个模块
        self.yolo_detector = YoloDetector(
            model_path=config["yolo"]["model_path"],
            confidence=config["yolo"]["confidence"],
            use_gpu=config["yolo"]["use_gpu"]
        )

        self.username_validator = UsernameValidator(
            whitelist=set(config["whitelist"]["users"])
        )

        self.contact_scanner = ContactScanner(
            yolo_detector=self.yolo_detector,
            username_validator=self.username_validator,
            ocr_engine=self.ocr_engine,
            scan_region=config["scanner"]["contact_list_region"]
        )

        self.ai_engine = AIEngine(config)
        self.message_handler = MessageHandler(config)

        self.scan_interval = config["scanner"]["scan_interval"] / 1000  # 转换为秒
        self.running = False

    def start(self):
        """启动主循环"""
        self.running = True
        logger.info("微信客户端启动")

        while self.running:
            try:
                self._main_loop()
            except Exception as e:
                logger.error(f"主循环异常: {e}", exc_info=True)
                time.sleep(self.scan_interval)

    def stop(self):
        """停止主循环"""
        self.running = False
        logger.info("微信客户端停止")

    def _main_loop(self):
        """主循环逻辑"""
        # 1. 截图
        screenshot = self._screenshot()

        # 2. 扫描白名单用户的新消息
        pending_messages = self.contact_scanner.scan(screenshot)

        if not pending_messages:
            time.sleep(self.scan_interval)
            return

        # 3. 处理每个白名单用户
        for message in pending_messages:
            if not self.running:
                break

            self._process_user_message(message)

        # 4. 等待下一次扫描
        time.sleep(self.scan_interval)

    def _process_user_message(self, message: PendingMessage):
        """
        处理单个用户的消息

        Args:
            message: 待处理消息
        """
        contact_name = message.contact_name
        logger.info(f"开始处理用户: {contact_name}")

        # 1. 点击联系人项，切换到该用户
        self._click_contact(message.contact_bbox)
        time.sleep(0.5)  # 等待界面切换

        # 2. 验证聊天标题（二次验证）
        if not self._verify_chat_title(contact_name):
            logger.warning(f"聊天标题验证失败: {contact_name}")
            return

        # 3. 检测新消息
        new_messages = self._detect_new_messages()

        if not new_messages:
            logger.info(f"用户 {contact_name} 无新消息")
            return

        # 4. 生成并发送回复
        for msg in new_messages:
            reply = self.ai_engine.generate_reply(contact_name, msg)
            self._send_reply(reply)

        logger.info(f"用户 {contact_name} 处理完成")

    def _verify_chat_title(self, expected_name: str) -> bool:
        """
        验证聊天标题

        Args:
            expected_name: 期望的聊天标题

        Returns:
            验证是否通过
        """
        screenshot = self._screenshot()

        # 检测聊天标题
        title_detection = self.yolo_detector.detect_chat_title(screenshot)

        if not title_detection:
            logger.warning("未检测到聊天标题")
            return False

        # OCR 识别
        title_crop = self.yolo_detector.crop_detection(screenshot, title_detection)
        detected_title = self.ocr_engine.ocr(title_crop)

        if not detected_title:
            logger.warning("OCR 未识别到聊天标题")
            return False

        # 验证
        validation = self.username_validator.validate_chat_title(detected_title)

        if validation.is_valid:
            return True

        logger.warning(f"聊天标题验证失败: 期望={expected_name}, 实际={detected_title}")
        return False

    def _detect_new_messages(self) -> List[str]:
        """
        检测新消息

        Returns:
            新消息内容列表
        """
        screenshot = self._screenshot()

        # 检测对方消息气泡
        bubbles = self.yolo_detector.detect_by_class(screenshot, ["bubble_other"])

        messages = []
        for bubble in bubbles:
            # OCR 提取消息内容
            bubble_crop = self.yolo_detector.crop_detection(screenshot, bubble)
            content = self.ocr_engine.ocr(bubble_crop)

            if content:
                messages.append(content)

        return messages

    def _screenshot(self) -> np.ndarray:
        """截取微信窗口"""
        # 实现截图逻辑
        pass

    def _click_contact(self, bbox: List[float]):
        """点击联系人项"""
        # 实现点击逻辑
        pass

    def _send_reply(self, reply: str):
        """发送回复"""
        # 实现发送逻辑
        pass
```

---

## 六、文件结构

```
wechat_custody/
├── core/
│   ├── __init__.py
│   ├── yolo_detector.py          # 新增：YOLO检测器
│   ├── contact_scanner.py        # 新增：联系人扫描器
│   ├── username_validator.py     # 新增：用户名验证器
│   ├── wechat_client.py          # 修改：适配新流程
│   ├── ai_engine.py              # 复用：AI回复引擎
│   ├── personality.py            # 复用：人设配置
│   ├── message_handler.py        # 修改：接口适配
│   ├── template_detector.py      # 保留：备用方案
│   └── ocr_engine.py             # 复用：OCR封装
│
├── training/                      # 新增：训练相关
│   ├── __init__.py
│   ├── train.py                  # 训练入口
│   ├── dataset.py                # 数据集处理
│   ├── collect_data.py           # 数据采集脚本
│   ├── wechat_ui.yaml            # YOLO数据集配置
│   └── README.md                 # 训练说明文档
│
├── data/
│   ├── dataset/                  # 新增：训练数据
│   │   ├── images/               # 原始图片
│   │   │   ├── train/            # 训练集 (~440张)
│   │   │   └── val/              # 验证集 (~110张)
│   │   └── labels/               # YOLO格式标注
│   │       ├── train/
│   │       └── val/
│   └── models/                   # 新增：训练好的模型
│       ├── wechat_yolov8s.pt     # PyTorch模型
│       └── wechat_yolov8s.onnx   # ONNX模型（可选）
│
├── config/
│   ├── __init__.py
│   ├── settings.json             # 修改：添加新配置
│   └── personality_config.json   # 复用：人设配置
│
├── storage/                      # 复用：数据存储
│   ├── __init__.py
│   ├── database.py               # 复用：数据库
│   └── conversation_history.py   # 复用：对话历史
│
├── web/                          # 复用：Web界面
│   └── ...
│
├── tray/                         # 复用：托盘图标
│   └── ...
│
├── tests/                        # 新增：测试用例
│   ├── __init__.py
│   ├── test_yolo_detector.py
│   ├── test_contact_scanner.py
│   ├── test_username_validator.py
│   └── test_integration.py
│
├── main.py                       # 修改：主入口
├── requirements.txt              # 修改：添加依赖
├── README.md                     # 修改：项目说明
└── YOLO_OCR_实施计划.md          # 本文档
```

---

## 七、配置文件

### 7.1 完整配置文件

```json
{
  "wechat": {
    "window_title": "微信",
    "window_class": "WeChatMainWnd"
  },

  "yolo": {
    "model_path": "data/models/wechat_yolov8s.pt",
    "onnx_path": "data/models/wechat_yolov8s.onnx",
    "confidence": 0.5,
    "iou_threshold": 0.45,
    "use_gpu": true,
    "device": "cuda:0"
  },

  "whitelist": {
    "users": [
      "张三 [wechat123]",
      "李四 [wechat456]",
      "王五 [wechat789]"
    ],
    "match_mode": "exact",
    "format_template": "昵称 [微信号]"
  },

  "scanner": {
    "scan_interval": 2000,
    "max_consecutive_skips": 10,
    "contact_list_region": {
      "top": 0,
      "left": 0,
      "bottom": 0.6,
      "right": 1.0
    },
    "chat_title_region": {
      "top": 0,
      "left": 0.3,
      "bottom": 0.1,
      "right": 0.7
    }
  },

  "ocr": {
    "engine": "paddleocr",
    "lang": "ch",
    "use_angle_cls": true,
    "drop_score": 0.5
  },

  "ai": {
    "provider": "openai",
    "model": "gpt-4o",
    "temperature": 0.7,
    "max_tokens": 500
  },

  "reply": {
    "auto_send": true,
    "delay_after_switch": 500,
    "delay_before_send": 300,
    "typing_simulation": true
  },

  "logging": {
    "level": "INFO",
    "log_validation_failures": true,
    "log_skip_non_whitelist": false,
    "log_file": "logs/wechat_custody.log"
  }
}
```

### 7.2 训练配置文件

```yaml
# training/wechat_ui.yaml
# YOLO 数据集配置文件

# 数据集路径
path: ../data/dataset
train: images/train
val: images/val

# 类别数量
nc: 5

# 类别名称
names:
  0: bubble_other
  1: bubble_own
  2: red_dot
  3: contact_item
  4: chat_title
```

---

## 八、实施时间表

### 8.1 总时间表（8天）

| 阶段 | 工时 | 天数 | 主要任务 |
|------|------|------|----------|
| **数据集准备** | 16h | 2天 | 采集截图、标注 |
| **模型训练** | 6h | 1天 | 环境配置、训练、导出 |
| **代码开发** | 32h | 4天 | 核心模块、集成、测试 |
| **验证优化** | 8h | 1天 | 功能验证、性能优化 |
| **总计** | **62h** | **8天** | |

### 8.2 详细时间表

#### Day 1-2：数据集准备

**Day 1 上午（4h）**：数据采集脚本开发
- [ ] 编写 `training/collect_data.py`
- [ ] 实现自动截图功能
- [ ] 实现分类保存功能

**Day 1 下午（4h）**：数据采集
- [ ] 采集 bubble_other 样本 150+ 张
- [ ] 采集 bubble_own 样本 100+ 张
- [ ] 采集 red_dot 样本 100+ 张

**Day 2 上午（4h）**：继续数据采集
- [ ] 采集 contact_item 样本 100+ 张
- [ ] 采集 chat_title 样本 100+ 张
- [ ] 检查数据多样性（不同主题、分辨率）

**Day 2 下午（4h）**：数据标注
- [ ] 安装 CVAT 标注工具
- [ ] 导入图片进行标注
- [ ] 导出 YOLO 格式标注文件
- [ ] 划分 train/val 数据集（8:2）

#### Day 3：模型训练

**Day 3 上午（2h）**：环境配置
- [ ] 安装 ultralytics、PyTorch GPU
- [ ] 配置 wechat_ui.yaml
- [ ] 验证数据集格式

**Day 3 下午（4h）**：模型训练
- [ ] 启动训练（50 epochs）
- [ ] 监控训练过程
- [ ] 评估模型性能（mAP@0.5 > 70%）

**Day 3 晚上（1h）**：模型导出
- [ ] 导出 PyTorch 模型
- [ ] （可选）导出 ONNX 模型

#### Day 4-5：核心模块开发

**Day 4（8h）**：
- [ ] `yolo_detector.py` 开发（4h）
- [ ] `username_validator.py` 开发（2h）
- [ ] 单元测试（2h）

**Day 5（8h）**：
- [ ] `contact_scanner.py` 开发（4h）
- [ ] `wechat_client.py` 适配（2h）
- [ ] 单元测试（2h）

#### Day 6-7：集成与适配

**Day 6（8h）**：
- [ ] 主循环开发（4h）
- [ ] 配置系统扩展（2h）
- [ ] 集成测试（2h）

**Day 7（8h）**：
- [ ] 端到端测试（4h）
- [ ] Bug 修复（4h）

#### Day 8：验证与优化

**Day 8（8h）**：
- [ ] 功能验证（2h）
- [ ] 性能测试（2h）
- [ ] 边界情况测试（2h）
- [ ] 文档编写（2h）

---

## 九、依赖管理

### 9.1 依赖文件

```txt
# requirements.txt

# YOLO 相关
ultralytics>=8.0.0
torch>=2.0.0
torchvision>=0.15.0
onnxruntime-gpu>=1.16.0

# OCR（已有）
paddleocr>=2.7.0
paddlepaddle>=2.5.0

# 图像处理
opencv-python>=4.8.0
pillow>=10.0.0
numpy>=1.24.0

# Windows 控制
pywin32>=305

# Web 框架
flask>=3.0.0
flask-cors>=4.0.0

# AI（已有）
openai>=1.0.0

# 数据库
sqlalchemy>=2.0.0

# 日志
loguru>=0.7.0

# 配置
pyyaml>=6.0
python-dotenv>=1.0.0
```

### 9.2 安装命令

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt

# 验证 GPU
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 十、测试方案

### 10.1 功能测试用例

| 编号 | 测试场景 | 测试步骤 | 预期结果 |
|------|----------|----------|----------|
| T1 | 单个白名单用户有新消息 | 1. 设置单个白名单用户<br>2. 发送新消息<br>3. 等待扫描 | 正确识别并回复 |
| T2 | 多个白名单用户有新消息 | 1. 设置多个白名单用户<br>2. 依次发送消息<br>3. 等待扫描 | 按从上到下依次处理 |
| T3 | 非白名单用户有新消息 | 1. 发送消息（非白名单）<br>2. 等待扫描 | 完全忽略，不影响使用 |
| T4 | 聊天标题验证失败 | 1. 模拟标题OCR错误<br>2. 发送新消息 | 记录日志，跳过该用户 |
| T5 | 联系人 OCR 识别错误 | 1. 模拟联系人OCR错误<br>2. 发送新消息 | 跳过该条目，继续扫描 |
| T6 | 亮色主题 | 1. 切换到亮色主题<br>2. 发送新消息 | 正常工作 |
| T7 | 暗黑主题 | 1. 切换到暗黑主题<br>2. 发送新消息 | 正常工作 |
| T8 | 窗口缩放 | 1. 调整窗口大小<br>2. 发送新消息 | 正常工作 |
| T9 | 白名单格式验证 | 1. 使用错误格式的白名单<br>2. 启动程序 | 提示格式错误或忽略 |
| T10 | 连续多次跳过 | 1. 设置 max_consecutive_skips=3<br>2. 连续4次无新消息 | 记录日志，继续运行 |

### 10.2 性能测试

| 指标 | 目标值 | 测试方法 |
|------|--------|----------|
| 单次扫描延迟 | < 500ms | 计时测试 |
| YOLO 检测延迟 | < 100ms | 计时测试 |
| OCR 单次延迟 | < 200ms | 计时测试 |
| CPU 占用（待机） | < 10% | 性能监控 |
| CPU 占用（扫描中） | < 40% | 性能监控 |
| GPU 占用（扫描中） | < 30% | 性能监控 |
| 内存占用 | < 1GB | 内存监控 |

### 10.3 边界情况测试

| 场景 | 处理方式 |
|------|----------|
| 消息被撤回 | 跳过，继续处理下一条 |
| 纯表情消息 | OCR 失败，跳过 |
| 链接消息 | 尝试 OCR，失败则跳过 |
| 图片消息 | OCR 失败，跳过 |
| 语音消息 | OCR 失败，跳过 |
| 空白消息 | 跳过 |
| 超长消息 | 正常处理 |
| 特殊字符消息 | 正常处理 |

---

## 十一、风险管理

### 11.1 风险矩阵

| 风险 | 影响 | 概率 | 风险等级 | 缓解措施 |
|------|------|------|----------|----------|
| 数据集不足 | 检测精度低 | 中 | 中 | 数据增强、迁移学习、增加样本 |
| 微信 UI 变化 | 模型失效 | 低 | 低 | 用户承诺近期不变 |
| OCR 识别错误 | 漏检/误检 | 中 | 中 | 完全匹配策略降低风险 |
| 用户备注格式错误 | 匹配失败 | 低 | 低 | 配置验证 + 格式检查 |
| GPU 显存不足 | 训练失败 | 低 | 低 | 使用 YOLOv8s，显存需求低 |
| 多用户同时消息 | 处理延迟 | 低 | 低 | 异步处理，留存在当前用户 |
| 网络波动 | AI 回复失败 | 中 | 中 | 重试机制 + 本地缓存 |

### 11.2 降级方案

| 情况 | 降级方案 |
|------|----------|
| YOLO 检测持续失败 | 切换到 template_detector（保留的备用方案） |
| OCR 持续失败 | 提示用户检查 OCR 配置 |
| AI 回复失败 | 使用预设模板回复 |
| 程序崩溃 | 自动重启 + 日志记录 |

---

## 十二、实施检查清单

### 12.1 准备阶段

- [ ] 确认白名单用户已按 `昵称 [微信号]` 格式修改备注
- [ ] 确认微信版本近期不会更新
- [ ] 确认 GPU 驱动已安装
- [ ] 创建虚拟环境
- [ ] 安装基础依赖

### 12.2 数据准备（Day 1-2）

- [ ] 开发数据采集脚本
- [ ] 采集 bubble_other 样本 150+ 张
- [ ] 采集 bubble_own 样本 100+ 张
- [ ] 采集 red_dot 样本 100+ 张
- [ ] 采集 contact_item 样本 100+ 张
- [ ] 采集 chat_title 样本 100+ 张
- [ ] 安装 CVAT 标注工具
- [ ] 完成所有标注
- [ ] 验证标注格式
- [ ] 划分 train/val 数据集（8:2）
- [ ] 检查数据多样性（主题、分辨率）

### 12.3 模型训练（Day 3）

- [ ] 安装 ultralytics + PyTorch GPU
- [ ] 验证 CUDA 可用
- [ ] 配置 wechat_ui.yaml
- [ ] 验证数据集格式
- [ ] 启动训练（50 epochs）
- [ ] 监控训练过程
- [ ] mAP@0.5 > 70%
- [ ] 导出 PyTorch 模型
- [ ] （可选）导出 ONNX 模型

### 12.4 代码开发（Day 4-5）

- [ ] `yolo_detector.py` 开发完成
- [ ] `yolo_detector.py` 单元测试通过
- [ ] `username_validator.py` 开发完成
- [ ] `username_validator.py` 单元测试通过
- [ ] `contact_scanner.py` 开发完成
- [ ] `contact_scanner.py` 单元测试通过
- [ ] `wechat_client.py` 适配完成
- [ ] 主循环开发完成
- [ ] 配置系统扩展完成

### 12.5 集成测试（Day 6-7）

- [ ] 单个白名单用户测试通过
- [ ] 多个白名单用户测试通过
- [ ] 非白名单用户忽略测试通过
- [ ] 聊天标题验证测试通过
- [ ] 亮色主题测试通过
- [ ] 暗黑主题测试通过
- [ ] 窗口缩放测试通过

### 12.6 性能验证（Day 8）

- [ ] 单次扫描延迟 < 500ms
- [ ] YOLO 检测延迟 < 100ms
- [ ] OCR 单次延迟 < 200ms
- [ ] CPU 占用（待机）< 10%
- [ ] CPU 占用（扫描中）< 40%
- [ ] 内存占用 < 1GB

### 12.7 交付准备

- [ ] 代码注释完善
- [ ] README 更新
- [ ] 配置文件示例
- [ ] 用户使用手册
- [ ] 问题排查指南

---

## 十三、附录

### 13.1 YOLO 训练命令

```bash
# 训练命令
python training/train.py \
  --model yolov8s.pt \
  --data training/wechat_ui.yaml \
  --epochs 50 \
  --batch 16 \
  --img 640 \
  --device 0 \
  --name wechat_ui

# 导出 ONNX
yolo export model=data/models/wechat_yolov8s.pt format=onnx
```

### 13.2 常用命令

```bash
# 运行主程序
python main.py

# 运行测试
pytest tests/

# 数据采集
python training/collect_data.py

# 查看训练结果
yolo detect predict model=data/models/wechat_yolov8s.pt source=test.jpg
```

### 13.3 目录结构创建脚本

```bash
# 创建目录结构
mkdir -p data/dataset/images/train
mkdir -p data/dataset/images/val
mkdir -p data/dataset/labels/train
mkdir -p data/dataset/labels/val
mkdir -p data/models
mkdir -p training
mkdir -p tests
mkdir -p logs
```

---

**文档版本**: v1.0 Final
**最后更新**: 2026-04-03
**状态**: 已批准，可开始实施
