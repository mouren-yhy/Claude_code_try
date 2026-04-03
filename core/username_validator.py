"""
用户名验证器模块
负责白名单匹配和聊天标题验证

使用完全匹配策略，确保只有白名单用户会被处理。
"""

import logging
from typing import Set, List, Optional, Dict
from dataclasses import dataclass, field
from enum import Enum

from utils.logger import logger


class ValidationReason(Enum):
    """验证失败原因"""
    NOT_IN_WHITELIST = "不在白名单"
    OCR_FAILED = "OCR 未识别到内容"
    FORMAT_MISMATCH = "格式不匹配"
    EMPTY_CONTENT = "内容为空"


@dataclass
class ValidationResult:
    """
    验证结果数据类

    Attributes:
        is_valid: 是否验证通过
        matched_name: 匹配到的用户名（验证通过时）
        reason: 失败原因（验证失败时）
        raw_input: 原始输入内容
    """
    is_valid: bool
    matched_name: Optional[str] = None
    reason: Optional[str] = None
    raw_input: Optional[str] = None

    def __str__(self) -> str:
        if self.is_valid:
            return f"✓ 验证通过: {self.matched_name}"
        return f"✗ 验证失败: {self.reason}"


@dataclass
class WhitelistConfig:
    """
    白名单配置

    Attributes:
        users: 白名单用户列表
        match_mode: 匹配模式 ("exact" 完全匹配, "fuzzy" 模糊匹配)
        format_template: 格式模板说明
        case_sensitive: 是否区分大小写
    """
    users: List[str] = field(default_factory=list)
    match_mode: str = "exact"
    format_template: str = "昵称 [微信号]"
    case_sensitive: bool = True

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "users": self.users,
            "match_mode": self.match_mode,
            "format_template": self.format_template,
            "case_sensitive": self.case_sensitive
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "WhitelistConfig":
        """从字典创建"""
        return cls(
            users=data.get("users", []),
            match_mode=data.get("match_mode", "exact"),
            format_template=data.get("format_template", "昵称 [微信号]"),
            case_sensitive=data.get("case_sensitive", True)
        )


class UsernameValidator:
    """
    用户名验证器

    使用完全匹配策略，确保只有白名单用户才会被处理。

    核心逻辑:
    1. 联系人列表 OCR 识别 → 白名单匹配 → 匹配成功则点击
    2. 点击后切换聊天 → 聊天标题 OCR 识别 → 二次验证
    3. 两次验证都通过才处理消息

    示例:
        config = WhitelistConfig(users=["张三 [wechat123]", "李四 [wechat456]"])
        validator = UsernameValidator(config)
        result = validator.validate_contact_name("张三 [wechat123]")
        if result.is_valid:
            print(f"匹配成功: {result.matched_name}")
    """

    def __init__(self, config: Optional[WhitelistConfig] = None):
        """
        初始化验证器

        Args:
            config: 白名单配置，None 则使用默认配置
        """
        self.config = config or WhitelistConfig()
        self._build_whitelist()

        logger.info(
            f"白名单验证器初始化完成，"
            f"共 {len(self.whitelist)} 个用户，"
            f"匹配模式: {self.config.match_mode}"
        )

    def _build_whitelist(self):
        """构建白名单集合"""
        if self.config.case_sensitive:
            self.whitelist: Set[str] = set(self.config.users)
        else:
            self.whitelist: Set[str] = {u.lower() for u in self.config.users}

    def validate_contact_name(self, detected_name: Optional[str]) -> ValidationResult:
        """
        验证联系人名字（第一次验证）

        Args:
            detected_name: OCR 检测到的用户名

        Returns:
            验证结果
        """
        # 保存原始输入
        raw_input = detected_name

        # 空值检查
        if not detected_name:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.OCR_FAILED.value,
                raw_input=raw_input
            )

        # 清理输入
        detected_name = detected_name.strip()

        if not detected_name:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.EMPTY_CONTENT.value,
                raw_input=raw_input
            )

        # 大小写处理
        check_name = detected_name if self.config.case_sensitive else detected_name.lower()

        # 完全匹配
        if check_name in self.whitelist:
            logger.debug(f"联系人验证通过: {detected_name}")
            return ValidationResult(
                is_valid=True,
                matched_name=detected_name,
                raw_input=raw_input
            )

        # 不在白名单：不记录日志（按需求）
        return ValidationResult(
            is_valid=False,
            reason=ValidationReason.NOT_IN_WHITELIST.value,
            raw_input=raw_input
        )

    def validate_chat_title(self, chat_title: Optional[str]) -> ValidationResult:
        """
        验证聊天标题（第二次验证）

        Args:
            chat_title: OCR 检测到的聊天标题

        Returns:
            验证结果
        """
        # 保存原始输入
        raw_input = chat_title

        # 空值检查
        if not chat_title:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.OCR_FAILED.value,
                raw_input=raw_input
            )

        # 清理输入
        chat_title = chat_title.strip()

        if not chat_title:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.EMPTY_CONTENT.value,
                raw_input=raw_input
            )

        # 大小写处理
        check_title = chat_title if self.config.case_sensitive else chat_title.lower()

        # 完全匹配
        if check_title in self.whitelist:
            logger.debug(f"聊天标题验证通过: {chat_title}")
            return ValidationResult(
                is_valid=True,
                matched_name=chat_title,
                raw_input=raw_input
            )

        # 二次验证失败需要记录
        logger.warning(f"聊天标题验证失败: {chat_title} (原因: {ValidationReason.NOT_IN_WHITELIST.value})")
        return ValidationResult(
            is_valid=False,
            reason=ValidationReason.NOT_IN_WHITELIST.value,
            raw_input=raw_input
        )

    def update_whitelist(self, new_config: WhitelistConfig):
        """
        更新白名单

        Args:
            new_config: 新的白名单配置
        """
        self.config = new_config
        self._build_whitelist()
        logger.info(f"白名单已更新，共 {len(self.whitelist)} 个用户")

    def add_user(self, username: str):
        """
        添加单个用户到白名单

        Args:
            username: 用户名（格式：昵称 [微信号]）
        """
        if username not in self.config.users:
            self.config.users.append(username)
            self._build_whitelist()
            logger.info(f"已添加用户到白名单: {username}")

    def remove_user(self, username: str):
        """
        从白名单移除用户

        Args:
            username: 用户名
        """
        if username in self.config.users:
            self.config.users.remove(username)
            self._build_whitelist()
            logger.info(f"已从白名单移除用户: {username}")

    def is_whitelisted(self, name: Optional[str]) -> bool:
        """
        快速检查是否在白名单中

        Args:
            name: 用户名

        Returns:
            是否在白名单中
        """
        if not name:
            return False

        check_name = name if self.config.case_sensitive else name.lower()
        return check_name in self.whitelist

    def get_whitelist(self) -> List[str]:
        """
        获取白名单列表

        Returns:
            白名单用户列表
        """
        return self.config.users.copy()

    def get_whitelist_count(self) -> int:
        """
        获取白名单用户数量

        Returns:
            用户数量
        """
        return len(self.whitelist)

    def validate_format(self, username: str) -> ValidationResult:
        """
        验证用户名格式是否符合要求

        要求格式: "昵称 [微信号]"

        Args:
            username: 用户名

        Returns:
            验证结果
        """
        if not username:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.EMPTY_CONTENT.value,
                raw_input=username
            )

        username = username.strip()

        # 检查是否包含 [ 和 ]
        if '[' not in username or ']' not in username:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.FORMAT_MISMATCH.value + ": 缺少 [ 或 ]",
                raw_input=username
            )

        # 检查格式是否正确
        if not username.count('[') == 1 or not username.count(']') == 1:
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.FORMAT_MISMATCH.value + ": 括号数量不正确",
                raw_input=username
            )

        # 检查 ] 是否在 [ 之后
        if username.index(']') < username.index('['):
            return ValidationResult(
                is_valid=False,
                reason=ValidationReason.FORMAT_MISMATCH.value + ": 括号顺序错误",
                raw_input=username
            )

        return ValidationResult(
            is_valid=True,
            matched_name=username,
            raw_input=username
        )

    def parse_username(self, formatted_name: str) -> Optional[Dict[str, str]]:
        """
        解析格式化的用户名

        Args:
            formatted_name: 格式如 "张三 [wechat123]" 的用户名

        Returns:
            包含 nickname 和 wechat_id 的字典，失败返回 None
        """
        if '[' not in formatted_name or ']' not in formatted_name:
            return None

        try:
            # 分离昵称和微信号
            parts = formatted_name.split('[')
            if len(parts) != 2:
                return None

            nickname = parts[0].strip()
            wechat_id = parts[1].rstrip(']').strip()

            if not nickname or not wechat_id:
                return None

            return {
                "nickname": nickname,
                "wechat_id": wechat_id,
                "full": formatted_name
            }
        except Exception:
            return None

    def find_similar_users(self, input_name: str, threshold: int = 2) -> List[str]:
        """
        查找相似的用户名（用于调试/提示）

        Args:
            input_name: 输入的用户名
            threshold: 编辑距离阈值

        Returns:
            相似的用户名列表
        """
        import difflib

        similar = difflib.get_close_matches(
            input_name,
            self.config.users,
            n=3,
            cutoff=0.6
        )

        return similar

    def export_config(self) -> Dict:
        """
        导出配置

        Returns:
            配置字典
        """
        return {
            "whitelist": {
                "users": self.config.users,
                "match_mode": self.config.match_mode,
                "format_template": self.config.format_template,
                "case_sensitive": self.config.case_sensitive,
                "count": len(self.whitelist)
            }
        }

    @staticmethod
    def create_from_users(users: List[str]) -> "UsernameValidator":
        """
        从用户列表创建验证器

        Args:
            users: 用户名列表

        Returns:
            UsernameValidator 实例
        """
        config = WhitelistConfig(users=users)
        return UsernameValidator(config)


# 便捷函数
def create_validator(users: List[str]) -> UsernameValidator:
    """
    创建用户名验证器的便捷函数

    Args:
        users: 白名单用户列表

    Returns:
        UsernameValidator 实例
    """
    return UsernameValidator.create_from_users(users)


def load_whitelist_from_file(file_path: str) -> WhitelistConfig:
    """
    从文件加载白名单

    Args:
        file_path: 文件路径（每行一个用户名）

    Returns:
        WhitelistConfig 实例
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            users = [line.strip() for line in f if line.strip() and not line.startswith('#')]

        logger.info(f"从文件加载白名单: {file_path}, 共 {len(users)} 个用户")
        return WhitelistConfig(users=users)
    except Exception as e:
        logger.error(f"加载白名单文件失败: {e}")
        return WhitelistConfig()
