"""
用户名验证器单元测试

运行方式: pytest tests/test_username_validator.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, patch

# 添加项目根目录到路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.username_validator import (
    UsernameValidator,
    WhitelistConfig,
    ValidationResult,
    ValidationReason,
    create_validator,
    load_whitelist_from_file
)
import tempfile
import os


class TestValidationResult:
    """ValidationResult 数据类测试"""

    def test_success_result(self):
        """测试成功的验证结果"""
        result = ValidationResult(
            is_valid=True,
            matched_name="张三 [wechat123]"
        )

        assert result.is_valid is True
        assert result.matched_name == "张三 [wechat123]"
        assert "✓" in str(result)

    def test_failure_result(self):
        """测试失败的验证结果"""
        result = ValidationResult(
            is_valid=False,
            reason=ValidationReason.NOT_IN_WHITELIST.value,
            raw_input="未知用户"
        )

        assert result.is_valid is False
        assert result.reason == "不在白名单"
        assert result.raw_input == "未知用户"
        assert "✗" in str(result)

    def test_result_with_all_fields(self):
        """测试包含所有字段的结果"""
        result = ValidationResult(
            is_valid=True,
            matched_name="李四 [wechat456]",
            raw_input=" 李四 [wechat456] "
        )

        assert result.matched_name == "李四 [wechat456]"
        assert result.raw_input == " 李四 [wechat456] "


class TestWhitelistConfig:
    """WhitelistConfig 数据类测试"""

    def test_default_config(self):
        """测试默认配置"""
        config = WhitelistConfig()

        assert config.users == []
        assert config.match_mode == "exact"
        assert config.format_template == "昵称 [微信号]"
        assert config.case_sensitive is True

    def test_config_with_users(self):
        """测试带用户的配置"""
        config = WhitelistConfig(
            users=["张三 [wechat123]", "李四 [wechat456]"],
            match_mode="exact",
            case_sensitive=True
        )

        assert len(config.users) == 2
        assert "张三 [wechat123]" in config.users

    def test_to_dict(self):
        """测试转换为字典"""
        config = WhitelistConfig(
            users=["用户A"],
            match_mode="fuzzy",
            case_sensitive=False
        )

        result = config.to_dict()
        assert result["users"] == ["用户A"]
        assert result["match_mode"] == "fuzzy"
        assert result["case_sensitive"] is False

    def test_from_dict(self):
        """测试从字典创建"""
        data = {
            "users": ["测试用户"],
            "match_mode": "exact",
            "format_template": "昵称 [微信号]",
            "case_sensitive": True
        }

        config = WhitelistConfig.from_dict(data)
        assert config.users == ["测试用户"]
        assert config.match_mode == "exact"

    def test_from_dict_with_defaults(self):
        """测试从字典创建（使用默认值）"""
        config = WhitelistConfig.from_dict({})

        assert config.users == []
        assert config.match_mode == "exact"
        assert config.case_sensitive is True


class TestUsernameValidator:
    """UsernameValidator 测试"""

    @pytest.fixture
    def sample_whitelist(self):
        """示例白名单"""
        return [
            "张三 [wechat123]",
            "李四 [wechat456]",
            "王五 [wechat789]"
        ]

    @pytest.fixture
    def validator(self, sample_whitelist):
        """创建验证器实例"""
        config = WhitelistConfig(users=sample_whitelist)
        return UsernameValidator(config)

    def test_initialization(self, validator):
        """测试初始化"""
        assert validator.get_whitelist_count() == 3
        assert validator.config.match_mode == "exact"

    def test_validate_contact_name_success(self, validator):
        """测试联系人验证成功"""
        result = validator.validate_contact_name("张三 [wechat123]")

        assert result.is_valid is True
        assert result.matched_name == "张三 [wechat123]"

    def test_validate_contact_name_not_in_whitelist(self, validator):
        """测试联系人不在白名单"""
        result = validator.validate_contact_name("陌生人 [abc]")

        assert result.is_valid is False
        assert result.reason == ValidationReason.NOT_IN_WHITELIST.value

    def test_validate_contact_name_empty(self, validator):
        """测试空输入"""
        result = validator.validate_contact_name("")

        assert result.is_valid is False
        assert result.reason == ValidationReason.OCR_FAILED.value

    def test_validate_contact_name_none(self, validator):
        """测试 None 输入"""
        result = validator.validate_contact_name(None)

        assert result.is_valid is False
        assert result.reason == ValidationReason.OCR_FAILED.value

    def test_validate_contact_name_with_spaces(self, validator):
        """测试带空格的输入（应被清理）"""
        result = validator.validate_contact_name("  张三 [wechat123]  ")

        assert result.is_valid is True
        assert result.matched_name == "张三 [wechat123]"

    def test_validate_chat_title_success(self, validator):
        """测试聊天标题验证成功"""
        result = validator.validate_chat_title("李四 [wechat456]")

        assert result.is_valid is True
        assert result.matched_name == "李四 [wechat456]"

    def test_validate_chat_title_failure(self, validator):
        """测试聊天标题验证失败"""
        result = validator.validate_chat_title("陌生人")

        assert result.is_valid is False
        assert result.reason == ValidationReason.NOT_IN_WHITELIST.value

    def test_case_sensitive_matching(self):
        """测试大小写敏感匹配"""
        config = WhitelistConfig(
            users=["Test [id123]"],
            case_sensitive=True
        )
        validator = UsernameValidator(config)

        # 大小写完全匹配
        assert validator.validate_contact_name("Test [id123]").is_valid is True

        # 大小写不匹配
        assert validator.validate_contact_name("test [id123]").is_valid is False

    def test_case_insensitive_matching(self):
        """测试大小写不敏感匹配"""
        config = WhitelistConfig(
            users=["Test [id123]"],
            case_sensitive=False
        )
        validator = UsernameValidator(config)

        # 大小写不同但应该匹配
        assert validator.validate_contact_name("Test [id123]").is_valid is True
        assert validator.validate_contact_name("test [id123]").is_valid is True
        assert validator.validate_contact_name("TEST [ID123]").is_valid is True

    def test_update_whitelist(self, validator):
        """测试更新白名单"""
        new_config = WhitelistConfig(users=["新用户 [new123]"])
        validator.update_whitelist(new_config)

        assert validator.get_whitelist_count() == 1
        assert validator.validate_contact_name("新用户 [new123]").is_valid is True

    def test_add_user(self, validator):
        """测试添加单个用户"""
        validator.add_user("赵六 [wechat000]")

        assert validator.get_whitelist_count() == 4
        assert "赵六 [wechat000]" in validator.get_whitelist()

    def test_add_duplicate_user(self, validator):
        """测试添加重复用户"""
        original_count = validator.get_whitelist_count()
        validator.add_user("张三 [wechat123]")  # 已存在

        assert validator.get_whitelist_count() == original_count

    def test_remove_user(self, validator):
        """测试移除用户"""
        validator.remove_user("张三 [wechat123]")

        assert validator.get_whitelist_count() == 2
        assert validator.validate_contact_name("张三 [wechat123]").is_valid is False

    def test_remove_nonexistent_user(self, validator):
        """测试移除不存在的用户"""
        original_count = validator.get_whitelist_count()
        validator.remove_user("不存在 [abc]")

        assert validator.get_whitelist_count() == original_count

    def test_is_whitelisted(self, validator):
        """测试快速检查是否在白名单"""
        assert validator.is_whitelisted("张三 [wechat123]") is True
        assert validator.is_whitelisted("陌生人") is False
        assert validator.is_whitelisted("") is False
        assert validator.is_whitelisted(None) is False

    def test_get_whitelist(self, validator):
        """测试获取白名单列表"""
        whitelist = validator.get_whitelist()

        assert len(whitelist) == 3
        assert "张三 [wechat123]" in whitelist

        # 确保返回的是副本
        whitelist.append("新用户")
        assert validator.get_whitelist_count() == 3

    def test_validate_format_valid(self, validator):
        """测试格式验证 - 有效格式"""
        result = validator.validate_format("张三 [wechat123]")

        assert result.is_valid is True
        assert result.matched_name == "张三 [wechat123]"

    def test_validate_format_no_brackets(self, validator):
        """测试格式验证 - 缺少括号"""
        result = validator.validate_format("张三 wechat123")

        assert result.is_valid is False
        assert "缺少" in result.reason and ("[" in result.reason or "]" in result.reason)

    def test_validate_format_wrong_bracket_order(self, validator):
        """测试格式验证 - 括号顺序错误"""
        result = validator.validate_format("张三 ]wechat123[")

        assert result.is_valid is False
        assert "括号顺序错误" in result.reason

    def test_validate_format_multiple_brackets(self, validator):
        """测试格式验证 - 多个括号"""
        result = validator.validate_format("张三 [wechat123] [extra]")

        assert result.is_valid is False
        assert "括号数量不正确" in result.reason

    def test_validate_format_empty(self, validator):
        """测试格式验证 - 空字符串"""
        result = validator.validate_format("")

        assert result.is_valid is False
        assert result.reason == ValidationReason.EMPTY_CONTENT.value

    def test_parse_username_valid(self, validator):
        """测试解析用户名 - 有效格式"""
        result = validator.parse_username("张三 [wechat123]")

        assert result is not None
        assert result["nickname"] == "张三"
        assert result["wechat_id"] == "wechat123"
        assert result["full"] == "张三 [wechat123]"

    def test_parse_username_no_brackets(self, validator):
        """测试解析用户名 - 无括号"""
        result = validator.parse_username("张三 wechat123")

        assert result is None

    def test_parse_username_empty_parts(self, validator):
        """测试解析用户名 - 空部分"""
        result = validator.parse_username(" [] ")

        assert result is None

    def test_parse_username_complex_nickname(self, validator):
        """测试解析用户名 - 复杂昵称"""
        result = validator.parse_username("张三·李四 [wechat123]")

        assert result is not None
        assert result["nickname"] == "张三·李四"

    def test_find_similar_users(self, validator):
        """测试查找相似用户"""
        similar = validator.find_similar_users("张三 [wechat12]")

        # 应该能找到相似的用户
        assert isinstance(similar, list)
        # 可能找到 "张三 [wechat123]" 作为相似结果

    def test_find_similar_users_empty_input(self, validator):
        """测试查找相似用户 - 空输入"""
        similar = validator.find_similar_users("")

        assert isinstance(similar, list)

    def test_export_config(self, validator):
        """测试导出配置"""
        config = validator.export_config()

        assert "whitelist" in config
        assert config["whitelist"]["count"] == 3
        assert config["whitelist"]["match_mode"] == "exact"

    def test_raw_input_preserved(self, validator):
        """测试原始输入被保留"""
        raw_name = "  张三 [wechat123]  "
        result = validator.validate_contact_name(raw_name)

        assert result.raw_input == raw_name


class TestConvenienceFunctions:
    """便捷函数测试"""

    def test_create_validator(self):
        """测试创建验证器的便捷函数"""
        users = ["用户A", "用户B"]
        validator = create_validator(users)

        assert isinstance(validator, UsernameValidator)
        assert validator.get_whitelist_count() == 2

    def test_create_validator_empty(self):
        """测试创建空验证器"""
        validator = create_validator([])

        assert validator.get_whitelist_count() == 0

    def test_load_whitelist_from_file(self):
        """测试从文件加载白名单"""
        # 创建临时文件
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            f.write("# 这是注释\n")
            f.write("张三 [wechat123]\n")
            f.write("\n")
            f.write("李四 [wechat456]\n")
            f.write("# 另一个注释\n")
            f.write("王五 [wechat789]\n")
            temp_path = f.name

        try:
            config = load_whitelist_from_file(temp_path)

            assert len(config.users) == 3
            assert "张三 [wechat123]" in config.users
            assert "李四 [wechat456]" in config.users
            assert "王五 [wechat789]" in config.users
            # 确保注释行没有被加载
            assert not any(u.startswith("#") for u in config.users)
        finally:
            os.unlink(temp_path)

    def test_load_whitelist_from_nonexistent_file(self):
        """测试从不存在的文件加载"""
        config = load_whitelist_from_file("/nonexistent/path/file.txt")

        assert config.users == []

    def test_load_empty_file(self):
        """测试加载空文件"""
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as f:
            # 只写入注释
            f.write("# 只有注释\n")
            f.write("# \n")
            temp_path = f.name

        try:
            config = load_whitelist_from_file(temp_path)
            assert config.users == []
        finally:
            os.unlink(temp_path)


class TestEdgeCases:
    """边界情况测试"""

    def test_validate_with_special_characters(self):
        """测试包含特殊字符的用户名"""
        config = WhitelistConfig(users=["用户@123 [id_123]"])
        validator = UsernameValidator(config)

        assert validator.validate_contact_name("用户@123 [id_123]").is_valid is True

    def test_validate_with_unicode(self):
        """测试包含 Unicode 的用户名"""
        config = WhitelistConfig(users=["用户😀 [emoji123]"])
        validator = UsernameValidator(config)

        assert validator.validate_contact_name("用户😀 [emoji123]").is_valid is True

    def test_very_long_username(self):
        """测试很长的用户名"""
        long_name = "A" * 100 + " [id123]"
        config = WhitelistConfig(users=[long_name])
        validator = UsernameValidator(config)

        assert validator.validate_contact_name(long_name).is_valid is True

    def test_whitespace_variations(self):
        """测试各种空格变体"""
        config = WhitelistConfig(users=["张三 [wechat123]"])
        validator = UsernameValidator(config)

        # 制表符
        assert validator.validate_contact_name("\t张三 [wechat123]\t").is_valid is True
        # 换行符会被 strip 去除
        assert validator.validate_contact_name("\n张三 [wechat123]\n").is_valid is True


if __name__ == "__main__":
    # 可以直接运行此文件进行测试
    pytest.main([__file__, "-v"])
