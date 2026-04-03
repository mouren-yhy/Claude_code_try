"""
错误处理装饰器

为 API 路由提供统一的错误处理和日志记录
"""
from functools import wraps
from flask import jsonify
import logging

logger = logging.getLogger(__name__)


def handle_api_error(func):
    """
    API 错误处理装饰器

    捕获异常并返回统一的错误响应格式
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ValueError as e:
            logger.warning(f"Validation error in {func.__name__}: {e}")
            return jsonify({"success": False, "error": str(e)}), 400
        except KeyError as e:
            logger.warning(f"Missing key in {func.__name__}: {e}")
            return jsonify({"success": False, "error": f"Missing required field: {str(e)}"}), 400
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            return jsonify({"success": False, "error": "Internal server error"}), 500
    return wrapper
