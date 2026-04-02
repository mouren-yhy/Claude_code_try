#!/usr/bin/env python3
"""
微信托管服务 - 主程序入口

使用本地 AI 模型自动回复微信私聊消息
"""
import asyncio
import sys
import threading
import time
from pathlib import Path

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import settings
from core.wechat_client import wechat_client
from core.message_handler import message_handler
from core.ai_engine import ai_engine
from storage.database import db
from tray.icon import tray_app
from utils.logger import logger
from web.app import flask_app


def check_dependencies():
    """检查依赖是否安装"""
    missing = []

    try:
        import wxauto
    except ImportError:
        missing.append("wxauto")

    try:
        import flask
    except ImportError:
        missing.append("flask")

    try:
        import requests
    except ImportError:
        missing.append("requests")

    try:
        import pystray
    except ImportError:
        missing.append("pystray")

    if missing:
        logger.error("缺少必要的依赖，请运行: pip install -r requirements.txt")
        logger.error(f"缺少的模块: {', '.join(missing)}")
        return False

    return True


def run_flask_app():
    """运行 Flask Web 服务器"""
    host = settings.get("web.host", "127.0.0.1")
    port = settings.get("web.port", 5001)
    debug = settings.get("web.debug", False)

    logger.info(f"Web 管理后台启动在: http://{host}:{port}")

    # 启用 Flask 的错误日志（用于调试）
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.INFO)

    flask_app.run(host=host, port=port, debug=debug, use_reloader=False)


def on_message_received(message_data: dict):
    """收到消息的回调函数"""
    try:
        # 在新的事件循环中处理异步任务
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(message_handler.handle_message(message_data))
        loop.close()
    except Exception as e:
        logger.error(f"消息处理异常: {e}")


def main(test_mode=False):
    """主函数"""
    # 打印启动信息
    print("\n" + "=" * 50)
    print(" 微信托管服务")
    print(" 使用本地 AI 模型自动回复微信私聊消息")
    if test_mode:
        print(" [测试模式 - 跳过微信连接]")
    print("=" * 50 + "\n")

    # 检查依赖
    if not check_dependencies():
        input("\n按回车键退出...")
        return

    # 检查 Ollama 连接
    logger.info("检查 Ollama 服务...")
    if not ai_engine.test_connection():
        logger.warning("Ollama 服务未连接，请确保已运行: ollama serve")
        logger.warning("程序将继续启动，但 AI 回复功能可能不可用")

    # 启动托盘应用
    logger.info("启动系统托盘...")
    tray_app.start()

    # 连接微信
    wechat_connected = False
    if not test_mode:
        logger.info("连接微信...")
        if wechat_client.connect():
            wechat_connected = True
            # 添加消息监听回调
            wechat_client.add_message_callback(on_message_received)
            # 开始监听微信消息
            logger.info("开始监听微信消息...")
            wechat_client.start_listening()
        else:
            logger.warning("微信连接失败，将仅启用 Web 管理后台和 AI 测试功能")
    else:
        logger.info("测试模式：跳过微信连接")

    # 启动 Web 服务器（在单独线程中）
    flask_thread = threading.Thread(target=run_flask_app, daemon=True)
    flask_thread.start()

    # 打印就绪信息
    print("\n" + "=" * 50)
    print(" 服务已启动！")
    print("=" * 50)
    print(f" 管理后台: http://127.0.0.1:{settings.get('web.port', 5001)}")
    print(f" 微信连接: {'已连接' if wechat_connected else '未连接'}")
    print(f" AI 模型: {settings.get('ollama.model')}")
    print("\n 提示:")
    if wechat_connected:
        print(" - 点击系统托盘图标可以暂停/恢复回复")
    print(" - 在管理后台中测试 AI 功能")
    print(" - 按 Ctrl+C 退出程序")
    print("=" * 50 + "\n")

    # 保持主线程运行
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("收到退出信号...")
    finally:
        # 清理资源
        logger.info("正在关闭服务...")
        if wechat_connected:
            wechat_client.stop_listening()
        tray_app.stop()
        logger.info("服务已关闭")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="微信托管服务")
    parser.add_argument("--test", action="store_true", help="测试模式：跳过微信连接")
    args = parser.parse_args()

    main(test_mode=args.test)
