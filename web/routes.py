"""
Flask 路由定义
"""
from flask import Blueprint, render_template, jsonify, request, redirect, url_for, flash
import json
import asyncio
from datetime import datetime

from config.settings import settings
from core.ai_engine import ai_engine
from core.message_handler import message_handler
from core.personality import style_learning
from storage.database import db, db_sync
from storage.models import Contact
from utils.logger import logger

bp = Blueprint("main", __name__)


def run_async(coro):
    """运行异步函数"""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


@bp.route("/")
def index():
    """首页 - 重定向到仪表盘"""
    return redirect(url_for("main.dashboard"))


@bp.route("/dashboard")
def dashboard():
    """仪表盘 - 显示运行状态"""
    # 获取统计数据
    stats = db_sync.get_stats_sync()
    return render_template("dashboard.html",
                          stats=stats,
                          is_paused=message_handler.is_paused(),
                          ollama_connected=ai_engine.test_connection())


@bp.route("/api/stats")
def api_stats():
    """获取统计数据"""
    stats = db_sync.get_stats_sync()
    return jsonify({
        "success": True,
        "data": stats
    })


@bp.route("/contacts")
def contacts():
    """联系人管理页面"""
    contacts_list = db_sync.get_all_contacts_sync()
    return render_template("contacts.html", contacts=contacts_list)


@bp.route("/api/contacts", methods=["GET"])
def api_contacts():
    """获取联系人列表"""
    whitelist_only = request.args.get("whitelist_only", "false").lower() == "true"
    contacts_list = db_sync.get_all_contacts_sync(whitelist_only=whitelist_only)
    return jsonify({
        "success": True,
        "data": [c.to_dict() for c in contacts_list]
    })


@bp.route("/api/contacts/<int:contact_id>", methods=["GET"])
def api_contact_get(contact_id):
    """获取单个联系人"""
    contacts_list = db_sync.get_all_contacts_sync()
    contact = next((c for c in contacts_list if c.id == contact_id), None)
    if contact:
        return jsonify({
            "success": True,
            "data": contact.to_dict()
        })
    return jsonify({"success": False, "error": "联系人不存在"}), 404


@bp.route("/api/contacts", methods=["POST"])
def api_contact_create():
    """创建联系人"""
    data = request.json
    contact = Contact(
        wx_id=data.get("wx_id", ""),
        name=data.get("name", ""),
        remark=data.get("remark"),
        is_whitelist=data.get("is_whitelist", False),
        system_prompt=data.get("system_prompt")
    )
    contact = db_sync.create_or_update_contact_sync(contact)
    return jsonify({
        "success": True,
        "data": contact.to_dict()
    })


@bp.route("/api/contacts/<int:contact_id>", methods=["PUT"])
def api_contact_update(contact_id):
    """更新联系人"""
    data = request.json
    contacts_list = db_sync.get_all_contacts_sync()
    existing = next((c for c in contacts_list if c.id == contact_id), None)

    if existing:
        contact = Contact(
            id=contact_id,
            wx_id=data.get("wx_id", existing.wx_id),
            name=data.get("name", existing.name),
            remark=data.get("remark", existing.remark),
            is_whitelist=data.get("is_whitelist", existing.is_whitelist),
            system_prompt=data.get("system_prompt", existing.system_prompt),
            style_profile=data.get("style_profile", existing.style_profile)
        )
        contact = db_sync.create_or_update_contact_sync(contact)
        return jsonify({
            "success": True,
            "data": contact.to_dict()
        })

    return jsonify({"success": False, "error": "联系人不存在"}), 404


@bp.route("/api/contacts/<int:contact_id>", methods=["DELETE"])
def api_contact_delete(contact_id):
    """删除联系人"""
    success = db_sync.delete_contact_sync(contact_id)
    if success:
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "删除失败"}), 500


@bp.route("/api/contacts/<int:contact_id>/whitelist", methods=["PUT"])
def api_contact_whitelist(contact_id):
    """设置白名单"""
    data = request.json
    is_whitelist = data.get("is_whitelist", False)
    contacts_list = db_sync.get_all_contacts_sync()
    contact = next((c for c in contacts_list if c.id == contact_id), None)

    if contact:
        contact.is_whitelist = is_whitelist
        db_sync.create_or_update_contact_sync(contact)
        return jsonify({
            "success": True,
            "data": {"is_whitelist": is_whitelist}
        })

    return jsonify({"success": False, "error": "联系人不存在"}), 404


@bp.route("/api/contacts/<int:contact_id>/clear-context", methods=["POST"])
def api_clear_context(contact_id):
    """清空对话上下文"""
    import asyncio
    asyncio.run(message_handler.clear_context(contact_id))
    return jsonify({"success": True})


@bp.route("/api/contacts/<int:contact_id>/messages", methods=["GET"])
def api_contact_messages(contact_id):
    """获取联系人的消息记录"""
    limit = request.args.get("limit", 50, type=int)
    messages = db_sync.get_messages_sync(contact_id, limit=limit)
    return jsonify({
        "success": True,
        "data": [msg.to_dict() for msg in messages]
    })


@bp.route("/chat-history")
def chat_history():
    """聊天历史页面"""
    contacts_list = db_sync.get_all_contacts_sync(whitelist_only=True)
    return render_template("chat_history.html", contacts=contacts_list)


@bp.route("/settings")
def settings_page():
    """设置页面"""
    config = settings.get_all()
    models = ai_engine.get_models()
    return render_template("settings.html",
                          config=config,
                          models=models,
                          current_model=settings.get("ollama.model"))


@bp.route("/api/settings", methods=["GET"])
def api_settings_get():
    """获取配置"""
    return jsonify({
        "success": True,
        "data": settings.get_all()
    })


@bp.route("/api/settings", methods=["PUT"])
def api_settings_update():
    """更新配置"""
    data = request.json
    settings.update(data)
    return jsonify({"success": True})


@bp.route("/api/ai/models", methods=["GET"])
def api_ai_models():
    """获取可用 AI 模型"""
    models = ai_engine.get_models()
    return jsonify({
        "success": True,
        "data": models
    })


@bp.route("/api/ai/test", methods=["POST"])
def api_ai_test():
    """测试 AI 连接"""
    test_message = request.json.get("message", "你好")
    reply = ai_engine.generate(test_message)
    return jsonify({
        "success": True,
        "data": {"reply": reply}
    })


@bp.route("/api/style/upload", methods=["POST"])
def api_style_upload():
    """上传聊天记录学习风格"""
    if "file" not in request.files:
        return jsonify({"success": False, "error": "没有文件"}), 400

    file = request.files["file"]
    contact_name = request.form.get("contact_name", "用户")

    if file.filename == "":
        return jsonify({"success": False, "error": "文件名为空"}), 400

    # 保存文件
    import os
    upload_dir = "data/chat_history"
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, f"{contact_name}_{file.filename}")

    file.save(file_path)

    # 学习风格
    profile = style_learning.learn_from_file(file_path)
    if profile:
        # 保存风格档案
        style_learning.save_profile(contact_name, profile)

        # 生成系统提示词
        system_prompt = style_learning.generate_system_prompt(profile, contact_name)

        return jsonify({
            "success": True,
            "data": {
                "profile": profile,
                "system_prompt": system_prompt
            }
        })

    return jsonify({"success": False, "error": "学习失败"}), 500


@bp.route("/api/control/pause", methods=["POST"])
def api_control_pause():
    """暂停自动回复"""
    message_handler.pause()
    return jsonify({"success": True})


@bp.route("/api/control/resume", methods=["POST"])
def api_control_resume():
    """恢复自动回复"""
    message_handler.resume()
    return jsonify({"success": True})


@bp.route("/api/control/status", methods=["GET"])
def api_control_status():
    """获取控制状态"""
    return jsonify({
        "success": True,
        "data": {
            "paused": message_handler.is_paused(),
            "ollama_connected": ai_engine.test_connection()
        }
    })
