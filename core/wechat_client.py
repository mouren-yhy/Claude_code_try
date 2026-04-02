"""
微信客户端模块 - 使用 wxauto 操作微信
"""
import time
from typing import Callable, Optional, List
from threading import Thread, Event
import queue

try:
    from wxauto import WeChat
    WXAUTO_AVAILABLE = True
except ImportError:
    WXAUTO_AVAILABLE = False
    WeChat = None

from config.settings import settings
from utils.logger import logger


class MessageType:
    """消息类型"""
    TEXT = "text"
    IMAGE = "image"
    FILE = "file"
    OTHER = "other"


class WeChatClient:
    """微信客户端封装类"""

    def __init__(self):
        self.wx: Optional["WeChat"] = None
        self._running = False
        self._listen_thread: Optional[Thread] = None
        self._stop_event = Event()
        self._message_queue = queue.Queue()
        self._message_callbacks: List[Callable] = []

    def connect(self) -> bool:
        """连接微信"""
        if not WXAUTO_AVAILABLE:
            logger.error("wxauto 未安装，请运行: pip install wxauto")
            return False

        try:
            self.wx = WeChat()
            logger.info("微信连接成功")
            return True
        except Exception as e:
            logger.error(f"微信连接失败: {e}")
            logger.error("请确保微信 PC 客户端已打开")
            return False

    def get_contacts(self) -> List[dict]:
        """获取联系人列表"""
        if not self.wx:
            logger.warning("微信未连接")
            return []

        try:
            # wxauto 获取最近聊天的联系人
            contacts = []
            session_list = self.wx.GetSessionList()  # 获取会话列表

            for session in session_list:
                # 解析会话信息
                # session 通常包含: 聊天对象、最后消息、时间等
                if hasattr(session, 'who'):
                    contacts.append({
                        "name": getattr(session, 'who', 'Unknown'),
                        "type": "friend"  # wxauto 主要处理私聊
                    })

            logger.info(f"获取到 {len(contacts)} 个联系人")
            return contacts

        except Exception as e:
            logger.error(f"获取联系人失败: {e}")
            return []

    def get_recent_messages(self, contact_name: str, limit: int = 10) -> List[dict]:
        """获取与某人的最近消息"""
        if not self.wx:
            return []

        try:
            # 切换到对应聊天窗口
            self.wx.ChatWith(contact_name)

            messages = []
            # wxauto 获取消息的方式
            # 注意：具体实现取决于 wxauto 版本
            msg_list = self.wx.GetListenMessage()  # 获取监听到的消息

            for msg in msg_list[-limit:]:
                messages.append({
                    "content": msg.get("content", ""),
                    "type": msg.get("type", "text"),
                    "sender": msg.get("sender", ""),
                    "time": msg.get("time", ""),
                })

            return messages

        except Exception as e:
            logger.error(f"获取消息失败: {e}")
            return []

    def send_message(self, contact_name: str, content: str) -> bool:
        """发送消息"""
        if not self.wx:
            logger.warning("微信未连接")
            return False

        try:
            # 切换到对应聊天窗口
            self.wx.ChatWith(contact_name)

            # 发送消息
            self.wx.SendMsg(content)
            logger.info(f"发送消息给 {contact_name}: {content[:50]}...")
            return True

        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            return False

    def add_message_callback(self, callback: Callable):
        """添加消息回调函数"""
        self._message_callbacks.append(callback)

    def start_listening(self):
        """开始监听消息"""
        if self._running:
            logger.warning("消息监听已在运行")
            return

        if not self.wx:
            if not self.connect():
                return

        self._running = True
        self._stop_event.clear()
        self._listen_thread = Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        logger.info("开始监听微信消息")

    def stop_listening(self):
        """停止监听消息"""
        if not self._running:
            return

        self._running = False
        self._stop_event.set()
        if self._listen_thread:
            self._listen_thread.join(timeout=5)
        logger.info("停止监听微信消息")

    def _listen_loop(self):
        """消息监听循环"""
        last_msgs = set()

        while self._running and not self._stop_event.is_set():
            try:
                # 获取新消息
                # wxauto 的消息监听方式
                msgs = self.wx.GetListenMessage()  # 返回消息列表

                for msg in msgs:
                    # 生成消息唯一标识
                    msg_id = f"{msg.get('who', '')}{msg.get('content', '')}{msg.get('time', '')}"

                    # 只处理新消息
                    if msg_id not in last_msgs:
                        last_msgs.add(msg_id)

                        # 限制集合大小
                        if len(last_msgs) > 1000:
                            last_msgs = set(list(last_msgs)[-500:])

                        # 处理消息
                        self._process_message(msg)

                # 休眠避免 CPU 占用过高
                time.sleep(0.5)

            except Exception as e:
                logger.error(f"监听消息异常: {e}")
                time.sleep(2)

    def _process_message(self, msg: dict):
        """处理收到的消息"""
        try:
            # 解析消息
            contact_name = msg.get("who", "")
            content = msg.get("content", "")
            msg_type = msg.get("type", "text")
            sender = msg.get("sender", "")  # 消息发送者

            # 过滤系统消息和自己发送的消息
            if not contact_name or not content:
                return

            # 只处理私聊消息（非群聊）
            # wxauto 中群聊通常会有特殊标识
            if "chatroom" in msg_type.lower():
                return

            # 只处理对方发送的消息
            if sender == "Self" or sender == "自己":
                return

            logger.info(f"收到来自 {contact_name} 的消息: {content[:50]}...")

            # 调用回调函数
            message_data = {
                "contact_name": contact_name,
                "content": content,
                "type": msg_type,
                "sender": sender,
                "time": msg.get("time", ""),
            }

            for callback in self._message_callbacks:
                try:
                    callback(message_data)
                except Exception as e:
                    logger.error(f"消息回调异常: {e}")

        except Exception as e:
            logger.error(f"处理消息异常: {e}")

    def get_current_chat(self) -> Optional[str]:
        """获取当前聊天窗口"""
        if not self.wx:
            return None
        try:
            return self.wx.GetCurrentChat()
        except:
            return None

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self.wx is not None

    def reconnect(self) -> bool:
        """重新连接"""
        self.stop_listening()
        self.wx = None
        return self.connect()


# 全局微信客户端实例
wechat_client = WeChatClient()
