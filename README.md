# wechat_custody

微信 AI 托管服务 - 使用本地 AI 模型自动回复微信私聊消息

## 功能特性

- 自动回复微信私聊消息（基于白名单）
- 使用本地 Ollama 模型（qwen2.5）
- 支持从聊天记录学习你的说话风格
- 按联系人设置不同 AI 人设
- 网页管理后台（5001 端口）
- 系统托盘应用，支持暂停/恢复
- 对话上下文持久化保存

## 环境要求

- Windows 11
- Python 3.10+
- 微信 PC 客户端 4.12+
- Ollama + qwen2.5 模型

## 安装

```bash
# 安装依赖
pip install -r requirements.txt

# 确保 Ollama 正在运行
ollama serve

# 启动程序
python main.py
```

## 使用说明

1. 首次运行会自动创建数据库
2. 打开网页后台 `http://localhost:5001`
3. 添加白名单联系人
4. 上传聊天记录以学习说话风格（可选）
5. 开始自动回复

## 项目结构

```
wechat_custody/
├── main.py              # 程序入口
├── config/              # 配置模块
├── core/                # 核心功能
├── storage/             # 数据存储
├── web/                 # Web 后台
├── tray/                # 托盘应用
├── utils/               # 工具函数
└── data/                # 数据目录
```

## 配置

配置文件位于 `config/settings.json`：

```json
{
  "web": {
    "host": "127.0.0.1",
    "port": 5001
  },
  "ollama": {
    "base_url": "http://localhost:11434",
    "model": "qwen2.5"
  },
  "ai": {
    "max_context": 20,
    "temperature": 0.7
  }
}
```

## 许可证

MIT
