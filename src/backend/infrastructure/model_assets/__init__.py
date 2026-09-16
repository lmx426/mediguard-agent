"""模型资产加载模块。

提供本地欺诈模型资产文件的 I/O 加载能力（pkl/csv/json），
不包含推理逻辑。LocalAssetsLoader 实现 ModelAssetsLoader 协议。
"""

from .local_loader import LocalAssetsLoader

__all__ = ["LocalAssetsLoader"]
