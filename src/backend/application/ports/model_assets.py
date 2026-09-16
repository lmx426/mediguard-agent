"""模型资产加载抽象接口。"""

from typing import Any, Protocol


class ModelAssetsLoader(Protocol):
    """从存储介质加载模型文件的只读接口。"""

    def load_model(self) -> Any: ...

    def load_scaler_stats(self) -> dict[str, Any]: ...

    def load_pca_matrix(self) -> Any: ...
