"""应用统一异常类型。"""


class ApplicationError(Exception):
    """所有可安全转换为客户端响应的应用异常基类。"""


class BusinessValidationError(ApplicationError, ValueError):
    """业务输入不满足领域约束。"""

    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("; ".join(errors))


class ResourceNotFoundError(ApplicationError):
    """请求的业务资源不存在。"""


class ResourceConflictError(ApplicationError):
    """资源当前状态与请求操作冲突。"""

    def __init__(
        self,
        message: str,
        *,
        code: str = "resource_conflict",
        context: dict | None = None,
    ) -> None:
        self.code = code
        self.context = context or {}
        super().__init__(message)


class ForbiddenDataError(BusinessValidationError):
    """输入包含禁止标签、身份信息或其他越界数据。"""
