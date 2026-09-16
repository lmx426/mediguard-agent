"""认证 API 请求/响应模型。

登录/登出/会话相关的 DTO。
LoginInput 和 AuthResponse 是纯 API 层定义，
不进入领域层。
"""

from pydantic import BaseModel, Field

from ...domain.audit.review.entities import AuthenticatedUser


class LoginInput(BaseModel):
    """登录请求体。

    属性:
        username: 审核人员账号名，1~80 字符。
        password: 登录密码，1~128 字符。
    """

    username: str = Field(min_length=1, max_length=80)
    password: str = Field(min_length=1, max_length=128)


class AuthResponse(BaseModel):
    """登录成功响应。

    属性:
        user: 当前登录审核人员信息。
    """

    user: AuthenticatedUser
