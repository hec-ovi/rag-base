"""Shared response models."""

from pydantic import BaseModel


class MessageResponse(BaseModel):
    message: str


class PaginationParams(BaseModel):
    offset: int = 0
    limit: int = 20
