"""Persistence layer (in-memory reference implementation)."""

from .repository import Repository, get_repository

__all__ = ["Repository", "get_repository"]
