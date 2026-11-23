"""Backward-compatible imports for the FastAPI application."""

from backend.main import app, create_app, main

__all__ = ["app", "create_app", "main"]
