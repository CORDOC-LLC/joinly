from contextvars import ContextVar, Token
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from joinly.core import (
    MeetingProvider,
    SpeechController,
    TranscriptionController,
)


class Settings(BaseSettings):
    """Settings for the meeting agent."""

    name: str = Field(default="Kurt")
    language: str = Field(default="en")

    # Gemini Live API settings
    gemini_model: str = Field(default="gemini-2.0-flash-exp")
    gemini_api_key: str | None = Field(default=None)
    gemini_temperature: float = Field(default=0.9)
    gemini_system_instruction: str | None = Field(default=None)

    meeting_provider: str | type[MeetingProvider] = Field(default="browser")
    transcription_controller: str | type[TranscriptionController] = Field(
        default="gemini_live"
    )
    speech_controller: str | type[SpeechController] = Field(default="gemini_live")

    meeting_provider_args: dict[str, Any] = Field(default_factory=dict)
    transcription_controller_args: dict[str, Any] = Field(default_factory=dict)
    speech_controller_args: dict[str, Any] = Field(default_factory=dict)
    gemini_service_args: dict[str, Any] = Field(default_factory=dict)

    model_config = SettingsConfigDict(
        env_prefix="JOINLY_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )


_current_settings: ContextVar[Settings] = ContextVar("settings", default=Settings())  # noqa: B039


def get_settings() -> Settings:
    """Get the current settings.

    Returns:
        Settings: The current settings.
    """
    return _current_settings.get()


def set_settings(settings: Settings) -> Token[Settings]:
    """Set the current settings.

    Args:
        settings (Settings): The settings to set.

    Returns:
        Token[Settings]: A token that can be used to reset the settings.
    """
    return _current_settings.set(settings)


def reset_settings(token: Token[Settings]) -> None:
    """Reset the current settings to the previous value.

    Args:
        token (Token[Settings]): The token returned by `set_settings`.
    """
    _current_settings.reset(token)
