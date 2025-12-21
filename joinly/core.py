import asyncio
from typing import Protocol

from joinly.types import (
    AudioChunk,
    AudioFormat,
    MeetingChatHistory,
    MeetingParticipant,
    Transcript,
    VideoSnapshot,
)
from joinly.utils.clock import Clock
from joinly.utils.events import EventBus


class AudioReader(Protocol):
    """Protocol for audio stream sources.

    Defines the interface for objects that provide audio data.

    Attributes:
        audio_format (AudioFormat): The format of the audio data being read.
    """

    audio_format: AudioFormat

    async def read(self) -> AudioChunk:
        """Read a chunk of audio data.

        Returns:
            AudioChunk: A chunk of audio data.
        """
        ...


class AudioWriter(Protocol):
    """Protocol for audio output destinations.

    Defines the interface for objects that consume audio data.

    Attributes:
        audio_format (AudioFormat): The format of the audio data being written.
        chunk_size (int): The smallest accepted size of an audio chunk in bytes.
    """

    audio_format: AudioFormat
    chunk_size: int

    async def write(self, data: bytes) -> None:
        """Write audio data to the sink.

        Args:
            data: Raw PCM audio data.
        """
        ...


class VideoReader(Protocol):
    """Protocol for video stream sources.

    Defines the interface for objects that provide video data.
    """

    async def snapshot(self) -> VideoSnapshot:
        """Capture a snapshot of the current video frame.

        Returns:
            VideoSnapshot: A snapshot of the current video frame.
        """
        ...


class MeetingProvider(Protocol):
    """Protocol defining the interface for meeting providers.

    A provider must implement audio input/output capabilities and meeting control
    functionality. This protocol ensures all providers have a consistent interface.
    """

    @property
    def audio_reader(self) -> AudioReader:
        """Get the audio reader for the provider.

        Returns:
            AudioReader: The audio input source.
        """
        ...

    @property
    def audio_writer(self) -> AudioWriter:
        """Get the audio writer for the provider.

        Returns:
            AudioWriter: The audio output destination.
        """
        ...

    @property
    def video_reader(self) -> VideoReader:
        """Get the video reader for the provider.

        Returns:
            VideoReader: The video input source.
        """
        ...

    async def join(
        self,
        url: str | None = None,
        name: str | None = None,
        passcode: str | None = None,
    ) -> None:
        """Join a meeting.

        Args:
            url: The meeting URL to join.
            name: The name to use in the meeting.
            passcode: The meeting password or passcode.
        """
        ...

    async def leave(self) -> None:
        """Leave the current meeting."""
        ...

    async def send_chat_message(self, message: str) -> None:
        """Send a chat message to the meeting.

        Args:
            message: The message to send.
        """
        ...

    async def get_chat_history(self) -> MeetingChatHistory:
        """Get the chat message history from the meeting.

        Returns:
            MeetingChatHistory: The chat history of the meeting.
        """
        ...

    async def get_participants(self) -> list[MeetingParticipant]:
        """Get the list of participants in the meeting.

        Returns:
            list[MeetingParticipant]: A list of participants in the meeting.
        """
        ...

    async def mute(self) -> None:
        """Mute yourself in the meeting."""
        ...

    async def unmute(self) -> None:
        """Unmute yourself in the meeting."""
        ...


class TranscriptionController(Protocol):
    """Protocol for controlling transcription processes.

    Defines the interface for starting and stopping transcriptions with Gemini Live.

    Attributes:
        reader (AudioReader): The audio reader to use for transcription.
    """

    reader: AudioReader

    @property
    def no_speech_event(self) -> asyncio.Event:
        """Get the event indicating no speech detected.

        Returns:
            asyncio.Event: An event that is set when no speech is detected.
        """
        ...

    async def start(
        self, clock: Clock, transcript: Transcript, event_bus: EventBus
    ) -> None:
        """Start the transcription process.

        Args:
            clock: The clock to use for timing.
            transcript: The transcript object to which the transcription will be added.
            event_bus: The event bus to publish events to.
        """
        ...

    async def stop(self) -> None:
        """Stop the transcription process."""
        ...


class SpeechController(Protocol):
    """Protocol for controlling speech output with Gemini Live.

    Defines the interface for speaking text using Gemini's audio generation.

    Attributes:
        writer (AudioWriter): The audio writer to use for output.
        no_speech_event (asyncio.Event): An event that is set when no speech is
            detected.
    """

    writer: AudioWriter
    no_speech_event: asyncio.Event

    async def start(
        self, clock: Clock, transcript: Transcript, event_bus: EventBus
    ) -> None:
        """Start the speech output process.

        Args:
            clock: The clock to use for timing.
            transcript: The transcript object to which the speech will be added.
            event_bus: The event bus to publish events to.
        """
        ...

    async def stop(self) -> None:
        """Stop the speech output process."""
        ...

    async def speak_text(self, text: str) -> None:
        """Speak the provided text.

        Args:
            text: The text to speak.

        Raises:
            SpeechInterruptedError: If the speech is interrupted before completion.
        """
        ...
