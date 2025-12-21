"""Gemini Live speech controller.

This controller manages speech output using Gemini Live API, which generates
audio responses directly without needing a separate TTS service.
"""

import asyncio
import logging
from typing import Self

from joinly.core import AudioWriter, SpeechController
from joinly.services.gemini_live import GeminiLiveService
from joinly.settings import get_settings
from joinly.types import SpeakerRole, Transcript, TranscriptSegment
from joinly.utils.audio import calculate_audio_duration
from joinly.utils.clock import Clock
from joinly.utils.events import EventBus

logger = logging.getLogger(__name__)


class GeminiLiveSpeechController(SpeechController):
    """Speech controller using Gemini Live API."""

    writer: AudioWriter
    no_speech_event: asyncio.Event

    def __init__(
        self,
        *,
        gemini_service: GeminiLiveService,
    ) -> None:
        """Initialize the Gemini Live speech controller.

        Args:
            gemini_service: The Gemini Live service instance
        """
        self.gemini_service = gemini_service
        self._clock: Clock | None = None
        self._transcript: Transcript | None = None
        self._event_bus: EventBus | None = None
        self._audio_output_task: asyncio.Task | None = None
        self._lock = asyncio.Lock()

    async def __aenter__(self) -> Self:
        """Enter the speech controller context."""
        return self

    async def __aexit__(self, *exc: object) -> None:  # noqa: ARG002
        """Stop the speech controller and clean up resources."""
        await self.stop()

    async def start(
        self, clock: Clock, transcript: Transcript, event_bus: EventBus
    ) -> None:
        """Start the speech controller.

        Args:
            clock: The clock to use for timing
            transcript: The transcript to write spoken text to
            event_bus: The event bus to publish events
        """
        if self._clock is not None or self._transcript is not None:
            msg = "Speech controller already active"
            raise RuntimeError(msg)

        self._clock = clock
        self._transcript = transcript
        self._event_bus = event_bus

        # Start audio output consumer
        self._audio_output_task = asyncio.create_task(self._audio_output_worker())

        logger.info("Gemini Live speech controller started")

    async def stop(self) -> None:
        """Stop the speech controller."""
        if self._audio_output_task is not None:
            self._audio_output_task.cancel()
            try:
                await self._audio_output_task
            except asyncio.CancelledError:
                pass
            self._audio_output_task = None

        self._clock = None
        self._transcript = None
        self._event_bus = None

        logger.info("Gemini Live speech controller stopped")

    def _notify(self, event_type: str) -> None:
        """Notify event bus of an event.

        Args:
            event_type: The event type to publish
        """
        if self._event_bus is None:
            return

        self._event_bus.publish(event_type)

    async def speak_text(self, text: str) -> None:
        """Speak the given text using Gemini Live API.

        This sends a text prompt to Gemini, which will generate an audio response.

        Args:
            text: The text to be spoken (sent as prompt to Gemini)
        """
        async with self._lock:
            logger.info("Kurt speaking via Gemini: %s", text)
            await self.gemini_service.send_text(text)

    async def _audio_output_worker(self) -> None:
        """Worker task to consume audio output from Gemini and write to speaker."""
        if self._transcript is None or self._clock is None:
            msg = "Speech controller not active"
            raise RuntimeError(msg)

        logger.info("Speech controller audio output worker started, waiting for audio from Gemini...")
        chunk_count = 0

        try:
            while True:
                # Get audio from Gemini Live service
                audio_chunk = await self.gemini_service.audio_output_queue.get()

                if not audio_chunk:
                    logger.warning("Received empty audio chunk from queue")
                    continue

                chunk_count += 1

                # Write audio to virtual microphone
                start_time = self._clock.now_s
                await self.writer.write(audio_chunk)
                end_time = self._clock.now_s

                # Calculate duration for transcript timing
                audio_duration = calculate_audio_duration(
                    len(audio_chunk), self.gemini_service.audio_format
                )

                logger.info(
                    "Played audio chunk #%d: %d bytes, %.2fs duration",
                    chunk_count,
                    len(audio_chunk),
                    audio_duration,
                )

        except asyncio.CancelledError:
            logger.debug("Audio output worker cancelled")
        except Exception:
            logger.exception("Error in audio output worker")
