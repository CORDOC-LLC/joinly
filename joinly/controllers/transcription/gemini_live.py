"""Gemini Live transcription controller.

This controller manages transcription using Gemini Live API, which handles
audio input and produces both audio responses and text transcripts.
"""

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
from typing import Self

from joinly.core import AudioReader, TranscriptionController
from joinly.services.gemini_live import GeminiLiveService
from joinly.types import AudioChunk, Transcript
from joinly.utils.clock import Clock
from joinly.utils.events import EventBus

logger = logging.getLogger(__name__)


class GeminiLiveTranscriptionController(TranscriptionController):
    """Transcription controller using Gemini Live API."""

    reader: AudioReader

    def __init__(
        self,
        *,
        gemini_service: GeminiLiveService,
        no_speech_event_delay: float = 0.4,
    ) -> None:
        """Initialize the Gemini Live transcription controller.

        Args:
            gemini_service: The Gemini Live service instance
            no_speech_event_delay: Delay before setting no-speech event
        """
        self.gemini_service = gemini_service
        self.no_speech_event_delay = no_speech_event_delay
        self._no_speech_event = asyncio.Event()
        self._audio_task: asyncio.Task | None = None
        self._transcript_task: asyncio.Task | None = None
        self._clock: Clock | None = None
        self._transcript: Transcript | None = None
        self._event_bus: EventBus | None = None

    @property
    def no_speech_event(self) -> asyncio.Event:
        """Get the event that is set when no speech is detected."""
        return self._no_speech_event

    async def __aenter__(self) -> Self:
        """Enter the transcription controller context."""
        return self

    async def __aexit__(self, *exc: object) -> None:  # noqa: ARG002
        """Clean up the transcription controller."""
        await self.stop()

    async def start(
        self, clock: Clock, transcript: Transcript, event_bus: EventBus
    ) -> None:
        """Start the transcription controller.

        Args:
            clock: The clock to use for timing
            transcript: The transcript to store segments
            event_bus: The event bus to publish events
        """
        if self._transcript_task is not None:
            msg = "Transcription controller already started"
            raise RuntimeError(msg)

        self._no_speech_event.set()
        self._clock = clock
        self._transcript = transcript
        self._event_bus = event_bus

        # Create audio input task
        self._audio_task = asyncio.create_task(
            self.gemini_service.start(
                audio_input_iterator=self._audio_chunk_iterator(),
                participant_name=None,  # Will be set by settings
            )
        )

        # Start transcript consumer
        self._transcript_task = asyncio.create_task(self._transcript_worker())

        logger.info("Gemini Live transcription controller started")

    async def stop(self) -> None:
        """Stop the transcription controller and clean up resources."""
        # Cancel audio task first
        if self._audio_task is not None:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task
            self._audio_task = None

        # Cancel transcript task
        if self._transcript_task is not None:
            self._transcript_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._transcript_task
            self._transcript_task = None

        await self.gemini_service.stop()

        self._no_speech_event.clear()
        self._clock = None
        self._transcript = None
        self._event_bus = None

        logger.info("Gemini Live transcription controller stopped")

    async def _audio_chunk_iterator(self) -> AsyncIterator[AudioChunk]:
        """Yield audio chunks from the reader."""
        logger.info("Audio chunk iterator started, reading from virtual speaker...")
        offset: int | None = None
        chunk_count = 0
        while True:
            chunk = await self.reader.read()
            if offset is None:
                offset = chunk.time_ns
                logger.info("First audio chunk received from virtual speaker")
            now_ns = chunk.time_ns - offset
            if self._clock is not None:
                self._clock.update(now_ns)

            # Log every 100 chunks to track flow
            chunk_count += 1
            if chunk_count % 100 == 0:
                logger.info("Processed %d audio chunks from virtual speaker", chunk_count)

            # Pass through chunks to Gemini
            yield AudioChunk(
                data=chunk.data,  # Gemini handles format conversion
                time_ns=now_ns,
                speaker=chunk.speaker,
            )

    async def _transcript_worker(self) -> None:
        """Process transcript segments from Gemini Live."""
        if self._transcript is None:
            msg = "Transcription controller not active"
            raise RuntimeError(msg)

        try:
            while True:
                # Get transcript segments from Gemini
                segment = await self.gemini_service.transcript_queue.get()

                # Add to transcript
                self._transcript.add_segment(segment)

                # Publish events
                if self._event_bus:
                    self._event_bus.publish("segment")

                logger.debug(
                    "%s: %s (%.2fs-%.2fs)",
                    segment.speaker or "User",
                    segment.text,
                    segment.start,
                    segment.end,
                )

                # Update no-speech event
                self._no_speech_event.clear()
                await asyncio.sleep(self.no_speech_event_delay)
                self._no_speech_event.set()

        except asyncio.CancelledError:
            logger.debug("Transcript worker cancelled")
        except Exception:
            logger.exception("Error in transcript worker")
