"""Gemini Live API service for bidirectional audio streaming.

This service replaces the traditional STT → LLM → TTS pipeline with Google's
Gemini 2.0 Multimodal Live API, which handles audio-to-audio streaming directly.
"""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Self

from google import genai
from google.genai import types

from joinly.types import AudioChunk, AudioFormat, TranscriptSegment

logger = logging.getLogger(__name__)


class GeminiLiveService:
    """Unified service for Gemini Live API bidirectional audio streaming."""

    def __init__(
        self,
        *,
        model: str = "gemini-2.0-flash-exp",
        api_key: str | None = None,
        system_instruction: str | None = None,
        temperature: float = 0.9,
        audio_format: AudioFormat | None = None,
    ) -> None:
        """Initialize the Gemini Live service.

        Args:
            model: Gemini model to use (default: "gemini-2.0-flash-exp")
            api_key: Google API key (reads from GOOGLE_API_KEY env if None)
            system_instruction: System prompt for Kurt's personality
            temperature: Model temperature for response generation
            audio_format: Audio format for input/output (16kHz PCM16 recommended)
        """
        self.model = model
        self.system_instruction = system_instruction
        self.temperature = temperature
        self.audio_format = audio_format or AudioFormat(
            sample_rate=16000,
            byte_depth=2,  # 16-bit PCM
        )

        # Initialize client
        self.client = genai.Client(api_key=api_key)

        self._session: genai.live.AsyncSession | None = None
        self._running = False
        self._audio_input_task: asyncio.Task | None = None
        self._audio_output_task: asyncio.Task | None = None
        self._transcript_task: asyncio.Task | None = None

        # Queues for communication
        self._input_audio_queue: asyncio.Queue[AudioChunk | None] = asyncio.Queue()
        self._output_audio_queue: asyncio.Queue[bytes] = asyncio.Queue()
        self._transcript_queue: asyncio.Queue[TranscriptSegment] = asyncio.Queue()

        # Track current speaker and timing
        self._current_time_s: float = 0.0
        self._participant_name: str | None = None

    @property
    def audio_output_queue(self) -> asyncio.Queue[bytes]:
        """Queue for output audio chunks to be played through virtual microphone."""
        return self._output_audio_queue

    @property
    def transcript_queue(self) -> asyncio.Queue[TranscriptSegment]:
        """Queue for transcript segments from Gemini."""
        return self._transcript_queue

    async def __aenter__(self) -> Self:
        """Enter async context."""
        return self

    async def __aexit__(self, *exc: object) -> None:  # noqa: ARG002
        """Exit async context and cleanup."""
        await self.stop()

    async def start(
        self,
        audio_input_iterator: AsyncIterator[AudioChunk],
        participant_name: str | None = None,
    ) -> None:
        """Start the Gemini Live session with bidirectional audio streaming.

        Args:
            audio_input_iterator: Async iterator of audio chunks from meeting
            participant_name: Name of the AI participant (for transcript)
        """
        if self._running:
            msg = "Gemini Live service already running"
            raise RuntimeError(msg)

        self._running = True
        self._participant_name = participant_name

        logger.info("Starting Gemini Live session with model %s", self.model)

        # Configure the Live API connection
        config = types.LiveConnectConfig(
            # Enable server-side voice activity detection for automatic turn-taking
            realtime_input_config=types.RealtimeInputConfig(
                automatic_activity_detection=types.AutomaticActivityDetection()
            ),
            # Speech configuration for voice
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name="Puck"  # Authoritative voice for Kurt
                    )
                )
            ),
        )

        logger.info("Gemini Live config: VAD enabled, voice=Puck, temp=%.1f", self.temperature)

        if self.system_instruction:
            config.system_instruction = types.Content(
                parts=[types.Part(text=self.system_instruction)]
            )

        try:
            # Connect to Live API
            async with self.client.aio.live.connect(
                model=self.model, config=config
            ) as session:
                self._session = session
                logger.info("Gemini Live session connected successfully")

                # Start background workers
                await asyncio.gather(
                    self._audio_input_worker(audio_input_iterator),
                    self._audio_output_worker(),
                    return_exceptions=True,
                )

        except Exception:
            logger.exception("Error in Gemini Live session")
            await self.stop()
            raise
        finally:
            self._session = None

    async def stop(self) -> None:
        """Stop the Gemini Live session and cleanup resources."""
        if not self._running:
            return

        logger.info("Stopping Gemini Live session")
        self._running = False

        # Clear queues
        while not self._input_audio_queue.empty():
            try:
                self._input_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._output_audio_queue.empty():
            try:
                self._output_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        while not self._transcript_queue.empty():
            try:
                self._transcript_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        logger.info("Gemini Live session stopped")

    async def send_text(self, text: str) -> None:
        """Send a text prompt to Gemini (triggers audio response).

        Args:
            text: Text to send to Gemini
        """
        if not self._session:
            msg = "Gemini Live session not started"
            raise RuntimeError(msg)

        logger.debug("Sending text to Gemini: %s", text)
        # Send text using send_realtime_input
        await self._session.send_realtime_input(text=text)

    async def _audio_input_worker(
        self, audio_iterator: AsyncIterator[AudioChunk]
    ) -> None:
        """Worker task to stream audio input to Gemini.

        Args:
            audio_iterator: Async iterator of audio chunks from meeting
        """
        if not self._session:
            return

        logger.info("Audio input worker started, waiting for audio chunks...")

        try:
            async for chunk in audio_iterator:
                if not self._running:
                    break

                # Update current time
                self._current_time_s = chunk.time_ns / 1e9

                # Send audio to Gemini using send_realtime_input
                # Specify full audio format: 16kHz, 16-bit signed integer PCM
                try:
                    await self._session.send_realtime_input(
                        audio=types.Blob(
                            mime_type="audio/pcm;rate=16000",
                            data=chunk.data
                        )
                    )
                except Exception as e:
                    if "ConnectionClosed" in str(type(e).__name__):
                        logger.warning("WebSocket connection closed, stopping audio input")
                        break
                    raise

                # Log every 100 chunks to track audio flow
                if hasattr(self, '_audio_chunk_count'):
                    self._audio_chunk_count += 1
                    if self._audio_chunk_count % 100 == 0:
                        logger.info("Sent %d audio chunks to Gemini", self._audio_chunk_count)
                else:
                    self._audio_chunk_count = 1
                    logger.info("Started streaming audio to Gemini Live")

        except asyncio.CancelledError:
            logger.debug("Audio input worker cancelled")
        except Exception:
            logger.exception("Error in audio input worker")

    async def _audio_output_worker(self) -> None:
        """Worker task to receive audio output from Gemini and queue for playback."""
        if not self._session:
            return

        logger.info("Audio output worker started, listening for Gemini responses...")
        response_count = 0

        try:
            # Listen for responses from Gemini
            async for response in self._session.receive():
                response_count += 1
                if response_count % 10 == 1:  # Log every 10th response
                    logger.info("Received response #%d from Gemini", response_count)

                if not self._running:
                    break

                # Log the response structure for debugging
                if response.server_content:
                    logger.info("Server content received: %s", type(response.server_content).__name__)

                    # Handle audio output
                    if response.server_content.model_turn:
                        logger.info("Model turn detected with %d parts", len(response.server_content.model_turn.parts))
                        for part in response.server_content.model_turn.parts:
                            # Audio data
                            if hasattr(part, "inline_data") and part.inline_data:
                                audio_data = part.inline_data.data
                                await self._output_audio_queue.put(audio_data)
                                logger.info(
                                    "Received audio chunk from Gemini: %d bytes",
                                    len(audio_data),
                                )

                            # Text transcript
                            if hasattr(part, "text") and part.text:
                                segment = TranscriptSegment(
                                    text=part.text,
                                    start=self._current_time_s,
                                    end=self._current_time_s + 1.0,  # Estimate
                                    speaker=self._participant_name,
                                )
                                await self._transcript_queue.put(segment)
                                logger.info(
                                    "%s: %s",
                                    self._participant_name or "Kurt",
                                    part.text,
                                )
                else:
                    # Log other types of responses
                    logger.info("Received non-content response: %s", type(response).__name__)

        except asyncio.CancelledError:
            logger.debug("Audio output worker cancelled")
        except Exception:
            logger.exception("Error in audio output worker")
