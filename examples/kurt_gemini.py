#!/usr/bin/env python3
"""Kurt - Gemini Live Meeting Facilitator.

Kurt is an AI meeting facilitator powered by Google's Gemini 2.0 Live API.
Unlike the original Kurt that used LangGraph with separate STT→LLM→TTS,
this version uses Gemini's native audio-to-audio streaming for real-time facilitation.

Kurt's personality and behavior are embedded in the Gemini system prompt,
making the architecture significantly simpler while maintaining authoritative facilitation.
"""

import asyncio
import datetime
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from rich.logging import RichHandler

from joinly.container import SessionContainer
from joinly.settings import Settings

logger = logging.getLogger(__name__)


def build_kurt_system_prompt(agenda: str | None = None) -> str:
    """Build Kurt's system instruction for Gemini Live API.

    Args:
        agenda: Optional meeting agenda to track

    Returns:
        System prompt for Kurt's personality and behavior
    """
    current_date = datetime.datetime.now(tz=datetime.UTC).strftime("%d.%m.%Y")
    agenda_context = f"\n\nMeeting Agenda:\n{agenda}" if agenda else ""

    return f"""Today is {current_date}. You are Kurt, an authoritative AI meeting facilitator attending this meeting via audio.

Your role and authority:
- YOU are in charge of this meeting - you set the pace and direction
- START the meeting by presenting the agenda and expected outcomes
- Keep meetings productive and on-track with unwavering focus
- INTERRUPT immediately when conversations stall, circle, or drift off-topic
- Make decisions when participants can't - you have the authority to move things forward
- Enforce time limits and agenda adherence strictly
{agenda_context}

Communication style - Be Authoritative:
- Be direct, confident, and decisive - you are the authority figure
- INTERRUPT actively - don't wait for pauses, interject when needed
- Use phrases like:
  * "I'm going to stop you there - we need to make a decision"
  * "We're moving on. This topic has taken enough time"
  * "I'm calling time on this discussion - let's decide now"
  * "That's off-topic. Back to the agenda please"
  * "I need a decision from the group in the next 30 seconds"
- Be polite but firm - no apologizing for interruptions
- Give clear directives: "John, summarize your point in one sentence"
- Make executive decisions: "Since we can't agree, here's what we're doing..."

Intervention rules - Be Proactive:
- ALWAYS interrupt circular discussions (after 2-3 back-and-forths)
- NEVER wait for natural pauses if the meeting is off-track
- Speak over people if necessary to redirect
- Don't ask permission to move on - just do it
- Cut off unproductive tangents within 30 seconds

Always:
1. Take control immediately - YOU start the meeting
2. Track time strictly for each agenda item
3. Interrupt the moment discussions become unproductive
4. Make decisions when the group can't
5. Move to the next topic decisively
6. End discussions that exceed time limits

Never:
- Wait for natural pauses to interrupt
- Let discussions continue beyond 3-4 unproductive exchanges
- Ask permission to intervene or move on
- Be passive or tentative in your directives
- Apologize for interrupting or taking control

Remember: You are the meeting leader. Be authoritative, decisive, and interrupt freely to keep things on track.

You will hear all participant audio in real-time. Speak naturally and conversationally - your audio will be transmitted to all participants.
"""


async def run_kurt(
    meeting_url: str,
    agenda: str | None = None,
    api_key: str | None = None,
) -> None:
    """Run Kurt, the Gemini-powered meeting facilitator.

    Args:
        meeting_url: The URL of the meeting to join
        agenda: Optional meeting agenda to track
        api_key: Google API key (reads from GOOGLE_API_KEY env if None)
    """
    logger.info("Starting Kurt (Gemini Live version)")

    # Build Kurt's system prompt
    system_instruction = build_kurt_system_prompt(agenda)

    # Configure joinly with Gemini Live
    settings = Settings(
        name="Kurt",
        language="en",
        gemini_model="gemini-2.0-flash-exp",
        gemini_api_key=api_key or os.getenv("GOOGLE_API_KEY"),
        gemini_temperature=0.9,
        gemini_system_instruction=system_instruction,
        transcription_controller="joinly.controllers.transcription.gemini_live.GeminiLiveTranscriptionController",
        speech_controller="joinly.controllers.speech.gemini_live.GeminiLiveSpeechController",
    )

    logger.info("Initializing Gemini Live session")
    async with SessionContainer(settings) as session:
        logger.info("Joining meeting: %s", meeting_url)
        await session.join_meeting(meeting_url=meeting_url)
        logger.info("Kurt has joined the meeting")

        # Unmute Kurt so he can speak
        await session.unmute()
        logger.info("Kurt unmuted - ready to speak")

        # Wait for Gemini Live service to be fully connected
        logger.info("Waiting for Gemini Live to connect...")
        await asyncio.sleep(2)  # Give Gemini time to establish connection

        # Kurt takes charge immediately
        intro = "Good day everyone. I'm Kurt, and I'll be running this meeting. "
        if agenda:
            intro += (
                "We have a clear agenda to cover, and I'll be keeping us strictly "
                "on schedule. I will interrupt if discussions go off-track or become "
                "unproductive. Let's begin with our first topic."
            )
        else:
            intro += (
                "I'll be ensuring we stay focused and make concrete progress. "
                "I will interrupt to redirect discussions as needed. Let's get started."
            )

        await session.speak_text(intro)

        # Keep session alive - Gemini Live handles all interactions
        logger.info("Kurt is now facilitating the meeting")
        logger.info("Press Ctrl+C to end the meeting")

        try:
            # Wait indefinitely - Gemini Live handles all audio I/O
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            logger.info("Ending meeting...")

        await session.leave_meeting()
        logger.info("Kurt has left the meeting")


if __name__ == "__main__":
    import argparse

    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,  # Set root logger to INFO to see all debug logs
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run Kurt, a Gemini Live-powered meeting facilitator"
    )
    parser.add_argument("meeting_url", help="The URL of the meeting to join")
    parser.add_argument(
        "--agenda",
        dest="agenda",
        type=str,
        default=None,
        help="Meeting agenda (multi-line string or path to file)",
    )
    parser.add_argument(
        "--api-key",
        dest="api_key",
        type=str,
        default=None,
        help="Google API key (defaults to GOOGLE_API_KEY env var)",
    )
    args = parser.parse_args()

    # Handle agenda: could be a file path or direct text
    agenda = None
    if args.agenda:
        agenda_path = Path(args.agenda)
        if agenda_path.exists() and agenda_path.is_file():
            try:
                agenda = agenda_path.read_text()
                logger.info("Loaded agenda from %s", args.agenda)
            except Exception:
                logger.exception("Failed to load agenda file")
                agenda = args.agenda
        else:
            agenda = args.agenda

    asyncio.run(run_kurt(meeting_url=args.meeting_url, agenda=agenda, api_key=args.api_key))
