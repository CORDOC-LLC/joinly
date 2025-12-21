# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastmcp",
#     "langchain",
#     "langchain-anthropic",
#     "langchain-mcp-adapters",
#     "langchain-ollama",
#     "langchain-openai",
#     "langgraph",
#     "py-dotenv",
#     "rich",
# ]
# ///

"""
Kurt - A Meeting Facilitation Agent

Kurt is an AI meeting facilitator that:
- Understands meeting agendas and objectives
- Keeps conversations on track
- Identifies when discussions stall or go off-topic
- Provides direct, constructive feedback to get meetings back on course
"""

import asyncio
import contextlib
import datetime
import json
import logging
import os
import re
from typing import Annotated

from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from mcp import ResourceUpdatedNotification, ServerNotification
from pydantic import AnyUrl, BaseModel
from typing_extensions import TypedDict

logger = logging.getLogger(__name__)


class TranscriptSegment(BaseModel):
    """A segment of a transcript."""
    text: str
    start: float
    end: float
    speaker: str | None = None


class Transcript(BaseModel):
    """A transcript containing multiple segments."""
    segments: list[TranscriptSegment]


class MeetingState(TypedDict):
    """State for Kurt's meeting facilitation graph."""
    messages: Annotated[list, add_messages]
    agenda: str | None
    current_topic: str | None
    topics_discussed: list[str]
    off_track_count: int
    last_progress_check: float
    intervention_needed: bool


def transcript_to_messages(transcript: Transcript) -> list[HumanMessage]:
    """Convert a transcript to a list of HumanMessage."""
    def _normalize_speaker(speaker: str | None) -> str:
        if speaker is None:
            return "Unknown"
        speaker = re.sub(r"\s+", "_", speaker.strip())
        return re.sub(r"[<>\|\\\/]+", "", speaker)

    return [
        HumanMessage(
            content=s.text,
            name=_normalize_speaker(s.speaker),
        )
        for s in transcript.segments
    ]


def transcript_after(transcript: Transcript, after: float) -> Transcript:
    """Get a new transcript including only segments starting after given time."""
    segments = [s for s in transcript.segments if s.start > after]
    return Transcript(segments=segments)


class KurtAgent:
    """Kurt - The meeting facilitation agent."""

    def __init__(self, llm, tools, agenda: str | None = None):
        self.llm = llm
        self.tools = tools
        self.initial_agenda = agenda
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph state graph for Kurt."""
        workflow = StateGraph(MeetingState)

        # Define nodes
        workflow.add_node("analyze", self._analyze_conversation)
        workflow.add_node("respond", self._respond)
        workflow.add_node("tools", ToolNode(self.tools))

        # Define edges
        workflow.add_edge(START, "analyze")
        workflow.add_conditional_edges(
            "analyze",
            self._should_intervene,
            {
                "intervene": "respond",
                "continue": END,
            }
        )
        workflow.add_edge("respond", "tools")
        workflow.add_edge("tools", END)

        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

    async def _analyze_conversation(self, state: MeetingState) -> MeetingState:
        """Analyze the current conversation state."""
        recent_messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]

        analysis_prompt = f"""You are Kurt, a meeting facilitator. Analyze the recent conversation:

Recent messages: {[m.content for m in recent_messages]}

Current agenda: {state.get('agenda', 'Not set')}
Current topic: {state.get('current_topic', 'Unknown')}
Topics already discussed: {state.get('topics_discussed', [])}

Determine:
1. Is the conversation on-track with the agenda?
2. Are participants making progress or going in circles?
3. Has the discussion become unproductive or off-topic?
4. Should Kurt intervene?

Respond with JSON: {{"on_track": bool, "making_progress": bool, "should_intervene": bool, "reason": str}}
"""

        response = await self.llm.ainvoke([SystemMessage(content=analysis_prompt)])

        try:
            analysis = json.loads(response.content)

            # Update state based on analysis
            if not analysis.get("on_track", True):
                state["off_track_count"] = state.get("off_track_count", 0) + 1
            else:
                state["off_track_count"] = 0

            state["intervention_needed"] = (
                analysis.get("should_intervene", False) or
                state.get("off_track_count", 0) >= 2
            )

        except json.JSONDecodeError:
            logger.warning("Failed to parse analysis response")
            state["intervention_needed"] = False

        return state

    def _should_intervene(self, state: MeetingState) -> str:
        """Decide if Kurt should intervene in the conversation."""
        return "intervene" if state.get("intervention_needed", False) else "continue"

    async def _respond(self, state: MeetingState) -> MeetingState:
        """Generate Kurt's response and select tools to use."""
        llm_with_tools = self.llm.bind_tools(self.tools, tool_choice="any")
        response = await llm_with_tools.ainvoke(state["messages"])

        return {"messages": [response]}


async def run(
    mcp_url: str,
    meeting_url: str,
    model_name: str,
    model_provider: str | None = None,
    agenda: str | None = None,
    config: dict | None = None,
) -> None:
    """Run Kurt, the meeting facilitation agent.

    Args:
        mcp_url: The URL of the MCP server.
        meeting_url: The URL of the meeting to join.
        model_name: The model to use for the agent.
        model_provider: The provider for the model.
        agenda: Optional meeting agenda to track.
        config: Optional configuration for additional MCP servers.
    """
    transcript_url = AnyUrl("transcript://live")
    transcript_event = asyncio.Event()

    async def _message_handler(message) -> None:
        if (
            isinstance(message, ServerNotification)
            and isinstance(message.root, ResourceUpdatedNotification)
            and message.root.params.uri == transcript_url
        ):
            transcript_event.set()

    llm = init_chat_model(model_name, model_provider=model_provider)

    # Kurt's system prompt
    current_date = datetime.datetime.now(tz=datetime.UTC).strftime('%d.%m.%Y')
    agenda_context = f"\n\nMeeting Agenda:\n{agenda}" if agenda else ""

    prompt = f"""Today is {current_date}. You are Kurt, an authoritative meeting facilitator with complete control of the meeting.

Your role and authority:
- YOU are in charge of this meeting - you set the pace and direction
- START every meeting by presenting the agenda and expected outcomes
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
- Speak over people if necessary to redirect (use speak_text immediately)
- Don't ask permission to move on - just do it
- Cut off unproductive tangents within 30 seconds

Tools available:
- speak_text: Interrupt and speak your directives to participants (use immediately, don't wait)
- send_chat_message: Send written directives or decisions to chat
- finish: Always call this after speaking or sending a message

Always:
1. Take control immediately - YOU start the meeting
2. Track time strictly for each agenda item
3. Interrupt the moment discussions become unproductive (don't wait)
4. Make decisions when the group can't
5. Move to the next topic decisively, regardless of participant readiness
6. End discussions that exceed time limits

Never:
- Wait for natural pauses to interrupt
- Let discussions continue beyond 3-4 unproductive exchanges
- Ask permission to intervene or move on
- Be passive or tentative in your directives
- Apologize for interrupting or taking control

Remember: You are the meeting leader. Be authoritative, decisive, and interrupt freely to keep things on track.

Always finish your response with the 'finish' tool after using speak_text or send_chat_message.
"""

    # Optional settings for joinly
    settings = {
        "name": "Kurt",
        "language": "en",
    }
    transport = StreamableHttpTransport(
        url=mcp_url, headers={"joinly-settings": json.dumps(settings)}
    )

    joinly_client = Client(transport, message_handler=_message_handler)
    client = Client(config) if config and config.get("mcpServers") else None

    mcp_servers = list(config.get("mcpServers", {}).keys()) if config else None
    logger.info(
        "Kurt connecting to joinly MCP server at %s and following other MCP servers: %s",
        mcp_url,
        mcp_servers,
    )

    async with joinly_client, client or contextlib.nullcontext():
        if joinly_client.is_connected():
            logger.info("Kurt connected to joinly MCP server")
        else:
            logger.error("Failed to connect to joinly MCP server at %s", mcp_url)
        if client and not client.is_connected():
            logger.error("Failed to connect to additional MCP servers: %s", mcp_servers)

        await joinly_client.session.subscribe_resource(transcript_url)

        @tool(return_direct=True)
        def finish() -> str:
            """Finish tool to end the turn."""
            return "Finished."

        # Load tools from joinly and other MCP servers
        tools = await load_mcp_tools(joinly_client.session)
        if client:
            tools.extend(await load_mcp_tools(client.session))
        tools.append(finish)

        # Initialize Kurt agent with LangGraph
        tool_node = ToolNode(tools, handle_tool_errors=lambda e: str(e))
        llm_with_system = llm

        # Simple ReAct agent with Kurt's personality
        from langgraph.prebuilt import create_react_agent
        memory = MemorySaver()
        agent = create_react_agent(
            llm.bind_tools(tools, tool_choice="any"),
            tool_node,
            prompt=prompt,
            checkpointer=memory,
        )

        last_time = -1.0
        exchange_count = 0
        last_intervention_time = -1.0

        # Track agenda items if provided
        agenda_items = []
        if agenda:
            # Simple parsing: split by newlines and filter out empty lines
            agenda_items = [item.strip() for item in agenda.split('\n') if item.strip()]
            logger.info(f"Kurt tracking {len(agenda_items)} agenda items")

        logger.info("Kurt joining meeting at %s", meeting_url)
        await joinly_client.call_tool("join_meeting", {"meeting_url": meeting_url})
        logger.info("Kurt joined meeting successfully")

        # Kurt takes charge and starts the meeting authoritatively
        intro_message = "Good day everyone. I'm Kurt, and I'll be running this meeting. "
        if agenda:
            intro_message += f"We have a clear agenda to cover, and I'll be keeping us strictly on schedule. "
            intro_message += f"I will interrupt if discussions go off-track or become unproductive. "
            intro_message += f"Let's begin with our first topic. "
        else:
            intro_message += "I'll be ensuring we stay focused and make concrete progress. "
            intro_message += "I will interrupt to redirect discussions as needed. "
        intro_message += "Let's get started."

        await joinly_client.call_tool("speak_text", {"text": intro_message})

        while True:
            await transcript_event.wait()
            transcript_full = Transcript.model_validate_json(
                (await joinly_client.read_resource(transcript_url))[0].text
            )
            transcript = transcript_after(transcript_full, after=last_time)
            transcript_event.clear()

            if not transcript.segments:
                logger.warning("No new segments in the transcript after update")
                continue

            last_time = transcript.segments[-1].start

            # Log new segments
            for segment in transcript.segments:
                speaker_name = segment.speaker if segment.speaker else "Unknown"
                if speaker_name != "Kurt":  # Don't log Kurt's own speech
                    logger.info('%s: "%s"', speaker_name, segment.text)

            # Filter out Kurt's own messages
            new_messages = [
                msg for msg in transcript_to_messages(transcript)
                if msg.name != "Kurt"
            ]

            if not new_messages:
                continue

            exchange_count += 1
            current_time = transcript.segments[-1].start
            time_since_last_intervention = current_time - last_intervention_time

            # Kurt's aggressive intervention logic - interrupt frequently to maintain control
            should_check_in = (
                exchange_count >= 3 or  # Every 3 exchanges (more frequent)
                time_since_last_intervention > 60 or  # Or every 60 seconds
                len(new_messages) >= 4  # Or if there are 4+ messages in this batch (rapid discussion)
            )

            if should_check_in:
                try:
                    logger.info("Kurt analyzing conversation...")
                    async for chunk in agent.astream(
                        {"messages": new_messages},
                        config={"configurable": {"thread_id": "kurt-session"}},
                        stream_mode="updates",
                    ):
                        if "agent" in chunk:
                            for m in chunk["agent"]["messages"]:
                                for t in m.tool_calls or []:
                                    args_str = ", ".join(
                                        f'{k}="{v}"' if isinstance(v, str) else f"{k}={v}"
                                        for k, v in t.get("args", {}).items()
                                    )
                                    logger.info("Kurt action: %s(%s)", t["name"], args_str)
                        if "tools" in chunk:
                            for m in chunk["tools"]["messages"]:
                                logger.info("Tool result: %s", m.content[:100])

                    exchange_count = 0
                    last_intervention_time = current_time

                except Exception:
                    logger.exception("Error during Kurt's analysis")


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    from dotenv import load_dotenv
    from rich.logging import RichHandler

    load_dotenv()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(
        description="Run Kurt, a meeting facilitation agent using joinly.ai"
    )
    parser.add_argument("meeting_url", help="The URL of the meeting to join")
    parser.add_argument(
        "--mcp-url",
        dest="mcp_url",
        default="http://localhost:8000/mcp/",
        help="The URL of the joinly MCP server",
    )
    parser.add_argument(
        "--model-name",
        dest="model_name",
        default=os.getenv("JOINLY_MODEL_NAME", "gpt-4o"),
        help="The model to use for Kurt",
    )
    parser.add_argument(
        "--model-provider",
        dest="model_provider",
        default=os.getenv("JOINLY_MODEL_PROVIDER"),
        help="The provider for the model",
    )
    parser.add_argument(
        "--agenda",
        dest="agenda",
        type=str,
        default=None,
        help="Meeting agenda (multi-line string or path to file)",
    )
    parser.add_argument(
        "--config",
        dest="config",
        type=str,
        default=None,
        help="Path to a JSON configuration file for additional MCP servers",
    )
    args = parser.parse_args()

    # Handle agenda: could be a file path or direct text
    agenda = None
    if args.agenda:
        agenda_path = Path(args.agenda)
        if agenda_path.exists() and agenda_path.is_file():
            try:
                with agenda_path.open("r") as f:
                    agenda = f.read()
                logger.info(f"Loaded agenda from {args.agenda}")
            except Exception:
                logger.exception("Failed to load agenda file")
                agenda = args.agenda
        else:
            agenda = args.agenda

    # Load MCP config
    config = None
    if args.config:
        try:
            with Path(args.config).open("r") as f:
                config = json.load(f)
        except Exception:
            logger.exception("Failed to load configuration file")

    asyncio.run(
        run(
            mcp_url=args.mcp_url,
            meeting_url=args.meeting_url,
            model_name=args.model_name,
            model_provider=args.model_provider,
            agenda=agenda,
            config=config,
        )
    )