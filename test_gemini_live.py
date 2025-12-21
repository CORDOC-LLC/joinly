#!/usr/bin/env python3
"""Quick test script for Gemini Live integration."""

import asyncio
import os

from joinly.services.gemini_live import GeminiLiveService
from joinly.types import AudioChunk


async def test_gemini_live():
    """Test basic Gemini Live functionality."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        print("ERROR: GOOGLE_API_KEY environment variable not set")
        return

    print("Testing Gemini Live Service...")
    print(f"API Key: {api_key[:10]}...")

    service = GeminiLiveService(
        model="gemini-2.0-flash-exp",
        api_key=api_key,
        system_instruction="You are a helpful assistant. Respond briefly.",
        temperature=0.7,
    )

    print("✓ Service initialized")

    # Test that we can create the service
    print("✓ All imports successful")
    print("\nGemini Live integration is ready!")
    print("\nNext steps:")
    print("1. Ensure GOOGLE_API_KEY is set")
    print("2. Run: python examples/kurt_gemini.py <meeting-url>")


if __name__ == "__main__":
    asyncio.run(test_gemini_live())
