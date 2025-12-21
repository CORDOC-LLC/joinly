import asyncio
import logging
import shutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _find_pactl() -> str:
    """Find pactl executable path.

    Returns:
        Path to pactl executable

    Raises:
        RuntimeError: If pactl is not found
    """
    # Try common locations
    paths = [
        "/usr/bin/pactl",  # Linux default
        "/opt/homebrew/bin/pactl",  # macOS Homebrew (Apple Silicon)
        "/usr/local/bin/pactl",  # macOS Homebrew (Intel)
    ]

    for path in paths:
        if Path(path).exists():
            return path

    # Try which command
    pactl_path = shutil.which("pactl")
    if pactl_path:
        return pactl_path

    msg = (
        "pactl (PulseAudio control utility) not found. Please install PulseAudio:\n"
        "  macOS: brew install pulseaudio\n"
        "  Ubuntu/Debian: sudo apt-get install pulseaudio\n"
        "  Fedora/RHEL: sudo dnf install pulseaudio"
    )
    raise RuntimeError(msg)


class PulseModuleManager:
    """A class to load and unload pulse modules via pactl."""

    async def _load_module(
        self, *cmd_args: str, env: dict[str, str] | None = None
    ) -> int:
        """Load a pulse module using pactl.

        Args:
            cmd_args: Arguments to pass to the pactl command.
            env: Optional environment variables to set for the command.

        Returns:
            The module id.
        """
        pactl_path = _find_pactl()
        cmd = [pactl_path, "load-module", *cmd_args]
        load_sink_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await load_sink_proc.communicate()
        if load_sink_proc.returncode != 0:
            msg = f"Failed to load pulse module: {stderr.decode()}"
            logger.error(msg)
            raise RuntimeError(msg)

        return int(stdout.decode().strip())

    async def _unload_module(
        self, module_id: int, env: dict[str, str] | None = None
    ) -> None:
        """Unload a pulse module using pactl.

        Args:
            module_id: The ID of the module to unload.
            env: Optional environment variables to set for the command.

        Raises:
            RuntimeError: If the module unload fails.
        """
        pactl_path = _find_pactl()
        cmd = [pactl_path, "unload-module", str(module_id)]
        unload_sink_proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        stdout, stderr = await unload_sink_proc.communicate()
        if unload_sink_proc.returncode != 0:
            msg = f"Failed to unload pulse module: {stderr.decode()}"
            logger.error(msg)
            raise RuntimeError(msg)
