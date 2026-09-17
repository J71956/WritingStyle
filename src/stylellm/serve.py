"""Supervise a local `llama-server` process for the test campaign.

The interactive CLI talks to a server the operator started by hand. The campaign
cannot: it sweeps several models, and each one needs its own server because
llama-server holds a single GGUF. This module gives the harness the same
load-once-per-model-slice shape it already had with `HFBackend` — start a server
for the slice, run the slice's cells against it, shut it down before the next
model's weights are loaded.

Inference still runs entirely on this machine; the server is a local process
bound to localhost, not a hosted API.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path


class LlamaServerError(RuntimeError):
    pass


def find_binary(explicit: str | None = None) -> str:
    """Locate the llama-server executable, preferring an explicit config path."""
    if explicit:
        if not Path(explicit).exists():
            raise LlamaServerError(f"llama-server not found at {explicit!r}")
        return explicit
    found = shutil.which("llama-server")
    if not found:
        raise LlamaServerError(
            "llama-server is not on PATH. Install llama.cpp and either add its "
            "bin directory to PATH or set generate.llamacpp_bin in the config."
        )
    return found


class LlamaServer:
    """Start a llama-server on a GGUF and stop it again, as a context manager.

    `hf_spec` is llama.cpp's `-hf` form, `<repo>:<quant>` (e.g.
    `unsloth/Qwen3.5-9B-GGUF:Q4_K_M`), which resolves through the same Hugging
    Face cache the rest of the pipeline uses — so a pre-downloaded GGUF needs no
    network, and `HF_HUB_OFFLINE=1` keeps that guarantee.
    """

    def __init__(
        self,
        hf_spec: str,
        port: int = 8080,
        n_gpu_layers: int = 99,
        ctx_size: int = 8192,
        binary: str | None = None,
        extra_args: list[str] | None = None,
        startup_timeout: int = 600,
    ):
        self.hf_spec = hf_spec
        self.port = port
        self.host = f"http://localhost:{port}"
        self.n_gpu_layers = n_gpu_layers
        self.ctx_size = ctx_size
        self.binary = find_binary(binary)
        self.extra_args = extra_args or []
        self.startup_timeout = startup_timeout
        self.proc: subprocess.Popen | None = None

    def command(self) -> list[str]:
        return [
            self.binary,
            "-hf", self.hf_spec,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx_size),
            # These GGUF repos ship a vision projector. Loading it would spend
            # VRAM on a modality this pipeline never uses.
            "--no-mmproj",
        ] + self.extra_args

    def start(self) -> "LlamaServer":
        if self._healthy():
            raise LlamaServerError(
                f"Something is already listening on {self.host}. Stop it first — "
                f"otherwise this slice would silently run against the wrong model."
            )
        self.proc = subprocess.Popen(
            self.command(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        self._wait_until_healthy()
        return self

    def _healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.host}/health", timeout=2) as resp:
                return json.loads(resp.read().decode("utf-8")).get("status") == "ok"
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False

    def _wait_until_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.proc and self.proc.poll() is not None:
                out = self.proc.stdout.read() if self.proc.stdout else ""
                raise LlamaServerError(
                    f"llama-server exited with code {self.proc.returncode} while "
                    f"loading {self.hf_spec!r}:\n{out[-2000:]}"
                )
            if self._healthy():
                return
            time.sleep(1.0)
        self.stop()
        raise LlamaServerError(
            f"llama-server did not come up within {self.startup_timeout}s for "
            f"{self.hf_spec!r}. A cold run downloads the GGUF first — pre-download "
            f"it, or raise the timeout."
        )

    def stop(self) -> None:
        """Terminate the server and wait for the GPU memory to actually be freed."""
        if not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=30)
        self.proc = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()
