from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect
import os  # Import os: read process environment variables (level 1 of the three-level configuration)
from abc import ABC, abstractmethod  # Import ABC and abstractmethod: define the backend abstract interface
from dataclasses import dataclass  # Import dataclass: generate constructors for the response/config classes
from pathlib import Path  # Import Path: locate the project root's .env file
from typing import Any  # Import Any for type annotations
"""LLM backend abstraction layer: LLMBackendConfig / LLMResponse / LLMBackend(ABC) pure interfaces.
Depends only on the standard library (os/abc/dataclasses/pathlib) and imports no in-project
module — keeping the llm package free of circular imports; the in-package parsers/matchers/
suggesters and the agent/engine/reasoning/web_ui layers all program against this interface
(backend injected via dependency injection), so swapping backends only means adding an
LLMBackend subclass.
Ships a lightweight built-in .env parser (deliberately not depending on python-dotenv);
configuration resolves with a three-level fallback:
environment variable > project-root .env > hardcoded default (llm.routing.config_from_state
also falls back the same way).
"""

def _load_project_env() -> dict[str, str]:  # Private helper: parse the project root .env into a key-value table
    """Parse the project root .env into {KEY: VALUE}; a missing file / bad lines / decode failure all silently return an empty table."""  # Function docstring: missing file / bad lines / decode failure all silently return an empty table
    env_path = Path(__file__).resolve().parents[2] / ".env"  # Two levels above the llm/ directory is the project root
    values: dict[str, str] = {}  # Initialize an empty key-value table
    if not env_path.exists():  # A missing .env is not an error: hardcoded defaults cover everything
        return values  # Return the empty table when the .env file is missing
    try:  # Wrap reading/parsing; silently return an empty table on exception
        # Parse line by line: skip empty lines, "#" comment lines, and malformed lines without "="
        for line in env_path.read_text(encoding="utf-8").splitlines():  # Read the .env as UTF-8, split into lines, and iterate
            stripped = line.strip()  # Strip the line's surrounding whitespace
            if not stripped or stripped.startswith("#") or "=" not in stripped:  # Skip empty lines, "#" comment lines, and malformed lines without an equals sign
                continue  # Move to the next line
            key, _, value = stripped.partition("=")  # Split on the first "=" only; the value may contain more "="
            values[key.strip()] = value.strip().strip('"').strip("'")  # Trim the key's trailing spaces and strip paired quotes from both sides of the value
    except Exception:  # Config reading is "best-effort"; it must never fail program startup
        return {}  # Return an empty table on any exception too (best-effort config, never blocks startup)
    return values  # Return the parsed key-value table


_PROJECT_ENV = _load_project_env()  # Parse and cache once at import time to avoid repeated disk reads


def _llm_env(name: str, default: str) -> str:  # Private helper: fetch the LLM env config with three-level priority
    """Fetch environment configuration with three-level priority: real process environment variables > project .env > hardcoded defaults."""  # Function docstring: environment variable > project .env > hardcoded default
    if name in os.environ:  # Deployment can override via environment variables without code changes
        return os.environ[name]  # Level 1: return the process environment variable on hit
    if name in _PROJECT_ENV:  # Level 2: consult the cached project .env table
        return _PROJECT_ENV[name]  # Return the value from .env on hit
    return default  # Level 3: return the hardcoded default


@dataclass  # dataclass decorator: generates the response class constructor
class LLMResponse:  # The unified response wrapper type
    """Unified response wrapper for one chat call (upper layers recognize only this type, decoupled from the concrete HTTP protocol)."""  # Class docstring: upper layers recognize only this type, decoupled from the HTTP protocol
    content: str  # The model's output text
    model: str = ""  # The model actually served (after server-side routing it may differ from the requested value)
    tokens_used: int = 0  # Total tokens consumed by this call
    finish_reason: str = ""  # Finish reason (stop/length/...; Ollama maps to done_reason)
    raw: Any = None  # The server's raw response payload (for debugging and fallback parsing)
    # Token breakdown: aggregated by run telemetry (run_telemetry); old callers are unaffected
    prompt_tokens: int = 0  # ~= context input usage
    completion_tokens: int = 0  # ~= amount generated this call


@dataclass  # dataclass decorator: generates the config class constructor
class LLMBackendConfig:  # Backend connection and inference parameter configuration
    """Central configuration of backend connection and inference parameters (endpoint, key and model come from LLM_API_BASE / LLM_API_KEY / LLM_MODEL_NAME; there is no built-in endpoint)."""  # Class docstring: connection settings come from the environment or .env
    # The endpoint / API key / model name can all be overridden by same-named environment variables to switch to remote services such as vLLM/OpenAI
    api_base: str = _llm_env("LLM_API_BASE", "")  # api_base field: OpenAI-compatible endpoint from LLM_API_BASE (environment variable or .env); empty by default, so it must be configured
    api_key: str = _llm_env("LLM_API_KEY", "")  # Auth key: never hardcode it in code — put LLM_API_KEY=your-key in the project root .env (or set the same-named environment variable); do not echo it in logs/output
    model_name: str = _llm_env("LLM_MODEL_NAME", "")  # model_name field: model served by the endpoint, from LLM_MODEL_NAME; empty by default, so it must be configured
    # temperature=0.1 keeps structured/JSON output stable and reproducible; the HTTP timeout is set to the thousand-second scale (local inference is slow)
    temperature: float = 0.1  # temperature field: defaults to 0.1; low temperature keeps structured output stable
    # max_tokens is the "unspecified" fallback: the previous 40960 default reserved roughly 40k tokens of generation quota, and when the
    # KV cache ballooned and VRAM tightened it failed outright with HTTP 500; callers (planning 768 / reflection 1.5k / mechanism 8k-24k / report 20k) already pass explicit values
    max_tokens: int = 10000  # max_tokens field: fallback generation cap of 8192 when not explicitly specified
    timeout_seconds: float = 1000.0  # timeout_seconds field: HTTP timeout on the thousand-second scale, fitting slow local inference
    # Cloud rate-limit protection: min_interval_seconds is the minimum interval between consecutive cloud requests (active throttling,
    # 0 = off); max_retries caps the exponential backoff retries on 429/5xx/network jitter (respecting Retry-After)
    min_interval_seconds: float = float(_llm_env("LLM_MIN_INTERVAL", "2.0"))  # Defaults to 2 seconds ~= capped at 30 requests/minute, fitting free-tier quotas
    max_retries: int = int(_llm_env("LLM_MAX_RETRIES", "6"))  # Backoff sequence 2^n seconds + random jitter; raises with the original semantics only once retries cannot save the call
    # Ollama's compatible endpoint ignores options.num_ctx and silently truncates an overlong prompt in half, returning empty content,
    # so calls go through the native /api/chat; the default 32768 pairs with "on HTTP 500, halve the window and retry" (40960-scale windows fail to allocate when VRAM is tight)
    num_ctx: int = int(_llm_env("LLM_NUM_CTX", "32768"))  # num_ctx field: context window in tokens, default 32768
    # Turn off the "thinking mode" of reasoning models such as qwen to output the content directly: shorter latency, keeps thinking segments out of content so JSON parsing is not disturbed
    disable_thinking: bool = _llm_env("LLM_DISABLE_THINKING", "1") not in ("0", "false", "False")  # disable_thinking field: thinking mode is disabled unless the environment variable is 0/false/False
    # enabled is the master switch; the next two control whether the LLM joins rule matching / report generation (falling back to pure-rule/template paths when off)
    enabled: bool = True  # enabled field: the LLM master switch, on by default
    use_for_rule_matching: bool = True  # use_for_rule_matching field: whether the LLM joins rule matching
    use_for_report_generation: bool = True  # use_for_report_generation field: whether the LLM joins report generation


class LLMBackend(ABC):  # Abstract base class of all LLM backends
    """Abstract base class of all LLM backends; openai_compatible.OpenAICompatibleBackend is the only built-in implementation."""  # Class docstring: the only built-in implementation is OpenAICompatibleBackend

    def __init__(self, config: LLMBackendConfig) -> None:  # Constructor: bind the backend config object
        """Construction binds the config; subclasses read the endpoint/model/timeout/num_ctx from self.config on every request."""  # Method docstring: subclasses read endpoint/model/timeout/num_ctx from self.config per request
        self.config = config  # Store the config as an instance attribute

    @abstractmethod  # abstractmethod decorator: declares an abstract method subclasses must implement
    def chat(self, messages: list[dict[str, str]], system_prompt: str = "",  # Abstract chat: send one non-streaming chat request (signature spans lines)
             **kwargs) -> LLMResponse:  # kwargs holds per-call overrides; returns the unified response type
        """Send one (non-streaming) chat request.
        messages uses the OpenAI style [{"role": ..., "content": ...}]; kwargs holds per-call
        overrides (temperature/max_tokens/num_ctx/response_format/think, etc.).
        Implementations must translate network errors into ConnectionError and other errors
        into RuntimeError, so upper layers can degrade and fall back.
        """
        ...  # Abstract method body placeholder (Ellipsis, no implementation)

    @abstractmethod  # abstractmethod decorator: declares an abstract method
    def is_available(self) -> bool:  # Abstract method: probe backend connectivity, return a bool
        """Probe backend connectivity (e.g. GET /v1/models); when unavailable, upper layers fall back entirely to pure-rule/template paths."""  # Method docstring: upper layers fall back entirely to pure-rule/template paths when unavailable
        ...  # Abstract method body placeholder (Ellipsis)

    def format_message(self, role: str, content: str) -> dict[str, str]:  # Concrete method: pack role+content into an OpenAI-style message
        """Pack role+content into an OpenAI-style message dict (keeps the message format consistent across the project)."""  # Method docstring: keeps the message format consistent project-wide
        return {"role": role, "content": content}  # Return a message dict with the role and content keys
