"""The main Agent's language-model connectivity gate (fail-fast; silent degradation forbidden).

The architecture treats the LLM as the thinking and dispatching subject of the main loop
(evidence planning, hierarchical interpretation, plate commanding); continuing to run
with an unreachable model would only produce reports that "look complete but contain
zero actual LLM calls", so an unreachable model is escalated to a hard error.
probe_llm() only probes liveness and returns a diagnostic dict without raising (for UI
display); require_llm() raises LLMUnavailableError immediately when probing fails, with
error text that directly states the address, model, cause, and way out. The deterministic
baseline (ablation comparison) can explicitly opt out via RunnerOptions.require_llm=False.
"""

from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect

from oregenesis.llm.base import LLMBackendConfig  # Import the backend config type LLMBackendConfig


class LLMUnavailableError(RuntimeError):  # Model-unavailable error: inherits RuntimeError
    """Raised when no language model is connected: the main Agent refuses to work without a model."""  # Class docstring: the main Agent refuses to work without a model

    def __init__(self, message: str, *, api_base: str = "", model: str = "") -> None:  # Constructor: carries extra api_base/model diagnostics (keyword-only arguments)
        """Carries api_base / model to ease upstream diagnosis and messaging."""  # Method docstring: diagnostic fields for upstream messaging
        super().__init__(message)  # Call the parent constructor to store the error message
        self.api_base = api_base  # Store the service address as an instance attribute
        self.model = model  # Store the model name as an instance attribute


def probe_llm(config: LLMBackendConfig | None = None) -> dict:  # Liveness probe: uses the default config when omitted, never raises
    """Probe the LLM backend's liveness and return {available, api_base, model, error} (never raises)."""  # Function docstring: returns the available/api_base/model/error diagnostic dict
    from oregenesis.llm.openai_compatible import OpenAICompatibleBackend  # Local import to avoid a circular import

    cfg = config or LLMBackendConfig()  # Instantiate the default config when none is passed
    available, error = OpenAICompatibleBackend(cfg).check_availability()  # Build the backend and probe; unpack availability and the failure cause
    return {  # Assemble the diagnostic dict (spans lines)
        "available": bool(available),  # available key: availability as a bool
        "api_base": cfg.api_base,  # api_base key: the service address used for probing
        "model": cfg.model_name,  # model key: the model name used for probing
        "error": "" if available else (error or "探活失败"),  # error key: empty string on success, the cause on failure (fallback text)
    }


def require_llm(config: LLMBackendConfig | None = None) -> dict:  # Gate function: raises immediately when the model cannot be reached
    """Gate: raises immediately when the model cannot be reached, leaving "go connect the model" as the only way out for the user."""  # Function docstring: hands "go connect the model" to the user as the only way out
    info = probe_llm(config)  # Probe first to obtain the diagnostics
    if not info["available"]:  # Trigger the hard error when unavailable
        raise LLMUnavailableError(  # Raise the model-unavailable error: the message's first segment names the address and model (arguments span lines)
            f"未连接到语言模型（{info['api_base']} · {info['model']}）："  # Error message, second segment: the concrete failure cause
            f"{info['error']}。"  # Error message, third segment: explains why it refuses to run without a model
            "主控 Agent 需要模型负责思考、判断与指挥工具，不会在没有模型的情况下出图出报告。"  # Error message, fourth segment: points to the way out of starting the model service
            "请先启动模型服务（如 Ollama 并 pull/加载该模型）或在设置里改地址与模型名，再重新运行。",  # Keyword argument: attaches the service address for diagnostics
            api_base=info["api_base"],  # Keyword argument: attaches the service address for diagnostics
            model=info["model"],  # Keyword argument: attaches the model name for diagnostics
        )
    return info  # Return the diagnostic dict when available
