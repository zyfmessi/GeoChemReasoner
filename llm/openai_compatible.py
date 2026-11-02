"""OpenAI-compatible API backend — the only built-in implementation of the LLMBackend abstraction (HTTP sent directly via the standard-library urllib).

Automatically supports two kinds of services: OpenAI-compatible endpoints /v1/chat/completions
(vLLM, OpenAI, Ollama /v1) and the Ollama-native endpoint /api/chat (the compatibility layer
ignores options.num_ctx and cannot turn off thinking; an overlong prompt is silently truncated
in half and returns empty content, so once Ollama is detected the call switches to the native
path automatically). Also provides chat_stream() for true streaming output (rendered segment
by segment in web_ui) and availability probing (is_available / check_availability, letting
upper layers decide whether to fall back to the pure-rule path). Deliberately avoids the
openai SDK: zero third-party dependencies, so the paper's reproduction environment runs with
a single command.
"""

from __future__ import annotations  # Enable deferred annotation evaluation: type annotations have no runtime effect

import json  # Import json: serialize request bodies and parse responses
import random  # Import random: add random jitter to backoff waits, avoiding fixed-rhythm collisions across processes
import sys  # Import sys: retry warnings go straight to stderr so they are visible in evaluation logs
import threading  # Import threading: lock the in-process throttle timer (the pace stays serialized even when multiple threads call the same backend)
import time  # Import time: monotonic clock for computing throttle intervals and backoff sleeps

from oregenesis.llm.base import LLMBackend, LLMResponse  # Import the abstract base class and the unified response type
import re  # Import re: clean tags out of thinking text
import urllib.request  # Import urllib.request: send HTTP directly via the standard library
import urllib.error  # Import urllib.error: distinguish network-layer errors from HTTP errors

# Transient HTTP status codes worth retrying: 429 = rate limited; 5xx = temporary server fault.
# Permanent errors such as 401/404 (wrong API key / model name) are pointless to retry and are excluded here
_RETRIABLE_HTTP_CODES = {429, 500, 502, 503, 504}

_THROTTLE_LOCK = threading.Lock()  # In-process throttle lock: waiting under the lock naturally queues concurrent requests into a serial pace
_LAST_REQUEST_TS = 0.0  # When the last cloud request was sent (monotonic clock)


def _throttle(min_interval: float) -> None:  # Private helper: active throttling, controlling the minimum interval between consecutive cloud requests
    """If less than min_interval seconds have passed since the last cloud request, sleep the remainder first and then release (process-level pace control)."""  # Function docstring: lowers the chance of hitting 429 at the source
    global _LAST_REQUEST_TS  # Declare mutation of the module-level timestamp
    if min_interval <= 0:  # An interval of 0/negative means throttling is off
        return  # Release immediately
    with _THROTTLE_LOCK:  # Under lock: threads queue up and release one by one, and the overall pace still respects the interval
        now = time.monotonic()  # Take the current monotonic clock (immune to system clock changes)
        wait = _LAST_REQUEST_TS + min_interval - now  # Compute how many seconds remain until the interval since the last request is met
        if wait > 0:  # When the interval has not elapsed
            time.sleep(wait)  # Sleep until the interval is satisfied
            now = time.monotonic()  # Re-take the clock after waking
        _LAST_REQUEST_TS = now  # Record this release's timestamp


def _warn(msg: str) -> None:  # Private helper: retry warnings go straight to stderr
    """Print retry/degradation events to stderr (still visible when evaluation output is redirected to a log file)."""  # Function docstring: the retry process stays observable, never silent
    print(msg, file=sys.stderr, flush=True)  # Flush immediately to avoid buffering delays


def _retry_delay(e: "urllib.error.HTTPError | None", attempt: int) -> float:  # Private helper: compute the wait in seconds for retry attempt N
    """Prefer the server's Retry-After header; otherwise exponential backoff of 2^n seconds plus 0-0.5 seconds of random jitter."""  # Function docstring: single-point implementation of the backoff policy
    delay = min(120.0, (2 ** attempt) + random.uniform(0, 0.5))  # Exponential backoff: 2/4/8/16/32/64 seconds, capped at 120 seconds
    if e is not None:  # An HTTP error response may carry the standard Retry-After header (the server knows best when to come back)
        retry_after = (e.headers or {}).get("Retry-After") if e.headers is not None else None  # Read the Retry-After header
        if retry_after:  # When present, try parsing it as integer seconds (HTTP-date format is rare; on parse failure fall back to the self-computed value)
            try:  # Wrap the integer parsing
                delay = max(delay, float(int(retry_after.strip())))  # Take the larger of the server-specified and self-computed values
            except ValueError:  # Non-integer format (e.g. an HTTP date)
                pass  # Ignore and use the self-computed backoff value
    return delay  # Return this retry's wait in seconds



def _is_ollama(api_base: str) -> bool:  # Private helper: decide whether the service address points at Ollama
    """Decide whether the backend is Ollama: the URL contains "ollama" or the port falls in 11434-11439."""  # Function docstring: judged by the URL text or the 11434-11439 ports
    base = api_base.lower()  # Lowercase the address for substring matching
    return "ollama" in base or any(f":{port}" in base for port in range(11434, 11440))  # Judged as Ollama when it contains "ollama" or any Ollama port number


def _ollama_native_base(api_base: str) -> str:  # Private helper: convert a compatible address into the Ollama native root address
    """Convert a .../v1-style compatible address into the Ollama native root address (for building /api/chat)."""  # Function docstring: strips the /v1 suffix for building /api/chat
    base = api_base.rstrip("/")  # Strip the trailing slash from the address
    if base.endswith("/v1"):  # When it ends with /v1 (the OpenAI compatibility layer suffix)
        base = base[:-3]  # Cut off the last 3 characters (/v1)
    return base.rstrip("/")  # Strip the trailing slash again and return the native root address


def _json_request(url: str, data: bytes, api_key: str) -> urllib.request.Request:  # Private helper: build a JSON POST request object
    """Build a JSON POST request (Content-Type + Bearer headers, boilerplate shared by all call paths)."""  # Function docstring: shared boilerplate of the Content-Type + Bearer headers
    return urllib.request.Request(url, data=data, headers={  # Create the request object with an inline header dict (parentheses span lines)
        "Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})  # Set the JSON content type and the Bearer auth header; close the parentheses and return


class OpenAICompatibleBackend(LLMBackend):  # OpenAI-compatible backend class, inheriting LLMBackend
    """OpenAI-compatible API backend (auto-routes to the native /api/chat when Ollama is detected)."""  # Class docstring: auto-routes to the native /api/chat on Ollama detection

    def chat(self, messages: list[dict[str, str]], system_prompt: str = "",  # Non-streaming chat implementation: message list and optional system prompt (signature spans lines)
             **kwargs) -> LLMResponse:  # kwargs holds per-call overrides; returns the unified response
        """Non-streaming chat main entry; raises ConnectionError on connection failure and RuntimeError on any other failure.

        kwargs: temperature / max_tokens / response_format (JSON mode) / num_ctx /
        think (Ollama-specific) / images (a list of base64 images for vision models such as
        qwen-vl, appended to the last user message), etc.; when omitted, the config defaults
        apply. Upper layers catch the exceptions and take the degradation fallback.
        """
        full_messages: list[dict[str, str]] = []  # Initialize the full message list
        if system_prompt:  # The system message carries role setting and output-format constraints, so it must go first
            full_messages.append({"role": "system", "content": system_prompt})  # Insert the system prompt as the first message
        full_messages.extend(messages)  # Append the caller's conversation messages
        images = kwargs.get("images") or []  # Vision input: a base64 list (used by VLM tools)
        if images:  # Appended to the last message: Ollama native uses the images field; compatible endpoints switch content to parts
            if _is_ollama(self.config.api_base):  # Ollama-native protocol branch
                full_messages[-1]["images"] = [str(b) for b in images]  # Attach the images field (base64 list) to the last message
            else:  # OpenAI-compatible protocol branch
                last = full_messages[-1]  # Take the last message to rewrite its content
                parts = [{"type": "text", "text": str(last.get("content", ""))}]  # Switch content to the parts structure: the original text segment goes first
                parts += [{"type": "image_url",  # Append the image-segment list (parentheses span lines)
                           "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]  # Each image encoded as a data URL segment; comprehension done
                last["content"] = parts  # Write the multi-part content back to the last message
        if _is_ollama(self.config.api_base):  # Only the native endpoint can apply num_ctx and disable thinking
            return self._chat_ollama_native(full_messages, kwargs)  # Route to the Ollama-native non-streaming implementation
        payload = {  # temperature/max_tokens follow "per-call kwargs take priority over config defaults"
            "model": self.config.model_name,  # Request body field: model name
            "messages": full_messages,  # Request body field: the full message list
            "temperature": kwargs.get("temperature", self.config.temperature),  # Request body field: temperature (per-call kwargs override config defaults)
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),  # Request body field: generation cap (per-call kwargs take priority)
        }
        if kwargs.get("response_format") is not None:  # Ask the server to force JSON output
            payload["response_format"] = kwargs["response_format"]  # Write the JSON-mode parameter into the request body
        # Reasoning-model compatibility: DeepSeek's flash/v4 series outputs its chain of thought (reasoning_content) first by
        # default, which can consume all of max_tokens, leaving content empty with finish=length. With disable_thinking
        # on, thinking is explicitly turned off (the parameter is officially supported by DeepSeek; applies only to deepseek endpoints, never touches the GLM path)
        if "deepseek" in self.config.api_base.lower() and self.config.disable_thinking:  # Reasoning model + thinking requested off
            payload["thinking"] = {"type": "disabled"}  # Write the thinking-disabled parameter
        data = json.dumps(payload).encode("utf-8")  # Serialize the request body into JSON bytes
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"  # The Bearer header is required by remote services
        attempt = 0  # Retries performed so far (incremented each loop until success)
        while True:  # Retry loop: transient failures retry with exponential backoff; only raise with the original semantics once retries cannot save it
            _throttle(self.config.min_interval_seconds)  # Active throttling: keep the minimum interval from the last cloud request (0 = off)
            try:  # Synchronous blocking request; local inference is slow so the timeout is thousand-second scale, and timeouts are caught by the except below too
                req = _json_request(url, data, self.config.api_key)  # Build the POST request with auth headers (rebuilt each loop: Request objects are not reusable)
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:  # Send the request and get the response context (timeout controlled by config)
                    body = json.loads(resp.read().decode("utf-8"))  # Read the response body, decode it, and parse it into a dict
                break  # Break out of the retry loop on success
            except urllib.error.HTTPError as e:  # HTTP error (a URLError subclass; must be caught before URLError)
                err_body = ""  # The error response body (used to detect response_format incompatibility and for the final error message)
                try:  # A failure reading the error body does not affect the main flow
                    err_body = e.read().decode("utf-8", errors="replace")  # Read the error response body
                except Exception:  # Keep the empty string on read failure
                    pass  # Continue to the decision logic
                # Compatibility shim: a few cloud services do not support the response_format forced-JSON mode (HTTP 400 error);
                # the parameter is dropped automatically and retried immediately (in later rounds the payload no longer has the key, pop returns None, this branch is skipped)
                if e.code == 400 and "response_format" in err_body and payload.pop("response_format", None) is not None:  # Identified as the JSON mode being unsupported
                    _warn(f"[兼容垫片] 服务端不支持 response_format，已去掉该参数重试: HTTP 400 {err_body[:120]}")  # Keep the warning visible
                    data = json.dumps(payload).encode("utf-8")  # Re-serialize the request body without the parameter
                    continue  # Retry immediately (not counted against the backoff retry budget)
                if e.code in _RETRIABLE_HTTP_CODES and attempt < self.config.max_retries:  # Transient failure with retry budget remaining
                    attempt += 1  # Increment the retry counter
                    delay = _retry_delay(e, attempt)  # Compute the wait (the server's Retry-After takes priority)
                    _warn(f"[限流重试] HTTP {e.code}，第 {attempt}/{self.config.max_retries} 次，等 {delay:.1f} 秒后重试")  # Keep the warning visible
                    time.sleep(delay)  # Sleep out the backoff and retry
                    continue  # Loop back and resend the request
                raise ConnectionError(f"LLM 后端返回 HTTP {e.code} ({url}): {e.reason} {err_body[:200]}")  # Permanent error or retries exhausted: translate and raise
            except urllib.error.URLError as e:  # Network-layer failure/timeout (socket timeouts included; mostly transient)
                if attempt < self.config.max_retries:  # Retry budget remaining
                    attempt += 1  # Increment the retry counter
                    delay = _retry_delay(None, attempt)  # Exponential backoff wait
                    _warn(f"[网络重试] {e}，第 {attempt}/{self.config.max_retries} 次，等 {delay:.1f} 秒后重试")  # Keep the warning visible
                    time.sleep(delay)  # Sleep out the backoff and retry
                    continue  # Loop back and resend the request
                raise ConnectionError(f"无法连接 LLM 后端 ({url}): {e}")  # Retries exhausted: translate and raise with the original semantics
            except Exception as e:  # Remaining failures (one-off errors such as decode failures)
                raise RuntimeError(f"LLM 调用失败: {e}")  # No retry; raise with the original semantics
        # Fault-tolerant parsing: take only the first choice, with .get fallbacks on the message/usage
        # fields to prevent crashes when a few compatible services return incomplete structures
        choice = body.get("choices", [{}])[0]  # Take the first choice (default empty dict, then the first element)
        message = choice.get("message", {})  # Take the message dict (empty by default)
        content_text = message.get("content") or ""  # Take the content field (null falls back to an empty string)
        if not str(content_text).strip() and message.get("reasoning_content"):  # Thinking exhausted the tokens and the content is empty
            _warn("[思考兜底] content 为空（finish=length），降级用思考文本作正文（可能含推理噪声）")  # Keep the warning visible
            content_text = str(message.get("reasoning_content"))  # Fall back to the thinking text (same policy as the Ollama branch)
        content_text = re.sub(r"<\|[A-Za-z_]+\|>", "", str(content_text))  # Clean up special tokens occasionally leaked by reasoning models (e.g. <|OPENAI|>); legitimate content can never contain this shape
        usage = body.get("usage", {})  # Take the usage stats dict (empty by default)
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)  # Take the prompt token count (None/0 fallback) as an int
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)  # Take the completion token count (None/0 fallback) as an int
        from oregenesis.llm.run_telemetry import emit as _emit_telemetry  # Local import of the telemetry publish function
        _emit_telemetry({  # Run telemetry: report token consumption and the context window (zero overhead when there are no subscribers)
            "kind": "llm",  # Event kind marked as llm
            "model": body.get("model", self.config.model_name),  # Report the model name (the server's return takes priority)
            "prompt_tokens": prompt_tokens,  # Report the input token count
            "completion_tokens": completion_tokens,  # Report the completion token count
            "context_window": int(self.config.num_ctx),  # Report the configured context window size
        })
        return LLMResponse(  # Wrap the unified response object (parentheses span lines)
            content=content_text,  # The content (fallen back to thinking text when necessary)
            model=body.get("model", self.config.model_name),  # The model name follows the server's return
            tokens_used=usage.get("total_tokens", 0),  # Total tokens from usage.total_tokens
            finish_reason=choice.get("finish_reason", ""),  # Finish reason from choice.finish_reason
            raw=body,  # Keep the server's raw payload
            prompt_tokens=prompt_tokens,  # Write the input token breakdown
            completion_tokens=completion_tokens,  # Write the completion token breakdown
        )

    def _chat_ollama_native(self, full_messages: list[dict[str, str]], kwargs: dict) -> LLMResponse:  # Ollama-native non-streaming implementation: takes messages and overrides
        """Ollama-native /api/chat non-streaming: sets num_ctx explicitly and disables thinking as needed.

        Ollama's OpenAI-compatible endpoint ignores options.num_ctx and silently truncates an
        overlong prompt, losing instructions and returning empty content; the native endpoint
        applies num_ctx correctly and supports think=false. Adaptive window: when VRAM is tight,
        a large num_ctx fails to allocate -> HTTP 500, and the window is halved for a retry
        (floor 8192), completing the call with the largest window the current VRAM can fit so the
        whole chain of LLM calls is not wiped out.
        """
        num_ctx = int(kwargs.get("num_ctx") or self.config.num_ctx)  # Context window: per-call override first, config value as fallback
        max_tokens = int(kwargs.get("max_tokens") or self.config.max_tokens)  # Generation cap: per-call override first, config value as fallback
        # num_ctx must hold both the prompt and the generation, otherwise the prompt is truncated to half (empty content returned):
        # estimate prompt tokens from character counts (mixed Chinese/English, about 1 token ~= 1.3 characters), budget floor 512, margin 256
        est_prompt = int(sum(len(str(m.get("content", ""))) for m in full_messages) / 1.3) + 16  # Rough prompt-token estimate as character count / 1.3, plus a margin of 16
        think = kwargs.get("think")  # Read whether this call explicitly specified the thinking mode
        if think is None:  # When unspecified, config.disable_thinking decides (thinking off by default)
            think = not self.config.disable_thinking  # Derive the think boolean from the configuration when unspecified
        url = f"{_ollama_native_base(self.config.api_base)}/api/chat"  # Build the Ollama-native chat endpoint URL
        while True:  # Retry loop: on HTTP 500, halve the window and try again
            budget = max(512, num_ctx - est_prompt - 256)  # Generation budget = window - prompt estimate - margin, floor 512
            call_max_tokens = min(max_tokens, budget)  # Keep the generation quota from crowding out the prompt's context space
            payload: dict[str, object] = {  # stream=False returns in one shot; options carries the dedicated parameters
                "model": self.config.model_name,  # Request body field: model name
                "messages": full_messages,  # Request body field: the full message list
                "stream": False,  # Request body field: non-streaming, one-shot return
                "think": bool(think),  # Request body field: whether thinking mode is on
                "options": {  # Request body field: the Ollama-specific options sub-dict
                    "num_ctx": num_ctx,  # Option: context window size
                    "temperature": kwargs.get("temperature", self.config.temperature),  # Option: temperature (per-call override first)
                    "num_predict": call_max_tokens,  # Option: this call's generation token cap
                },
            }
            if kwargs.get("response_format") is not None:  # OpenAI's JSON mode maps to "format"
                payload["format"] = "json"  # Write Ollama's JSON output format switch
            req = _json_request(url, json.dumps(payload).encode("utf-8"), self.config.api_key)  # Serialize the request body and build the POST request
            # Exception translation matches chat(); HTTPError is a URLError subclass and must be caught first
            try:  # try wraps this request; failures route by exception type
                with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:  # Send the request and get the response
                    body = json.loads(resp.read().decode("utf-8"))  # Read and parse the response JSON
                break  # Break out of the retry loop on success
            except urllib.error.HTTPError as e:  # Catch HTTPError first (a URLError subclass must be caught first)
                # 500 ~= context/VRAM allocation failure: halve the window and retry (floor 8192); raise only when it still fails at the floor
                if e.code == 500 and num_ctx > 8192:  # On 500 with room left to halve, enter the adaptive down-shift
                    import logging  # Local import of the logging module
                    new_ctx = max(8192, num_ctx // 2)  # New window = half the current, floor 8192
                    logging.getLogger(__name__).warning(  # Log the down-shift warning (parentheses span lines)
                        "Ollama 返回 500（疑似 %d 上下文窗口显存分配失败），降为 %d 重试", num_ctx, new_ctx,  # Log content: the old and new window sizes
                    )
                    num_ctx = new_ctx  # Update the window and retry
                    continue  # Loop back to while and resend the request
                raise ConnectionError(f"LLM 后端返回 HTTP {e.code} ({url}): {e.reason}")  # Not 500, or already at the floor: translate and raise as ConnectionError
            except urllib.error.URLError as e:  # Then catch network-layer errors
                raise ConnectionError(f"无法连接 LLM 后端 ({url}): {e}")  # Translate and raise as ConnectionError
            except Exception as e:  # Catch-all for remaining exceptions
                raise RuntimeError(f"LLM 调用失败: {e}")  # Translate and raise as RuntimeError
        # Response field adaptation: message.content is the content ("or ''" guards null; in thinking mode the content may go entirely through thinking);
        # prompt_eval_count / eval_count are the prompt and generated token counts respectively
        message = body.get("message") or {}  # Take the response message dict (null falls back to an empty dict)
        content = message.get("content") or ""  # Take the content field (null falls back to an empty string)
        if not content.strip():  # When the content is blank, try recovering from thinking
            # Under Ollama, qwen3-vl writes the entire content into message.thinking (content stays empty,
            # even think=false cannot stop it), wrapped in an extra layer of literal think open/close tags; without
            # recovering from thinking, every LLM call in this package behaves as "the model returned empty" — the evidence loop aborts in its first round
            thinking = re.sub(r"<\s*/?\s*think\s*>", "", message.get("thinking") or "").strip()  # Take the thinking text, strip the think open/close tags, then trim the surrounding whitespace
            content = thinking  # Use the recovered thinking content as the content
        prompt_tokens = int(body.get("prompt_eval_count") or 0)  # Prompt token count from prompt_eval_count
        gen_tokens = int(body.get("eval_count") or 0)  # Generated token count from eval_count
        from oregenesis.llm.run_telemetry import emit as _emit_telemetry  # Local import of the telemetry publish function
        _emit_telemetry({  # Report token consumption and the actually effective context window (adaptive down-shifts included)
            "kind": "llm",  # Event kind marked as llm
            "model": body.get("model", self.config.model_name),  # Report the model name (the server's return takes priority)
            "prompt_tokens": prompt_tokens,  # Report the input token count
            "completion_tokens": gen_tokens,  # Report the completion token count
            "context_window": num_ctx,  # Report the actually effective window (possibly already adaptively down-shifted)
        })
        return LLMResponse(  # Wrap the unified response object (parentheses span lines)
            content=content,  # The content (recovered from thinking when necessary)
            model=body.get("model", self.config.model_name),  # The model name follows the server's return
            tokens_used=prompt_tokens + gen_tokens,  # Total tokens = input + generated
            finish_reason=str(body.get("done_reason") or ""),  # Finish reason mapped to the done_reason string
            raw=body,  # Keep the raw response payload
            prompt_tokens=prompt_tokens,  # Write the input token breakdown
            completion_tokens=gen_tokens,  # Write the completion token breakdown
        )

    def chat_stream(self, messages: list[dict[str, str]], system_prompt: str = "", **kwargs):  # True streaming chat generator entry
        """True streaming chat: yields {"type": "reasoning"|"content", "content": str} segment by segment.
        Ollama goes through the native /api/chat (ndjson stream, thinking -> the reasoning channel);
        other OpenAI-compatible endpoints go through SSE (delta.reasoning_content -> reasoning).
        The caller aggregates the full content; on failure it raises the same exceptions as chat().
        """
        full_messages: list[dict[str, str]] = []  # Initialize the full message list
        if system_prompt:  # When there is a system prompt
            full_messages.append({"role": "system", "content": system_prompt})  # Insert the system message first
        full_messages.extend(messages)  # Append the conversation messages
        if _is_ollama(self.config.api_base):  # Only the native endpoint applies num_ctx correctly
            yield from self._stream_ollama_native(full_messages, kwargs)  # Delegate to the Ollama-native streaming implementation, yielding segment by segment
            return  # The generator ends
        payload = {  # With stream=True the server returns "data: ..." lines block by block via SSE (text/event-stream)
            "model": self.config.model_name,  # Request body field: model name
            "messages": full_messages,  # Request body field: the full message list
            "temperature": kwargs.get("temperature", self.config.temperature),  # Request body field: temperature (per-call override first)
            "max_tokens": kwargs.get("max_tokens", self.config.max_tokens),  # Request body field: generation cap (per-call override first)
            "stream": True,  # Request body field: enable streaming
        }
        url = f"{self.config.api_base.rstrip('/')}/chat/completions"  # Build the chat-completions endpoint URL
        req = _json_request(url, json.dumps(payload).encode("utf-8"), self.config.api_key)  # Serialize and build the authenticated POST request
        try:  # Wrap the connection phase
            resp = urllib.request.urlopen(req, timeout=self.config.timeout_seconds)  # Open the streaming response (closed later manually via with)
        except urllib.error.URLError as e:  # Connection-failure branch
            raise ConnectionError(f"无法连接 LLM 后端 ({url}): {e}")  # Translate and raise as ConnectionError
        with resp:  # Use a context manager to ensure the stream closes
            # SSE line-by-line parsing: only "data:"-prefixed lines are processed; a single line failing to parse is skipped outright (fragment tolerance)
            for raw in resp:  # Iterate the response stream line by line
                line = raw.decode("utf-8", errors="replace").strip()  # Decode the line (bad bytes replaced) and trim the surrounding whitespace
                if not line.startswith("data:"):  # Skip lines without the data: prefix (e.g. blank lines)
                    continue  # Process the next line
                chunk = line[5:].strip()  # Strip the data: prefix to get the payload
                if chunk == "[DONE]":  # End marker
                    break  # Exit the loop when the end marker arrives
                try:  # Wrap the single-line JSON parse
                    body = json.loads(chunk)  # Parse the chunk's JSON
                except json.JSONDecodeError:  # Bad-line (fragment) tolerance branch
                    continue  # Skip and continue with the next line
                # delta is this round's incremental fragment; thinking and content are reported as two event types
                delta = (body.get("choices") or [{}])[0].get("delta") or {}  # Take the first choice's delta field (multiple fallbacks guard against emptiness)
                reasoning = delta.get("reasoning_content")  # Take the thinking increment (the compatible-endpoint field name)
                content = delta.get("content")  # Take the content increment
                if reasoning:  # When there is a thinking increment
                    yield {"type": "reasoning", "content": reasoning}  # Yield as a reasoning-type event
                if content:  # When there is a content increment
                    yield {"type": "content", "content": content}  # Yield as a content-type event

    def _stream_ollama_native(self, full_messages: list[dict[str, str]], kwargs: dict):  # Ollama-native streaming implementation (generator)
        """Ollama-native /api/chat streaming (ndjson), with thinking as the reasoning channel.

        Unlike the non-streaming path: the payload carries no think field, thinking content
        streams out through the thinking field, and the caller decides whether to display it.
        """
        num_ctx = int(kwargs.get("num_ctx") or self.config.num_ctx)  # Context window: per-call override first, config value as fallback
        # Same token estimation and num_predict clamping policy as _chat_ollama_native
        est_prompt = int(sum(len(str(m.get("content", ""))) for m in full_messages) / 1.3) + 16  # Rough prompt-token estimate as character count / 1.3, plus a margin of 16
        max_tokens = min(  # Generation cap takes the smaller of the two (parentheses span lines)
            int(kwargs.get("max_tokens") or self.config.max_tokens),  # Candidate 1: per-call override or the config default
            max(512, num_ctx - est_prompt - 256),  # Candidate 2: the window budget with a floor of 512
        )
        payload: dict[str, object] = {  # Mostly identical to the non-streaming payload, except stream=True
            "model": self.config.model_name,  # Request body field: model name
            "messages": full_messages,  # Request body field: the full message list
            "stream": True,  # Request body field: enable streaming
            "options": {  # Request body field: the Ollama-specific options sub-dict
                "num_ctx": num_ctx,  # Option: context window
                "temperature": kwargs.get("temperature", self.config.temperature),  # Option: temperature
                "num_predict": max_tokens,  # Option: generation cap (already clamped to the budget)
            },
        }
        url = f"{_ollama_native_base(self.config.api_base)}/api/chat"  # Build the native chat endpoint URL
        req = _json_request(url, json.dumps(payload).encode("utf-8"), self.config.api_key)  # Build the authenticated POST request
        with urllib.request.urlopen(req, timeout=self.config.timeout_seconds) as resp:  # Open the ndjson streaming response
            for raw in resp:  # ndjson line-by-line parsing: blank/bad lines are skipped outright (tolerance without breaking the whole stream)
                line = raw.decode("utf-8", errors="replace").strip()  # Decode the line (bad bytes replaced) and trim the surrounding whitespace
                if not line:  # Skip blank lines
                    continue  # Process the next line
                try:  # Wrap the single-line JSON parse
                    body = json.loads(line)  # Parse the frame's JSON
                except json.JSONDecodeError:  # Bad-frame tolerance branch
                    continue  # Skip and continue with the next frame
                message = body.get("message") or {}  # thinking = thinking increment, content = content increment
                thinking = message.get("thinking")  # Take the thinking-increment field
                content = message.get("content")  # Take the content-increment field
                if thinking:  # When there is a thinking increment
                    yield {"type": "reasoning", "content": thinking}  # Yield as a reasoning event
                if content:  # When there is a content increment
                    yield {"type": "content", "content": content}  # Yield as a content event
                if body.get("done"):  # done=True is Ollama's final frame (with token stats attached)
                    break  # End stream reading

    def is_available(self) -> bool:  # Lightweight liveness probe: returns a bool
        """Lightweight probe (GET <api_base>/models with a 5-second short timeout); any exception returns False."""  # Method docstring: 5-second short timeout; exceptions always return False
        try:  # Wrap the probe request
            url = f"{self.config.api_base.rstrip('/')}/models"  # Build the model-list endpoint as the probe address
            req = urllib.request.Request(  # Build the probe request (parentheses span lines)
                url, headers={"Authorization": f"Bearer {self.config.api_key}"})  # Attach the Bearer header and close the parentheses
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # Send the probe with a 5-second short timeout
                return resp.status == 200  # HTTP 200 counts as available
        except Exception:  # Probe failure is a normal branch (service not started, etc.); report unavailable instead of raising
            return False  # Any exception counts as unavailable

    def check_availability(self) -> tuple[bool, str]:  # Enhanced probe: returns an (availability, failure cause) pair
        """Enhanced probe: returns the failure cause in addition to the boolean result (for the diagnostics panel to display)."""  # Method docstring: the failure cause feeds the diagnostics panel
        try:  # Wrap the probe request
            url = f"{self.config.api_base.rstrip('/')}/models"  # Build the model-list endpoint
            req = urllib.request.Request(  # Build the probe request (parentheses span lines)
                url, headers={"Authorization": f"Bearer {self.config.api_key}"})  # Attach the Bearer header and close the parentheses
            with urllib.request.urlopen(req, timeout=5.0) as resp:  # Send the probe with a short timeout
                if resp.status == 200:  # HTTP 200 means available
                    return True, ""  # Available with no error message
                return False, f"HTTP {resp.status}"  # Non-200 (e.g. a proxy returning 404/502) also counts as unavailable
        except urllib.error.URLError as e:  # Network-layer error: carrying reason back helps locate port/service problems
            return False, f"无法连接 {self.config.api_base}: {e.reason}"  # Return unavailable plus the connection-failure cause (reason carried back for diagnosis)
        except Exception as e:  # Catch-all for remaining exceptions
            return False, str(e)  # Return unavailable plus the exception text
