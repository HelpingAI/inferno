# Copilot / AI Agent Instructions for Inferno

Summary: concise, task-focused guidance so an AI coding assistant can be immediately productive in this repository.

Quick architecture
- Two developer surfaces:
  - CLI (inferno command / python -m inferno) defined in inferno/cli/commands.py (Typer + Rich)
  - API server (FastAPI) with two flavours:
    - OpenAI-compatible endpoints in inferno/api/server.py (e.g. /v1/chat/completions). Uses SSE streaming (`text/event-stream`, `data: ...`) when stream=True.
    - Ollama/alternative endpoints in inferno/api/routes.py (e.g. /api/chat, /api/generate). Uses JSON-line streaming (`application/json` or newline-delimited JSON) in some routes.
- Core model logic: inferno/core
  - ModelManager (inferno/core/model_manager.py): download/list/remove/copy/quantize models; stores per-model info.json in ~/.inferno/models/<model>/info.json.
  - LLMInterface (inferno/core/llm.py): wrapper around llama-cpp-python Llama class. Accepts many llm parameters and auto-detects RoPE params from GGUF metadata using inferno/core/gguf_reader.py.
  - GGUF reader (inferno/core/gguf_reader.py): canonical place to extract metadata (max context, rope params, quantization_type).
  - Quantizer (inferno/core/quantizer.py): model quantize/compare helpers.
- Native client: inferno/client/client.py implements an OpenAI-compatible client (Inference + streaming parsing). The client expects SSE streaming `data: ...` lines (see _handle_streaming_response).

Important conventions & behaviors
- Model identifiers: supported formats include
  - repo_id:filename (explicit GGUF file)
  - repo_id (Hugging Face repo)
  - hf:repo_id or hf:repo_id:QUANT (ModelManager.parse_model_string handles these)
  - Local models are stored under ~/.inferno/models/<name>/info.json; info includes name, path, filename, downloaded_at, max_context, quantization_type
- Streaming semantics differ between server modules — be careful when changing server responses:
  - inferno/api/server.py (OpenAI-compatible): SSE format (prefixed with "data: ") and ends with "[DONE]".
  - inferno/api/routes.py (other API): uses newline-delimited JSON (JSON per line). Client code and tests must target the correct streaming style.
- keep_alive: endpoints accept strings like "5m", "0"; parse_keep_alive() and per-route parsing are implemented in server/routes; "0" means unload immediately (model.llm set to None / background task to unload).
- llama-cpp-python must be installed with the correct hardware backend BEFORE installing/running Inferno (see README). LLMInterface dynamically imports llama-cpp-python and surfaces a clear import error if it's missing.
- RoPE detection: LLMInterface attempts to extract rope_freq_base / rope_freq_scale from GGUF metadata; prefer using simple_gguf_info / GGUFReader utilities when adding features that depend on GGUF metadata.
- CLI uses INFERNO_DEBUG environment variable to toggle extra traceback output in several places — useful when adding diagnostics or modifying error messages.

Developer workflows
- Local development: pip install -e .; run commands with `inferno --help` or `python -m inferno`.
- Running server in dev: `inferno serve <model>` (downloads if needed and starts uvicorn server). Passing --host/--port/n_gpu_layers/n_ctx/n_threads are common options.
- Install llama-cpp-python with correct CMake flags for target hardware. Examples (from README):
  - CUDA: CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --upgrade --no-cache-dir
  - Metal: CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python ...
- Debugging: set INFERNO_DEBUG=1 to enable additional tracebacks in the CLI.

Testing & PR guidance
- There are currently no automated tests in `tests/`. Add targeted unit tests for:
  - ModelManager.parse_model_string (various input formats)
  - download_model behavior (use monkeypatch to stub huggingface_hub functions)
  - LLMInterface.load_model when llama-cpp-python is missing (assert it raises the informative import error)
  - Streaming behavior for /v1 (SSE) and /api endpoints (JSON-lines)
- Follow .github/pull_request_template.md for PR descriptions and include tests and docs updates for behavioral changes.

Typical small tasks & examples
- To add a new CLI command: add a function decorated with @app.command in inferno/cli/commands.py using typer; follow existing style (rich tables/panels). Add tests that invoke the function via subprocess or invoke Typer's CliRunner.
- To add a new API route: prefer adding Pydantic models in api/server.py or api/routes.py depending on target compatibility (OpenAI vs other). Use BackgroundTasks to schedule unloads and StreamingResponse for streaming output. Keep response shape consistent with existing endpoints and update inferno/client when you change streaming format.

Files to reference when editing behavior
- CLI: inferno/cli/commands.py
- Server: inferno/api/server.py (OpenAI-compatible) and inferno/api/routes.py (other API shapes)
- Core model logic: inferno/core/llm.py, inferno/core/model_manager.py, inferno/core/gguf_reader.py, inferno/core/quantizer.py
- Client: inferno/client/client.py (stream parsing expectations)
- Config: inferno/utils/config.py (models_dir default, config.json location)

If anything is ambiguous or you need examples for a specific change (tests, endpoint, or CLI behavior), ask for the exact files/behavior to modify and which compatibility surface (OpenAI / Ollama / CLI) to target.

Tool set to use
use modern Python best practices, including type hints, async programming where appropriate, and following PEP8 guidelines.
use `uvx ty check .` to verify type correctness.
use `ruff .` to verify linting.