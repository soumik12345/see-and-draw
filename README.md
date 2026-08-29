# see-and-draw

`see-and-draw` is an asynchronous Python agent that uses Tau and
[`cyyprezz/krita-codex-mcp`](https://github.com/cyyprezz/krita-codex-mcp) to
inspect a live Krita canvas and turn text prompts into editable artwork. Each
normal run saves an `artwork.kra`, an `artwork.png`, and a replayable
`rollout.json` under `runs/`. The included demo uses a vision-capable model
through OpenRouter and renders a live Rich trace while the agent works.

## Requirements

- Linux on x86-64, `git`, and `curl`.
- Python 3.14 or newer and [`uv`](https://docs.astral.sh/uv/).
- An [OpenRouter](https://openrouter.ai/) API key.
- [Krita 5.3.2.1](https://krita.org/en/posts/2026/krita-5.3.2.1-released/),
  the build tested by Krita Codex MCP.

> [!NOTE]
> Krita Codex MCP 0.2.0 is Windows-first upstream. The explicit environment
> variables below are the Linux compatibility setup used by this project.

## Setup

### 1. Install the project

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Clone the repository, install its locked dependencies, and create the allowed
artifact directory:

```bash
git clone https://github.com/soumik12345/see-and-draw.git
cd see-and-draw
uv sync --locked
mkdir -p runs
```

The project pins `krita-codex-mcp==0.2.0`; do not install the older
`nanayax3/krita-mcp` server for this agent.

### 2. Install Krita 5.3.2.1

Ubuntu 24.04's package repository provides Krita 5.2.2, which does not expose
the native stroke APIs required by `krita_paint_path`. Install the tested
AppImage instead:

```bash
mkdir -p "$HOME/.local/opt/krita"
curl -fL \
  -o "$HOME/.local/opt/krita/krita-5.3.2.1-x86_64.AppImage" \
  https://download.kde.org/stable/krita/5.3.2.1/krita-5.3.2.1-x86_64.AppImage
chmod +x "$HOME/.local/opt/krita/krita-5.3.2.1-x86_64.AppImage"
```

### 3. Install the authenticated Krita bridge

Close every Krita window, then install the wheel-bundled bridge. The allowed
root must already exist and should remain as narrow as possible:

```bash
env \
  APPDATA="$HOME/.local/share" \
  LOCALAPPDATA="$HOME/.config" \
  USERPROFILE="$HOME" \
  KRITA_CODEX_CONFIG="$HOME/.config/krita-codex-mcp/config.json" \
  uv run krita-codex-install install --allowed-root "$PWD/runs"
```

In Krita, open **Settings > Configure Krita > Python Plugin Manager**:

1. Disable the old **Krita MCP** plugin if it is present.
2. Enable **Krita Codex Bridge**.
3. Close Krita completely.

### 4. Start Krita with the bridge environment

Always start the tested AppImage with the same configuration path. The
`--appimage-extract-and-run` option also works when `/dev/fuse` is unavailable:

```bash
env \
  APPDATA="$HOME/.local/share" \
  LOCALAPPDATA="$HOME/.config" \
  USERPROFILE="$HOME" \
  KRITA_CODEX_CONFIG="$HOME/.config/krita-codex-mcp/config.json" \
  "$HOME/.local/opt/krita/krita-5.3.2.1-x86_64.AppImage" \
  --appimage-extract-and-run
```

Leave Krita open. In another terminal, verify the installed plugin, shared
token, loopback bridge, and runtime capabilities:

```bash
env \
  APPDATA="$HOME/.local/share" \
  LOCALAPPDATA="$HOME/.config" \
  USERPROFILE="$HOME" \
  KRITA_CODEX_CONFIG="$HOME/.config/krita-codex-mcp/config.json" \
  uv run krita-codex-install check
```

A successful check ends with output similar to:

```text
Krita 5.3.2.1: compatible (compatibility profile 1, tested_build).
Ready for productive use.
```

### 5. Configure the model provider

Create a local `.env` file; it is ignored by Git:

```dotenv
OPENROUTER_API_KEY=replace-with-your-key
LOCALAPPDATA=${HOME}/.config
KRITA_CODEX_CONFIG=${HOME}/.config/krita-codex-mcp/config.json
```

### 6. Run the agent

Keep Krita open with the bridge running, then execute:

```bash
uv run python run_agent.py
```

The sample prompt and model are configured in `run_agent.py`. Successful runs
are written to a numbered directory under `runs/` with editable artwork, a PNG
preview, and the complete replay rollout.

## Verification

Run the project checks after changing the agent or its Krita integration:

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
```

If `krita-codex-install check` reports that the bridge is unavailable, confirm
that Krita is still open, **Krita Codex Bridge** is enabled, and Krita was
started with the environment shown above. If `krita_state` reports native paint
capabilities as unavailable, confirm that the running application is the
5.3.2.1 AppImage rather than `/usr/bin/krita` 5.2.2.
