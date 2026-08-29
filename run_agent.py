import asyncio
import os

from dotenv import load_dotenv
from tau_ai import OpenAICompatibleConfig, OpenAICompatibleProvider

from see_and_draw.agent import KritaSeeAndDrawAgent
from see_and_draw.trace import RichTraceRenderer


async def main() -> None:
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
            provider_name="OpenRouter",
            supports_images=True,
            timeout_seconds=120,
            max_retries=5,
        )
    )

    agent = KritaSeeAndDrawAgent(
        provider=provider,
        model="qwen/qwen3.8-flash",
        runs_dir="runs",
    )

    renderer = RichTraceRenderer()
    unsubscribe = agent.subscribe(renderer)
    try:
        async for _ in agent.run("Create a 1024x1024 sketch of a 3D cube."):
            pass

        if agent.last_run_artifacts is not None:
            renderer.console.print(
                f"[bold]Run directory:[/bold] {agent.last_run_artifacts.directory}"
            )
    finally:
        unsubscribe()
        await agent.aclose()
        await provider.aclose()


load_dotenv()
asyncio.run(main())
