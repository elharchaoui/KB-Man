from __future__ import annotations

import asyncio
import os
import sys
from functools import wraps
from pathlib import Path

import click
from openai import AsyncOpenAI

from kb import KnowledgeBase
from kb.config import Config
from kb.injection import injector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def async_command(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))
    return wrapper


def _make_kb(config: Config) -> KnowledgeBase:
    api_key = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    llm_model = os.environ.get("KB_LLM_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        click.echo(
            "Error: set OPENROUTER_API_KEY (or OPENAI_API_KEY) environment variable.",
            err=True,
        )
        sys.exit(1)

    llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    embedding_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    return KnowledgeBase(
        llm_client=llm_client,
        embedding_client=embedding_client,
        llm_model=llm_model,
        config=config,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to config.json (default: ~/.kb/config.json)",
)
@click.pass_context
def cli(ctx: click.Context, config_path: str | None) -> None:
    """KB-Man — personal knowledge base."""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config.load(config_path)


@cli.command()
@click.argument("input")
@click.option("--tags", "-t", multiple=True, help="Tags (repeatable: -t ai -t rag)")
@click.pass_context
@async_command
async def add(ctx: click.Context, input: str, tags: tuple[str, ...]) -> None:
    """Add a URL, text note, or code snippet to the KB."""
    config: Config = ctx.obj["config"]
    kb = _make_kb(config)
    await kb.startup()

    click.echo(f"Ingesting: {input[:80]}{'...' if len(input) > 80 else ''}")
    with click.progressbar(length=4, label="Processing") as bar:
        # We can't hook into internals, so just show a spinner
        bar.update(1)
        result = await kb.add(input, list(tags))
        bar.update(3)

    click.echo(f"\n✓ Stored: {result.title!r} [{result.type}]")
    click.echo(f"  document_id : {result.document_id}")
    click.echo(f"  chunks      : {result.child_chunks_stored}")
    if result.auto_tags:
        click.echo(f"  auto-tags   : {', '.join(result.auto_tags)}")
    if list(tags):
        click.echo(f"  user-tags   : {', '.join(tags)}")


@cli.command()
@click.argument("query")
@click.option("--k", default=5, show_default=True, help="Number of results")
@click.option("--raw", is_flag=True, help="Print raw injection context instead of formatted output")
@click.pass_context
@async_command
async def search(ctx: click.Context, query: str, k: int, raw: bool) -> None:
    """Search the knowledge base."""
    config: Config = ctx.obj["config"]
    kb = _make_kb(config)
    await kb.startup()

    click.echo(f"Searching: {query!r}\n")
    output = await kb.search(query, k=k)

    if not output.final_results:
        click.echo("No results above threshold.")
        return

    if raw:
        context = injector.build(output, config)
        click.echo(context or "No results.")
        return

    # Formatted output
    click.echo(f"{'─' * 60}")
    click.echo(f"Iterations: {len(output.iterations)}")
    for rec in output.iterations:
        status = click.style("converged", fg="green") if rec.converged else click.style(f"ratio={rec.convergence_ratio:.0%}", fg="yellow")
        click.echo(f"  [{rec.iteration}] {rec.query!r} → {status}")
    click.echo(f"{'─' * 60}\n")

    for i, r in enumerate(output.final_results, 1):
        title = r.payload.get("title", "untitled")
        source = r.payload.get("source", "user")
        level_color = "cyan" if r.level == "summary" else "blue"
        click.echo(
            f"{i}. {click.style(title, bold=True)} "
            f"[{click.style(r.level, fg=level_color)}] "
            f"score={r.score:.3f}"
        )
        click.echo(f"   source: {source}")
        click.echo(f"   doc_id: {r.document_id}")
        if r.level == "chunk" and r.payload.get("chunk_header"):
            click.echo(f"   section: {r.payload['chunk_header']}")
        click.echo()
        # Print a preview of the text
        preview = r.text[:300].replace("\n", " ")
        if len(r.text) > 300:
            preview += "..."
        click.echo(f"   {preview}")
        click.echo()


@cli.command("list")
@click.pass_context
@async_command
async def list_docs(ctx: click.Context) -> None:
    """List all documents in the KB."""
    config: Config = ctx.obj["config"]
    kb = _make_kb(config)
    await kb.startup()

    docs = kb.list()
    if not docs:
        click.echo("Knowledge base is empty.")
        return

    click.echo(f"{'─' * 60}")
    click.echo(f"{'ID':<36}  {'TYPE':<6}  {'ADDED':<12}  TITLE")
    click.echo(f"{'─' * 60}")
    for d in docs:
        tags_str = f"  [{', '.join(d.tags)}]" if d.tags else ""
        click.echo(
            f"{d.document_id}  {d.type:<6}  {str(d.added_at.date()):<12}  "
            f"{d.title}{tags_str}"
        )


@cli.command()
@click.argument("document_id")
@click.confirmation_option(prompt="Delete this document and all its chunks?")
@click.pass_context
@async_command
async def delete(ctx: click.Context, document_id: str) -> None:
    """Delete a document by its document_id."""
    config: Config = ctx.obj["config"]
    kb = _make_kb(config)
    await kb.startup()

    result = await kb.delete(document_id)
    click.echo(
        f"Deleted: {result.deleted_summaries} summary, "
        f"{result.deleted_chunks} chunks (doc_id: {result.document_id})"
    )


@cli.command()
@click.pass_context
@async_command
async def stats(ctx: click.Context) -> None:
    """Show KB storage stats."""
    config: Config = ctx.obj["config"]
    kb = _make_kb(config)
    await kb.startup()

    s = await kb._store.collection_stats()
    click.echo(f"Summaries : {s['summaries']}")
    click.echo(f"Chunks    : {s['chunks']}")
    click.echo(f"Store     : {Path(config.qdrant_path).expanduser()}")
