"""
KB-Man Benchmark Suite
======================
Measures ingest and retrieval performance across a curated GenAI research corpus.

Usage:
    uv run python eval/benchmark.py [--skip-ingest] [--qdrant-path PATH]
"""
from __future__ import annotations

import argparse
import asyncio
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

# ── project root on path ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from kb import KnowledgeBase
from kb.config import Config
from kb.models import AddResult, RetrievalOutput


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader — no external dependency required."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


_load_dotenv(ROOT / ".env")

# ─────────────────────────────────────────────────────────────────────────────
# CORPUS  — mix of real URLs + rich text notes from recent GenAI research
# ─────────────────────────────────────────────────────────────────────────────

DOCUMENTS: list[dict] = [
    # ── URLs ─────────────────────────────────────────────────────────────────
    {
        "input": "https://ai.meta.com/blog/llama-4-multimodal-intelligence/",
        "tags": ["llm", "meta", "multimodal", "moe"],
        "label": "Llama 4 — Meta blog",
        "kind": "url",
    },
    {
        "input": "https://simonwillison.net/2025/Dec/31/the-year-in-llms/",
        "tags": ["llm", "review", "2025"],
        "label": "Year in LLMs 2025 — Simon Willison",
        "kind": "url",
    },
    {
        "input": "https://blogs.nvidia.com/blog/mixture-of-experts-frontier-models/",
        "tags": ["moe", "nvidia", "inference", "architecture"],
        "label": "NVIDIA — MoE frontier models",
        "kind": "url",
    },
    {
        "input": "https://blog.premai.io/speculative-decoding-2-3x-faster-llm-inference-2026/",
        "tags": ["inference", "speculative-decoding", "optimization"],
        "label": "Speculative Decoding 2026 — Prem AI",
        "kind": "url",
    },
    {
        "input": "https://arxiv.org/abs/2501.09136",
        "tags": ["rag", "agentic", "survey"],
        "label": "Agentic RAG — arXiv 2501.09136",
        "kind": "url",
    },
    # ── Text notes (rich summaries from research) ─────────────────────────────
    {
        "input": (
            "## LLM Landscape 2025–2026: Key Developments\n\n"
            "### OpenAI GPT-5 Family\n"
            "GPT-5 launched throughout 2025 with configurable reasoning effort, native multimodal input, "
            "and context windows exceeding one million tokens. GPT-5.4 (March 2026) unified general-purpose "
            "and coding lines into a single flagship with native computer use. Variant 'Thinking' trades "
            "latency for depth of reasoning; 'Instant' optimises for speed.\n\n"
            "### Anthropic Claude 4 Family\n"
            "Claude Opus 4.1 and Sonnet 4 were released in August 2025, supporting 1 million token context. "
            "The current lineup — Opus 4.6, Sonnet 4.6, Haiku 4.5 — spans a capability/cost spectrum. "
            "Claude Enterprise offers 500 K token context for deep document analysis. Claude excels at "
            "instruction following, long-context synthesis, and safety alignment.\n\n"
            "### Google Gemini 3 Family\n"
            "Gemini 3 Pro launched November 2025; Gemini 3.1 Pro (February 2026) doubled ARC-AGI-2 "
            "performance and leads 12 of 18 tracked benchmarks. Gemini 3 Flash delivers Pro-level "
            "intelligence at lower cost and is the default across Google consumer products.\n\n"
            "### DeepSeek & Open-Weight Trends\n"
            "DeepSeek-R1 popularised chain-of-thought reasoning in open models. DeepSeek-V3.2 and Qwen3 "
            "continue to close the gap with proprietary frontier models at a fraction of the training cost. "
            "Efficiency-focused training — synthetic data, speculative decoding at train time — is now standard.\n\n"
            "### Key Cross-Model Trends\n"
            "- Reasoning models (o1-style) trade speed for accuracy via extended thinking budgets\n"
            "- Multimodal inputs (text, image, audio, video) now standard at frontier\n"
            "- Cost per million tokens dropped ~10x between 2024 and 2026 for GPT-4-equivalent quality\n"
            "- Context windows: 128 K → 10 M tokens in the open-weight space (Llama 4 Scout)\n"
        ),
        "tags": ["llm", "gpt5", "claude", "gemini", "deepseek", "survey"],
        "label": "LLM Landscape 2025-2026 (text note)",
        "kind": "text",
    },
    {
        "input": (
            "## RAG — Retrieval-Augmented Generation: State of the Art 2025–2026\n\n"
            "### Modular & Agentic Phase\n"
            "Modern RAG has evolved from naive retrieve-then-generate into multi-stage, self-correcting "
            "pipelines. Agentic RAG embeds autonomous agents that apply reflection, planning, tool use, "
            "and multi-agent collaboration to iteratively refine retrieval strategies.\n\n"
            "### Adaptive Retrieval\n"
            "Adaptive mechanisms adjust retrieval depth based on query complexity, using reinforcement "
            "learning to select data sources in real time. Multi-stage re-ranking with semantic filters "
            "delivers ~15% precision improvement on legal documents.\n\n"
            "### Multimodal & Real-Time RAG\n"
            "RAG now retrieves images, videos, structured tables, and live sensor data. Future RAG "
            "integrates auto-updating knowledge graphs for real-time domains (legal rulings, market data).\n\n"
            "### Specialised Techniques\n"
            "- **RAFT** (Retrieval-Augmented Fine-Tuning): combines RAG + fine-tuning on synthetic datasets\n"
            "- **SELF-RAG**: dynamically decides when to retrieve and self-critiques outputs\n"
            "- **Long RAG**: processes longer retrieval units to preserve context for lengthy documents\n"
            "- **RA-ISF**: decomposes tasks into sub-modules to reduce hallucinations via iterative feedback\n"
            "- **GraphRAG**: combines vector search with knowledge graphs, achieving up to 99% search precision\n\n"
            "### Hybrid Search\n"
            "Dense (embedding) + sparse (BM25/SPLADE) fusion with Reciprocal Rank Fusion (RRF) is now "
            "the default for production RAG. Parent-child chunking retains summary context while enabling "
            "fine-grained passage retrieval.\n"
        ),
        "tags": ["rag", "retrieval", "graphrag", "agentic", "hybrid-search"],
        "label": "RAG State of the Art 2025-2026 (text note)",
        "kind": "text",
    },
    {
        "input": (
            "## Agentic AI Frameworks: Landscape 2025–2026\n\n"
            "### LangChain / LangGraph\n"
            "LangGraph (built on LangChain) models multi-agent workflows as directed cyclic graphs, "
            "enabling stateful, branching, and looping agent topologies. It is the fastest framework "
            "by latency across standard benchmarks and is most popular in production deployments. "
            "In late 2025 LangChain introduced DeepAgents — a batteries-included harness supporting "
            "long-horizon planning, tool-calling-in-a-loop, context offloading to filesystem, and "
            "sub-agent orchestration.\n\n"
            "### Microsoft AutoGen / AG2\n"
            "AutoGen (open-sourced by Microsoft Research, late 2023) is the default framework for "
            "multi-agent research. Agents collaborate, share information, and act autonomously. AG2 "
            "is the community-driven continuation with improved orchestration primitives.\n\n"
            "### CrewAI\n"
            "Role-based multi-agent framework where each agent has a specialised function (researcher, "
            "writer, critic, etc.). Best for rapid prototyping with clear task divisions.\n\n"
            "### Market Trajectory\n"
            "Gartner forecasts 33% of enterprise software applications will incorporate agentic AI by 2028 "
            "(vs <1% in 2024). Key patterns: ReAct, Reflexion, Plan-and-Execute, and LATS (Language "
            "Agent Tree Search).\n\n"
            "### OpenAI Agents SDK\n"
            "Released in early 2025, the OpenAI Agents SDK provides first-party primitives: Agents, "
            "Handoffs, and Guardrails, directly integrated with the Assistants API and tool-calling.\n"
        ),
        "tags": ["agentic", "langgraph", "autogen", "crewai", "multi-agent"],
        "label": "Agentic AI Frameworks 2025-2026 (text note)",
        "kind": "text",
    },
    {
        "input": (
            "## LLM Inference Optimization: Techniques & Benchmarks 2025–2026\n\n"
            "### Speculative Decoding\n"
            "A small draft model (1–7B) generates 3–12 candidate tokens per step; the target model "
            "verifies all in one parallel forward pass. Acceptance rates of 70–90% on domain tasks "
            "yield 2–3x speedups with zero quality loss (rejected tokens resampled from target). "
            "P-EAGLE (integrated in vLLM) extends this with parallel speculative decoding — K drafts "
            "in a single forward pass.\n\n"
            "### Quantization\n"
            "- FP16 → INT8: 2x memory reduction, ~50% cost cut, 95–99% accuracy retention\n"
            "- FP16 → INT4: 4x memory reduction, lower accuracy floor\n"
            "- Google TurboQuant (2026): KV-cache compressed to 3-bit, 6x memory reduction, zero accuracy loss\n"
            "- NVFP4 (NVIDIA): emerging ultra-low precision for Blackwell GPUs\n\n"
            "### Stacking Optimizations\n"
            "FP8 + Flash Attention 3 + continuous batching + speculative decoding on H100 → 5–8x "
            "cost efficiency vs naive FP16 + static batching. Effects are multiplicative, not additive: "
            "AMD MI300X benchmarks show 3.6x total speedup on Llama 3.1-405B with FP8 + speculative.\n\n"
            "### Serving Engines (H100, 100 concurrent requests)\n"
            "| Engine | Throughput | Cold Start |\n"
            "|---|---|---|\n"
            "| TensorRT-LLM | 2780 tok/s | 28 min |\n"
            "| SGLang | 2460 tok/s | ~60 s |\n"
            "| vLLM | 2400 tok/s | ~60 s |\n\n"
            "### KV Cache Optimizations\n"
            "PagedAttention (vLLM), RadixAttention (SGLang), and FlashInfer reduce KV cache memory "
            "waste and enable higher batch sizes. Prefix caching amortises prompt cost across repeated "
            "system prompts.\n"
        ),
        "tags": ["inference", "vllm", "quantization", "speculative-decoding", "optimization"],
        "label": "LLM Inference Optimization 2025-2026 (text note)",
        "kind": "text",
    },
    {
        "input": (
            "## Mixture of Experts (MoE) in Large Language Models: 2025–2026\n\n"
            "### Why MoE Dominates Frontier Models\n"
            "Since early 2025, nearly all leading frontier models use MoE designs. MoE decouples "
            "model capacity (total parameters) from compute per token: only a subset of 'expert' "
            "FFN layers is activated per token, keeping FLOPs/token constant while scaling knowledge.\n\n"
            "### Llama 4 MoE Architecture\n"
            "Llama 4 Maverick: 17B active parameters, 128 experts (total ~400B parameters). "
            "Beats GPT-4o and Gemini 2.0 Flash on broad multimodal benchmarks. Llama 4 Scout extends "
            "context to 10M tokens — possible partly due to reduced memory pressure from sparse activation.\n\n"
            "### Routing Advances\n"
            "- **MaxScore routing**: formulates expert selection as constrained optimisation, eliminating "
            "token dropping and padding waste from hard capacity constraints\n"
            "- **Similarity-preserving load balancing**: stabilises expert assignment for related inputs, "
            "prevents expert collapse, enables faster training\n"
            "- Research focus shifted from scale (more experts) to reliability (consistent routing "
            "under long training runs and deployment drift)\n\n"
            "### System-Level Optimizations\n"
            "Four critical axes: (1) hybrid parallel computing, (2) memory management, (3) fine-grained "
            "communication scheduling for expert all-to-all, (4) adaptive load balancing. On NVIDIA "
            "Blackwell NVL72, MoE models run 10x faster at 1/10th the token cost vs dense equivalents.\n\n"
            "### Interpretability\n"
            "MoE-Lens (ICLR 2025 Workshop) provides a three-pronged framework for understanding actual "
            "expert behaviour in deployed MoE LLMs — moving toward mechanistic interpretability.\n"
        ),
        "tags": ["moe", "architecture", "llama4", "routing", "efficiency"],
        "label": "MoE Architecture 2025-2026 (text note)",
        "kind": "text",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# QUERY SET  — ground-truth relevance tags (for precision computation)
# ─────────────────────────────────────────────────────────────────────────────

QUERIES: list[dict] = [
    {
        "query": "What is Llama 4's architecture, how many experts does it have, and what context length does it support?",
        "relevant_tags": {"llm", "meta", "multimodal", "moe", "llama4"},
        "label": "Llama 4 architecture",
    },
    {
        "query": "How does speculative decoding speed up LLM inference and what speedup can I expect?",
        "relevant_tags": {"inference", "speculative-decoding", "optimization"},
        "label": "Speculative decoding speedup",
    },
    {
        "query": "What are the main techniques in Agentic RAG and how does it differ from standard RAG?",
        "relevant_tags": {"rag", "agentic", "retrieval"},
        "label": "Agentic RAG techniques",
    },
    {
        "query": "What is mixture of experts and which recent models use it?",
        "relevant_tags": {"moe", "architecture", "nvidia"},
        "label": "MoE models overview",
    },
    {
        "query": "What are the top LLM inference serving engines and their throughput benchmarks?",
        "relevant_tags": {"inference", "vllm", "optimization"},
        "label": "Inference serving benchmarks",
    },
    {
        "query": "How did Claude and GPT-5 models evolve in 2025 in terms of context window?",
        "relevant_tags": {"llm", "claude", "gpt5", "survey"},
        "label": "Claude & GPT-5 evolution",
    },
    {
        "query": "What agentic AI framework should I use for multi-agent research vs production?",
        "relevant_tags": {"agentic", "langgraph", "autogen", "multi-agent"},
        "label": "Agent framework selection",
    },
    {
        "query": "How does quantization reduce LLM memory and what precision formats are available?",
        "relevant_tags": {"inference", "quantization", "optimization"},
        "label": "Quantization techniques",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class IngestRecord:
    label: str
    kind: str
    success: bool
    elapsed_s: float
    chunks: int = 0
    title: str = ""
    error: str = ""


@dataclass
class QueryRecord:
    label: str
    query: str
    success: bool
    elapsed_s: float
    iterations: int = 0
    results_count: int = 0
    top1_score: float = 0.0
    mean_score: float = 0.0
    scores: list[float] = field(default_factory=list)
    precision_at_k: float = 0.0   # fraction of top-k results from relevant docs
    converged: bool = False
    error: str = ""


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _compute_precision(result: RetrievalOutput, relevant_tags: set[str], doc_tags: dict[str, list[str]]) -> float:
    """Tag-based precision@k: fraction of final results whose source doc shares a tag with the query."""
    if not result.final_results:
        return 0.0
    hits = 0
    for r in result.final_results:
        doc_id = r.document_id
        tags = set(doc_tags.get(doc_id, []))
        if tags & relevant_tags:
            hits += 1
    return hits / len(result.final_results)


def _bar(value: float, max_val: float = 1.0, width: int = 20) -> str:
    filled = int(round(value / max_val * width)) if max_val > 0 else 0
    return "█" * filled + "░" * (width - filled)


def _color(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def _green(t: str) -> str: return _color(t, "32")
def _red(t: str) -> str:   return _color(t, "31")
def _yellow(t: str) -> str: return _color(t, "33")
def _bold(t: str) -> str:  return _color(t, "1")
def _cyan(t: str) -> str:  return _color(t, "36")
def _dim(t: str) -> str:   return _color(t, "2")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

class Benchmark:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.ingest_records: list[IngestRecord] = []
        self.query_records: list[QueryRecord] = []
        # maps document_id → tags (populated during ingest)
        self._doc_tags: dict[str, list[str]] = {}

    async def run_ingest(self, documents: list[dict]) -> None:
        print(_bold("\n╔══════════════════════════════════════════════════╗"))
        print(_bold("║          PHASE 1 — INGEST BENCHMARK             ║"))
        print(_bold("╚══════════════════════════════════════════════════╝\n"))

        for i, doc in enumerate(documents, 1):
            label = doc["label"]
            print(f"  [{i:02d}/{len(documents)}] {_cyan(label)} ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                result: AddResult = await self.kb.add(doc["input"], tags=doc.get("tags", []))
                elapsed = time.perf_counter() - t0
                self._doc_tags[result.document_id] = doc.get("tags", [])
                rec = IngestRecord(
                    label=label,
                    kind=doc["kind"],
                    success=True,
                    elapsed_s=elapsed,
                    chunks=result.child_chunks_stored,
                    title=result.title,
                )
                print(f"→ {_green('✓')} {result.child_chunks_stored} chunks  {elapsed:.1f}s  [{result.title[:50]}]")
            except Exception as exc:
                elapsed = time.perf_counter() - t0
                rec = IngestRecord(
                    label=label, kind=doc["kind"],
                    success=False, elapsed_s=elapsed, error=str(exc)[:120],
                )
                print(f"→ {_red('✗')} {rec.error}")
            self.ingest_records.append(rec)

    async def run_queries(self, queries: list[dict]) -> None:
        print(_bold("\n╔══════════════════════════════════════════════════╗"))
        print(_bold("║          PHASE 2 — RETRIEVAL BENCHMARK          ║"))
        print(_bold("╚══════════════════════════════════════════════════╝\n"))

        for i, q in enumerate(queries, 1):
            label = q["label"]
            print(f"  [{i:02d}/{len(queries)}] {_cyan(label)}")
            print(f"         Query: {_dim(q['query'][:90])}")
            t0 = time.perf_counter()
            try:
                output: RetrievalOutput = await self.kb.search(q["query"], k=5)
                elapsed = time.perf_counter() - t0

                scores = [r.score for r in output.final_results]
                iterations = len(output.iterations)
                converged = output.iterations[-1].converged if output.iterations else False
                precision = _compute_precision(output, q["relevant_tags"], self._doc_tags)
                top1   = scores[0] if scores else 0.0
                mean_s = statistics.mean(scores) if scores else 0.0

                rec = QueryRecord(
                    label=label,
                    query=q["query"],
                    success=True,
                    elapsed_s=elapsed,
                    iterations=iterations,
                    results_count=len(output.final_results),
                    top1_score=top1,
                    mean_score=mean_s,
                    scores=scores,
                    precision_at_k=precision,
                    converged=converged,
                )
                conv_marker = _green("converged") if converged else _yellow("iterated")
                empty_note = f"  {_yellow('(0 results above threshold)')}" if not scores else ""
                print(
                    f"         → {_green('✓')} {elapsed:.2f}s  "
                    f"{iterations} iter ({conv_marker})  "
                    f"top1={top1:.3f}  mean={mean_s:.3f}  "
                    f"P@k={precision:.2f}  results={len(scores)}{empty_note}\n"
                )
            except Exception as exc:
                import traceback
                elapsed = time.perf_counter() - t0
                rec = QueryRecord(
                    label=label, query=q["query"],
                    success=False, elapsed_s=elapsed, error=str(exc)[:120],
                )
                print(f"         → {_red('✗')} {rec.error}")
                traceback.print_exc()
                print()
            self.query_records.append(rec)

    # ── Dashboard ─────────────────────────────────────────────────────────────

    def print_dashboard(self) -> None:
        W = 60
        sep = "─" * W

        print(_bold(f"\n{'═' * W}"))
        print(_bold(f"{'KB-MAN BENCHMARK RESULTS':^{W}}"))
        print(_bold(f"{'═' * W}\n"))

        # ── Ingest metrics ───────────────────────────────────────────────────
        print(_bold("  INGEST METRICS"))
        print(f"  {sep}")

        ok = [r for r in self.ingest_records if r.success]
        fail = [r for r in self.ingest_records if not r.success]

        total_chunks  = sum(r.chunks for r in ok)
        times         = [r.elapsed_s for r in ok]
        url_times     = [r.elapsed_s for r in ok if r.kind == "url"]
        text_times    = [r.elapsed_s for r in ok if r.kind == "text"]
        chunk_counts  = [r.chunks for r in ok]

        def _stat(vals: list[float]) -> str:
            if not vals:
                return "n/a"
            return f"min={min(vals):.1f}s  avg={statistics.mean(vals):.1f}s  max={max(vals):.1f}s"

        print(f"  Documents attempted  : {len(self.ingest_records)}")
        print(f"  Succeeded            : {_green(str(len(ok)))}  Failed: {_red(str(len(fail)))}")
        print(f"  Total chunks stored  : {_bold(str(total_chunks))}")
        print(f"  Chunks per doc       : avg={statistics.mean(chunk_counts):.1f}  "
              f"min={min(chunk_counts)}  max={max(chunk_counts)}" if chunk_counts else "  n/a")
        print(f"  Time — URLs          : {_stat(url_times)}")
        print(f"  Time — Text notes    : {_stat(text_times)}")
        print(f"  Time — all           : {_stat(times)}")
        print(f"  Total ingest time    : {sum(times):.1f}s\n")

        print(f"  {'Label':<42} {'Kind':<5} {'Chunks':>6} {'Time':>7}  Status")
        print(f"  {'─'*42} {'─'*5} {'─'*6} {'─'*7}  ──────")
        for r in self.ingest_records:
            status = _green("✓ ok") if r.success else _red("✗ err")
            chunks = str(r.chunks) if r.success else "—"
            t      = f"{r.elapsed_s:.1f}s"
            print(f"  {r.label:<42} {r.kind:<5} {chunks:>6} {t:>7}  {status}")

        # ── Retrieval metrics ─────────────────────────────────────────────────
        print(f"\n  {sep}")
        print(_bold("  RETRIEVAL METRICS"))
        print(f"  {sep}")

        qok   = [r for r in self.query_records if r.success]
        qfail = [r for r in self.query_records if not r.success]

        q_times    = [r.elapsed_s for r in qok]
        q_iters    = [r.iterations for r in qok]
        q_top1     = [r.top1_score for r in qok]
        q_mean     = [r.mean_score for r in qok]
        q_prec     = [r.precision_at_k for r in qok]
        q_conv     = sum(1 for r in qok if r.converged)

        def _qstat(vals: list[float], fmt: str = ".3f") -> str:
            if not vals:
                return "n/a"
            f = f"{{:{fmt}}}"
            return (f"min=" + f.format(min(vals)) +
                    "  avg=" + f.format(statistics.mean(vals)) +
                    "  max=" + f.format(max(vals)))

        print(f"  Queries attempted    : {len(self.query_records)}")
        print(f"  Succeeded            : {_green(str(len(qok)))}  Failed: {_red(str(len(qfail)))}")
        print(f"  Converged on 1st try : {_green(str(q_conv))}/{len(qok)}")
        print(f"  Latency              : {_qstat(q_times, '.2f')}")
        print(f"  Iterations used      : {_qstat(q_iters, '.1f')}")
        print(f"  Top-1 score          : {_qstat(q_top1)}")
        print(f"  Mean score           : {_qstat(q_mean)}")
        print(f"  Precision@k (tag)    : {_qstat(q_prec)}")
        print(f"  Overall avg P@k      : {_bold(f'{statistics.mean(q_prec):.3f}') if q_prec else 'n/a'}\n")

        # Per-query table
        print(f"  {'Label':<34} {'Time':>6} {'Iter':>4} {'Top-1':>6} {'Mean':>6} {'P@k':>5}  Conv")
        print(f"  {'─'*34} {'─'*6} {'─'*4} {'─'*6} {'─'*6} {'─'*5}  ────")
        for r in self.query_records:
            if not r.success:
                print(f"  {r.label:<34} {'—':>6} {'—':>4} {'—':>6} {'—':>6} {'—':>5}  {_red('err')}")
                continue
            conv_s = _green("yes") if r.converged else _yellow("no")
            print(
                f"  {r.label:<34} {r.elapsed_s:>5.2f}s {r.iterations:>4d} "
                f"{r.top1_score:>6.3f} {r.mean_score:>6.3f} {r.precision_at_k:>5.2f}  {conv_s}"
            )

        # ── Score distribution histogram ──────────────────────────────────────
        all_scores = [s for r in qok for s in r.scores]
        if all_scores:
            print(f"\n  {sep}")
            print(_bold("  SCORE DISTRIBUTION  (all retrieved results)"))
            print(f"  {sep}")
            buckets = {f"{lo:.1f}–{lo+0.1:.1f}": 0 for lo in [i/10 for i in range(0, 10)]}
            for s in all_scores:
                key = f"{min(int(s * 10) / 10, 0.9):.1f}–{min(int(s * 10) / 10 + 0.1, 1.0):.1f}"
                buckets[key] = buckets.get(key, 0) + 1
            max_count = max(buckets.values()) or 1
            for bucket, count in sorted(buckets.items()):
                bar = _bar(count, max_count, 30)
                pct = count / len(all_scores) * 100
                print(f"  [{bucket}] {bar} {count:>3} ({pct:5.1f}%)")
            print(f"\n  Total scored results : {len(all_scores)}")
            print(f"  Mean  : {statistics.mean(all_scores):.4f}")
            print(f"  Stdev : {statistics.stdev(all_scores):.4f}" if len(all_scores) > 1 else "")
            print(f"  Min   : {min(all_scores):.4f}   Max: {max(all_scores):.4f}")

        # ── Summary verdict ───────────────────────────────────────────────────
        print(f"\n  {sep}")
        print(_bold("  SUMMARY"))
        print(f"  {sep}")

        avg_prec    = statistics.mean(q_prec) if q_prec else 0
        avg_latency = statistics.mean(q_times) if q_times else 0
        avg_iter    = statistics.mean(q_iters) if q_iters else 0

        verdict_parts = []
        if avg_prec >= 0.7:
            verdict_parts.append(_green(f"Good relevance (P@k={avg_prec:.2f})"))
        elif avg_prec >= 0.4:
            verdict_parts.append(_yellow(f"Moderate relevance (P@k={avg_prec:.2f})"))
        else:
            verdict_parts.append(_red(f"Low relevance (P@k={avg_prec:.2f})"))

        if avg_latency <= 3.0:
            verdict_parts.append(_green(f"Fast queries ({avg_latency:.2f}s avg)"))
        elif avg_latency <= 8.0:
            verdict_parts.append(_yellow(f"Moderate latency ({avg_latency:.2f}s avg)"))
        else:
            verdict_parts.append(_red(f"Slow queries ({avg_latency:.2f}s avg)"))

        if avg_iter <= 1.5:
            verdict_parts.append(_green(f"Efficient iteration ({avg_iter:.1f} avg)"))
        else:
            verdict_parts.append(_yellow(f"Multi-pass iteration ({avg_iter:.1f} avg)"))

        for v in verdict_parts:
            print(f"  • {v}")
        ingest_rate = f"{len(ok)/len(self.ingest_records)*100:.0f}%" if self.ingest_records else "n/a (skipped)"
        print(f"\n  {_bold('Overall ingest success rate')}: {ingest_rate}")
        print(f"  {_bold('Overall query success rate')} : {len(qok)/len(self.query_records)*100:.0f}%")
        print(f"\n{'═' * W}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

async def main(skip_ingest: bool, qdrant_path: Optional[str]) -> None:
    api_key  = os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY")
    base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model    = os.environ.get("KB_LLM_MODEL", "openai/gpt-4o-mini")

    if not api_key:
        print(_red("ERROR: OPENROUTER_API_KEY or OPENAI_API_KEY not set."))
        sys.exit(1)

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    cfg = Config.load()
    if qdrant_path:
        cfg.qdrant_path = qdrant_path

    kb = KnowledgeBase(
        llm_client=client,
        embedding_client=client,
        llm_model=model,
        config=cfg,
    )

    print(_bold(f"\n  KB-Man Benchmark — startup"))
    print(f"  Model       : {model}")
    print(f"  Base URL    : {base_url}")
    print(f"  Qdrant path : {cfg.qdrant_path}")
    print(f"  Corpus size : {len(DOCUMENTS)} docs  ({sum(1 for d in DOCUMENTS if d['kind']=='url')} URLs + {sum(1 for d in DOCUMENTS if d['kind']=='text')} text notes)")
    print(f"  Query set   : {len(QUERIES)} queries")

    await kb.startup()

    bench = Benchmark(kb)

    if not skip_ingest:
        await bench.run_ingest(DOCUMENTS)
    else:
        print(_yellow("\n  [--skip-ingest] Skipping ingestion phase, running queries only.\n"))
        # Precision computation needs doc_tags — reconstruct from Qdrant payloads
        texts = await kb._store.fetch_all_texts()
        # fetch_all_texts only returns text; we need payloads for tags.
        # Scroll summaries collection directly for tags.
        offset = None
        from kb.store.qdrant import COLLECTION_SUMMARIES
        while True:
            result, offset = await kb._store._client.scroll(
                COLLECTION_SUMMARIES, limit=100, offset=offset, with_payload=True,
            )
            for p in result:
                doc_id = p.payload.get("document_id", str(p.id))
                bench._doc_tags[doc_id] = p.payload.get("tags", [])
            if offset is None:
                break

    # ── Threshold sensitivity test ─────────────────────────────────────────
    # Show how many raw results exist before threshold filtering
    print(_bold("\n  Threshold sensitivity probe (raw searcher, before dedup+threshold)"))
    from kb.retrieval.searcher import Searcher as _S
    thresholds = [0.75, 0.60, 0.50]
    for thr in thresholds:
        cfg.similarity_threshold = thr
        hits = 0
        for q in QUERIES:
            raw = await kb._retriever._searcher.search(q["query"])
            kept = [r for r in raw if r.score >= thr]
            if kept:
                hits += 1
        cfg.similarity_threshold = 0.75  # restore
        print(f"    threshold={thr:.2f} → {hits}/{len(QUERIES)} queries return ≥1 result")

    # Run queries at default threshold (0.75), then at 0.55 so both are visible
    print(_bold("  Benchmark run A — default threshold (0.75)"))
    cfg.similarity_threshold = 0.75
    await bench.run_queries(QUERIES)

    print(_bold("  Benchmark run B — adjusted threshold (0.55)"))
    cfg.similarity_threshold = 0.55
    bench_b = Benchmark(kb)
    bench_b._doc_tags = bench._doc_tags
    await bench_b.run_queries(QUERIES)

    cfg.similarity_threshold = 0.75  # restore

    print(_bold("\n══ RUN A (threshold=0.75) ══"))
    bench.print_dashboard()
    print(_bold("\n══ RUN B (threshold=0.55) ══"))
    bench_b.print_dashboard()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="KB-Man benchmark suite")
    parser.add_argument("--skip-ingest", action="store_true",
                        help="Skip ingestion phase (use existing KB data)")
    parser.add_argument("--qdrant-path", default=None,
                        help="Override Qdrant storage path (default: ~/.kb/qdrant)")
    args = parser.parse_args()
    asyncio.run(main(args.skip_ingest, args.qdrant_path))
