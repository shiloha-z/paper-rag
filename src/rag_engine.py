from openai import OpenAI

from .config import Config
from .embedder import create_embedder, APIEmbedder, LocalEmbedder
from .vector_store import VectorStore
from .retriever import Retriever


SYSTEM_PROMPT = """You are a research assistant with access to a local paper knowledge base.
Answer the user's question based on the provided paper excerpts.

Guidelines:
- Base your answer primarily on the provided paper chunks.
- Cite papers by title and authors when you reference them.
- If the provided chunks don't contain the answer, say so — don't make up information.
- When comparing or synthesizing, explicitly mention which paper each claim comes from.
- Keep answers concise but thorough.
- If asked for a summary, focus on key findings and methodology."""


class RAGEngine:
    """End-to-end RAG query engine: retrieve → augment → generate."""

    def __init__(self, config: Config):
        self.config = config

        self.embedder = create_embedder(config.embedding)

        self.store = VectorStore(
            persist_path=config.vector_store.path,
            collection_name=config.vector_store.collection,
        )

        self.retriever = Retriever(self.embedder, self.store, config.retrieval)

        if not config.llm.api_key:
            raise ValueError("LLM API key is required. Set in config.yaml or PAPERRAG_LLM_API_KEY env var.")

        self.llm_client = OpenAI(
            api_key=config.llm.api_key,
            base_url=config.llm.api_base,
        )
        self.llm_model = config.llm.model

    def query(
        self,
        question: str,
        top_k: int | None = None,
        show_sources: bool = True,
        stream: bool = False,
    ) -> dict:
        """Run a full RAG query.

        Returns dict with keys: answer, sources (list of hit dicts)
        """
        # Retrieve
        hits = self.retriever.search(question, top_k=top_k)

        if not hits:
            return {
                "answer": "I couldn't find any relevant papers in the knowledge base for this question.",
                "sources": [],
            }

        # Build context from retrieved chunks
        context_parts = []
        for i, h in enumerate(hits):
            m = h.get("metadata", {})
            src_line = f"[Source {i+1}] Title: {m.get('title', 'Unknown')}"
            if m.get("authors"):
                src_line += f" | Authors: {m['authors']}"
            if m.get("year"):
                src_line += f" | Year: {m['year']}"
            context_parts.append(f"{src_line}\n{h['text']}")

        context = "\n\n---\n\n".join(context_parts)

        # Generate
        user_prompt = f"""Below are excerpts from academic papers relevant to the question.

QUESTION: {question}

PAPER EXCERPTS:
{context}

Please answer the question based on these excerpts. Cite sources by their [Source N] label."""

        response = self.llm_client.chat.completions.create(
            model=self.llm_model,
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=stream,
        )

        if stream:
            return {"answer": response, "sources": hits if show_sources else []}

        answer = response.choices[0].message.content
        return {
            "answer": answer,
            "sources": hits if show_sources else [],
        }

    def ask(self, question: str, top_k: int | None = None) -> str:
        """Simple ask: return just the answer string."""
        result = self.query(question, top_k=top_k, show_sources=False)
        return result["answer"]

    def chat(self):
        """Interactive chat loop (used by CLI)."""
        from rich.console import Console
        from rich.markdown import Markdown
        from rich.panel import Panel

        console = Console()
        console.print(Panel.fit(
            "[bold]Paper-RAG Chat[/bold]\n"
            "Ask questions about your paper library.\n"
            "Type [yellow]/search[/yellow] to search without LLM.\n"
            "Type [yellow]/papers[/yellow] to group results by paper.\n"
            "Type [yellow]/quit[/yellow] to exit.",
            border_style="blue",
        ))

        while True:
            try:
                user_input = console.input("\n[bold green]> [/bold green]").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye.[/dim]")
                break

            if not user_input:
                continue

            if user_input.lower() in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye.[/dim]")
                break

            if user_input.lower().startswith("/search "):
                query = user_input[8:].strip()
                hits = self.retriever.search(query)
                console.print(self.retriever.format_hits(hits))
                continue

            if user_input.lower().startswith("/papers "):
                query = user_input[8:].strip()
                grouped = self.retriever.search_by_paper(query)
                self._print_grouped(grouped)
                continue

            # RAG query
            with console.status("[bold yellow]Thinking...[/bold yellow]"):
                result = self.query(user_input, show_sources=True)

            console.print(Markdown(result["answer"]))

            if result["sources"]:
                console.print("\n[dim]Sources:[/dim]")
                seen = set()
                for s in result["sources"]:
                    title = s["metadata"].get("title", "Unknown")
                    if title not in seen:
                        seen.add(title)
                        authors = s["metadata"].get("authors", "")
                        year = s["metadata"].get("year", "")
                        console.print(
                            f"  [dim]• {title}"
                            + (f" — {authors}" if authors else "")
                            + (f" ({year})" if year else "")
                            + f"  [score: {s['score']}][/dim]"
                        )

    def _print_grouped(self, grouped: dict[str, list[dict]]):
        from rich.console import Console
        console = Console()

        for i, (title, chunks) in enumerate(grouped.items()):
            meta = chunks[0]["metadata"]
            author = meta.get("authors", "")
            year = meta.get("year", "")
            console.print(f"\n[bold]{i+1}. {title}[/bold]")
            if author or year:
                console.print(f"   [dim]{author}  {year}[/dim]")
            for ch in chunks[:2]:
                chunk_type = ch["metadata"].get("chunk_type", "")
                label = f" [{chunk_type}]" if chunk_type else ""
                console.print(f"   [dim]score: {ch['score']}{label}[/dim]")
                console.print(f"   {ch['text'][:200]}...")
