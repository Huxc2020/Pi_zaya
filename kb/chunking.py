from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Block:
    kind: str  # "heading" | "text"
    text: str
    heading_path: str


def _parse_blocks(md: str) -> list[Block]:
    blocks: list[Block] = []
    heading_stack: list[tuple[int, str]] = []

    def current_heading_path() -> str:
        return " / ".join([t for _, t in heading_stack])

    lines = md.splitlines()
    buf: list[str] = []

    def flush_buf() -> None:
        nonlocal buf
        s = "\n".join(buf).strip("\n")
        if s.strip():
            blocks.append(Block(kind="text", text=s, heading_path=current_heading_path()))
        buf = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            # Flush previous text block
            flush_buf()

            level = len(stripped) - len(stripped.lstrip("#"))
            title = stripped[level:].strip()

            # Maintain stack
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))

            blocks.append(Block(kind="heading", text=stripped, heading_path=current_heading_path()))
            continue

        # Keep paragraph structure; blank lines separate paragraphs.
        if stripped == "":
            buf.append("")
        else:
            buf.append(line)

    flush_buf()
    return blocks


def _merge_blocks_into_chunks(
    blocks: list[Block],
    source_path: str,
    chunk_size: int,
    overlap: int,
) -> list[dict]:
    chunks: list[dict] = []
    cur: list[str] = []
    cur_len = 0
    cur_heading_path = ""

    def flush(force: bool = False) -> None:
        nonlocal cur, cur_len, cur_heading_path
        if not cur:
            return
        text = "\n".join(cur).strip()
        if not text:
            cur = []
            cur_len = 0
            return

        chunks.append(
            {
                "text": text,
                "meta": {
                    "source_path": source_path,
                    "heading_path": cur_heading_path,
                    "char_len": len(text),
                },
            }
        )

        if force or overlap <= 0:
            cur = []
            cur_len = 0
            return

        # Keep tail as overlap
        tail = text[-overlap:]
        cur = [tail]
        cur_len = len(tail)

    for b in blocks:
        if b.kind == "heading":
            # Start a new chunk at headings to help retrieval & navigation.
            flush(force=True)
            cur_heading_path = b.heading_path
            cur = [b.text]
            cur_len = len(b.text)
            continue

        if not cur:
            cur_heading_path = b.heading_path

        if cur_len + len(b.text) + 1 > chunk_size and cur_len > 200:
            flush(force=False)

        cur.append(b.text)
        cur_len += len(b.text) + 1

    flush(force=True)
    return chunks


def chunk_markdown(
    md: str,
    source_path: str,
    chunk_size: int = 1400,
    overlap: int = 200,
) -> list[dict]:
    blocks = _parse_blocks(md)
    return _merge_blocks_into_chunks(
        blocks=blocks,
        source_path=source_path,
        chunk_size=chunk_size,
        overlap=overlap,
    )

