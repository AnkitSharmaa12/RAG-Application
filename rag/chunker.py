"""Markdown-aware chunker: one chunk per heading section, long sections split with overlap."""
import re
from dataclasses import dataclass, asdict

from . import config

HEADING_RE = re.compile(r"^(#{1,3})\s+(.*\S)\s*$")


@dataclass
class Chunk:
    id: int
    heading: str  # e.g. "Refund Policy (summary...) > 2. Milestone-Based Work"
    text: str  # heading path + body; this is what gets embedded and shown to the LLM

    def to_dict(self) -> dict:
        return asdict(self)


def _split_long(body: str, max_chars: int, overlap: int) -> list[str]:
    """Split on blank lines / lines, keeping tables and Q&A lines intact where possible."""
    if len(body) <= max_chars:
        return [body]
    pieces, current = [], ""
    for para in re.split(r"\n\s*\n", body):
        if current and len(current) + len(para) + 2 > max_chars:
            pieces.append(current.strip())
            current = current[-overlap:] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}" if current else para
        # a single giant paragraph (e.g. a long table): hard-split by lines
        while len(current) > max_chars * 1.5:
            cut = current.rfind("\n", 0, max_chars)
            cut = cut if cut > 0 else max_chars
            pieces.append(current[:cut].strip())
            current = current[max(cut - overlap, 0):]
    if current.strip():
        pieces.append(current.strip())
    return pieces


def chunk_markdown(md: str) -> list[Chunk]:
    doc_title = "MecroTech"
    sections: list[tuple[list[str], list[str]]] = []  # (heading path, body lines)
    path: list[str] = []
    in_front_matter = md.startswith("---")
    body_lines: list[str] = []

    def flush():
        if any(line.strip() for line in body_lines):
            sections.append((list(path), list(body_lines)))

    for i, line in enumerate(md.splitlines()):
        if in_front_matter:
            if i > 0 and line.strip() == "---":
                in_front_matter = False
            continue
        if line.strip() == "---":  # horizontal rule between sections
            continue
        m = HEADING_RE.match(line)
        if m:
            flush()
            body_lines = []
            level = len(m.group(1))
            title = m.group(2)
            if level == 1:
                path = []  # document title; chunks are prefixed with "MecroTech" instead
            else:
                path = path[: level - 2] + [title]
        else:
            body_lines.append(line)
    flush()

    chunks: list[Chunk] = []
    for heading_path, lines in sections:
        if not heading_path:  # preamble before the first ## section (usage notes for humans)
            continue
        heading = " > ".join([doc_title, *heading_path])
        body = "\n".join(lines).strip()
        for piece in _split_long(body, config.MAX_CHUNK_CHARS, config.CHUNK_OVERLAP_CHARS):
            chunks.append(Chunk(id=len(chunks), heading=heading, text=f"[{heading}]\n{piece}"))
    return chunks
