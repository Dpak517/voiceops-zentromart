"""Load kb/*.md and split into one chunk per '## [SECTION-ID] Heading' section."""
import os
import re
from dataclasses import dataclass

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
KB_DIR = os.path.join(ROOT, "kb")

_SECTION = re.compile(r"^##\s+\[([^\]]+)\]\s*(.*)$")


@dataclass
class Chunk:
    id: str
    doc_id: str
    doc_type: str
    status: str
    version: str
    effective_from: str
    doc_title: str
    heading: str
    text: str
    note: str = ""          # e.g. "> This version was replaced by version 4 ..."
    source_file: str = ""

    def search_text(self) -> str:
        return f"{self.doc_title}. {self.heading}. {self.heading}. {self.text}"

    def render(self) -> str:
        """Text shown to the LLM, with the metadata it needs to apply precedence."""
        tag = f"{self.doc_type} - {self.status} - v{self.version} - effective {self.effective_from}"
        out = f"[{self.id}] {self.doc_title} > {self.heading}  ({tag})\n"
        if self.status != "current":
            out += f"WARNING: this document is {self.status.upper()}. {self.note}\n"
        if self.doc_type == "marketing":
            out += "NOTE: marketing text never overrides a policy.\n"
        return out + self.text


def _front_matter(raw):
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, re.S)
    if not m:
        return {}, raw
    meta = {}
    for line in m.group(1).splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            meta[k.strip()] = v.strip()
    return meta, m.group(2)


def load_chunks(kb_dir: str = KB_DIR) -> list:
    chunks = []
    for name in sorted(os.listdir(kb_dir)):
        if not name.endswith(".md"):
            continue
        raw = open(os.path.join(kb_dir, name), encoding="utf-8").read().replace("\r\n", "\n")
        meta, body = _front_matter(raw)
        note_lines, cur_id, cur_head, cur_lines = [], None, "", []

        def flush():
            if cur_id:
                chunks.append(Chunk(
                    id=cur_id, doc_id=meta.get("doc_id", ""), doc_type=meta.get("doc_type", ""),
                    status=meta.get("status", "current"), version=meta.get("version", ""),
                    effective_from=meta.get("effective_from", ""), doc_title=meta.get("title", ""),
                    heading=cur_head, text="\n".join(cur_lines).strip(),
                    note=" ".join(note_lines).strip(), source_file=name))

        for line in body.splitlines():
            m = _SECTION.match(line)
            if m:
                flush()
                cur_id, cur_head, cur_lines = m.group(1).strip(), m.group(2).strip(), []
            elif cur_id is None:
                if line.startswith(">"):
                    note_lines.append(line.lstrip("> ").strip())
            else:
                cur_lines.append(line)
        flush()
    return chunks


def catalog_labels(chunks) -> dict:
    """Parse the RET-2 table: catalog code -> category label (e.g. fashion -> 'Clothing, footwear and bags')."""
    for c in chunks:
        if c.id == "RET-2":
            return {m.group(2): m.group(1).strip() for m in
                    re.finditer(r"\|\s*([^|]+?)\s*\|[^|]*\|\s*`(\w+)`\s*\|", c.text)}
    return {}
