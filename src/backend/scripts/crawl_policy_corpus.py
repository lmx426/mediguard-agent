"""Crawl official policy sources into the stage-1 policy corpus.

This script uses Crawl4AI for JS-rendered page loading and Markdown conversion,
then writes raw HTML, clean Markdown, downloaded attachment files, and a JSONL
manifest. It intentionally avoids LLM filtering, LLM rewriting, embedding, and
vector-store writes.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import html as html_lib
import json
import re
import sys
import urllib.parse
import urllib.request
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.backend.infrastructure.policy_corpus import (  # noqa: E402
    PolicyAttachmentRecord,
    PolicySourceManifestRecord,
    load_manifest,
    sha256_file,
    upsert_manifest_record,
    write_manifest_records,
)


DEFAULT_CORPUS_ROOT = PROJECT_ROOT / "src" / "backend" / "policy_corpus"
DEFAULT_SEED_CONFIG = DEFAULT_CORPUS_ROOT / "seeds" / "p0_sources.yaml"
DEFAULT_MANIFEST_PATH = DEFAULT_CORPUS_ROOT / "manifest" / "policy_source_manifest.jsonl"
ATTACHMENT_CONTENT_TYPES = {
    "application/pdf",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-works",
    "application/octet-stream",
    "application/zip",
    "text/csv",
    "application/json",
    "image/png",
    "image/jpeg",
}


@dataclass(frozen=True)
class SeedSource:
    source_id: str
    title: str
    jurisdiction: str
    policy_domain: str
    doc_type: str
    legal_weight: str
    issuing_authority: str | None
    url: str
    case_tags: list[str]


@dataclass(frozen=True)
class QueueItem:
    url: str
    seed: SeedSource
    depth: int
    link_text: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Crawl official policy sources into src/backend/policy_corpus."
    )
    parser.add_argument(
        "--seed-config",
        type=Path,
        default=DEFAULT_SEED_CONFIG,
        help="YAML seed config path.",
    )
    parser.add_argument(
        "--corpus-root",
        type=Path,
        default=DEFAULT_CORPUS_ROOT,
        help="Policy corpus root directory.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest JSONL output path.",
    )
    parser.add_argument(
        "--crawl4ai-root",
        type=Path,
        default=Path("F:/crawl4ai"),
        help="Local Crawl4AI project root to add to sys.path.",
    )
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-total-pages", type=int, default=None)
    parser.add_argument("--max-pages-per-seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-crawl URLs already in manifest.")
    parser.add_argument("--validate-manifest", action="store_true")
    parser.add_argument(
        "--prune-noise",
        action="store_true",
        help="Remove non-seed records whose title matches exclude keywords.",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("PyYAML is required to read seed YAML files.") from exc

    with path.open("r", encoding="utf-8") as file_obj:
        payload = yaml.safe_load(file_obj)
    if not isinstance(payload, dict):
        raise ValueError(f"Seed config must be a YAML object: {path}")
    return payload


def load_seed_sources(payload: dict[str, Any]) -> list[SeedSource]:
    sources = payload.get("sources", [])
    if not isinstance(sources, list):
        raise ValueError("Seed config field 'sources' must be a list.")

    result: list[SeedSource] = []
    for raw in sources:
        result.append(
            SeedSource(
                source_id=str(raw["source_id"]),
                title=str(raw["title"]),
                jurisdiction=str(raw["jurisdiction"]),
                policy_domain=str(raw["policy_domain"]),
                doc_type=str(raw["doc_type"]),
                legal_weight=str(raw["legal_weight"]),
                issuing_authority=raw.get("issuing_authority"),
                url=str(raw["url"]),
                case_tags=list(raw.get("case_tags", [])),
            )
        )
    return result


def import_crawl4ai(crawl4ai_root: Path):
    if crawl4ai_root.exists() and str(crawl4ai_root) not in sys.path:
        sys.path.insert(0, str(crawl4ai_root))

    try:
        from crawl4ai import (  # type: ignore
            AsyncWebCrawler,
            BM25ContentFilter,
            BrowserConfig,
            CacheMode,
            CrawlerRunConfig,
            DefaultMarkdownGenerator,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Crawl4AI is not importable. Pass --crawl4ai-root or install crawl4ai."
        ) from exc
    return {
        "AsyncWebCrawler": AsyncWebCrawler,
        "BM25ContentFilter": BM25ContentFilter,
        "BrowserConfig": BrowserConfig,
        "CacheMode": CacheMode,
        "CrawlerRunConfig": CrawlerRunConfig,
        "DefaultMarkdownGenerator": DefaultMarkdownGenerator,
    }


def normalize_url(url: str, base_url: str | None = None) -> str | None:
    if base_url:
        url = urllib.parse.urljoin(base_url, url)
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None

    query_pairs = [
        (key, value)
        for key, value in urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith(("utm_", "spm", "from"))
    ]
    clean_query = urllib.parse.urlencode(query_pairs, doseq=True)
    clean = parsed._replace(fragment="", query=clean_query)
    return urllib.parse.urlunparse(clean)


def host_of(url: str) -> str:
    return urllib.parse.urlparse(url).netloc.lower()


def domain_allowed(url: str, allowed_domains: Iterable[str]) -> bool:
    host = host_of(url)
    for domain in allowed_domains:
        domain = domain.lower()
        if host == domain or host.endswith("." + domain):
            return True
    return False


def same_domain(url: str, seed_url: str) -> bool:
    return host_of(url) == host_of(seed_url)


def extension_of_url(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    if parsed.path.lower().endswith("/downfile.jsp") or "downfile.jsp" in parsed.path.lower():
        return ".download"
    if "downloadfilebyuser.html" in parsed.path.lower():
        return ".download"
    suffix = Path(urllib.parse.unquote(parsed.path)).suffix.lower()
    if suffix:
        return suffix

    query = urllib.parse.parse_qs(parsed.query)
    for values in query.values():
        for value in values:
            suffix = Path(urllib.parse.unquote(value)).suffix.lower()
            if suffix:
                return suffix
    return ""


def is_attachment_url(url: str, attachment_exts: set[str]) -> bool:
    ext = extension_of_url(url)
    return ext in attachment_exts or ext == ".download"


def keyword_hit_count(text: str, keywords: Iterable[str]) -> int:
    normalized = text.lower()
    return sum(1 for keyword in keywords if keyword.lower() in normalized)


def should_enqueue_link(
    url: str,
    item: QueueItem,
    crawl_policy: dict[str, Any],
    keywords: list[str],
    link_text: str,
) -> bool:
    if item.depth + 1 > int(crawl_policy["max_depth"]):
        return False
    if not domain_allowed(url, crawl_policy.get("allowed_domains", [])):
        return False
    if crawl_policy.get("same_domain_only", True) and not same_domain(url, item.seed.url):
        return False

    if item.depth == 0:
        return True

    return keyword_hit_count(url + " " + link_text, keywords) > 0


def should_save_page(
    url: str,
    seed: SeedSource,
    depth: int,
    title: str,
    markdown: str,
    keywords: list[str],
) -> bool:
    if depth == 0 and normalize_url(url) == normalize_url(seed.url):
        return True
    return keyword_hit_count(" ".join([url, title, markdown[:12000]]), keywords) > 0


def safe_filename(value: str, suffix: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    if not normalized:
        normalized = hashlib.sha1(value.encode("utf-8")).hexdigest()[:12]
    return f"{normalized[:100]}{suffix}"


def attachment_filename_from_url(url: str, fallback_stem: str | None = None) -> str:
    """Build a stable attachment filename, including query-hosted downloads."""

    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)

    candidates: list[str] = []
    for key in ("filename", "fileName", "file", "name"):
        candidates.extend(query.get(key, []))

    path_name = Path(urllib.parse.unquote(parsed.path)).name
    if path_name and path_name.lower() not in {"downfile.jsp", "downloadfilebyuser.html"}:
        candidates.append(path_name)

    for candidate in candidates:
        candidate = urllib.parse.unquote(str(candidate)).strip()
        suffix = Path(candidate).suffix.lower()
        if candidate and suffix:
            return safe_filename(candidate, "")

    ext = extension_of_url(url)
    if ext == ".download":
        ext = ""
    stem = fallback_stem or hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return safe_filename(f"{stem}_{digest}", ext)


def discovered_source_id(seed_id: str, url: str, depth: int) -> str:
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    return f"{seed_id}_d{depth}_{digest}"


def extract_title(result: Any, fallback: str) -> str:
    metadata = getattr(result, "metadata", None) or {}
    for key in ("title", "og:title"):
        value = metadata.get(key)
        if value:
            return str(value).strip()

    html = getattr(result, "html", "") or ""
    match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
    if match:
        return re.sub(r"\s+", " ", match.group(1)).strip()
    return fallback


def parse_policy_date(text: str, labels: Iterable[str]) -> str | None:
    label_pattern = "|".join(re.escape(label) for label in labels)
    pattern = re.compile(
        rf"(?:{label_pattern})\s*[:：]?\s*"
        r"(?P<year>20\d{2})[年\-/.]"
        r"(?P<month>\d{1,2})[月\-/.]"
        r"(?P<day>\d{1,2})日?",
        re.S,
    )
    match = pattern.search(text[:20000])
    if not match:
        return None
    year = int(match.group("year"))
    month = int(match.group("month"))
    day = int(match.group("day"))
    try:
        return f"{year:04d}-{month:02d}-{day:02d}"
    except ValueError:
        return None


def infer_status(text: str, default: str) -> str:
    head = text[:20000]
    if any(marker in head for marker in ("废止", "失效", "停止执行")):
        return "expired"
    if any(marker in head for marker in ("现行有效", "继续有效", "有效")):
        return "active"
    return default


def markdown_text(result: Any) -> str:
    markdown = getattr(result, "markdown", None)
    html = getattr(result, "html", "") or ""
    if markdown is None:
        return extract_markdown_from_html(html)

    fit_markdown = getattr(markdown, "fit_markdown", None)
    raw_markdown = str(markdown)
    chosen = fit_markdown if fit_markdown and len(fit_markdown.strip()) >= 80 else raw_markdown
    if not chosen or len(chosen.strip()) < 80:
        chosen = extract_markdown_from_html(html)
    chosen = re.sub(r"\n{3,}", "\n\n", chosen)
    return chosen.strip()


def extract_markdown_from_html(html: str) -> str:
    """Deterministically extract policy body text when Crawl4AI filtering is empty."""

    if not html:
        return ""

    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = re.sub(r"<(script|style)[^>]*>.*?</\1>", "", html, flags=re.I | re.S)
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.I)
        text = re.sub(r"</p>|</div>|</h[1-6]>", "\n", text, flags=re.I)
        text = re.sub(r"<[^>]+>", "", text)
        return normalize_markdown_text(text)

    soup = BeautifulSoup(html, "lxml")
    for selector in [
        ".TRS_UEDITOR",
        ".trs_paper_default",
        ".view",
        ".article",
        ".article-content",
        ".content",
        "#zoom",
        "article",
        "main",
    ]:
        node = soup.select_one(selector)
        if node and len(node.get_text("", strip=True)) >= 80:
            return html_node_to_markdown(node)

    body = soup.body or soup
    return html_node_to_markdown(body)


def html_node_to_markdown(node: Any) -> str:
    lines: list[str] = []
    for element in node.find_all(["h1", "h2", "h3", "h4", "p", "li", "tr"], recursive=True):
        text = element.get_text(" ", strip=True)
        if not text:
            continue
        if element.name == "h1":
            lines.append(f"# {text}")
        elif element.name == "h2":
            lines.append(f"## {text}")
        elif element.name in {"h3", "h4"}:
            lines.append(f"### {text}")
        elif element.name == "li":
            lines.append(f"- {text}")
        elif element.name == "tr":
            cells = [cell.get_text(" ", strip=True) for cell in element.find_all(["th", "td"])]
            if cells:
                lines.append(" | ".join(cells))
        else:
            lines.append(text)
    return normalize_markdown_text("\n\n".join(lines))


def normalize_markdown_text(text: str) -> str:
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def front_matter(record: dict[str, Any]) -> str:
    lines = ["---"]
    for key, value in record.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False)}")
    lines.append("---")
    return "\n".join(lines) + "\n\n"


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def save_raw_html(corpus_root: Path, jurisdiction: str, source_id: str, html: str) -> Path:
    raw_dir = corpus_root / "raw" / jurisdiction
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / safe_filename(source_id, ".html")
    raw_path.write_text(html, encoding="utf-8", newline="\n")
    return raw_path


def save_clean_markdown(
    corpus_root: Path,
    jurisdiction: str,
    source_id: str,
    metadata: dict[str, Any],
    markdown: str,
) -> Path:
    clean_dir = corpus_root / "clean" / "markdown" / jurisdiction
    clean_dir.mkdir(parents=True, exist_ok=True)
    clean_path = clean_dir / safe_filename(source_id, ".md")
    clean_path.write_text(front_matter(metadata) + markdown + "\n", encoding="utf-8", newline="\n")
    return clean_path


def extract_links(result: Any, base_url: str) -> list[tuple[str, str]]:
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    raw_links = getattr(result, "links", None) or {}
    for group in ("internal", "external"):
        for link in raw_links.get(group, []) or []:
            href = link.get("href") or link.get("url")
            if not href:
                continue
            normalized = normalize_url(html_lib.unescape(str(href)), base_url)
            if not normalized:
                continue
            if normalized in seen:
                continue
            seen.add(normalized)
            text = html_lib.unescape(str(link.get("text") or link.get("title") or "")).strip()
            links.append((normalized, text))

    html = getattr(result, "html", "") or ""
    for match in re.finditer(r'<a\s+[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', html, re.I | re.S):
        normalized = normalize_url(html_lib.unescape(match.group(1)), base_url)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        text = html_lib.unescape(re.sub(r"<[^>]+>", "", match.group(2)))
        text = re.sub(r"\s+", " ", text).strip()
        links.append((normalized, text))

    for match in re.finditer(r"https?://[^\s\"'<>]+", html):
        normalized = normalize_url(html_lib.unescape(match.group(0)), base_url)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        links.append((normalized, ""))
    return links


def download_attachment(url: str, output_dir: Path) -> PolicyAttachmentRecord | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = attachment_filename_from_url(url)
    target_path = output_dir / filename
    if target_path.exists():
        stem = target_path.stem
        suffix = target_path.suffix
        digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
        target_path = output_dir / safe_filename(f"{stem}_{digest}", suffix)
        filename = target_path.name

    content_type, byte_count = download_url_to_path(url, target_path)

    return PolicyAttachmentRecord(
        url=url,
        filename=filename,
        raw_path=relative_path(target_path),
        content_type=content_type,
        bytes=byte_count,
        sha256=sha256_file(target_path),
        fetched_at=datetime.now(timezone.utc),
    )


def download_url_to_path(url: str, target_path: Path) -> tuple[str, int]:
    """Download a URL to a target path and return content type and byte count."""

    target_path.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": "MediGuardPolicyCorpusCrawler/1.0",
            "Accept": "*/*",
        },
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        content = response.read()
        content_type = response.headers.get("Content-Type")

    target_path.write_bytes(content)
    if not content_type or not any(
            content_type.lower().startswith(allowed) for allowed in ATTACHMENT_CONTENT_TYPES
    ):
        content_type = content_type or "unknown"

    return content_type, len(content)


def save_direct_attachment_seed(
    item: QueueItem,
    corpus_root: Path,
    manifest_path: Path,
    metadata_defaults: dict[str, Any],
) -> tuple[bool, list[tuple[str, str]]]:
    """Save a seed whose URL points directly to a source attachment."""

    ext = extension_of_url(item.url) or ".bin"
    raw_dir = corpus_root / "raw" / item.seed.jurisdiction
    if ext == ".download":
        filename = attachment_filename_from_url(item.url, item.seed.source_id)
        raw_path = raw_dir / filename
    else:
        raw_path = raw_dir / safe_filename(item.seed.source_id, ext)
    try:
        content_type, byte_count = download_url_to_path(item.url, raw_path)
    except Exception as exc:  # noqa: BLE001 - keep crawl run alive.
        print(f"[WARN] direct attachment failed: {item.url} :: {exc}")
        return False, []

    record = PolicySourceManifestRecord(
        source_id=item.seed.source_id,
        title=item.seed.title,
        jurisdiction=item.seed.jurisdiction,  # type: ignore[arg-type]
        policy_domain=item.seed.policy_domain,
        url=item.url,
        issuing_authority=item.seed.issuing_authority,
        publish_date=None,
        effective_date=None,
        expiry_date=None,
        status=metadata_defaults.get("status", "unknown"),  # type: ignore[arg-type]
        version_note=(
            metadata_defaults.get("version_note")
            or "Direct source attachment; version fields require manual review."
        ),
        supersedes=list(metadata_defaults.get("supersedes", [])),
        doc_type=item.seed.doc_type,  # type: ignore[arg-type]
        legal_weight=item.seed.legal_weight,  # type: ignore[arg-type]
        case_tags=item.seed.case_tags,
        raw_path=relative_path(raw_path),
        clean_path=None,
        sha256=sha256_file(raw_path),
        fetched_at=datetime.now(timezone.utc),
        attachments=[],
        crawl_metadata={
            "depth": item.depth,
            "seed_source_id": item.seed.source_id,
            "link_text": item.link_text,
            "content_type": content_type,
            "bytes": byte_count,
            "cleaning": {
                "raw_file_preserved": True,
                "markdown_generated": False,
                "llm_filter": False,
                "llm_rewrite": False,
                "attachment_parse": False,
            },
        },
    )
    upsert_manifest_record(manifest_path, record)
    print(f"[OK] {item.seed.source_id} -> {relative_path(raw_path)}")
    return True, []


def parse_date_or_none(value: str | None):
    if not value:
        return None
    return value


async def crawl_one(
    crawler: Any,
    run_config: Any,
    item: QueueItem,
    corpus_root: Path,
    manifest_path: Path,
    crawl_policy: dict[str, Any],
    metadata_defaults: dict[str, Any],
    keywords: list[str],
) -> tuple[bool, list[tuple[str, str]]]:
    result = await crawler.arun(url=item.url, config=run_config)
    if not getattr(result, "success", False):
        print(f"[WARN] crawl failed: {item.url} :: {getattr(result, 'error_message', '')}")
        return False, []

    links = extract_links(result, item.url)

    title = extract_title(result, item.seed.title)
    markdown = markdown_text(result)
    html = getattr(result, "html", "") or ""

    if not should_save_page(item.url, item.seed, item.depth, title, markdown, keywords):
        return False, links

    source_id = (
        item.seed.source_id
        if item.depth == 0 and normalize_url(item.url) == normalize_url(item.seed.url)
        else discovered_source_id(item.seed.source_id, item.url, item.depth)
    )
    raw_path = save_raw_html(corpus_root, item.seed.jurisdiction, source_id, html)

    searchable_text = "\n".join([title, markdown, html[:20000]])
    publish_date = parse_policy_date(searchable_text, ["发布日期", "发布时间", "发文日期", "成文日期"])
    effective_date = parse_policy_date(searchable_text, ["施行日期", "执行日期", "生效日期", "自"])
    expiry_date = parse_policy_date(searchable_text, ["废止日期", "失效日期"])
    status = infer_status(searchable_text, metadata_defaults.get("status", "unknown"))

    attachment_records: list[PolicyAttachmentRecord] = []
    if crawl_policy.get("download_attachments", True):
        attachment_exts = set(crawl_policy.get("attachment_extensions", []))
        attachment_dir = corpus_root / "raw" / "attachments" / item.seed.jurisdiction / source_id
        for link_url, _link_text in links:
            if not is_attachment_url(link_url, attachment_exts):
                continue
            if not domain_allowed(link_url, crawl_policy.get("allowed_domains", [])):
                continue
            try:
                record = download_attachment(link_url, attachment_dir)
            except Exception as exc:  # noqa: BLE001 - keep crawl running and report.
                print(f"[WARN] attachment failed: {link_url} :: {exc}")
                continue
            if record:
                attachment_records.append(record)

    clean_path: Path | None = None
    if crawl_policy.get("save_clean_markdown", True):
        clean_metadata = {
            "source_id": source_id,
            "title": title,
            "jurisdiction": item.seed.jurisdiction,
            "policy_domain": item.seed.policy_domain,
            "url": item.url,
            "depth": item.depth,
            "seed_source_id": item.seed.source_id,
            "fetched_at": datetime.now(timezone.utc).isoformat(),
        }
        clean_path = save_clean_markdown(
            corpus_root,
            item.seed.jurisdiction,
            source_id,
            clean_metadata,
            markdown,
        )

    record = PolicySourceManifestRecord(
        source_id=source_id,
        title=title,
        jurisdiction=item.seed.jurisdiction,  # type: ignore[arg-type]
        policy_domain=item.seed.policy_domain,
        url=item.url,
        issuing_authority=item.seed.issuing_authority,
        publish_date=parse_date_or_none(publish_date),
        effective_date=parse_date_or_none(effective_date),
        expiry_date=parse_date_or_none(expiry_date),
        status=status,  # type: ignore[arg-type]
        version_note=metadata_defaults.get(
            "version_note",
            "Dates and replacement relation require deterministic extraction or manual review.",
        ),
        supersedes=list(metadata_defaults.get("supersedes", [])),
        doc_type=item.seed.doc_type,  # type: ignore[arg-type]
        legal_weight=item.seed.legal_weight,  # type: ignore[arg-type]
        case_tags=item.seed.case_tags,
        raw_path=relative_path(raw_path),
        clean_path=relative_path(clean_path) if clean_path else None,
        sha256=sha256_file(raw_path),
        fetched_at=datetime.now(timezone.utc),
        attachments=attachment_records,
        crawl_metadata={
            "depth": item.depth,
            "seed_source_id": item.seed.source_id,
            "link_text": item.link_text,
            "status_code": getattr(result, "status_code", None),
            "redirected_url": getattr(result, "redirected_url", None),
            "keyword_hits": keyword_hit_count(searchable_text, keywords),
            "cleaning": {
                "raw_html_preserved": True,
                "markdown_generated": bool(clean_path),
                "llm_filter": False,
                "llm_rewrite": False,
                "attachment_parse": False,
            },
        },
    )
    upsert_manifest_record(manifest_path, record)
    print(f"[OK] {source_id} -> {relative_path(raw_path)}")
    return True, links


async def crawl_all(args: argparse.Namespace) -> None:
    payload = load_yaml(args.seed_config)
    crawl_policy = dict(payload.get("crawl_policy", {}))
    metadata_defaults = dict(payload.get("metadata_defaults", {}))

    if args.max_depth is not None:
        crawl_policy["max_depth"] = args.max_depth
    if args.max_total_pages is not None:
        crawl_policy["max_total_pages_first_run"] = args.max_total_pages
    if args.max_pages_per_seed is not None:
        crawl_policy["max_pages_per_seed"] = args.max_pages_per_seed

    sources = load_seed_sources(payload)
    keywords = list(crawl_policy.get("include_keywords", []))
    attachment_exts = set(crawl_policy.get("attachment_extensions", []))

    if args.dry_run:
        print(f"Seed config: {args.seed_config}")
        print(f"Corpus root: {args.corpus_root}")
        print(f"Manifest: {args.manifest_path}")
        print(
            "Crawl limits: "
            f"depth={crawl_policy.get('max_depth')}, "
            f"per_seed={crawl_policy.get('max_pages_per_seed')}, "
            f"total={crawl_policy.get('max_total_pages_first_run')}"
        )
        for source in sources:
            print(f"- {source.source_id}: {source.url}")
        return

    crawl4ai = import_crawl4ai(args.crawl4ai_root)
    cache_mode = crawl4ai["CacheMode"].BYPASS
    if str(crawl_policy.get("cache_mode", "")).lower() == "enabled":
        cache_mode = crawl4ai["CacheMode"].ENABLED

    bm25_filter = crawl4ai["BM25ContentFilter"](
        user_query=" ".join(keywords),
        bm25_threshold=0.8,
        use_stemming=False,
    )
    markdown_generator = crawl4ai["DefaultMarkdownGenerator"](
        content_filter=bm25_filter,
        options={
            "ignore_links": False,
            "body_width": 0,
        },
    )
    browser_config = crawl4ai["BrowserConfig"](
        headless=True,
        java_script_enabled=bool(crawl_policy.get("render_js", True)),
        text_mode=False,
        verbose=False,
        accept_downloads=False,
    )
    run_config = crawl4ai["CrawlerRunConfig"](
        markdown_generator=markdown_generator,
        cache_mode=cache_mode,
        wait_until=crawl_policy.get("wait_until", "domcontentloaded"),
        page_timeout=int(crawl_policy.get("request_timeout_ms", 60000)),
        delay_before_return_html=float(
            crawl_policy.get("delay_before_return_html_seconds", 0.5)
        ),
        screenshot=bool(crawl_policy.get("screenshot", False)),
        pdf=bool(crawl_policy.get("pdf_snapshot", False)),
        remove_forms=True,
        excluded_tags=["nav", "footer", "header", "aside", "script", "style", "form", "iframe", "noscript"],
        verbose=False,
    )

    args.corpus_root.mkdir(parents=True, exist_ok=True)
    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)

    _existing_records = load_manifest(args.manifest_path)
    visited: set[str] = set()
    total_saved_or_seen = 0
    max_total = int(crawl_policy.get("max_total_pages_first_run", 1200))
    max_per_seed = int(crawl_policy.get("max_pages_per_seed", 120))

    async with crawl4ai["AsyncWebCrawler"](config=browser_config) as crawler:
        for seed in sources:
            seed_seen = 0
            queue: deque[QueueItem] = deque([QueueItem(url=seed.url, seed=seed, depth=0)])
            while queue and seed_seen < max_per_seed and total_saved_or_seen < max_total:
                item = queue.popleft()
                normalized = normalize_url(item.url)
                if not normalized:
                    continue
                if normalized in visited:
                    continue
                visited.add(normalized)

                item = QueueItem(url=normalized, seed=item.seed, depth=item.depth, link_text=item.link_text)
                seed_seen += 1
                total_saved_or_seen += 1
                if is_attachment_url(normalized, attachment_exts):
                    saved, links = save_direct_attachment_seed(
                        item=item,
                        corpus_root=args.corpus_root,
                        manifest_path=args.manifest_path,
                        metadata_defaults=metadata_defaults,
                    )
                    _ = saved
                    _ = links
                    continue

                saved, links = await crawl_one(
                    crawler=crawler,
                    run_config=run_config,
                    item=item,
                    corpus_root=args.corpus_root,
                    manifest_path=args.manifest_path,
                    crawl_policy=crawl_policy,
                    metadata_defaults=metadata_defaults,
                    keywords=keywords,
                )
                _ = saved

                for link_url, link_text in links:
                    if is_attachment_url(link_url, attachment_exts):
                        continue
                    if link_url in visited:
                        continue
                    if should_enqueue_link(link_url, item, crawl_policy, keywords, link_text):
                        queue.append(
                            QueueItem(
                                url=link_url,
                                seed=seed,
                                depth=item.depth + 1,
                                link_text=link_text,
                            )
                        )


def validate_manifest(path: Path) -> None:
    records = load_manifest(path)
    by_source: set[str] = set()
    by_url: set[str] = set()
    duplicate_sources: list[str] = []
    duplicate_urls: list[str] = []
    missing_raw: list[str] = []

    for record in records:
        if record.source_id in by_source:
            duplicate_sources.append(record.source_id)
        by_source.add(record.source_id)

        url = str(record.url).rstrip("/")
        if url in by_url:
            duplicate_urls.append(url)
        by_url.add(url)

        raw_path = PROJECT_ROOT / record.raw_path
        if not raw_path.exists():
            missing_raw.append(record.raw_path)

    print(f"records={len(records)}")
    if duplicate_sources:
        print("duplicate_source_id=" + ",".join(sorted(set(duplicate_sources))))
    if duplicate_urls:
        print("duplicate_url=" + ",".join(sorted(set(duplicate_urls))))
    if missing_raw:
        print("missing_raw=" + ",".join(sorted(set(missing_raw))))


def prune_noise_records(seed_config: Path, manifest_path: Path) -> None:
    payload = load_yaml(seed_config)
    crawl_policy = dict(payload.get("crawl_policy", {}))
    exclude_keywords = list(crawl_policy.get("exclude_keywords", []))
    records = load_manifest(manifest_path)

    kept: list[PolicySourceManifestRecord] = []
    removed: list[PolicySourceManifestRecord] = []
    for record in records:
        depth = int(record.crawl_metadata.get("depth", 0))
        title = record.title or ""
        if depth > 0 and any(keyword in title for keyword in exclude_keywords):
            removed.append(record)
            continue
        kept.append(record)

    for record in removed:
        for path_value in [record.raw_path, record.clean_path]:
            if not path_value:
                continue
            path = PROJECT_ROOT / path_value
            if path.exists() and path.is_file():
                path.unlink()

    write_manifest_records(manifest_path, kept)
    print(f"kept={len(kept)} removed={len(removed)}")
    for record in removed:
        print(f"[PRUNE] {record.source_id} | {record.title}")


def main() -> None:
    args = parse_args()
    if args.prune_noise:
        prune_noise_records(args.seed_config, args.manifest_path)
        return
    if args.validate_manifest:
        validate_manifest(args.manifest_path)
        return
    asyncio.run(crawl_all(args))


if __name__ == "__main__":
    main()
