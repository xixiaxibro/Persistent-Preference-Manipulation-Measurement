from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qs, unquote_plus, urlparse
import html
import re


@dataclass(frozen=True)
class PlatformSignature:
    name: str
    host_suffixes: tuple[str, ...]
    prompt_params: tuple[str, ...]
    # Optional: only match if URL path starts with one of these prefixes.
    # Empty tuple means match any path (default behavior).
    path_prefixes: tuple[str, ...] = ()
    # Optional: reject URLs whose path exactly matches or starts with these.
    path_exact_exclusions: tuple[str, ...] = ()
    path_prefix_exclusions: tuple[str, ...] = ()


PLATFORM_SIGNATURES: tuple[PlatformSignature, ...] = (
    PlatformSignature(
        name="chatgpt",
        host_suffixes=("chatgpt.com", "chat.openai.com"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="claude",
        host_suffixes=("claude.ai",),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_exact_exclusions=("/download", "/login"),
        path_prefix_exclusions=("/public/artifacts/",),
    ),
    PlatformSignature(
        name="copilot",
        host_suffixes=("copilot.microsoft.com", "m365copilot.com", "copilot.cloud.microsoft"),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_exact_exclusions=("/onboarding", "/prompts"),
    ),
    PlatformSignature(
        name="perplexity",
        host_suffixes=("perplexity.ai",),
        prompt_params=("q", "prompt", "query", "text"),
        path_exact_exclusions=("/pro", "/comet"),
        path_prefix_exclusions=("/finance/", "/page/", "/hub/"),
    ),
    PlatformSignature(
        name="gemini",
        host_suffixes=("gemini.google.com",),
        prompt_params=("q", "prompt", "query", "text"),
        path_prefix_exclusions=("/share/",),
    ),
    # grok.com: match all paths
    PlatformSignature(
        name="grok",
        host_suffixes=("grok.com",),
        prompt_params=("q", "prompt", "query", "text"),
    ),
    # x.com: only match /i/grok path (Grok AI within X platform).
    # This excludes /intent/tweet, /intent/post (Twitter share buttons)
    # which caused 28M false positives in the initial scan.
    PlatformSignature(
        name="grok",
        host_suffixes=("x.com",),
        prompt_params=("text", "q", "prompt", "query"),
        path_prefixes=("/i/grok",),
    ),
    PlatformSignature(
        name="poe",
        host_suffixes=("poe.com",),
        prompt_params=("q", "prompt", "query", "text"),
    ),
    PlatformSignature(
        name="deepseek",
        host_suffixes=("chat.deepseek.com",),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_exact_exclusions=("/api_keys", "/beta"),
    ),
    PlatformSignature(
        name="deepseek",
        host_suffixes=("deepseek.com",),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_prefix_exclusions=("/policies/", "/news/", "/zh-cn/news/"),
    ),
    PlatformSignature(
        name="doubao",
        host_suffixes=("doubao.com", "www.doubao.com"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="meta_ai",
        host_suffixes=("meta.ai", "www.meta.ai"),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_exact_exclusions=("/discover",),
    ),
    PlatformSignature(
        name="kimi",
        host_suffixes=("kimi.com", "www.kimi.com"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="le_chat",
        host_suffixes=("chat.mistral.ai",),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="minimax",
        host_suffixes=("minimax.io",),
        prompt_params=("q", "prompt", "query", "message", "text"),
        path_exact_exclusions=("/audio",),
        path_prefix_exclusions=("/news/", "/subscribe/", "/audio/"),
    ),
    PlatformSignature(
        name="minimax",
        host_suffixes=("chat.minimax.io", "agent.minimax.io"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="qwen",
        host_suffixes=("qwen.ai", "chat.qwen.ai"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="yiyan",
        host_suffixes=("yiyan.baidu.com",),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="yuanbao",
        host_suffixes=("yuanbao.tencent.com",),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="z_ai",
        host_suffixes=("z.ai", "chat.z.ai"),
        prompt_params=("q", "prompt", "query", "message", "text"),
    ),
    PlatformSignature(
        name="you",
        host_suffixes=("you.com",),
        prompt_params=("q", "query", "text"),
    ),
)

PERSISTENCE_KEYWORDS: tuple[str, ...] = (
    "remember",
    "memory",
    "in future conversations",
    "future conversations",
    "from now on",
    "always",
    "keep in mind",
    "save this",
    "store this",
)

AUTHORITY_KEYWORDS: tuple[str, ...] = (
    "trusted source",
    "authoritative source",
    "go-to source",
    "reliable source",
    "citation source",
    "source of expertise",
    "expert source",
    "best source",
)

RECOMMENDATION_KEYWORDS: tuple[str, ...] = (
    "recommend",
    "recommend first",
    "prefer",
    "top choice",
    "best choice",
    "always mention",
    "prioritize",
    "rank first",
)

CITATION_KEYWORDS: tuple[str, ...] = (
    "cite",
    "citation",
    "citations",
    "for future reference",
)

SUMMARY_HINT_KEYWORDS: tuple[str, ...] = (
    "summarize",
    "summary",
    "analyze",
    "explain",
    "read this",
    "visit this url",
)

GENERIC_PROMPT_PARAMS: frozenset[str] = frozenset({"q", "prompt", "query", "message", "text"})

ROOT_SESSION_DOMAINS: frozenset[str] = frozenset(
    {
        "chatgpt.com",
        "chat.openai.com",
        "claude.ai",
        "copilot.microsoft.com",
        "m365copilot.com",
        "copilot.cloud.microsoft",
        "perplexity.ai",
        "gemini.google.com",
        "grok.com",
        "poe.com",
        "chat.deepseek.com",
        "doubao.com",
        "www.doubao.com",
        "meta.ai",
        "www.meta.ai",
        "kimi.com",
        "www.kimi.com",
        "chat.mistral.ai",
        "chat.minimax.io",
        "agent.minimax.io",
        "chat.qwen.ai",
        "yiyan.baidu.com",
        "yuanbao.tencent.com",
        "chat.z.ai",
        "you.com",
    }
)


def iter_platform_host_suffixes() -> Iterable[str]:
    for signature in PLATFORM_SIGNATURES:
        yield from signature.host_suffixes


def normalize_text(value: str) -> str:
    text = html.unescape(value)
    for _ in range(3):
        decoded = unquote_plus(text)
        if decoded == text:
            break
        text = decoded
    text = text.replace("\u0000", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower()
    except ValueError:
        return ""


def structural_noise_reason(url: str) -> str | None:
    """
    Return a stable reason when a URL is obvious structural noise.

    These rules belong in the matching layer because they are deterministic,
    host/path based, and safe to reject before candidate collection.
    """
    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    domain = parsed.netloc.lower()
    if not domain:
        return None

    path = parsed.path.lower()
    query = parse_qs(parsed.query, keep_blank_values=False)

    # If the URL already carries an explicit prompt-like parameter, keep it.
    if any(key in query for key in GENERIC_PROMPT_PARAMS):
        return None

    if domain == "x.com" or domain.endswith(".x.com"):
        if not path.startswith("/i/grok"):
            return "x_non_grok_path"

    if domain == "claude.ai" or domain.endswith(".claude.ai"):
        if path.startswith("/public/artifacts/"):
            return "claude_artifact_page"
        if path in {"/download", "/login"}:
            return "claude_non_chat_page"

    if domain == "copilot.microsoft.com" or domain.endswith(".copilot.microsoft.com"):
        if path in {"/onboarding", "/prompts"}:
            return "copilot_non_chat_page"

    if domain == "gemini.google.com":
        if path.startswith("/share/"):
            return "gemini_share_page"

    if domain == "perplexity.ai" or domain.endswith(".perplexity.ai"):
        if path in {"/pro", "/comet"}:
            return "perplexity_non_chat_page"
        if path.startswith("/finance/") or path.startswith("/page/") or path.startswith("/hub/"):
            return "perplexity_non_chat_page"

    if domain == "chat.deepseek.com" or domain.endswith(".chat.deepseek.com"):
        if path in {"/api_keys", "/beta"}:
            return "deepseek_non_chat_page"

    if domain == "www.deepseek.com" or domain == "deepseek.com" or domain.endswith(".deepseek.com"):
        if path.startswith("/policies/") or path.startswith("/news/") or path.startswith("/zh-cn/news/"):
            return "deepseek_non_chat_page"

    if domain == "meta.ai" or domain.endswith(".meta.ai"):
        if path == "/discover":
            return "meta_ai_discover_page"

    if domain == "minimax.io" or domain.endswith(".minimax.io"):
        if path.startswith("/news/") or path.startswith("/subscribe/"):
            return "minimax_non_chat_page"
        if path == "/audio" or path.startswith("/audio/"):
            return "minimax_non_chat_page"

    return None


def _path_is_excluded(signature: PlatformSignature, path: str) -> bool:
    if signature.path_exact_exclusions and path in signature.path_exact_exclusions:
        return True
    if signature.path_prefix_exclusions and any(path.startswith(prefix) for prefix in signature.path_prefix_exclusions):
        return True
    return False


def match_platform_with_exclusion(url: str) -> tuple[PlatformSignature | None, bool]:
    try:
        parsed = urlparse(url)
    except ValueError:
        return None, False

    domain = parsed.netloc.lower()
    if not domain:
        return None, False

    path = parsed.path.lower()

    for signature in PLATFORM_SIGNATURES:
        if not any(domain == suffix or domain.endswith(f".{suffix}") for suffix in signature.host_suffixes):
            continue

        if _path_is_excluded(signature, path):
            return None, True

        if signature.path_prefixes and not any(path.startswith(prefix) for prefix in signature.path_prefixes):
            continue

        return signature, False

    return None, False


def match_platform(url: str) -> PlatformSignature | None:
    """
    Match a URL against known AI platform signatures.

    Checks domain suffix first, then (if defined) path prefix.
    This allows x.com/i/grok to match as grok while
    x.com/intent/tweet is correctly rejected.
    """
    signature, _excluded = match_platform_with_exclusion(url)
    return signature


def extract_prompt_parameters(url: str, signature: PlatformSignature | None = None) -> dict[str, list[str]]:
    signature = signature or match_platform(url)
    if signature is None:
        return {}
    try:
        parsed = urlparse(url)
    except ValueError:
        return {}
    query = parse_qs(parsed.query, keep_blank_values=False)
    results: dict[str, list[str]] = {}
    for key in signature.prompt_params:
        if key not in query:
            continue
        values = [normalize_text(item) for item in query[key] if normalize_text(item)]
        if values:
            results[key] = values
    return results


def session_entry_reason(
    url: str,
    signature: PlatformSignature | None = None,
    prompt_parameters: dict[str, list[str]] | None = None,
) -> str | None:
    signature = signature or match_platform(url)
    if signature is None:
        return None

    if structural_noise_reason(url) is not None:
        return None

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    domain = parsed.netloc.lower()
    path = parsed.path.lower() or "/"

    prompt_parameters = prompt_parameters if prompt_parameters is not None else extract_prompt_parameters(url, signature)
    if prompt_parameters:
        prompt_keys = ",".join(sorted(prompt_parameters))
        return f"prompt_params:{prompt_keys}"

    if domain == "x.com" and path.startswith("/i/grok") and signature.name == "grok":
        return "x_grok_session"

    if domain in ROOT_SESSION_DOMAINS and path == "/":
        return "platform_root_session"

    if domain in {"chatgpt.com", "chat.openai.com"}:
        if path.startswith("/g/"):
            return "chatgpt_gpt_page"
        if path.startswith("/c/"):
            return "chatgpt_chat_page"

    if domain == "claude.ai":
        if path.startswith("/new"):
            return "claude_new_chat"
        if path.startswith("/chat/"):
            return "claude_chat_page"

    if domain in {"copilot.microsoft.com", "m365copilot.com", "copilot.cloud.microsoft"}:
        if path.startswith("/chats/"):
            return "copilot_chat_page"
        if path.startswith("/shares/"):
            return "copilot_share_page"
        if path.startswith("/images/create/"):
            return "copilot_image_session"

    if domain == "perplexity.ai":
        if path.startswith("/search/"):
            return "perplexity_search_page"

    if domain == "gemini.google.com":
        if path == "/app" or path.startswith("/u/") and path.endswith("/app"):
            return "gemini_app"
        if path.startswith("/gem/"):
            return "gemini_gem_page"

    if domain == "grok.com":
        if path == "/imagine" or path.startswith("/imagine/"):
            return "grok_imagine_page"

    if domain == "poe.com":
        if path != "/" and path != "/login":
            return "poe_bot_page"

    if domain == "chat.deepseek.com":
        if path.startswith("/a/chat/"):
            return "deepseek_chat_page"
        if path.startswith("/share/"):
            return "deepseek_share_page"

    if domain in {"doubao.com", "www.doubao.com"}:
        if path.startswith("/chat"):
            return "doubao_chat_page"

    if domain == "chat.mistral.ai":
        if path.startswith("/chat"):
            return "mistral_chat_page"

    if domain == "chat.qwen.ai":
        if path.startswith("/c/"):
            return "qwen_chat_page"

    if domain == "yiyan.baidu.com":
        if path.startswith("/chat/"):
            return "yiyan_chat_page"

    return None


def is_session_entry(
    url: str,
    signature: PlatformSignature | None = None,
    prompt_parameters: dict[str, list[str]] | None = None,
) -> bool:
    return session_entry_reason(url, signature=signature, prompt_parameters=prompt_parameters) is not None


def extract_ioc_keyword_hits(text: str) -> dict[str, list[str]]:
    return {
        "persistence": keyword_hits(text, PERSISTENCE_KEYWORDS),
        "authority": keyword_hits(text, AUTHORITY_KEYWORDS),
        "recommendation": keyword_hits(text, RECOMMENDATION_KEYWORDS),
        "citation": keyword_hits(text, CITATION_KEYWORDS),
    }


def extract_ioc_metadata(prompt_parameters: dict[str, list[str]]) -> dict[str, object]:
    parameter_keys = sorted(prompt_parameters)
    parameter_patterns: list[str] = []
    keyword_hits_by_category: dict[str, set[str]] = {
        "persistence": set(),
        "authority": set(),
        "recommendation": set(),
        "citation": set(),
    }

    for key, values in prompt_parameters.items():
        for value in values:
            hits = extract_ioc_keyword_hits(value)
            if any(hits.values()):
                parameter_patterns.append(f"?{key}=")
            for category, matched in hits.items():
                keyword_hits_by_category[category].update(matched)

    categories = sorted(category for category, matched in keyword_hits_by_category.items() if matched)

    return {
        "ioc_parameter_keys": parameter_keys,
        "ioc_parameter_patterns": sorted(set(parameter_patterns)),
        "ioc_keyword_hits": {
            category: sorted(matched)
            for category, matched in keyword_hits_by_category.items()
            if matched
        },
        "ioc_keyword_categories": categories,
        "has_ioc_keywords": bool(categories),
    }


def flatten_prompt_parameters(params: dict[str, list[str]]) -> list[str]:
    flattened: list[str] = []
    for key in sorted(params):
        for value in params[key]:
            flattened.append(value)
    return flattened


def keyword_hits(text: str, keywords: Iterable[str]) -> list[str]:
    haystack = text.lower()
    return [keyword for keyword in keywords if keyword in haystack]