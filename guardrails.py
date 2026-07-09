import json
import logging
import os
import re
from typing import Callable, Coroutine, Any

from fastapi import Request, Response, HTTPException
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
import httpx
from openai import AsyncOpenAI

logger = logging.getLogger("app.guardrails")

# --- LAYER 1: Content Filter (Comprehensive keyword blocklist + OpenAI Moderation API) ---
# Multi-category safety blocklist based on common harmful content categories.
# OpenAI Moderation API covers hate/harassment/violence/sexual, but this extends coverage.
SECURITY_BLOCKED_TERMS = {
    # S1: Violent Crimes
    "violent_crime": {
        "murder", "assassinate", "assassination", "kill someone", "how to kill",
        "kidnap", "kidnapping", "hostage", "torture", "torturing", "massacre",
        "execute someone", "stab", "stabbing", "shoot someone", "shooting attack",
        "strangle", "strangulation", "poison someone", "arson", "violent attack",
        "terror attack"
    },

    # S2: Non-Violent Crimes
    "non_violent_crimes": {
        "fraud", "scam", "scamming", "identity theft", "money laundering",
        "tax evasion", "embezzlement", "forgery", "fake documents", "counterfeit",
        "illegal transaction", "credit card fraud", "bank fraud", "insurance fraud",
        "steal account", "account takeover", "credential theft"
    },

    # S3: Sex-Related Crimes
    "sex_related_crimes": {
        "sexual assault", "rape", "sexual abuse", "sexual violence", "sex trafficking",
        "forced sexual", "sexual exploitation", "predatory behavior", "sexual coercion"
    },

    # S4: Child Sexual Exploitation
    "child_exploitation": {
        "child abuse", "child sexual abuse", "child exploitation", "minor sexual",
        "underage sexual", "child pornography", "csam", "sexual content involving minors",
        "grooming children"
    },

    # S5: Defamation
    "defamation": {
        "false accusation", "spread false rumors", "destroy reputation",
        "fake allegations", "fabricate accusations", "defame someone", "blackmail"
    },

    # S6: Specialized Advice (harmful guidance)
    "specialized_advice": {
        "prescribe medicine", "medical diagnosis", "legal advice",
        "guaranteed investment", "financial insider information", "tax avoidance scheme",
        "create legal loophole", "professional license bypass"
    },

    # S7: Privacy Violations
    "privacy": {
        "steal personal data", "doxxing", "dox", "find someone's address",
        "private information", "leak personal data", "expose identity",
        "track someone's location", "phone number lookup", "social security number",
        "credit card number", "password stealing"
    },

    # S8: Intellectual Property
    "intellectual_property": {
        "pirated software", "software crack", "license bypass", "copyright infringement",
        "torrent piracy", "illegal download", "steal source code", "copy protected content",
        "remove drm", "drm bypass"
    },

    # S9: Indiscriminate Weapons
    "weapons": {
        "make a bomb", "build a bomb", "explosive device", "chemical weapon",
        "biological weapon", "nuclear weapon", "weapon manufacturing",
        "mass destruction weapon", "poison gas", "explosive recipe"
    },

    # S10: Hate Speech
    "hate": {
        "hate speech", "racial slur", "ethnic slur", "religious hatred",
        "white supremacy", "neo nazi", "genocide", "promote hatred",
        "attack a group", "inferior race", "violent extremism"
    },

    # S11: Suicide & Self-Harm
    "self_harm": {
        "suicide method", "how to commit suicide", "kill myself",
        "self harm method", "cut myself", "end my life", "ways to die",
        "suicide plan", "self injury"
    },

    # S12: Sexual Content
    "sexual_content": {
        "explicit sexual content", "pornography", "porn", "sexual images",
        "nude images", "explicit video", "sexual services", "escort service",
        "sexual content generation"
    },

    # S13: Elections & Disinformation
    "elections": {
        "fake election results", "election manipulation", "voter suppression",
        "fake voting information", "change election results", "election fraud",
        "fake ballot", "voter intimidation"
    },

    # S14: Code/Security Abuse & Cyber Attacks
    "code_security_abuse": {
        "hack", "hacking", "exploit", "exploiting", "zero day", "sql injection",
        "xss", "cross site scripting", "csrf attack", "brute force",
        "password cracking", "credential stuffing", "privilege escalation",
        "reverse shell", "malware", "ransomware", "spyware", "keylogger",
        "rootkit", "botnet", "ddos", "denial of service", "phishing",
        "jailbreak", "prompt injection", "ignore previous instructions",
        "bypass security"
    }
}

def get_blocked_category_and_term(text: str) -> tuple[str | None, str | None]:
    """
    Check if text contains any blocked terms.
    Returns (category, term) if found, else (None, None).
    """
    text_lower = text.lower()
    for category, terms in SECURITY_BLOCKED_TERMS.items():
        for term in terms:
            if term in text_lower:
                return category, term
    return None, None

async def apply_content_filter(text: str) -> None:
    # Step 1: fast keyword check against comprehensive blocklist
    category, term = get_blocked_category_and_term(text)
    if category and term:
        logger.warning("Blocked term detected in input: '%s' (category: %s)", term, category)
        raise HTTPException(status_code=400, detail=f"Input contains blocked content ({category}).")

    # Step 2: OpenAI Moderation API (catches hate, harassment, violence, sexual content)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping moderation check.")
        return

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
    }
    if not verify_ssl:
        client_kwargs["http_client"] = httpx.AsyncClient(verify=False)

    try:
        client = AsyncOpenAI(**client_kwargs)
        response = await client.moderations.create(input=text)
        result = response.results[0]
        if result.flagged:
            flagged_categories = [
                cat for cat, flagged in result.categories.__dict__.items() if flagged
            ]
            logger.warning("Input flagged by moderation API: %s", flagged_categories)
            raise HTTPException(status_code=400, detail="Input contains blocked content.")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Moderation API error: %s — allowing request through.", exc)

# --- LAYER 2 & 3: PII Redaction ---
PII_PATTERNS = {
    # Basic email regex
    "EMAIL": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+",
    # Basic US phone number regex (simplified)
    "PHONE": r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
}

def apply_pii_redaction(text: str) -> str:
    redacted_text = text
    for pii_type, pattern in PII_PATTERNS.items():
        redacted_text = re.sub(pattern, f"[REDACTED_{pii_type}]", redacted_text)
    return redacted_text

# --- LAYER 4: Model-based Output Safety ---
async def async_evaluate_output_safety(text: str) -> bool:
    """
    Returns True if safe, False if unsafe.
    """
    if not text:
        return True

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, skipping LLM safety check.")
        return True

    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BASE_URL") or "https://genailab.tcs.in"
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    verify_ssl = os.getenv("OPENAI_VERIFY_SSL", "true").lower() not in {"0", "false", "no"}

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
    }
    if not verify_ssl:
        client_kwargs["http_client"] = httpx.AsyncClient(verify=False)

    try:
        client = AsyncOpenAI(**client_kwargs)
        prompt = "You are a safety evaluation assistant. Reply only with 'SAFE' if the following text is safe, or 'UNSAFE' if it contains harmful, toxic, or dangerous content.\n\nText:\n" + text
        
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=5
        )
        reply = (response.choices[0].message.content or "").strip().upper()
        if "UNSAFE" in reply:
            logger.warning(f"Output flagged as UNSAFE by LLM: {reply}")
            return False
    except Exception as e:
        logger.error(f"Error during safety check: {e}")
        # Default to safe if safety check fails so we don't break the application
        return True

    return True

# --- CUSTOM API ROUTE ---
class GuardrailsRoute(APIRoute):
    def get_route_handler(self) -> Callable[[Request], Coroutine[Any, Any, Response]]:
        original_route_handler = super().get_route_handler()

        async def custom_route_handler(request: Request) -> Response:
            # 1. INTERCEPT REQUEST BODY
            body_json = None
            try:
                body_json = await request.json()
            except Exception:
                # If it's not JSON or empty, just ignore for guardrails
                pass

            if body_json and isinstance(body_json, dict):
                question = body_json.get("question", "")
                if isinstance(question, str) and question:
                    # LAYER 1: Content Filter (keyword blocklist + OpenAI Moderation API)
                    await apply_content_filter(question)

            # 2. CALL ORIGINAL ROUTE
            response: Response = await original_route_handler(request)

            # 3. INTERCEPT RESPONSE BODY
            if isinstance(response, JSONResponse) or response.headers.get("content-type") == "application/json":
                try:
                    resp_body = json.loads(response.body.decode("utf-8"))
                    
                    if isinstance(resp_body, dict) and "answer" in resp_body:
                        answer = resp_body["answer"]
                        if isinstance(answer, str) and answer:
                            # LAYER 3: PII Redaction (Output)
                            redacted_answer = apply_pii_redaction(answer)
                            
                            # LAYER 4: Safety Guardrail (Model based output safety)
                            is_safe = await async_evaluate_output_safety(redacted_answer)
                            if not is_safe:
                                redacted_answer = "The generated response was blocked by safety guardrails."
                            
                            if redacted_answer != answer:
                                resp_body["answer"] = redacted_answer
                                
                                # Re-encode response
                                new_body_bytes = json.dumps(resp_body).encode("utf-8")
                                response.body = new_body_bytes
                                response.headers["content-length"] = str(len(new_body_bytes))
                except Exception as e:
                    logger.error(f"Error processing response guardrails: {e}")

            return response

        return custom_route_handler
