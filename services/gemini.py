import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

"""
Gemini AI analysis service.
API key is stored server-side in .env — never exposed to the client.
"""

import json  # noqa: E402
import os  # noqa: E402

import httpx  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

from services.geometry import GeometryResult  # noqa: E402

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


class GeminiError(Exception):
    pass


def _build_prompt(geom: GeometryResult, cogs_summary: dict) -> str:
    face_text = (
        f"{geom.face_count} B-Rep faces (STEP)"
        if geom.face_count
        else f"{geom.triangle_count} triangles (STL)"
    )
    assy_text = (
        f", Assembly components: {', '.join(geom.components)}"
        if geom.is_assembly and geom.components
        else ""
    )
    nearshore_100u = cogs_summary.get("nearshore_100u", "N/A")

    return f"""You are an expert hardware engineering analyst and manufacturing consultant.

REAL CAD GEOMETRY DATA (parsed directly from file — do not hallucinate):
- Format: {geom.file_format} ({geom.format_detail})
- Volume: {geom.volume_cm3} cm³ ({geom.volume_method})
- Surface area: {geom.surface_area_cm2} cm²
- Bounding box: {geom.bounding_box_mm.x} × {geom.bounding_box_mm.y} × {geom.bounding_box_mm.z} mm
- Complexity score: {geom.complexity_score}/10
- {face_text}{assy_text}
- Nearshore COGS at 100 units: ${nearshore_100u}/unit

Based ONLY on this real geometry, provide:
1. Structural integrity score (0–100) for CNC machined Al 6061-T6
2. A specific DFM (Design for Manufacturability) warning referencing actual dimensions
3. Estimated retail price (USD), gross margin health
4. Prototype cost, tooling CapEx, total seed capital ask
5. Recommended supply chain route, lead time, risk level, and 3 real supplier names
6. Exactly 3 geometry-specific engineering optimisation recommendations

Rules:
- Do NOT estimate COGS (already computed separately)
- Reference specific measurements in your DFM warning
- Suppliers must be real, specific companies
- Seed ask should reflect actual hardware startup needs

Respond ONLY with valid JSON matching this exact schema, no markdown, no preamble:
{{
  "physics": {{
    "score": 85,
    "warning": "specific DFM warning referencing dimensions",
    "status": "PASS"
  }},
  "economics": {{
    "retail": 149.99,
    "health": "STRONG",
    "gross_margin_pct": 62.5
  }},
  "capital": {{
    "prototype": 450.00,
    "capex": 5200.00,
    "total_ask": 5650.00
  }},
  "supply_chain": {{
    "lead_time": "14–18 days",
    "route": "Nearshore CNC (Turkey / E. Europe)",
    "risk": "Low",
    "suppliers": ["Xometry EU", "3D Hubs", "Protocase"]
  }},
  "optimizations": [
    "tip 1 referencing real geometry",
    "tip 2 referencing real geometry",
    "tip 3 referencing real geometry"
  ]
}}"""


def _fallback_analysis(geom: GeometryResult) -> dict:
    """Rule-based fallback when Gemini is unavailable."""
    import math
    v = geom.volume_cm3
    cs = geom.complexity_score
    bbox = geom.bounding_box_mm

    score = min(98, max(60, int(90 - cs * 2 + math.log(max(v, 0.1)) * 0.5)))
    retail = round(v * 8.5 + 45, 2)

    if cs > 7:
        warning = (
            f"High complexity ({cs}/10) — verify minimum wall thickness ≥ 0.8mm. "
            f"At {bbox.z:.1f}mm depth, undercuts likely require 5-axis or EDM."
        )
    elif cs > 4:
        warning = (
            f"Complexity {cs}/10 — check draft angles on features "
            f"deeper than {round(bbox.z * 0.3, 1)}mm. "
            f"Part envelope {bbox.x}×{bbox.y}×{bbox.z}mm fits standard 3-axis vise."
        )
    else:
        warning = (
            f"Geometry appears machinable on 3-axis. "
            f"Verify draft angles > {round(bbox.z * 0.3, 1)}mm deep. "
            f"Bounding box {bbox.x}×{bbox.y}×{bbox.z}mm — standard fixturing applies."
        )

    return {
        "physics": {"score": score, "warning": warning, "status": "ESTIMATED"},
        "economics": {
            "retail": retail,
            "health": "STRONG" if cs < 6 else "MODERATE",
            "gross_margin_pct": None,
        },
        "capital": {
            "prototype": round(380 + v * 12, 2),
            "capex": round(4200 + v * 45 * cs, 2),
            "total_ask": round(4580 + v * 57 * cs, 2),
        },
        "supply_chain": {
            "lead_time": f"{round(10 + cs * 2)} days",
            "route": (
                "Multi-source → nearshore integration"
                if geom.is_assembly
                else "Nearshore CNC (Turkey / E. Europe)"
            ),
            "risk": "Medium" if cs > 5 else "Low",
            "suppliers": (
                ["Xometry EU", "3D Hubs", "Protocase"]
                if not geom.is_assembly
                else ["Xometry EU", "3D Hubs", "Fictiv"]
            ),
        },
        "optimizations": [
            f"Volume {v} cm³ — hollow non-structural internals to reduce material ~18%.",
            (
                f"Complexity {cs}/10: redesign undercuts → ~$12/unit saving at 100+ units."
                if cs > 4
                else f"Complexity {cs}/10: well-optimised for 3-axis single setup."
            ),
            (
                "Consolidate 2–3 assembly parts into single CNC component to cut labour."
                if geom.is_assembly
                else f"At {bbox.x}×{bbox.y}mm, nearshore CNC (Turkey) optimal at 100+ units."
            ),
        ],
    }


async def search_part_prices(bom_items: list) -> dict:
    """
    Use Gemini with Google Search grounding to find real-time AliExpress + Amazon prices.
    Only searches standard/off-the-shelf parts (bearings, fasteners, gears, motors, electronics).
    Returns: {item_name: {aliexpress_price, amazon_price, aliexpress_url, amazon_url}}
    """
    if not GEMINI_API_KEY:
        return {}

    searchable = [b for b in bom_items if b.get('part_type') not in ('machined', 'service', 'finish')]
    if not searchable:
        return {}

    parts_list = "\n".join([
        f"- {b['item']} (type: {b['part_type']}, estimated: ${b['unit_cost']})"
        for b in searchable[:10]
    ])

    prompt = f"""Search the web to find current retail prices for these mechanical/electronic components.

For each part find the unit price (when buying 10+ pieces) from:
1. AliExpress.com - cheapest reasonable-quality listing
2. Amazon.com - standard quality, Prime-eligible if possible

Parts to search:
{parts_list}

Return ONLY a JSON object with no markdown, no preamble, no explanation:
{{
  "prices": {{
    "EXACT_PART_NAME_FROM_LIST": {{
      "aliexpress_price": 0.05,
      "amazon_price": 0.18,
      "aliexpress_url": "https://www.aliexpress.com/item/...",
      "amazon_url": "https://www.amazon.com/dp/..."
    }}
  }}
}}

Use the EXACT part names from the list as JSON keys. If a price is not found, omit that marketplace key."""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                GEMINI_URL,
                params={"key": GEMINI_API_KEY},
                json={
                    "tools": [{"google_search": {}}],
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.1,
                        "maxOutputTokens": 2048,
                        # NOTE: response_mime_type cannot be used with tools
                    },
                },
            )
            response.raise_for_status()
            data = response.json()

        # Collect all text parts (grounding may add extra parts)
        raw_text = ""
        for part in data.get("candidates", [{}])[0].get("content", {}).get("parts", []):
            if part.get("text"):
                raw_text += part["text"]

        if not raw_text:
            return {}

        # Extract JSON from response
        start = raw_text.find('{')
        end = raw_text.rfind('}') + 1
        if start >= 0 and end > start:
            parsed = json.loads(raw_text[start:end])
            return parsed.get("prices", {})

    except json.JSONDecodeError as e:
        print(f"[Gemini price search] JSON parse failed: {e}")
    except Exception as e:
        print(f"[Gemini price search] Failed: {e}")

    return {}


def _extract_fields_manually(text: str) -> dict:
    """Last-resort: pull individual fields from Gemini output using regex."""
    import re as _re

    def _num(pattern, default):
        m = _re.search(pattern, text)
        return float(m.group(1)) if m else default

    def _str(pattern, default):
        m = _re.search(pattern, text, _re.DOTALL)
        if m:
            val = m.group(1).strip().replace('\n', ' ')
            return val[:300]
        return default

    def _list(pattern):
        m = _re.search(pattern, text, _re.DOTALL)
        if not m:
            return []
        items = _re.findall(r'"([^"]{5,})"', m.group(1))
        return items[:3]

    score   = _num(r'"score"\s*:\s*(\d+)', 75)
    warning = _str(r'"warning"\s*:\s*"([^"]*(?:\\"[^"]*)*)"', 'Review DFM requirements carefully.')
    status  = _str(r'"status"\s*:\s*"([^"]*)"', 'ESTIMATED')
    retail  = _num(r'"retail"\s*:\s*([\d.]+)', 149.99)
    health  = _str(r'"health"\s*:\s*"([^"]*)"', 'MODERATE')
    margin  = _num(r'"gross_margin_pct"\s*:\s*([\d.]+)', 50.0)
    proto   = _num(r'"prototype"\s*:\s*([\d.]+)', 450.0)
    capex   = _num(r'"capex"\s*:\s*([\d.]+)', 5000.0)
    total   = _num(r'"total_ask"\s*:\s*([\d.]+)', 5450.0)
    lead    = _str(r'"lead_time"\s*:\s*"([^"]*)"', '14-21 days')
    route   = _str(r'"route"\s*:\s*"([^"]*)"', 'Nearshore CNC')
    risk    = _str(r'"risk"\s*:\s*"([^"]*)"', 'Medium')
    suppliers = _list(r'"suppliers"\s*:\s*\[([^\]]*)\]')
    if not suppliers:
        suppliers = ['Xometry EU', '3D Hubs', 'Protocase']
    opts = _list(r'"optimizations"\s*:\s*\[([^\]]*)\]')
    if not opts:
        opts = ['Review geometry for DFM compliance.', 'Check wall thicknesses.', 'Optimise for 3-axis machining.']

    return {
        'physics':      {'score': int(score), 'warning': warning, 'status': status},
        'economics':    {'retail': retail, 'health': health, 'gross_margin_pct': margin},
        'capital':      {'prototype': proto, 'capex': capex, 'total_ask': total},
        'supply_chain': {'lead_time': lead, 'route': route, 'risk': risk, 'suppliers': suppliers},
        'optimizations': opts,
    }


async def analyze_geometry(geom: GeometryResult, cogs_data: dict) -> tuple[dict, bool]:
    """
    Returns (analysis_dict, ai_was_used).
    Falls back to rule-based if Gemini key missing or call fails.
    """
    if not GEMINI_API_KEY:
        return _fallback_analysis(geom), False

    nearshore = cogs_data.get("regions", {}).get("nearshore", {})
    nearshore_100u = nearshore.get("tiers", {}).get("100", {}).get("total", "N/A")
    prompt = _build_prompt(geom, {"nearshore_100u": nearshore_100u})

    try:
        # Retry up to 3 times for transient 5xx errors
        response = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=45.0) as client:
                    response = await client.post(
                        GEMINI_URL,
                        params={"key": GEMINI_API_KEY},
                        json={
                            "contents": [{"parts": [{"text": prompt}]}],
                            "generationConfig": {
                                "temperature": 0.1,
                                "maxOutputTokens": 4096,
                                "response_mime_type": "application/json",
                            },
                        },
                    )
                response.raise_for_status()
                break  # success
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (500, 503, 529) and attempt < 2:
                    import asyncio
                    await asyncio.sleep(2 ** attempt)
                    continue
                raise

        if response is None:
            return _fallback_analysis(geom), False

        data = response.json()
        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

        # Strip markdown fences
        clean = raw_text.strip()
        for prefix in ("```json", "```"):
            if clean.startswith(prefix):
                clean = clean[len(prefix):]
        if clean.endswith("```"):
            clean = clean[:-3]
        clean = clean.strip()

        # Extract outermost JSON object
        start = clean.find('{')
        end = clean.rfind('}') + 1
        if start >= 0 and end > start:
            clean = clean[start:end]

        # Attempt 1: direct parse
        try:
            return json.loads(clean), True
        except json.JSONDecodeError:
            pass

        # Attempt 2: character-level repair
        import re as _re
        fixed = clean
        fixed = _re.sub(r'[\u2013\u2014]', '-', fixed)
        fixed = _re.sub(r'[\u201c\u201d]', '"', fixed)
        fixed = _re.sub(r'[\u2018\u2019]', "'", fixed)
        fixed = _re.sub(r',\s*([\]}])', r'\1', fixed)

        # Fix bare newlines inside quoted strings
        out, in_str, i = [], False, 0
        while i < len(fixed):
            ch = fixed[i]
            if ch == '\\' and in_str:
                out.append(ch)
                i += 1
                if i < len(fixed):
                    out.append(fixed[i])
            elif ch == '"':
                in_str = not in_str
                out.append(ch)
            elif in_str and ch == '\n':
                out.append('\\n')
            elif in_str and ch == '\r':
                pass
            elif in_str and ch == '\t':
                out.append(' ')
            else:
                out.append(ch)
            i += 1
        fixed = ''.join(out)

        try:
            return json.loads(fixed), True
        except json.JSONDecodeError:
            pass

        # Attempt 3: regex field extraction (last resort)
        try:
            return _extract_fields_manually(clean), True
        except Exception:
            pass

    except Exception as e:
        print(f"[Gemini] Failed: {e} — using fallback")

    return _fallback_analysis(geom), False
