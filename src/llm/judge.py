"""LLM-as-Judge for medical exam theme equivalence.

Optional module — only activated when use_llm=True is passed.
Uses OpenAI-compatible API (works with OpenAI, Azure, local Ollama, etc.)
to validate low-confidence matches and suggest alternatives.

Design principles:
  - Never blocks the pipeline: if the LLM call fails, falls back to the
    original embedding-based score silently.
  - Batches requests to minimize API calls and cost.
  - Prompt is optimized for medical residency exam terminology (PT-BR).
  - Structured JSON output for reliable parsing.
  - Institution-agnostic: works with FAMERP, USP, UNICAMP, etc.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from src.utils.logging import logger


@dataclass
class LLMVerdict:
    """Result from the LLM judge for a single match."""

    is_equivalent: bool
    confidence: float  # 0.0 - 1.0
    suggested_match: str | None  # LLM's suggested DB tema if different
    reasoning: str  # Short explanation


SYSTEM_PROMPT = """\
Voce e um especialista em curriculos de residencia medica brasileira.

Sua tarefa: dado um tema de prova medica (INPUT) e uma lista de candidatos encontrados no banco de dados, \
voce deve decidir qual candidato e o melhor match — ou se nenhum candidato e equivalente.

O sistema ja fez uma busca automatica e encontrou um MATCH ATUAL com um score de confianca. \
Alem disso, fizemos buscas adicionais no banco e encontramos CANDIDATOS ALTERNATIVOS. \
Voce deve analisar TODOS os candidatos e escolher o melhor.

Regras:
1. Considere sinonimos medicos (ex: "IAM" = "Infarto Agudo do Miocardio" = "Sindrome Coronariana Aguda")
2. Subtemas podem estar contidos em temas mais amplos (ex: "Triagem Neonatal" contem "Teste do Coracaozinho")
3. Considere abreviacoes comuns (PCR, ITU, DPOC, RN, SCA, TEP, etc.)
4. Diferentes bancas (FAMERP, USP, UNICAMP, UNIFESP) usam terminologias diferentes para o mesmo assunto
5. Seja rigoroso: temas de especialidades diferentes NAO sao equivalentes
6. Se um candidato alternativo for MELHOR que o match atual, sugira-o em "suggested_match"
7. Se o match atual estiver correto, retorne is_equivalent=true e suggested_match=null
8. Se nenhum candidato for bom, retorne is_equivalent=false e suggested_match=null
9. PREFIRA SEMPRE o candidato MAIS ESPECIFICO: se o INPUT mencionar um subtopico especifico \
(ex: "Inflamatorio", "Obstrutivo", "Cardiogenico", "Hipoglicemia"), escolha o candidato cujo subtema \
melhor corresponda a essa especificidade. Evite "Aspectos Gerais" quando existir um candidato mais especifico disponivel.
10. O valor de "suggested_match" DEVE ser copiado EXATAMENTE do texto de um dos candidatos listados \
(incluindo "tema | subtema" com a mesma grafia). NAO invente texto novo.

Exemplos:
- INPUT "Avaliacao do RN | Triagem Neonatal" + candidato "Triagem neonatal | Teste do Coracaozinho" → EQUIVALENTES
- INPUT "Trauma | Abordagem Inicial" + candidato "Politraumatizado | Atendimento Inicial" → EQUIVALENTES
- INPUT "Pneumonia" + candidato "DPOC" → NAO EQUIVALENTES (doencas distintas)
- INPUT "Queimaduras" (Cirurgia) + candidato "Queimaduras" (Dermatologia) → NAO EQUIVALENTES
- INPUT "Abdome agudo Inflamatorio | Apendicite" + candidatos ["Abdome agudo | Aspectos Gerais", \
"Abdome agudo | Apendicite aguda"] → PREFIRA "Abdome agudo | Apendicite aguda" (mais especifico)
- INPUT "Choque | Choque Cardiogenico" + candidatos ["Cardiointensivismo | Choque (exceto choque septico)", \
"Cardiointensivismo | Aspectos Gerais"] → PREFIRA "Cardiointensivismo | Choque (exceto choque septico)"

Responda SEMPRE em JSON valido:
{
  "results": [
    {
      "index": 0,
      "is_equivalent": true/false,
      "confidence": 0.0-1.0,
      "suggested_match": "tema | subtema EXATAMENTE como listado nos candidatos, ou null",
      "reasoning": "Explicacao curta do por que (1-2 frases)"
    }
  ]
}"""


def _build_batch_prompt(
    items: list[dict[str, Any]],
    db_candidates: list[list[dict[str, Any]]],
) -> str:
    """Build a single prompt for multiple items to judge in one API call."""
    parts = []
    for i, (item, candidates) in enumerate(zip(items, db_candidates)):
        input_tema = item.get("input_tema", "")
        input_subtema = item.get("input_subtema", "")
        current_match = item.get("current_match", "")
        current_score = item.get("current_score", 0)

        cand_lines = []
        seen = set()
        # Include current match as first candidate
        if current_match:
            cand_lines.append(
                f"  * [MATCH ATUAL] {current_match} (score={current_score:.0%})"
            )
            seen.add(current_match.lower())

        for c in candidates[:10]:
            t = c.get("tema", "")
            s = c.get("subtema", "")
            nq = c.get("num_questions", 0)
            label = f"{t} | {s}" if s else t
            if label.lower() not in seen:
                seen.add(label.lower())
                cand_lines.append(f"  - {label} ({nq} questoes)")

        parts.append(
            f'[{i}] INPUT: tema="{input_tema}"'
            + (f' subtema="{input_subtema}"' if input_subtema else "")
            + "\n    CANDIDATOS ENCONTRADOS NO BANCO:"
            + ("\n" + "\n".join(cand_lines) if cand_lines else "\n  (nenhum)")
            + "\n    TAREFA: Qual candidato e o melhor match para o INPUT? O match atual esta correto ou existe um candidato melhor?"
        )

    return "\n\n".join(parts)


class LLMJudge:
    """Validates and improves match quality using an LLM."""

    # Models that do not accept a custom temperature (only the API default is allowed)
    _NO_TEMPERATURE_MODELS: frozenset[str] = frozenset({
        "gpt-5-mini-2025-08-07",
        "gpt-5-mini",
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4-mini",
    })

    # Models that do not support response_format=json_object.
    # For these, JSON output is requested via the system prompt only.
    _NO_JSON_RESPONSE_FORMAT_MODELS: frozenset[str] = frozenset({
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4-mini",
    })

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-5-mini-2025-08-07",
        base_url: str | None = None,
        max_batch_size: int = 10,
        temperature: float = 0.1,
    ):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=api_key,
            **({"base_url": base_url} if base_url else {}),
        )
        self._model = model
        self._max_batch_size = max_batch_size
        self._temperature = temperature
        self._supports_temperature = model not in self._NO_TEMPERATURE_MODELS
        self._supports_json_response_format = model not in self._NO_JSON_RESPONSE_FORMAT_MODELS
        logger.info(
            "[LLMJudge] initialized: model=%s batch_size=%d temperature=%s json_format=%s",
            model,
            max_batch_size,
            f"{temperature}" if self._supports_temperature else "default (not sent)",
            "json_object" if self._supports_json_response_format else "prompt-only",
        )

    def judge_batch(
        self,
        items: list[dict[str, Any]],
        db_candidates: list[list[dict[str, Any]]],
    ) -> list[LLMVerdict]:
        """Judge multiple items in batches. Returns one verdict per item.

        Args:
            items: list of dicts with keys:
                input_tema, input_subtema, current_match, current_score
            db_candidates: for each item, a list of candidate dicts from DB
                with keys: tema, subtema, num_questions
        """
        all_verdicts: list[LLMVerdict] = []

        for batch_start in range(0, len(items), self._max_batch_size):
            batch_items = items[batch_start : batch_start + self._max_batch_size]
            batch_cands = db_candidates[
                batch_start : batch_start + self._max_batch_size
            ]

            verdicts = self._judge_single_batch(batch_items, batch_cands)
            all_verdicts.extend(verdicts)

        return all_verdicts

    def _judge_single_batch(
        self,
        items: list[dict[str, Any]],
        db_candidates: list[list[dict[str, Any]]],
    ) -> list[LLMVerdict]:
        """Send a single batch to the LLM and parse results."""
        user_prompt = _build_batch_prompt(items, db_candidates)

        try:
            call_kwargs: dict = {
                "model": self._model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            }
            if self._supports_temperature:
                call_kwargs["temperature"] = self._temperature
            if self._supports_json_response_format:
                call_kwargs["response_format"] = {"type": "json_object"}
            response = self._client.chat.completions.create(**call_kwargs)

            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            results = parsed.get("results", [])

            # Build index dict once — O(1) per item instead of O(n) linear scan
            results_by_index = {r.get("index"): r for r in results}
            verdicts = []
            for item_idx in range(len(items)):
                result = results_by_index.get(item_idx)
                if result:
                    verdicts.append(
                        LLMVerdict(
                            is_equivalent=bool(result.get("is_equivalent", False)),
                            confidence=float(result.get("confidence", 0)),
                            suggested_match=result.get("suggested_match"),
                            reasoning=str(result.get("reasoning", "")),
                        )
                    )
                else:
                    # LLM didn't return result for this index — keep original
                    verdicts.append(
                        LLMVerdict(
                            is_equivalent=True,
                            confidence=0.0,
                            suggested_match=None,
                            reasoning="LLM nao retornou resultado para este item",
                        )
                    )

            usage = response.usage
            if usage:
                logger.info(
                    "[LLMJudge] batch of %d items: %d prompt + %d completion tokens",
                    len(items),
                    usage.prompt_tokens,
                    usage.completion_tokens,
                )

            return verdicts

        except Exception as exc:
            logger.warning(
                "[LLMJudge] API call failed, falling back to original scores: %s",
                exc,
            )
            return [
                LLMVerdict(
                    is_equivalent=True,
                    confidence=0.0,
                    suggested_match=None,
                    reasoning=f"Fallback: erro na API ({exc})",
                )
                for _ in items
            ]

    def judge_single(
        self,
        input_tema: str,
        input_subtema: str | None,
        current_match: str,
        current_score: float,
        candidates: list[dict[str, Any]],
    ) -> LLMVerdict:
        """Convenience: judge a single item."""
        item = {
            "input_tema": input_tema,
            "input_subtema": input_subtema or "",
            "current_match": current_match,
            "current_score": current_score,
        }
        results = self.judge_batch([item], [candidates])
        return results[0]
