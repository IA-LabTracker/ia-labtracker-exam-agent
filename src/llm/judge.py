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
Sua tarefa e avaliar se dois temas/subtemas de provas medicas sao equivalentes.

Regras:
1. Considere sinonimos medicos (ex: "IAM" = "Infarto Agudo do Miocardio" = "Sindrome Coronariana Aguda")
2. Subtemas podem estar contidos em temas mais amplos (ex: "Triagem Neonatal" contem "Teste do Coracaozinho" e "Teste da Oximetria")
3. Considere abreviacoes comuns (PCR = Parada Cardiorrespiratoria, ITU = Infeccao do Trato Urinario, DPOC = Doenca Pulmonar Obstrutiva Cronica)
4. Diferentes bancas (FAMERP, USP, UNICAMP, UNIFESP, etc.) usam terminologias diferentes para o mesmo assunto
5. Seja rigoroso: temas de especialidades diferentes NAO sao equivalentes mesmo que compartilhem palavras
6. Para cada item, primeiro explique brevemente o que cada tema significa, depois de o veredito

Exemplos de equivalencia:
- "PCR" = "Parada Cardiorrespiratoria" = "Parada Cardiaca" → EQUIVALENTES (sinonimos)
- "Avaliacao do Recem-nascido | Triagem Neonatal" ≈ "Triagem neonatal | Teste do Coracaozinho" → EQUIVALENTES (subtema contido)
- "Diabetes" ≈ "Diabetes Mellitus | Tipo 2" → EQUIVALENTES (tema amplo contem subtema)
- "Queimaduras" (Cirurgia) != "Queimaduras" (Dermatologia) → NAO EQUIVALENTES (especialidades diferentes)
- "Pneumonia" != "DPOC" → NAO EQUIVALENTES (doencas distintas)
- "Trauma | Abordagem Inicial" ≈ "Politraumatizado | Atendimento Inicial" → EQUIVALENTES (mesma abordagem, terminologia diferente)

Responda SEMPRE em JSON valido com esta estrutura exata:
{
  "results": [
    {
      "index": 0,
      "is_equivalent": true/false,
      "confidence": 0.0-1.0,
      "suggested_match": "tema sugerido do banco ou null se o match atual esta correto",
      "reasoning": "O que cada tema significa e por que sao/nao sao equivalentes (1-2 frases)"
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
        for c in candidates[:5]:
            t = c.get("tema", "")
            s = c.get("subtema", "")
            nq = c.get("num_questions", 0)
            label = f"{t} | {s}" if s else t
            cand_lines.append(f"  - {label} ({nq} questoes)")

        parts.append(
            f'[{i}] INPUT: tema="{input_tema}"'
            + (f' subtema="{input_subtema}"' if input_subtema else "")
            + f'\n    MATCH ATUAL: "{current_match}" (score={current_score:.0%})'
            + f"\n    CANDIDATOS DO BANCO:"
            + ("\n" + "\n".join(cand_lines) if cand_lines else "\n  (nenhum)")
            + "\n    PERGUNTA: O match atual esta correto? Se nao, qual candidato e melhor?"
        )

    return "\n\n".join(parts)


class LLMJudge:
    """Validates and improves match quality using an LLM."""

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
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
        logger.info(
            "[LLMJudge] initialized: model=%s batch_size=%d",
            model,
            max_batch_size,
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
            response = self._client.chat.completions.create(
                model=self._model,
                temperature=self._temperature,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content or "{}"
            parsed = json.loads(content)
            results = parsed.get("results", [])

            verdicts = []
            for item_idx in range(len(items)):
                # Find the result for this index
                result = next(
                    (r for r in results if r.get("index") == item_idx),
                    None,
                )
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
