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
Voce e um especialista senior em curriculos de residencia medica brasileira, com profundo conhecimento \
de todas as bancas (FAMERP, USP, UNICAMP, UNIFESP, HIAE, HCPA, SCMSP, etc.).

Sua tarefa CRITICA: dado um tema de prova medica (INPUT) e uma lista de candidatos do banco de dados, \
voce deve decidir qual candidato e o MELHOR match semantico — ou rejeitar TODOS se nenhum for equivalente.

VOCE E A ULTIMA LINHA DE DEFESA. Matches incorretos prejudicam diretamente os alunos de residencia.

## PROCESSO DE RACIOCINIO (siga EXATAMENTE estes passos):
1. IDENTIFIQUE o assunto medico especifico do INPUT (doenca, procedimento, conceito)
2. IDENTIFIQUE a especialidade medica do INPUT (cardiologia, cirurgia, pediatria, etc.)
3. Para CADA candidato, avalie:
   a) O candidato trata da MESMA doenca/procedimento/conceito? (sinonimos contam)
   b) O candidato pertence a MESMA especialidade ou contexto clinico?
   c) O nivel de especificidade e compativel? (especifico → especifico, geral → geral)
4. ESCOLHA o candidato mais especifico que satisfaz (a), (b) e (c)
5. Se NENHUM candidato satisfaz os 3 criterios, retorne is_equivalent=false

## REGRAS DE EQUIVALENCIA (RIGOROSAS):

### ACEITAR como equivalente:
R1. Sinonimos medicos diretos: "IAM" = "Infarto Agudo do Miocardio" = "SCA com supra de ST"
R2. Abreviacoes reconhecidas: PCR, ITU, DPOC, RN, SCA, TEP, DM, HAS, ICC, AVC, TCE, etc.
R3. Subtema contido em tema: "Triagem Neonatal" contem "Teste do Coracaozinho", "Teste do Pezinho"
R4. Mesma entidade clinica com nomes diferentes entre bancas: \
"Politraumatizado | Atendimento Inicial" = "Trauma | Abordagem Inicial ao Politraumatizado"
R5. Hierarquia clinica: "Abdome agudo | Apendicite" pode mapear para "Abdome agudo | Apendicite aguda"
R6. Equivalencias conceituais: "Reanimacao neonatal" = "Ressuscitacao do RN em sala de parto"

### REJEITAR como NAO equivalente:
R7. Doencas DIFERENTES da mesma especialidade: Pneumonia ≠ DPOC, Apendicite ≠ Colecistite, \
Meningite ≠ Encefalite, Asma ≠ Bronquite, Diabetes ≠ Hipotireoidismo
R8. Especialidades DIFERENTES: Queimadura (Cirurgia) ≠ Queimadura (Dermatologia), \
Dor toracica (Cardiologia) ≠ Dor toracica (Pneumologia) — EXCETO quando o candidato e claramente \
uma abordagem sindrômica que engloba ambas
R9. Patologia especifica vs organizacao/politica: "TCE" ≠ "Semiologia neurologica", \
"Pneumonia" ≠ "Politica de saude publica"
R10. Generico vs Especifico SEM relacao direta: "Aspectos Gerais" so serve se NAO existir \
candidato especifico que corresponda ao INPUT
R11. Orgaos/sistemas diferentes: "Abscesso pulmonar" ≠ "Sinusite" (pulmao ≠ seios paranasais)
R12. Procedimentos diferentes: "Parto cesario" ≠ "Parto normal", "Intubacao" ≠ "Traqueostomia" \
(a menos que o contexto seja "via aerea" e ambos se encaixem)

## REGRA DE ESPECIFICIDADE (PRIORITARIA):
- Se o INPUT menciona subtopico especifico (ex: "Inflamatorio", "Obstrutivo", "Cardiogenico", \
"Apendicite", "Colecistite", "Reposicao volemica"), SEMPRE prefira o candidato que corresponda \
a essa especificidade.
- So aceite "Aspectos Gerais" se NENHUM candidato especifico estiver disponivel.
- Se o INPUT e generico (ex: "Abdome agudo") e existe candidato "Abdome agudo | Aspectos Gerais", \
este e um bom match.

## REGRA DO suggested_match:
- O valor de "suggested_match" DEVE ser copiado EXATAMENTE do texto de um dos candidatos \
(incluindo "tema | subtema" com a mesma grafia exata). NAO invente texto novo.
- Se o match atual ja e o melhor, retorne is_equivalent=true e suggested_match=null.
- Se um candidato alternativo e melhor, retorne is_equivalent=false e suggested_match com o texto exato.
- Se nenhum candidato e bom, retorne is_equivalent=false e suggested_match=null.

## ESCALA DE CONFIANCA:
- 0.95-1.00: Sinonimo direto ou match exato (IAM = Infarto Agudo do Miocardio)
- 0.85-0.94: Equivalencia clara com pequena diferenca de terminologia
- 0.70-0.84: Equivalencia provavel — mesmo assunto, terminologia bem diferente
- 0.50-0.69: Relacao parcial — mesma area mas pode nao ser a mesma entidade
- 0.00-0.49: Sem equivalencia — assuntos diferentes

## EXEMPLOS DETALHADOS:

### EQUIVALENTES (aceitar):
1. INPUT "Avaliacao do RN | Triagem Neonatal" + "Triagem neonatal | Teste do Coracaozinho" \
→ EQUIVALENTES (confidence=0.90) — subtema contido no tema (R3)
2. INPUT "Trauma | Abordagem Inicial" + "Politraumatizado | Atendimento Inicial" \
→ EQUIVALENTES (confidence=0.92) — mesma entidade clinica (R4)
3. INPUT "Abdome agudo Inflamatorio | Apendicite" + candidatos ["Abdome agudo | Aspectos Gerais", \
"Abdome agudo | Apendicite aguda"] → PREFIRA "Abdome agudo | Apendicite aguda" (confidence=0.95) (R5, especificidade)
4. INPUT "Choque | Choque Cardiogenico" + ["Cardiointensivismo | Choque (exceto choque septico)", \
"Cardiointensivismo | Aspectos Gerais"] → PREFIRA "Cardiointensivismo | Choque (exceto choque septico)" \
(confidence=0.88) (especificidade)
5. INPUT "HAS" + "Hipertensao arterial sistemica | Tratamento farmacologico" \
→ EQUIVALENTES (confidence=0.85) (R2 + R3)
6. INPUT "Insuficiencia cardiaca" + "ICC | Diagnostico e tratamento" \
→ EQUIVALENTES (confidence=0.95) (R1)
7. INPUT "Saude da mulher | Pre-natal" + "Assistencia pre-natal | Acompanhamento" \
→ EQUIVALENTES (confidence=0.88) (R4)

### NAO EQUIVALENTES (rejeitar — erros CRITICOS a evitar):
1. INPUT "Pneumonia" + "DPOC" → REJEITAR (R7 — doencas distintas do mesmo sistema respiratorio)
2. INPUT "Queimaduras" (Cirurgia) + "Queimaduras" (Dermatologia) → REJEITAR (R8 — contextos diferentes)
3. INPUT "Atendimento ao politraumatizado | Vias Aereas" + \
"Politica nacional de atencao basica | Atribuicoes" → REJEITAR (R9 — trauma ≠ politica de saude)
4. INPUT "Pneumonia | Covid" + "Infeccoes respiratorias | Coqueluche" → REJEITAR (R7 — agentes diferentes)
5. INPUT "Pneumonia | Abscesso Pulmonar" + "Infeccoes respiratorias | Sinusite" → REJEITAR (R11 — orgaos diferentes)
6. INPUT "Avaliacao do RN | Dermatoses do RN" + "Alojamento conjunto" → REJEITAR (R9 — conteudo ≠ organizacao)
7. INPUT "Avaliacao neurologica | TCE" + "Semiologia | Exame fisico e neuroanatomia" → REJEITAR (R9 — patologia ≠ propedeutica)
8. INPUT "Queimado | Reposicao volemica" + "Queimadura | Aspectos Gerais" \
→ REJEITAR se existir candidato mais especifico de reposicao (R10, especificidade)
9. INPUT "Diarreia aguda | Viral" + "Diarreia cronica | Doenca celiaca" → REJEITAR (R7 — doencas diferentes)
10. INPUT "Ictericia neonatal | Fototerapia" + "Ictericia | Hepatite" → REJEITAR (R7 — etiologias diferentes)
11. INPUT "Sepse neonatal" + "Sepse | Choque septico no adulto" → REJEITAR (R8 — populacoes diferentes)
12. INPUT "Diabetes mellitus tipo 1" + "Diabetes mellitus tipo 2" → REJEITAR (R7 — doencas distintas)

Responda SEMPRE em JSON valido:
{
  "results": [
    {
      "index": 0,
      "is_equivalent": true/false,
      "confidence": 0.0-1.0,
      "suggested_match": "tema | subtema EXATAMENTE como listado nos candidatos, ou null",
      "reasoning": "1) Assunto do INPUT: [X]. 2) Melhor candidato: [Y]. 3) Conclusao: [aceito/rejeito] porque [motivo]"
    }
  ]
}"""


def _clean_suggested_match(raw: str) -> str:
    """Strip prompt annotations from LLM-returned suggested_match.

    The candidate list shown to the LLM includes prefixes like '[MATCH ATUAL]'
    and suffixes like '(8 questoes)' or '(score=45%)'.  If the LLM copies one
    of those strings literally, the DB lookup will fail silently.  This
    function strips only known prompt annotations so that real DB names that
    contain parentheses (e.g. "Choque (exceto choque séptico)") are preserved.
    """
    import re

    s = raw.strip()
    # Remove leading markers: "* [MATCH ATUAL] ", "- ", "* " etc.
    s = re.sub(r"^[\*\-]\s*\[MATCH ATUAL\]\s*", "", s)
    s = re.sub(r"^[\*\-]\s*", "", s)
    # Remove ONLY known prompt-generated suffixes at end of string:
    #   (N questoes)            – question count annotation
    #   (score=N%)              – score annotation
    #   (sem match automatico)  – no-match annotation
    # These patterns are unambiguous and never appear in real DB names.
    s = re.sub(r"\s*\(\d+\s+questoes?\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(score=\d+%\)\s*$", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s*\(sem match[^)]*\)\s*$", "", s, flags=re.IGNORECASE)
    return s.strip()


def _build_batch_prompt(
    items: list[dict[str, Any]],
    db_candidates: list[list[dict[str, Any]]],
) -> str:
    """Build a single prompt for multiple items to judge in one API call."""
    parts = []
    for i, (item, candidates) in enumerate(zip(items, db_candidates)):
        input_tema = item.get("input_tema", "")
        input_subtema = item.get("input_subtema", "")
        equivalencia = item.get("equivalencia", "")
        current_match = item.get("current_match", "")
        current_score = item.get("current_score", 0)
        match_method = item.get("match_method", "")

        cand_lines = []
        seen = set()
        # Include current match as first candidate.
        if current_match:
            if current_score > 0:
                score_str = f"score={current_score:.0%}, metodo={match_method}"
            else:
                score_str = "sem match automatico"
            cand_lines.append(
                f"  * [MATCH ATUAL] {current_match} ({score_str})"
            )
            seen.add(current_match.lower())

        for c in candidates[:12]:
            t = c.get("tema", "")
            s = c.get("subtema", "")
            nq = c.get("num_questions", 0)
            label = f"{t} | {s}" if s else t
            if label.lower() not in seen:
                seen.add(label.lower())
                cand_lines.append(f"  - {label} ({nq} questoes)")

        no_candidates_hint = (
            "\n  (nenhum candidato encontrado — retorne is_equivalent=false e suggested_match=null)"
        )

        input_line = f'[{i}] INPUT: tema="{input_tema}"'
        if input_subtema:
            input_line += f' subtema="{input_subtema}"'
        if equivalencia:
            input_line += f'\n    EQUIVALENCIA INFORMADA PELO USUARIO: "{equivalencia}"'

        parts.append(
            input_line
            + "\n    CANDIDATOS ENCONTRADOS NO BANCO:"
            + ("\n" + "\n".join(cand_lines) if cand_lines else no_candidates_hint)
            + "\n    TAREFA: Siga o PROCESSO DE RACIOCINIO. Identifique o assunto do INPUT, compare com CADA candidato,"
            + " e escolha o mais especifico e correto. Se a EQUIVALENCIA do usuario apontar para um candidato valido, considere-a."
            + ' Em "suggested_match" use APENAS o texto "tema" ou "tema | subtema" sem prefixos nem sufixos.'
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
        max_batch_size: int = 5,
        temperature: float = 0.0,
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
