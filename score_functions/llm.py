"""OpenAI 호출 공통부. 추출(ex_traits)과 항목 비교(ex)가 함께 쓴다."""

import os
import re
import sys
from typing import Type, TypeVar

from dotenv import load_dotenv
from openai import BadRequestError, OpenAI
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)

_client: OpenAI | None = None
_warned_undated = False
# 모델이 지원하지 않는다고 거절한 파라미터. 한 번 거절당하면 이후 호출에서 뺀다.
_unsupported: set[str] = set()


def resolve_model() -> str:
    """사용할 모델 이름. 날짜가 고정된 스냅샷을 OPENAI_MODEL로 지정해야 한다."""
    global _warned_undated
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    if not model:
        raise RuntimeError(
            "OPENAI_MODEL이 설정되어 있지 않습니다. 날짜가 고정된 스냅샷 이름을 지정해 주세요. "
            "(실행마다 점수가 흔들리지 않도록 기본값을 두지 않습니다.)"
        )
    if "latest" in model.lower():
        raise RuntimeError(f"OPENAI_MODEL={model!r}: latest 별칭은 쓸 수 없습니다. 날짜가 고정된 스냅샷을 지정해 주세요.")
    if not _warned_undated and not re.search(r"\d{4}-?\d{2}-?\d{2}", model):
        _warned_undated = True
        print(
            f"경고: OPENAI_MODEL={model!r}에 날짜가 보이지 않습니다. 스냅샷이 아닌 별칭이면 점수가 달라질 수 있습니다.",
            file=sys.stderr,
        )
    return model


def get_client() -> OpenAI:
    global _client
    if _client is None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 설정되어 있지 않습니다.")
        _client = OpenAI(max_retries=4, timeout=120)
    return _client


def _rejected_param(err: BadRequestError) -> str | None:
    text = f"{getattr(err, 'param', '') or ''} {err}".lower()
    for name in ("temperature", "reasoning_effort"):
        if name in text:
            return name
    return None


def parse_chat(model: str, system: str, messages: list[dict], response_format: Type[T]) -> tuple[T, str]:
    """구조화 응답을 받아 (파싱된 객체, 원본 JSON 문자열)을 돌려준다.

    temperature는 쓸 수 있는 모델에서만 쓰고, 거절당하면 빼고 다시 부른다.
    (점수 일관성은 temperature가 아니라 3회 합의와 캐시가 보장한다.)
    """
    client = get_client()
    for _ in range(3):
        kwargs: dict = {
            "model": model,
            "messages": [{"role": "system", "content": system}, *messages],
            "response_format": response_format,
        }
        if "temperature" not in _unsupported:
            kwargs["temperature"] = float(os.getenv("OPENAI_TEMPERATURE", "0"))
        effort = os.getenv("OPENAI_REASONING_EFFORT", "none").strip()
        if effort and "reasoning_effort" not in _unsupported:
            kwargs["reasoning_effort"] = effort
        try:
            completion = client.chat.completions.parse(**kwargs)
        except BadRequestError as err:
            rejected = _rejected_param(err)
            if rejected and rejected not in _unsupported:
                _unsupported.add(rejected)
                continue
            raise
        message = completion.choices[0].message
        if message.refusal:
            raise RuntimeError(f"모델이 응답을 거부함: {message.refusal}")
        if message.parsed is None:
            raise RuntimeError("모델이 구조화 응답을 반환하지 않음")
        return message.parsed, message.content or ""
    raise RuntimeError("모델 파라미터 조정 후에도 요청이 거절되었습니다.")
