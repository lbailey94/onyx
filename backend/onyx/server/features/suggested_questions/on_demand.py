"""An experiment: write a person's suggested questions inside the request.

Nothing is stored or read from the shelves. Every call runs the whole
pipeline (choose documents, read chunks, triage, write, search test) and
returns up to SQ_SHOWN questions with the seconds each step took, so the
synchronous path can be measured against the Celery runs.
"""

import time
from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel
from sqlalchemy.orm import Session

from onyx.db.models import User
from onyx.secondary_llm_flows.suggested_questions import (
    build_triage_clients,
    build_writing_llms,
    triage_chunks,
    write_questions,
)
from onyx.server.features.suggested_questions.config import SQ_SHOWN
from onyx.server.features.suggested_questions.models import ShownQuestion
from onyx.server.features.suggested_questions.selection import (
    best_chunks_per_document,
    choose_documents,
    fetch_chunks,
)
from onyx.server.features.suggested_questions.store import role_for
from onyx.server.features.suggested_questions.verification import run_search_test
from onyx.server.settings.store import load_settings

T = TypeVar("T")


class StepCounts(BaseModel):
    documents: int
    chunks: int
    passages: int
    drafts: int
    passed_search_test: int


class OnDemandSuggestions(BaseModel):
    questions: list[ShownQuestion]
    # Seconds per step, in pipeline order, and the whole request.
    timings_s: dict[str, float]
    counts: StepCounts


def generate_on_demand(
    db_session: Session, user: User, tenant_id: str
) -> OnDemandSuggestions:
    timings: dict[str, float] = {}

    def timed(step: str, fn: Callable[[], T]) -> T:
        start = time.perf_counter()
        result = fn()
        timings[step] = round(time.perf_counter() - start, 3)
        return result

    started = time.perf_counter()
    # No exclusions: this call does not know what earlier calls wrote.
    candidates = timed(
        "choose_documents",
        lambda: choose_documents(db_session, [user], set(), {user.id: set()}),
    )
    chunks = timed(
        "fetch_chunks", lambda: fetch_chunks(db_session, candidates, tenant_id)
    )
    triaged = timed(
        "triage_chunks", lambda: triage_chunks(chunks, build_triage_clients())
    )
    passages = best_chunks_per_document(triaged)
    rank = {c.document_id: i for i, c in enumerate(candidates)}
    passages.sort(key=lambda p: rank.get(p.chunk.document_id, len(rank)))
    role = role_for(user)
    drafts = timed(
        "write_questions",
        lambda: write_questions(
            passages,
            build_writing_llms(),
            load_settings().company_name,
            [role] if role else [],
        ),
    )
    # Searched as the person, so the questions are ones they can open.
    kept = timed("run_search_test", lambda: run_search_test(drafts, user=user))
    timings["total"] = round(time.perf_counter() - started, 3)

    private_documents = {c.document_id for c in candidates if c.owner_ids}
    shown: list[ShownQuestion] = []
    seen_documents: set[str] = set()
    for question in kept:
        if question.document_id in seen_documents:
            continue
        seen_documents.add(question.document_id)
        shown.append(
            ShownQuestion(
                id=question.id,
                message=question.text,
                private=question.document_id in private_documents,
            )
        )
        if len(shown) == SQ_SHOWN:
            break
    return OnDemandSuggestions(
        questions=shown,
        timings_s=timings,
        counts=StepCounts(
            documents=len(candidates),
            chunks=len(chunks),
            passages=len(passages),
            drafts=len(drafts),
            passed_search_test=len(kept),
        ),
    )
