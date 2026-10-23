"""Offline evaluation harness for the support agent.

Runs every case in eval_cases.json end-to-end through the real agent loop (retrieval, LLM, tools, DB)
and scores three things per case:
  * tool routing   - did the agent call the expected tool (or correctly call none)?
  * answer quality - does the reply contain the expected facts?
  * retrieval      - was the expected knowledge-base document retrieved?

Usage:  python -m evals.run_evals [--min-pass-rate 0.9]
Set LLM_PROVIDER=openai|anthropic to evaluate a real model with the same suite.
Exits non-zero when the pass rate is below the threshold, so CI blocks prompt/model regressions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.agent import SupportAgent
from app.config import get_settings
from app.db import init_db
from app.llm import get_llm
from app.main import seed_demo_bookings
from app.models import Conversation
from app.retrieval import KnowledgeBase

CASES = Path(__file__).with_name("eval_cases.json")


def run(min_pass_rate: float) -> int:
    settings = get_settings()
    engine = create_engine("sqlite:///:memory:")
    init_db(engine)
    Session = sessionmaker(bind=engine, expire_on_commit=False)

    knowledge = KnowledgeBase()
    knowledge.load_directory(settings.knowledge_dir)
    agent = SupportAgent(get_llm(settings), knowledge, settings.agent_max_steps, settings.retrieval_top_k)

    cases = json.loads(CASES.read_text())
    passed = tool_ok = answer_ok = retrieval_ok = retrieval_total = 0
    with Session() as session:
        seed_demo_bookings(session)
        for case in cases:
            conversation = Conversation(customer_ref=f"eval-{case['id']}")
            session.add(conversation)
            session.flush()
            result = agent.run(session, conversation, case["input"])

            called = result.tool_calls[0]["name"] if result.tool_calls else None
            t_ok = called == case["expect_tool"]
            a_ok = all(fact.lower() in result.reply.lower() for fact in case["expect_in_reply"])
            r_ok = True
            if case["expect_source"]:
                retrieval_total += 1
                r_ok = case["expect_source"] in result.sources
                retrieval_ok += r_ok
            ok = t_ok and a_ok and r_ok
            tool_ok += t_ok
            answer_ok += a_ok
            passed += ok
            print(f"{'PASS' if ok else 'FAIL'}  {case['id']:<16} tool={called!s:<18} reply={result.reply[:70]!r}")

    n = len(cases)
    pass_rate = passed / n
    print("\n--- summary ---")
    print(f"model            : {agent.llm.name}")
    print(f"cases            : {n}")
    print(f"tool routing     : {tool_ok}/{n}")
    print(f"answer accuracy  : {answer_ok}/{n}")
    print(f"retrieval hit    : {retrieval_ok}/{retrieval_total}")
    print(f"pass rate        : {pass_rate:.0%} (threshold {min_pass_rate:.0%})")
    return 0 if pass_rate >= min_pass_rate else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    sys.exit(run(parser.parse_args().min_pass_rate))
