"""最小 QA 评测：替换 answer_question() 即可测试自己的应用。"""
import os
from pathlib import Path

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "1")

from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name(".env"))

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import LocalModel
from deepeval.test_case import LLMTestCase, SingleTurnParams


# 演示用应用，不调用生成模型；实际项目可替换成模型或 HTTP API 调用。
def answer_question(question: str) -> str:
    answers = {
        "Python 的列表和元组有什么区别？": "列表可修改，元组不可修改。两者都可以存储多个元素。",
        "什么是二分查找？": "二分查找在有序序列中每次排除一半的搜索范围，时间复杂度为 O(log n)。",
    }
    return answers.get(question, "不知道。")


CASES = [
    ("Python 的列表和元组有什么区别？", "列表是可变序列，元组是不可变序列。"),
    ("什么是二分查找？", "针对有序数据，每次将搜索范围缩小一半，时间复杂度为 O(log n)。"),
]


@pytest.fixture(scope="session")
def judge():
    api_key = os.getenv("QA_JUDGE_API_KEY")
    model = os.getenv("QA_JUDGE_MODEL")
    if not api_key or not model:
        pytest.fail("请复制 .env.example 为 .env，并填写 QA_JUDGE_API_KEY 和 QA_JUDGE_MODEL。")
    return LocalModel(
        model=model,
        api_key=api_key,
        base_url=os.getenv("QA_JUDGE_BASE_URL") or "https://api.openai.com/v1",
        temperature=0,
    )


@pytest.mark.parametrize("question,expected", CASES, ids=["list_vs_tuple", "binary_search"])
def test_qa(question, expected, judge):
    test_case = LLMTestCase(
        input=question,
        actual_output=answer_question(question),
        expected_output=expected,
    )
    metric = GEval(
        name="Answer Correctness",
        criteria="判断实际回答是否正确回答问题、覆盖参考答案的关键事实，且没有事实错误。允许措辞不同，不要求逐字一致。",
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT, SingleTurnParams.EXPECTED_OUTPUT],
        model=judge,
        threshold=0.7,
        async_mode=False,
    )
    assert_test(test_case=test_case, metrics=[metric], run_async=False)
