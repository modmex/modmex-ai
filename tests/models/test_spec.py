import pytest

from modmex_ai import FallbackModel, model_from
from modmex_ai.models.spec import parse
from modmex_ai.providers.anthropic import AnthropicMessagesModel
from modmex_ai.providers.bedrock import BedrockConverseModel
from modmex_ai.providers.openai import OpenAIResponsesModel


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "o")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a")


def test_one_name_is_that_providers_client_with_its_key():
    claude = model_from("anthropic:claude-sonnet-5-5")
    assert isinstance(claude, AnthropicMessagesModel) and claude.model == "claude-sonnet-5-5" and claude.api_key == "a"
    gpt = model_from(" openai : gpt-5.6-terra ")
    assert isinstance(gpt, OpenAIResponsesModel) and gpt.model == "gpt-5.6-terra" and gpt.api_key == "o"
    assert isinstance(model_from("bedrock:anthropic.claude-sonnet-5-5", region="us-west-2"), BedrockConverseModel)


def test_several_names_fall_back_in_order_and_skip_providers_without_key(monkeypatch):
    both = model_from("anthropic:claude-sonnet-5-5,openai:gpt-5.6-terra")
    assert isinstance(both, FallbackModel)
    assert [type(model) for model in both.models] == [AnthropicMessagesModel, OpenAIResponsesModel]
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert isinstance(model_from("anthropic:claude-sonnet-5-5,openai:gpt-5.6-terra"), OpenAIResponsesModel)


def test_options_reach_the_client():
    assert model_from("openai:gpt-5.6-terra", api_key="explicit").api_key == "explicit"


@pytest.mark.parametrize("spec", ["", "claude-sonnet-5-5", "anthropic:", "mistral:large", " , "])
def test_bad_names_say_how_to_write_them(spec):
    with pytest.raises(ValueError):
        parse(spec)
