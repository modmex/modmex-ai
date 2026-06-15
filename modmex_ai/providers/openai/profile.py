from modmex_ai.models import ModelProfile


OPENAI_RESPONSES_PROFILE = ModelProfile(
    supports_tools=True,
    supports_parallel_tool_calls=True,
    supports_structured_output=True,
    supports_system_messages=True,
    tool_schema_mode="openai",
    output_mode="json_schema",
)


OPENAI_CHAT_PROFILE = ModelProfile(
    supports_tools=True,
    supports_parallel_tool_calls=True,
    supports_structured_output=True,
    supports_system_messages=True,
    tool_schema_mode="openai",
    output_mode="json_schema",
)

