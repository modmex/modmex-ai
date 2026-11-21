from modmex_ai.models import ModelProfile


ANTHROPIC_MESSAGES_PROFILE = ModelProfile(
    supports_tools=True,
    supports_parallel_tool_calls=True,
    supports_structured_output=True,
    supports_system_messages=True,
    tool_schema_mode="json_schema",
    output_mode="json_schema",
)
