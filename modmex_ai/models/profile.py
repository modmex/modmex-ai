from __future__ import annotations

from typing import Literal

from modmex import BaseModel


ToolSchemaMode = Literal["openai", "json_schema"]
OutputMode = Literal["json_schema", "json_object", "prompted_json", "none"]


class ModelProfile(BaseModel):
    """Capabilities a model adapter declares.

    Fields that change Agent behavior:

    - ``supports_structured_output``: when False, the agent adds the output JSON
      schema to its instructions instead of relying on native structured output.
    - ``supports_tools``: when False, running an agent with tools or handoffs on
      this model fails fast.

    The remaining fields are informational (hosts may use them to route or
    validate); the runtime does not act on them.
    """

    supports_tools: bool = True
    supports_parallel_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_system_messages: bool = True
    tool_schema_mode: ToolSchemaMode = "json_schema"
    output_mode: OutputMode = "prompted_json"
