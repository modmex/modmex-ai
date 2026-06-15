from __future__ import annotations

from typing import Literal

from modmex import BaseModel


ToolSchemaMode = Literal["openai", "json_schema"]
OutputMode = Literal["json_schema", "json_object", "prompted_json", "none"]


class ModelProfile(BaseModel):
    supports_tools: bool = True
    supports_parallel_tool_calls: bool = False
    supports_structured_output: bool = False
    supports_system_messages: bool = True
    tool_schema_mode: ToolSchemaMode = "json_schema"
    output_mode: OutputMode = "prompted_json"
