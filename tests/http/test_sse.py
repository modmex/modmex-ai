from modmex_ai.http import parse_sse_lines


def test_parse_sse_lines_yields_json_events():
    events = list(parse_sse_lines([
        'data: {"type":"delta"}\n',
        "\n",
        "data: [DONE]\n",
        "\n",
    ]))

    assert events == [{"type": "delta"}]


def test_parse_sse_lines_flushes_final_event_without_done():
    events = list(parse_sse_lines([
        'data: {"type":"delta"}\n',
        "\n",
    ]))

    assert events == [{"type": "delta"}]
