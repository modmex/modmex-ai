from modmex_ai import Usage


def test_usage_add_sums_token_fields_and_keeps_raw_items():
    usage = Usage(
        input_tokens=2,
        output_tokens=1,
        total_tokens=3,
        details={"raw": {"input_tokens": 2, "output_tokens": 1}},
    )

    returned = usage.add(Usage(
        input_tokens=3,
        output_tokens=4,
        total_tokens=7,
        cached_input_tokens=1,
        reasoning_output_tokens=2,
        details={"raw": {"input_tokens": 3, "output_tokens": 4}},
    ))

    assert returned is usage
    assert usage.input_tokens == 5
    assert usage.output_tokens == 5
    assert usage.total_tokens == 10
    assert usage.cached_input_tokens == 1
    assert usage.reasoning_output_tokens == 2
    assert usage.details["raw_items"] == [
        {"input_tokens": 3, "output_tokens": 4},
    ]


def test_usage_copy_returns_independent_value():
    usage = Usage(input_tokens=1, details={"raw": {"x": 1}})

    copied = usage.copy()
    copied.input_tokens = 2
    copied.details["raw"] = {"x": 2}

    assert usage.input_tokens == 1
    assert usage.details["raw"] == {"x": 1}
