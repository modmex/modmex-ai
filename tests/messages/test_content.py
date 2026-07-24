import pytest

from modmex_ai.messages import FileInput, ImageInput, InputDetail, TextInput


def test_file_input_requires_exactly_one_source():
    assert FileInput(url="https://files.example/rate-confirmation.pdf").url == (
        "https://files.example/rate-confirmation.pdf"
    )

    with pytest.raises(ValueError, match="exactly one"):
        FileInput()

    with pytest.raises(ValueError, match="exactly one"):
        FileInput(url="https://files.example/document.pdf", file_id="file-123")


def test_image_input_requires_exactly_one_source_and_preserves_detail():
    image = ImageInput(data="YWJj", media_type="image/png", detail=InputDetail.HIGH)

    assert image.detail is InputDetail.HIGH

    with pytest.raises(ValueError, match="exactly one"):
        ImageInput()

    with pytest.raises(ValueError, match="exactly one"):
        ImageInput(url="https://files.example/page.png", data="YWJj")


def test_content_parts_have_stable_neutral_types():
    assert TextInput(text="Read this document.").type == "text"
    assert FileInput(file_id="file-123").type == "file"
    assert ImageInput(url="https://files.example/page.png").type == "image"
