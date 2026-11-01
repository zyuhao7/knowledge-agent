from knowledge_agent.chunking import chunk_text


def test_short_text_is_one_chunk() -> None:
    assert chunk_text("第一段。\n\n第二段。", 100, 20) == ["第一段。\n第二段。"]


def test_long_text_is_split_with_bounded_size() -> None:
    text = "\n\n".join("这是一段用于测试分块的中文内容，包含足够多的字符。" * 4 for _ in range(10))
    chunks = chunk_text(text, 120, 30)
    assert len(chunks) > 1
    assert all(chunk.strip() for chunk in chunks)
    assert all(len(chunk) <= 150 for chunk in chunks)


def test_overlap_must_be_smaller_than_size() -> None:
    try:
        chunk_text("abc", 50, 50)
    except ValueError as error:
        assert "overlap" in str(error)
    else:
        raise AssertionError("expected ValueError")
