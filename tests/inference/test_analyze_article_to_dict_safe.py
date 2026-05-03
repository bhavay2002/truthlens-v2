from src.inference.analyze_article import ArticleAnalyzer


class _Obj:
    def to_dict(self):
        return {"a": 1}


def test_to_dict_safe_handles_dict_and_object():
    assert ArticleAnalyzer._to_dict_safe({"x": 1}) == {"x": 1}
    assert ArticleAnalyzer._to_dict_safe(_Obj()) == {"a": 1}
    assert ArticleAnalyzer._to_dict_safe(None) == {}
