from pathlib import Path

from kb.converter.config import ConvertConfig
from kb.converter.pipeline import PDFConverter


def _make_converter(tmp_path):
    cfg = ConvertConfig(
        pdf_path=tmp_path / "dummy.pdf",
        out_dir=tmp_path,
        translate_zh=False,
        start_page=0,
        end_page=-1,
        skip_existing=False,
        keep_debug=False,
        llm=None,
    )
    return PDFConverter(cfg)


class _DummyLLMWorker:
    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    def call_llm_page_to_markdown(
        self,
        png_bytes,
        *,
        page_number,
        total_pages,
        hint,
        speed_mode,
        is_references_page,
    ):
        self.calls.append(
            {
                "page_number": page_number,
                "total_pages": total_pages,
                "hint": hint,
                "speed_mode": speed_mode,
                "is_references_page": is_references_page,
            }
        )
        if self._outputs:
            return self._outputs.pop(0)
        return None


def test_fragmented_math_detector_flags_split_equation():
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    assert PDFConverter._looks_fragmented_math_output(broken) is True


def test_fragmented_math_detector_flags_mixed_complete_plus_shards():
    mixed = """
$$
L = \\text{loss\\_func} \\sum_{n=1}^{M}\\left(DNN(u_n^*,v_n^*)\\right)
$$

M

(DNN(u *), v *)

)(15)

$$
n = 1
$$
"""
    assert PDFConverter._looks_fragmented_math_output(mixed) is True


def test_fragmented_math_detector_accepts_coherent_equation():
    clean = "$$L = \\text{loss\\_func} \\sum_{n=1}^{N}(DNN(u_n),u_n) \\tag{16}$$"
    assert PDFConverter._looks_fragmented_math_output(clean) is False


def test_guardrails_retry_then_fallback_when_still_fragmented(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, broken])
    converter.llm_worker = dummy
    monkeypatch.setenv("KB_PDF_VISION_FRAGMENT_FALLBACK", "1")

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: "FALLBACK_OK",
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=5,
        total_pages=12,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == "FALLBACK_OK"
    assert len(dummy.calls) == 2


def test_guardrails_accepts_retry_result_when_fixed(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    clean = "$$L = \\text{loss\\_func} \\sum_{n=1}^{N}(DNN(u_n),u_n) \\tag{16}$$"
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, clean])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("fallback should not run when retry succeeded")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=2,
        total_pages=8,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == clean
    assert len(dummy.calls) == 2


def test_guardrails_skip_math_fragment_check_for_references_page(tmp_path, monkeypatch):
    broken = """
$$
\\sum^{N}
$$

N
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("references page should not enter fallback in this test")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=1,
        total_pages=6,
        page_hint="references page",
        speed_mode="normal",
        is_references_page=True,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == broken
    assert len(dummy.calls) == 1


def test_restore_formula_placeholders_exact_and_fuzzy():
    md = "Before [[EQ_1]] middle [ EQ_2 ] after."
    mapping = {
        "[[EQ_1]]": "$$\nA=B\n$$",
        "[[EQ_2]]": "$$\nC=D\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "[[EQ_1]]" not in out
    # Single-bracket variant is intentionally not matched now.
    assert "[ EQ_2 ]" in out
    assert "$$\nA=B\n$$" in out
    assert "$$\nC=D\n$$" not in out


def test_restore_formula_placeholders_leaves_missing_unmodified():
    md = "Only one token: [[EQ_1]]"
    mapping = {
        "[[EQ_1]]": "$$\nA=B\n$$",
        "[[EQ_2]]": "$$\nC=D\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "$$\nA=B\n$$" in out
    assert "$$\nC=D\n$$" not in out


def test_restore_formula_placeholders_backslash_safe():
    md = "Math token [[EQ_1]] done."
    mapping = {
        "[[EQ_1]]": "$$\nL=\\sum_{n=1}^{N}\\mu_n\\text{ok}\n$$",
    }
    out = PDFConverter._restore_formula_placeholders(md, mapping)
    assert "\\sum_{n=1}^{N}" in out
    assert "\\mu_n" in out
    assert "\\text{ok}" in out


def test_guardrails_default_keeps_vl_output_when_fragmented(tmp_path, monkeypatch):
    broken = """
$$
\\frac{N}{T}
$$

N

$$
\\sum^{N}
$$

( DNN ( u n
), u n
)(16)
"""
    converter = _make_converter(tmp_path)
    dummy = _DummyLLMWorker([broken, broken])
    converter.llm_worker = dummy

    monkeypatch.setattr(
        converter,
        "_process_page",
        lambda page, page_index, pdf_path, assets_dir: (_ for _ in ()).throw(
            AssertionError("fallback should be disabled by default")
        ),
    )

    out = converter._convert_page_with_vision_guardrails(
        png_bytes=b"fake",
        page=object(),
        page_index=4,
        total_pages=10,
        page_hint="",
        speed_mode="normal",
        is_references_page=False,
        pdf_path=Path("dummy.pdf"),
        assets_dir=tmp_path,
    )

    assert out == broken


def test_legacy_extra_cleanup_toggle(monkeypatch, tmp_path):
    converter = _make_converter(tmp_path)

    monkeypatch.delenv("KB_PDF_LEGACY_EXTRA_CLEANUP", raising=False)
    assert converter._legacy_extra_cleanup_enabled() is False

    monkeypatch.setenv("KB_PDF_LEGACY_EXTRA_CLEANUP", "1")
    assert converter._legacy_extra_cleanup_enabled() is True


def test_inject_missing_page_image_links_for_figure_caption():
    md = """
Some paragraph.
Figure 5. Example caption text.
More paragraph.
""".strip()
    out = PDFConverter._inject_missing_page_image_links(
        md,
        page_index=4,
        image_names=["page_5_fig_1.png"],
        is_references_page=False,
    )
    assert "![Figure 5](./assets/page_5_fig_1.png)" in out
    assert out.index("![Figure 5](./assets/page_5_fig_1.png)") < out.index("Figure 5. Example caption text.")


def test_inject_missing_page_image_links_skips_references_page():
    md = "Figure 20. This line should stay unchanged."
    out = PDFConverter._inject_missing_page_image_links(
        md,
        page_index=19,
        image_names=["page_20_fig_1.png"],
        is_references_page=True,
    )
    assert out == md
