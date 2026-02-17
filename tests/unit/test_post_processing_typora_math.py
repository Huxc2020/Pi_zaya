from kb.converter.post_processing import postprocess_markdown


def test_typora_normalizes_display_argmin():
    src = r"""
$$
\theta^* = \argmin_{\theta}\left\|DNN_{\theta}(O) - T(x,y)\right\|_2^2 \tag{17}
$$
"""
    out = postprocess_markdown(src)
    assert r"\operatorname*{arg\,min}_{\theta}" in out
    assert r"\argmin_" not in out


def test_typora_normalizes_inline_argmax_and_arg_min():
    src = r"""
We optimize $x^*=\argmax_x f(x)$ and $y^*=\arg\min_y g(y)$ in one paragraph.
"""
    out = postprocess_markdown(src)
    assert r"\operatorname*{arg\,max}_x" in out
    assert r"\operatorname*{arg\,min}_y" in out
    assert r"\argmax_x" not in out
    assert r"\arg\min_y" not in out


def test_typora_keeps_existing_operatorname_form():
    src = r"""
Already normalized: $\operatorname*{arg\,min}_{\theta} J(\theta)$.
"""
    out = postprocess_markdown(src)
    assert out.count(r"\operatorname*{arg\,min}_{\theta}") == 1


def test_typora_normalizes_unicode_symbols_and_old_commands():
    src = r"""
$$
\rm{x} + α ≤ β + γ × δ + \mbox{subject to } x ∈ Ω
$$
"""
    out = postprocess_markdown(src)
    assert r"\mathrm{x}" in out
    assert r"\alpha" in out
    assert r"\leq" in out
    assert r"\times" in out
    assert r"\text{subject to }" in out
    assert r"\in" in out
    assert r"\Omega" in out
    assert "α" not in out
    assert "≤" not in out


def test_typora_normalizes_declare_math_operator_and_equation_env():
    src = r"""
$$
\DeclareMathOperator*{\fooop}{foo\,op}
\begin{equation}
y = \fooop_x f(x)
\end{equation}
$$
"""
    out = postprocess_markdown(src)
    assert r"\DeclareMathOperator" not in out
    assert r"\begin{equation}" not in out
    assert r"\end{equation}" not in out
    assert r"\operatorname*{foo\,op}_x" in out


def test_typora_moves_tag_out_of_aligned_environment():
    src = r"""
$$
\begin{aligned}
\min \left( a+b \right) \tag{10}
\end{aligned}
$$
"""
    out = postprocess_markdown(src)
    assert r"\begin{aligned}" in out
    assert r"\end{aligned} \tag{10}" in out
    assert r"\tag{10} \end{aligned}" not in out
