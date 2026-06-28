from ui.components.page_layout import render_page

from ui.views.explainability import render_explainability

render_page(
    title="🧠 Explainability",
    content_function=render_explainability,
)