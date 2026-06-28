from ui.components.page_layout import render_page

from ui.views.scoring import render_scoring

render_page(
    title="📈 Batch Scoring",
    content_function=render_scoring,
)