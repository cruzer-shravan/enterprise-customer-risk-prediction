from ui.components.page_layout import render_page

from ui.views.about import render_about

render_page(
    title="ℹ About",
    content_function=render_about,
)