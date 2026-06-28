from ui.components.page_layout import render_page

from ui.views.dashboard import render_dashboard

render_page(
    title="📊 Dashboard",
    content_function=render_dashboard,
)