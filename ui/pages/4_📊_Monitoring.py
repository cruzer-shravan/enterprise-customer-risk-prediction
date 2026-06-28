from ui.components.page_layout import render_page

from ui.views.monitoring import render_monitoring

render_page(
    title="📊 Monitoring",
    content_function=render_monitoring,
)