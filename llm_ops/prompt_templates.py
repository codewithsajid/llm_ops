from jinja2 import Template

SYSTEM = (
    "You are a helpful assistant. Answer with ≤ 4 sentences. "
    "Cite sources like [1], [2] at the end of each fact."
)

QA_TEMPLATE = """Answer the following question using provided context. Focus on accuracy and relevance.

Context:
{%- for hit in context %}
[{{loop.index}}] {{ hit.text }}
{%- endfor %}

Question: {{question}}

Instructions:
1. Give a direct, concise answer (3-4 sentences)
2. Include relevant citations [1], [2] etc.
3. Only answer what's supported by the context

Available Sources:
{%- for hit in context %}
[{{loop.index}}] {% if hit.source == 'web' %}{{hit.title}} ({{hit.url}}){% else %}{{hit.doc_id}}{% endif %}
{%- endfor %}
"""
