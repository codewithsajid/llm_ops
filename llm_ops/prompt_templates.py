# llm_ops/prompt_templates.py
from jinja2 import Template

SYSTEM = (
    "You are a helpful assistant. "
    "Answer in 3‑4 sentences and add citations like [1], [2] at the end of each fact."
)

QA_TEMPLATE = """
### Task
Answer the following question based on the provided context. Your answer should be concise (3–4 sentences), use only information present in the context, and cite every fact with [n], where n is the snippet index.

### Context
{% for passage in context %}
[{{ loop.index }}] (Source: {{ passage.meta.source }}) {{ passage.text }}
{% endfor %}

### Question
{{ question }}

### Answer
"""

# Modify the QA template to request more specific details
WEB_FOCUSED_QA_TEMPLATE = """
### Task
Answer the following question using ONLY the following web-retrieved snippets.  
Do NOT add or infer any information not present in these snippets.  
Cite every statement by appending [n], matching the snippet index.

### Context
{% for passage in context if passage.meta.source == "web" %}
[{{ loop.index }}] {{ passage.text }}
{% endfor %}

### Question
{{ question }}

### Answer
"""
