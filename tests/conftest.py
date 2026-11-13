import os

# Tests never talk to LangSmith.
os.environ["LANGSMITH_TRACING"] = "false"
