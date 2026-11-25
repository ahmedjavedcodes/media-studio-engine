"""Push the Art Director, Prompt Writer and critic system prompts to LangSmith Hub with a version tag.

    uv run python playground/push_prompts.py          # tags with dev (or prod when APP_ENV=prod)

This WRITES to your LangSmith workspace, so run it yourself. The app pulls
`media-art-director:<tag>`, `media-prompt-writer:<tag>` and `media-critic:<tag>`, and falls back to prompts/create/txt2img/*.md
when the Hub has nothing. Checked against the real Hub on 2026-11-25 (langsmith 0.16):
pulling `<name>:<tag>` works. It tags the newest commit, and is safe to run again.
"""

from langchain_core.prompts import ChatPromptTemplate
from langsmith import Client
from langsmith.utils import LangSmithConflictError

from app.core.config import get_settings, prompts_dir
from app.core.prompts import LOCAL_FILES


def main() -> None:
    s = get_settings()
    client = Client(api_key=s.langsmith_api_key)
    refs = {
        "art_director": s.art_director_prompt_ref,
        "prompt_writer": s.writer_prompt_ref,
        "critic": s.critic_prompt_ref,
    }
    for name, path in LOCAL_FILES.items():
        text = (prompts_dir() / path).read_text("utf-8")
        # mustache keeps the single braces in the JSON examples from being read as variables
        prompt = ChatPromptTemplate.from_messages([("system", text)], template_format="mustache")
        # `commit_tags` tags the new commit. (`tags` would only label the repository.)
        try:
            url = client.push_prompt(refs[name], object=prompt, commit_tags=[s.hub_tag])
        except LangSmithConflictError:  # the text is the same as the newest commit: nothing to add
            url = "(unchanged)"
        newest = next(iter(client.list_prompt_commits(f"-/{refs[name]}", limit=1)))
        try:
            client.pull_prompt_commit(f"{refs[name]}:{s.hub_tag}")
        except Exception:  # the text did not change, so no new commit took the tag: tag this one
            # The tag call wants the commit's UUID (`id`), not its hash.
            client._create_commit_tags(f"-/{refs[name]}", str(newest.id), [s.hub_tag])
        latest = client.pull_prompt_commit(f"{refs[name]}:{s.hub_tag}")  # the tag must now resolve
        print(f"{refs[name]}:{s.hub_tag} -> {latest.commit_hash[:8]}  {url}")


if __name__ == "__main__":
    main()
