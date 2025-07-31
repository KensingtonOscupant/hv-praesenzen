from enum import Enum
from pathlib import Path
import weave

class Prompt(Enum):
    SYSTEM = "system"
    BASE = "base"

    @property
    def path(self) -> Path:
        """gets path to .txt file where prompt is stored"""
        return Path("prompts") / f"{self.value}_prompt.txt"

    def ensure_ref(self) -> weave.trace.refs.ObjectRef:
        """
        Return a Weave ref; publish a new version if the .txt
        file was edited since the last run.
        """
        text = self.path.read_text()

        try:
            latest = weave.ref(f"{self.value}_prompt")
            if latest.get().content.strip() == text.strip():
                return latest
        except ValueError:
            pass
                # first publish ever or deleted artifact

        # Text changed → create a new version
        return weave.publish(weave.StringPrompt(text), name=self.value)
