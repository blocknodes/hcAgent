"""临时：往 STDOUT 打 round-2 请求体，用于诊断远端注入的 shortMemory。用后即删。"""

import json
import os

_LOG = "/tmp/hc_req.jsonl"
_TMP_TAG = os.environ.get("HC_REQLOG", "")


def tlog(req) -> None:
    with open(_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(req.model_dump(), ensure_ascii=False) + "\n")