#!/usr/bin/env python3
from _bootstrap import add_src_to_path

add_src_to_path()

from unveiling_persistent.classify_prompt_links import main

if __name__ == "__main__":
    raise SystemExit(main())
