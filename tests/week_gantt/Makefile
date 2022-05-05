PYTHON_EXEC   ?= python3
GANTT_SCRIPT   = ../../scripts/pygantt.py

# Put it first so that "make" without argument is like "make help".
export COMMENT_EXTRACT

gantt:
	$(PYTHON_EXEC) $(GANTT_SCRIPT) --input_csv gantt.csv --colormap_csv colormap.csv --output_figure gantt.jpg

# Functions to extract comments from Makefiles
define COMMENT_EXTRACT
import re
with open ('Makefile', 'r' ) as f:
    matches = re.finditer('^([a-zA-Z-_]*):.*\n#(.*)', f.read(), flags=re.M)
    for _, match in enumerate(matches, start=1):
        header, content = match[1], match[2]
        print(f"  {header:10} {content}")
endef
