REPORT_MD := final_report.md
REPORT_TEX := final_report.tex
REPORT_PDF := final_report.pdf
BIB := references.bib

.PHONY: all tex pdf clean check

all: pdf

tex:
	pandoc $(REPORT_MD) \
	  --from markdown \
	  --to latex \
	  --standalone \
	  --citeproc \
	  --bibliography $(BIB) \
	  --csl ieee.csl \
	  --output $(REPORT_TEX)

pdf:
	pandoc $(REPORT_MD) \
	  --from markdown \
	  --to pdf \
	  --standalone \
	  --citeproc \
	  --bibliography $(BIB) \
	  --csl ieee.csl \
	  --pdf-engine=xelatex \
	  --output $(REPORT_PDF)

check:
	python3 -c 'from pathlib import Path; p=Path("final_report.md"); text=p.read_text(); abstract=text.split("# 1. Introduction",1)[0].replace("# Abstract",""); words=[w for w in abstract.split() if w and not w.startswith("<!--")]; print(f"abstract_words={len(words)}"); print(f"markdown_bytes={p.stat().st_size}"); print(f"markdown_lines={sum(1 for _ in p.open())}")'

clean:
	rm -f $(REPORT_TEX) $(REPORT_PDF)
