# Build with latexmk (recommended) or plain pdflatex
.PHONY: all clean view

PAPER = main

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	latexmk -pdf -interaction=nonstopmode -halt-on-error $(PAPER).tex

clean:
	latexmk -C
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

view: $(PAPER).pdf
	@echo "PDF: $$(pwd)/$(PAPER).pdf"
