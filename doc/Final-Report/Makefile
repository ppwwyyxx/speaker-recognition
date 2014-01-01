TARGET=report
TEX=xelatex -shell-escape
BIBTEX=biber
READER=evince

all: view

rebuild  output/$(TARGET).pdf: *.tex *.bib output
	cd output && rm -f *.tex *.bib && ln -fs ../*.tex ../*.bib ../img .
	cd output && $(TEX) $(TARGET).tex && $(BIBTEX) $(TARGET) && $(TEX) $(TARGET).tex

output:
	mkdir output
	cd output && rm -f data res src && ln -s ../data ../res ../src .

view: output/$(TARGET).pdf
	$(READER) output/$(TARGET).pdf &

clean:
	rm -rf output

dist: output/$(TARGET).pdf
	rm -rf paper
	mkdir paper
	cp output/$(TARGET).pdf paper/
	7z a -tzip paper.zip paper
	rm -rf paper

.PHONY: all view clean rebuild dist
