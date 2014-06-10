TARGET=report
TEX=xelatex -shell-escape
BIBTEX=biber
READER=mupdf

all: rebuild

rebuild output/$(TARGET).pdf: *.tex *.bib output
	cd output && rm -f *.tex *.bib && ln -fs ../*.tex ../*.bib ../img .
	pgrep -a $(TEX) || cd output && $(TEX) $(TARGET).tex && $(BIBTEX) $(TARGET) #&& $(TEX) $(TARGET).tex

output:
	mkdir output
	cd output && rm -f data res src && ln -s ../data ../res ../src .

view: output/$(TARGET).pdf
	$(READER) output/$(TARGET).pdf &
	(inotifywait -mqe CLOSE_WRITE output/report.pdf | while read; do killall -SIGHUP mupdf; done)

clean:
	rm -rf output

run: view

dist: output/$(TARGET).pdf
	rm -rf paper
	mkdir paper
	cp output/$(TARGET).pdf paper/
	7z a -tzip paper.zip paper
	rm -rf paper

.PHONY: all view clean rebuild dist
