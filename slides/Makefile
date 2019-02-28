LATEXMK = latexmk -r mklatex.rc

all: getting-to-grips-with-databricks.pdf

%.pdf: %.tex
	$(LATEXMK) -gg -pdf -xelatex -use-make $<
	$(LATEXMK) -c

clean:
	$(LATEXMK) -C
