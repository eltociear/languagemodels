all: test

test: lint
	python3 -m doctest languagemodels/*.py

lint:
	flake8 --max-line-length 88 languagemodels/*.py

format:
	black languagemodels/*.py

doc:
	mkdir -p doc
	python3 -m pdoc -o doc languagemodels

paper.pdf: paper.md paper.bib
	pandoc $< --citeproc -o $@

spellcheck:
	aspell -c --dont-backup readme.md
	aspell -c --dont-backup paper.md

upload:
	python3 setup.py sdist bdist_wheel
	python3 -m twine upload dist/*

clean:
	rm -rf tmp
	rm -rf languagemodels.egg-info
	rm -rf languagemodels/__pycache__
	rm -rf dist
	rm -rf build
	rm -rf doc
	rm -rf .ipynb_checkpoints
	rm -rf examples/.ipynb_checkpoints
	cd vendor/llama.cpp && make clean
	cd vendor/llama.cpp && rm libllama.so
	rm -rf _skbuild
	rm llama_cpp/libllama.so

update:
	poetry install
	git submodule update --init --recursive

update.vendor:
	cd vendor/llama.cpp && git pull origin master

build:
	python3 setup.py develop

build.cuda:
	CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 python3 setup.py develop

build.opencl:
	CMAKE_ARGS="-DLLAMA_CLBLAST=on" FORCE_CMAKE=1 python3 setup.py develop

build.openblas:
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on" FORCE_CMAKE=1 python3 setup.py develop

build.blis:
	CMAKE_ARGS="-DLLAMA_OPENBLAS=on -DLLAMA_OPENBLAS_VENDOR=blis" FORCE_CMAKE=1 python3 setup.py develop

build.sdist:
	python3 setup.py sdist

deploy.pypi:
	python3 -m twine upload dist/*

deploy.gh-docs:
	mkdocs build
	mkdocs gh-deploy

.PHONY: \
	update \
	update.vendor \
	build \
	build.cuda \
	build.opencl \
	build.openblas \
	build.sdist \
	deploy.pypi \
	deploy.gh-docs \
	clean
