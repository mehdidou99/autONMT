all: prepare
	python3 -m build

prepare: src/prepare.py
	cp src/prepare.py src/prepare
	chmod +x src/prepare

clean:
	rm -f src/prepare
	rm -rf build dist
	rm -f src/*.egg-info

