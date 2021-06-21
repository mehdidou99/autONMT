all: prepare
	python3 -m build

prepare: src/prepare.py
	cp src/autonmt.py src/autonmt
	chmod +x src/autonmt

clean:
	rm -f src/prepare
	rm -rf build dist
	rm -rf src/*.egg-info

