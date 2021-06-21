all: prepare
	python3 -m build

prepare: src/autonmt.py
	cp src/autonmt.py src/autonmt
	chmod +x src/autonmt

clean:
	rm -f src/autonmt
	rm -rf build dist
	rm -rf src/*.egg-info

