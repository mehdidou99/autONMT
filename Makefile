all: prepare
	python3 -m build

prepare: src/autonmt.py
	cp src/autonmt.py src/autonmt_cli
	chmod +x src/autonmt_cli

clean:
	rm -f src/autonmt_cli
	rm -rf build dist
	rm -rf src/*.egg-info

