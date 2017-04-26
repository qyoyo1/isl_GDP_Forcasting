import sys
from importlib import import_module

def main():
	script = sys.argv[1:2][0].replace("/", ".").replace("\\", ".")
	mod = import_module(script)
	mod.main()

if __name__ == "__main__":
	main()
