from jaxtyping import install_import_hook

with install_import_hook(["src"], "beartype.beartype"):
  from src.example import main

if __name__ == "__main__":
  main()
