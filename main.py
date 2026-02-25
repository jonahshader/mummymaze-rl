from jaxtyping import install_import_hook

with install_import_hook(["src", "src.env"], "beartype.beartype"):
  pass

if __name__ == "__main__":
  pass
