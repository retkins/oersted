# Notes for Building the Project

## `.cargo/config.toml`

Should look like this:
```toml
[build]
rustdocflags = [
    "--html-in-header", 
    # This must be an abs path:
    "PATH_TO_PROJECT/oersted/src/docs-header.html", 
]
rustflags = [
    "-C", "target-cpu=native"
]
```

Note that this is non-portable and therefore is not included in the repo.

To do this locally and one-off, build the docs like this instead:
```shell
$ RUSTDOCFLAGS="--html-in-header src/docs-header.html" cargo doc --open
```

Show docs with feature flags:
```shell
RUSTDOCFLAGS="--cfg docsrs --html-in-header src/docs-header.html" cargo +nightly doc --no-deps --all-features --open
```