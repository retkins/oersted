# Run checks before git commits 
cargo fmt 
cargo test 
maturin develop --release
ruff format 
ruff check --fix 
ty check
uv run pytest