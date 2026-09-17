set minimum-version := "1.58.0"
set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set script-interpreter := ["bash", "-eu", "-o", "pipefail"]
set positional-arguments
set default-list

# Setup the development environment
setup:
    uv sync

# Format code
format:
    uv run ruff format
    uv run ruff check --fix
    git ls-files "*.toml" | xargs uv run taplo fmt
    git ls-files "*.md" | xargs uv run mdformat
    git ls-files "*.yml" "*.yaml" | xargs uv run yamlfix

# Lint code
lint:
    uv run ruff format --check
    uv run ruff check
    uv run ty check --no-progress
    git ls-files "*.toml" | xargs uv run taplo fmt --check
    git ls-files "*.md" | xargs uv run mdformat --check
    git ls-files "*.yml" "*.yaml" | xargs uv run yamlfix --check
    uv run typos

# Run tests
test *pytest_args="--numprocesses=auto":
    uv run pytest -v tests/ "$@"

# Run tests with coverage
coverage:
    uv run pytest -v tests/ --numprocesses=auto --cov=osam --cov-report=term-missing

# Prepare a release
[script]
release version="":
    version="$1"
    if test -z "$version"; then
        shopt -s nullglob
        fragments=(changelog.d/*.{added,changed,deprecated,removed,fixed,security}.md)
        latest=$(git tag --sort=-v:refname | awk '/^v[0-9]+\.[0-9]+\.[0-9]+$/ && !found { print; found=1 }')
        if test "${#fragments[@]}" -gt 0 && test -n "$latest"; then
            current=${latest#v}
            major=${current%%.*}
            remainder=${current#*.}
            minor=${remainder%%.*}
            patch=${remainder#*.}
            if grep -q '\*\*Breaking:\*\*' "${fragments[@]}"; then
                next=$((major + 1)).0.0
            else
                minor_fragments=(changelog.d/*.{added,changed,deprecated,removed}.md)
                if test "${#minor_fragments[@]}" -gt 0; then
                    next=$major.$((minor + 1)).0
                else
                    next=$major.$minor.$((patch + 1))
                fi
            fi
            echo "suggested: just release $next" >&2
        else
            echo "usage: just release X.Y.Z" >&2
        fi
        echo "recent releases:" >&2
        git tag --sort=-v:refname | sed -n '1,5s/^/  /p' >&2
        exit 1
    fi
    uv run towncrier build --yes --version "$version"
    uv run mdformat CHANGELOG.md
    git add CHANGELOG.md
    printf "\n\033[1;32mNext steps\033[0m\n"
    echo "  git commit -am \"chore: prep $version release\""
    echo "  git tag v$version"
    echo "  git push origin main v$version"
