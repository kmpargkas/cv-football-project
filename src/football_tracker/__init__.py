"""Football CV Tracker — detection, pitch homography, player and ball tracking,
ReID, and 2D pitch projection from tactical-camera footage."""

from importlib.metadata import version

# Read from the installed package metadata rather than duplicating the literal in
# pyproject.toml, where the two would drift apart on the next release bump.
__version__ = version("football-tracker")


def main() -> None:
    """Print the version and the pipeline entry point."""
    print(f"football-tracker {__version__}")
    print(
        "Run the pipeline with:\n"
        "  uv run python scripts/run_pipeline.py --config configs/<clip>.yaml"
    )
