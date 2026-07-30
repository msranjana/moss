from moss_cli.commands.playground import PLAYGROUND_HTML


def test_playground_html_asset_exists():
    assert PLAYGROUND_HTML.exists(), (
        f"Playground HTML not found at {PLAYGROUND_HTML}. "
        "Check that the asset is included in the package data."
    )