from ._tabulate import tabulate


def test_tabulate_aligns_columns_across_rows() -> None:
    table = tabulate(
        rows=[["a", "v1"], ["ccc", "v2"]],
        headers=["NAME", "ID"],
    )
    lines = table.split("\n")
    assert len(lines) == 3
    assert lines[0].index("ID") == lines[1].index("v1") == lines[2].index("v2")


def test_tabulate_widens_column_to_longest_cell() -> None:
    table = tabulate(
        rows=[["a-very-long-name", "v1"], ["short", "v2"]],
        headers=["NAME", "ID"],
    )
    lines = table.split("\n")
    id_start = lines[0].index("ID")
    assert id_start > len("a-very-long-name")
    assert lines[1].index("v1") == id_start == lines[2].index("v2")


def test_tabulate_with_only_headers_has_no_rows() -> None:
    table = tabulate(rows=[], headers=["NAME", "ID"])
    assert "\n" not in table
    assert "NAME" in table
    assert "ID" in table
