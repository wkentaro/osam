import collections


def tabulate(rows, headers):
    col_widths = collections.defaultdict(int)
    for row in [headers] + rows:
        for c, item in enumerate(row):
            col_widths[c] = max(col_widths[c], len(item))

    table = ""

    for c, header in enumerate(headers):
        table += header.ljust(col_widths[c] + 4)

    for row in rows:
        table += "\n"
        for c, item in enumerate(row):
            table += item.ljust(col_widths[c] + 4)
    return table
