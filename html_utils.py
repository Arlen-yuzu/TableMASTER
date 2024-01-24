from html import escape


def format_html(tab):
    """
    Formats HTML code from html annotation of table img
    """
    html_code = tab['html']['structure']['tokens'].copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], tab['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code
    return html_code

def format_html_pred(tab):
    """
    Formats HTML code from tokenized annotation of table img
    """
    html_code = tab.copy()
    to_insert = [i for i, tag in enumerate(html_code) if tag in ('<td>', '>')]
    for i, cell in zip(to_insert[::-1], tab['html']['cells'][::-1]):
        if cell['tokens']:
            cell = [escape(token) if len(token) == 1 else token for token in cell['tokens']]
            cell = ''.join(cell)
            html_code.insert(i + 1, cell)
    html_code = ''.join(html_code)
    html_code = '''<html><body><table>%s</table></body></html>''' % html_code
    return html_code