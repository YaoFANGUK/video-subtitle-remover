#!/usr/bin/python
# -*- coding: utf-8 -*-

# The MIT License (MIT)
#
# Copyright (c) 2014 Matheus Vieira Portela
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""simpletable.py - v0.1 2014-07-31 Matheus Vieira Portela

This module provides simple classes and interfaces to generate simple HTML
tables based on Python native types, such as lists.

Author's website: http://matheusvportela.wordpress.com/

v0.4 2019-05-24 by Walter Schwenger
"""

### CHANGES ###
# 2014-07-31: v0.1 MVP:
#   - First version
# 2014-08-05: v0.2 MVP:
#   - Method for defining header rows
#   - SimpleTable method to create a SimpleTable from lists
#   - Method to create a table from a simple list of elements and a column size
# 2014-08-20: v0.3 MVP:
#   - Enable SimplePage to accept a list of tables
#   - Enable SimplePage to iterate over its tables
# 2019-05-24: v0.4 WS:
#   - Added SimpleTableImage class to handle adding images to tables
#   - Added test images and image example to __main__

### REFERENCES ###
# Decalage HTML.py module: http://www.decalage.info/python/html

import codecs


# noinspection PyCompatibility,PyUnresolvedReferences
def quote(string):
    try:
        from urllib.parse import quote
        return quote(string)
    except ModuleNotFoundError:
        from urllib import pathname2url
        return pathname2url(string)


class SimpleTableCell(object):
    """A table class to create table cells.

    Example:
    cell = SimpleTableCell('Hello, world!')
    """

    def __init__(self, text, header=False):
        """Table cell constructor.

        Keyword arguments:
        text -- text to be displayed
        header -- flag to indicate this cell is a header cell.
        """
        self.text = text
        self.header = header

    def __str__(self):
        """Return the HTML code for the table cell."""
        if self.header:
            return '<th>%s</th>' % (self.text)
        else:
            return '<td>%s</td>' % (self.text)


class SimpleTableImage(object):
    """A table class to create table cells with an image.

    Example:
    cell = SimpleTableImage('images/image_1.jpg')
    """

    def __init__(self, image_file, width=None, height=None):
        """Table cell constructor.

        Keyword arguments:
        image_file -- relative filepath to image file to display.
        width -- (optional) width of the image in pixels
        height -- (optional) height of the image in pixels
        """
        self.image_file = image_file
        if width:
            self.width = round(width)
        else:
            self.width = width
        if height:
            self.height = round(height)
        else:
            self.height = height

    def __str__(self):
        """Return the HTML code for the table cell with the image."""
        safe_filename = quote(self.image_file)
        output = '<a href="%s" target="_blank">' % (safe_filename)
        output += '<img src="%s"' % (safe_filename)
        if self.height:
            output += ' height="%s"' % (self.height)
        if self.width:
            output += ' width="%s"' % (self.width)
        output += '></a>'

        return output


class SimpleTableRow(object):
    """A table class to create table rows, populated by table cells.

    Example:
    # Row from list
    row = SimpleTableRow(['Hello,', 'world!'])

    # Row from SimpleTableCell
    cell1 = SimpleTableCell('Hello,')
    cell2 = SimpleTableCell('world!')
    row = SimpleTableRow([cell1, cell2])
    """

    def __init__(self, cells=None, header=False):
        """Table row constructor.

        Keyword arguments:
        cells -- iterable of SimpleTableCell (default None)
        header -- flag to indicate this row is a header row.
                  if the cells are SimpleTableCell, it is the programmer's
                  responsibility to verify whether it was created with the
                  header flag set to True.
        """
        cells = cells or []
        if isinstance(cells[0], SimpleTableCell):
            self.cells = cells
        else:
            self.cells = [SimpleTableCell(cell, header=header) for cell in cells]

        self.header = header

    def __str__(self):
        """Return the HTML code for the table row and its cells as a string."""
        row = []

        row.append('<tr>')

        for cell in self.cells:
            row.append(str(cell))

        row.append('</tr>')

        return '\n'.join(row)

    def __iter__(self):
        """Iterate through row cells"""
        for cell in self.cells:
            yield cell

    def add_cell(self, cell):
        """Add a SimpleTableCell object to the list of cells."""
        self.cells.append(cell)

    def add_cells(self, cells):
        """Add a list of SimpleTableCell objects to the list of cells."""
        for cell in cells:
            self.cells.append(cell)


class SimpleTable(object):
    """A table class to create HTML tables, populated by HTML table rows.

    Example:
    # Table from lists
    table = SimpleTable([['Hello,', 'world!'], ['How', 'are', 'you?']])

    # Table with header row
    table = SimpleTable([['Hello,', 'world!'], ['How', 'are', 'you?']],
                      header_row=['Header1', 'Header2', 'Header3'])

    # Table from SimpleTableRow
    rows = SimpleTableRow(['Hello,', 'world!'])
    table = SimpleTable(rows)
    """

    def __init__(self, rows=None, header_row=None, css_class=None):
        """Table constructor.

        Keyword arguments:
        rows -- iterable of SimpleTableRow
        header_row -- row that will be displayed at the beginning of the table.
                      if this row is SimpleTableRow, it is the programmer's
                      responsibility to verify whether it was created with the
                      header flag set to True.
        css_class -- table CSS class
        """
        rows = rows or []
        if isinstance(rows[0], SimpleTableRow):
            self.rows = rows
        else:
            self.rows = [SimpleTableRow(row) for row in rows]

        if header_row is None:
            self.header_row = None
        elif isinstance(header_row, SimpleTableRow):
            self.header_row = header_row
        else:
            self.header_row = SimpleTableRow(header_row, header=True)

        self.css_class = css_class

    def __str__(self):
        """Return the HTML code for the table as a string."""
        table = []

        if self.css_class:
            table.append('<table class=%s>' % self.css_class)
        else:
            table.append('<table>')

        if self.header_row:
            table.append(str(self.header_row))

        for row in self.rows:
            table.append(str(row))

        table.append('</table>')

        return '\n'.join(table)

    def __iter__(self):
        """Iterate through table rows"""
        for row in self.rows:
            yield row

    def add_row(self, row):
        """Add a SimpleTableRow object to the list of rows."""
        self.rows.append(row)

    def add_rows(self, rows):
        """Add a list of SimpleTableRow objects to the list of rows."""
        for row in rows:
            self.rows.append(row)


class HTMLPage(object):
    """A class to create HTML pages containing CSS and tables."""

    def __init__(self, tables=None, css=None, encoding="utf-8"):
        """HTML page constructor.

        Keyword arguments:
        tables -- List of SimpleTable objects
        css -- Cascading Style Sheet specification that is appended before the
               table string
        encoding -- Characters encoding. Default: UTF-8
        """
        self.tables = tables or []
        self.css = css
        self.encoding = encoding

    def __str__(self):
        """Return the HTML page as a string."""
        page = []

        if self.css:
            page.append('<style type="text/css">\n%s\n</style>' % self.css)

        # Set encoding
        page.append('<meta http-equiv="Content-Type" content="text/html;'
                    'charset=%s">' % self.encoding)

        for table in self.tables:
            page.append(str(table))
            page.append('<br />')

        return '\n'.join(page)

    def __iter__(self):
        """Iterate through tables"""
        for table in self.tables:
            yield table

    def save(self, filename):
        """Save HTML page to a file using the proper encoding"""
        with codecs.open(filename, 'w', self.encoding) as outfile:
            for line in str(self):
                outfile.write(line)

    def add_table(self, table):
        """Add a SimpleTable to the page list of tables"""
        self.tables.append(table)


def fit_data_to_columns(data, num_cols):
    """Format data into the configured number of columns in a proper format to
    generate a SimpleTable.

    Example:
    test_data = [str(x) for x in range(20)]
    fitted_data = fit_data_to_columns(test_data, 5)
    table = SimpleTable(fitted_data)
    """
    num_iterations = len(data) / num_cols

    if len(data) % num_cols != 0:
        num_iterations += 1

    return [data[num_cols * i:num_cols * i + num_cols] for i in range(num_iterations)]
