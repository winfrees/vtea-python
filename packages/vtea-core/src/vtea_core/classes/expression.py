"""The little boolean language a class definition is written in.

A class is "which objects count as this cell type", and the honest answer is
usually a sentence rather than a shape: *CD3 positive and not in the
tubule ROI*, *mean_ch2 between 50 and 150*, *cluster 3 or cluster 7*. That
is what this parses.

    mean_ch2 >= 50 AND mean_ch2 <= 150
    50 <= mean_ch2 <= 150                  (the same thing, said once)
    gate_high AND NOT roi_tubule
    kmeans_1 in [3, 7] XOR gate_dim

Deliberately not Python's `eval`. A protocol is a document that gets saved,
mailed to a collaborator and re-opened a year later; `eval` on a string out
of a JSON file is a remote-code-execution hole with a friendly name, and no
amount of "our users wouldn't" makes that acceptable in software people run
on data they were sent. So there is a tokenizer, a recursive-descent parser
over an explicit grammar, and an evaluator that can only ever produce a
boolean mask over the columns of one table.

The operators are the ones the request named plus the two that complete the
set - every binary boolean function of two inputs is here or is a
combination of these:

    NOT  AND  OR  XOR  XNOR  NAND  NOR

with `~ & | ^` accepted as synonyms for the first four, `AND` binding
tighter than `OR` (as everywhere else), and parentheses to override that.
A bare column is true where it is true (a boolean column) or non-zero (a
numeric one), so `gate_high AND kmeans_1 == 3` reads the way it looks.
Column names that are not identifiers - `nuclei_1.mean_ch0` is, `mean ch0`
is not - can be written in backticks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Binary operators, and what they do to two boolean arrays. Written out
# rather than composed (XNOR as ~(a ^ b)) so the table itself is the
# documentation of what each name means.
BOOLEAN_OPERATORS = {
    "AND": lambda a, b: a & b,
    "OR": lambda a, b: a | b,
    "XOR": lambda a, b: a ^ b,
    "XNOR": lambda a, b: ~(a ^ b),
    "NAND": lambda a, b: ~(a & b),
    "NOR": lambda a, b: ~(a | b),
}

# Tighter binds first. AND above OR is the convention every language and
# every reader already has; the exotic ones sit with AND because they are
# conjunction-shaped, and anyone mixing them without parentheses is asking
# for a reading nobody can predict anyway.
_PRECEDENCE = {"OR": 1, "NOR": 1, "XOR": 2, "XNOR": 2, "AND": 3, "NAND": 3}

_SYNONYMS = {"&": "AND", "|": "OR", "^": "XOR", "~": "NOT", "!": "NOT"}

_COMPARISONS = {
    "==": lambda values, other: values == other,
    "!=": lambda values, other: values != other,
    "<=": lambda values, other: values <= other,
    ">=": lambda values, other: values >= other,
    "<": lambda values, other: values < other,
    ">": lambda values, other: values > other,
}

_TOKEN_PATTERN = re.compile(
    r"""
    \s*(?:
        (?P<backticked>`[^`]*`)
      | (?P<number>-?\d+\.?\d*(?:[eE][-+]?\d+)?)
      | (?P<string>'[^']*'|"[^"]*")
      | (?P<comparison><=|>=|==|!=|<|>)
      | (?P<punctuation>[()\[\],])
      | (?P<symbol>[&|^~!])
      | (?P<name>[A-Za-z_][A-Za-z0-9_.:%-]*)
    )
    """,
    re.VERBOSE,
)


class ExpressionError(ValueError):
    """A class definition that cannot be read, said in terms of the text."""


@dataclass(frozen=True)
class Token:
    kind: str
    text: str
    position: int


def tokenize(expression: str) -> list[Token]:
    tokens: list[Token] = []
    position = 0
    while position < len(expression):
        if expression[position].isspace():
            position += 1
            continue
        match = _TOKEN_PATTERN.match(expression, position)
        if match is None or match.end() == position:
            raise ExpressionError(
                f"cannot read {expression[position]!r} at position {position} of {expression!r}"
            )
        kind = match.lastgroup
        text = match.group(kind)
        tokens.append(Token(kind, text, match.start(kind)))
        position = match.end()
    return tokens


# -- the syntax tree ------------------------------------------------------
#
# Four node types is the whole language: a column, a comparison against it,
# a range around it, and the boolean combinations. Each knows how to
# evaluate itself against a table, which keeps the parser free of any
# knowledge of pandas.


@dataclass(frozen=True)
class Column:
    name: str

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        return _truth(_column(frame, self.name))

    def columns(self) -> set[str]:
        return {self.name}


@dataclass(frozen=True)
class Comparison:
    column: str
    operator: str
    value: Any

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        values = _column(frame, self.column)
        if isinstance(self.value, (list, tuple)):
            return np.asarray(values.isin(list(self.value)))
        return np.asarray(_COMPARISONS[self.operator](values, self.value))

    def columns(self) -> set[str]:
        return {self.column}


@dataclass(frozen=True)
class Range:
    """`low <= column <= high`, the form a range of data is usually said in."""

    column: str
    low: float
    high: float
    include_low: bool = True
    include_high: bool = True

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        values = _column(frame, self.column)
        above = values >= self.low if self.include_low else values > self.low
        below = values <= self.high if self.include_high else values < self.high
        return np.asarray(above & below)

    def columns(self) -> set[str]:
        return {self.column}


@dataclass(frozen=True)
class Not:
    operand: Any

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        return ~self.operand.evaluate(frame)

    def columns(self) -> set[str]:
        return self.operand.columns()


@dataclass(frozen=True)
class BinaryOperation:
    operator: str
    left: Any
    right: Any

    def evaluate(self, frame: pd.DataFrame) -> np.ndarray:
        return BOOLEAN_OPERATORS[self.operator](
            self.left.evaluate(frame), self.right.evaluate(frame)
        )

    def columns(self) -> set[str]:
        return self.left.columns() | self.right.columns()


def _column(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame.columns:
        raise ExpressionError(
            f"no column {name!r} in the table (columns: {', '.join(map(str, frame.columns))})"
        )
    return frame[name]


def _truth(values: pd.Series) -> np.ndarray:
    """A bare column as a condition: true where true, or where non-zero.

    A boolean column - a gate's membership, another class - is its own
    answer. A numeric one is read the way every language reads a number in a
    condition, which makes `roi_tubule` (an ROI id per object, 0 for "in no
    ROI") mean "in some ROI" without anyone having to write `!= 0`.
    """
    if pd.api.types.is_bool_dtype(values):
        return np.asarray(values, dtype=bool)
    if pd.api.types.is_numeric_dtype(values):
        return np.asarray(values.fillna(0) != 0)
    return np.asarray(values.notna() & (values.astype(str) != ""))


class _Parser:
    """Recursive descent over the token list. One pass, no backtracking."""

    def __init__(self, tokens: list[Token], expression: str):
        self.tokens = tokens
        self.expression = expression
        self.position = 0

    # -- token helpers ----------------------------------------------------

    def peek(self) -> Token | None:
        return self.tokens[self.position] if self.position < len(self.tokens) else None

    def next(self) -> Token:
        token = self.peek()
        if token is None:
            raise ExpressionError(f"{self.expression!r} ends unexpectedly")
        self.position += 1
        return token

    def _keyword(self, token: Token | None) -> str | None:
        """The operator a token names, if it names one."""
        if token is None:
            return None
        if token.kind == "symbol":
            return _SYNONYMS.get(token.text)
        if token.kind == "name":
            upper = token.text.upper()
            if upper in _PRECEDENCE or upper == "NOT":
                return upper
        return None

    # -- grammar ----------------------------------------------------------

    def parse(self):
        node = self.parse_binary(0)
        if self.peek() is not None:
            token = self.peek()
            raise ExpressionError(
                f"unexpected {token.text!r} at position {token.position} of {self.expression!r}"
            )
        return node

    def parse_binary(self, minimum_precedence: int):
        left = self.parse_unary()
        while True:
            keyword = self._keyword(self.peek())
            precedence = _PRECEDENCE.get(keyword or "", 0)
            if keyword is None or precedence < minimum_precedence or precedence == 0:
                return left
            self.next()
            right = self.parse_binary(precedence + 1)
            left = BinaryOperation(keyword, left, right)

    def parse_unary(self):
        if self._keyword(self.peek()) == "NOT":
            self.next()
            return Not(self.parse_unary())
        return self.parse_atom()

    def parse_atom(self):
        token = self.next()
        if token.kind == "punctuation" and token.text == "(":
            node = self.parse_binary(0)
            closing = self.next()
            if closing.text != ")":
                raise ExpressionError(f"expected ')' in {self.expression!r}, got {closing.text!r}")
            return node
        if token.kind == "number":
            # Only meaningful as the low end of `50 <= x <= 150`.
            return self.parse_leading_range(float(token.text))
        if token.kind in ("name", "backticked"):
            return self.parse_column(_column_name(token))
        raise ExpressionError(
            f"expected a column or '(' at position {token.position} of {self.expression!r}, "
            f"got {token.text!r}"
        )

    def parse_leading_range(self, low: float):
        """`50 <= mean_ch2 <= 150` - a range written the way it is spoken."""
        operator = self.next()
        if operator.kind != "comparison" or operator.text not in ("<", "<="):
            raise ExpressionError(
                f"a number can only start a range ('50 <= x <= 150') in {self.expression!r}"
            )
        column = self.next()
        if column.kind not in ("name", "backticked"):
            raise ExpressionError(f"expected a column name in the range in {self.expression!r}")
        second = self.next()
        if second.kind != "comparison" or second.text not in ("<", "<="):
            raise ExpressionError(
                f"a range needs a second bound ('50 <= x <= 150') in {self.expression!r}"
            )
        high = self.next()
        if high.kind != "number":
            raise ExpressionError(f"expected a number to close the range in {self.expression!r}")
        return Range(
            column=_column_name(column),
            low=low,
            high=float(high.text),
            include_low=operator.text == "<=",
            include_high=second.text == "<=",
        )

    def parse_column(self, name: str):
        token = self.peek()
        if token is None:
            return Column(name)
        if token.kind == "comparison":
            self.next()
            value = self.parse_value()
            # `50 <= x <= 150` also arrives here when written `x >= 50` and
            # then closed - but a second comparison after a column is the
            # chained form, which is a range around this column.
            follow = self.peek()
            if (
                follow is not None
                and follow.kind == "comparison"
                and token.text in ("<", "<=")
                and follow.text in ("<", "<=")
            ):
                self.next()
                high = self.parse_value()
                return Range(name, float(value), float(high), token.text == "<=", follow.text == "<=")
            return Comparison(name, token.text, value)
        if token.kind == "name" and token.text.lower() == "in":
            self.next()
            return Comparison(name, "in", self.parse_list())
        return Column(name)

    def parse_value(self):
        token = self.next()
        if token.kind == "number":
            return float(token.text)
        if token.kind == "string":
            return token.text[1:-1]
        if token.kind == "backticked":
            return token.text[1:-1]
        if token.kind == "name":
            lowered = token.text.lower()
            if lowered in ("true", "false"):
                return lowered == "true"
            return token.text
        raise ExpressionError(
            f"expected a value at position {token.position} of {self.expression!r}, "
            f"got {token.text!r}"
        )

    def parse_list(self) -> list:
        opening = self.next()
        if opening.text not in ("[", "("):
            raise ExpressionError(f"expected a list after 'in' in {self.expression!r}")
        closing = "]" if opening.text == "[" else ")"
        values = []
        while True:
            token = self.peek()
            if token is None:
                raise ExpressionError(f"unclosed list in {self.expression!r}")
            if token.text == closing:
                self.next()
                return values
            if token.text == ",":
                self.next()
                continue
            values.append(self.parse_value())


def _column_name(token: Token) -> str:
    return token.text[1:-1] if token.kind == "backticked" else token.text


def parse(expression: str):
    """The syntax tree for a class definition, or ExpressionError."""
    if not expression or not expression.strip():
        raise ExpressionError("an empty class definition selects nothing; write a condition")
    return _Parser(tokenize(expression), expression).parse()


def evaluate(expression: str, frame: pd.DataFrame) -> np.ndarray:
    """Which rows of `frame` satisfy `expression`, as a boolean array."""
    mask = parse(expression).evaluate(frame)
    return np.asarray(mask, dtype=bool)


def referenced_columns(expression: str) -> set[str]:
    """Which columns a definition reads - for checking one against a table
    before running it, and for saying what a class depends on."""
    return parse(expression).columns()
