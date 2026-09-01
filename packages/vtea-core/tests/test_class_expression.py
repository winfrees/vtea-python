import numpy as np
import pandas as pd
import pytest

from vtea_core.classes import ExpressionError, evaluate, referenced_columns


def a_table():
    return pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4, 5],
            "mean_ch2": [10.0, 60.0, 120.0, 160.0, np.nan],
            "kmeans_1": [0, 3, 7, 3, 1],
            "gate_high": [False, True, True, False, False],
            "roi_tubule": [0, 2, 2, 0, 1],
            "cell type": ["T", "B", "T", "B", "T"],
        }
    )


class TestSingleConditions:
    def test_a_boolean_column_is_its_own_condition(self):
        assert list(evaluate("gate_high", a_table())) == [False, True, True, False, False]

    def test_a_numeric_column_is_true_where_it_is_not_zero(self):
        """`roi_tubule` holds a region id per object, 0 for none - so naming
        it means "in some region" without anyone writing `!= 0`."""
        assert list(evaluate("roi_tubule", a_table())) == [False, True, True, False, True]

    def test_a_comparison(self):
        assert list(evaluate("mean_ch2 > 100", a_table())) == [False, False, True, True, False]

    def test_equality_against_a_cluster_id(self):
        assert list(evaluate("kmeans_1 == 3", a_table())) == [False, True, False, True, False]

    def test_membership_in_a_list(self):
        assert list(evaluate("kmeans_1 in [3, 7]", a_table())) == [False, True, True, True, False]

    def test_a_string_value(self):
        assert list(evaluate("`cell type` == 'T'", a_table())) == [True, False, True, False, True]

    def test_a_chained_range(self):
        """The form the request writes it in: mean_ch2 from 50 to 150."""
        assert list(evaluate("50 <= mean_ch2 <= 150", a_table())) == [
            False, True, True, False, False
        ]

    def test_a_range_written_the_other_way_round(self):
        assert list(evaluate("mean_ch2 >= 50 AND mean_ch2 <= 150", a_table())) == [
            False, True, True, False, False
        ]

    def test_an_exclusive_range_excludes_its_bounds(self):
        assert list(evaluate("60 < mean_ch2 < 160", a_table())) == [
            False, False, True, False, False
        ]


class TestCombinations:
    def test_and(self):
        assert list(evaluate("gate_high AND kmeans_1 == 3", a_table())) == [
            False, True, False, False, False
        ]

    def test_or(self):
        assert list(evaluate("gate_high OR kmeans_1 == 1", a_table())) == [
            False, True, True, False, True
        ]

    def test_not(self):
        assert list(evaluate("NOT gate_high", a_table())) == [True, False, False, True, True]

    def test_xor(self):
        assert list(evaluate("gate_high XOR roi_tubule", a_table())) == [
            False, False, False, False, True
        ]

    def test_xnor_is_the_negation_of_xor(self):
        table = a_table()
        assert list(evaluate("gate_high XNOR roi_tubule", table)) == list(
            ~evaluate("gate_high XOR roi_tubule", table)
        )

    def test_nand_and_nor(self):
        table = a_table()
        assert list(evaluate("gate_high NAND roi_tubule", table)) == list(
            ~evaluate("gate_high AND roi_tubule", table)
        )
        assert list(evaluate("gate_high NOR roi_tubule", table)) == list(
            ~evaluate("gate_high OR roi_tubule", table)
        )

    def test_and_binds_tighter_than_or(self):
        table = a_table()
        loose = evaluate("kmeans_1 == 1 OR gate_high AND roi_tubule == 2", table)
        explicit = evaluate("kmeans_1 == 1 OR (gate_high AND roi_tubule == 2)", table)
        assert list(loose) == list(explicit)

    def test_parentheses_override_precedence(self):
        table = a_table()
        grouped = evaluate("(kmeans_1 == 1 OR gate_high) AND roi_tubule == 2", table)
        assert list(grouped) == [False, True, True, False, False]

    def test_symbols_are_synonyms_for_the_words(self):
        table = a_table()
        assert list(evaluate("gate_high & ~roi_tubule", table)) == list(
            evaluate("gate_high AND NOT roi_tubule", table)
        )

    def test_the_whole_request_in_one_definition(self):
        """A gate, a napari ROI, a clustering output and a range, combined -
        which is what a class is for."""
        table = a_table()
        mask = evaluate(
            "gate_high AND roi_tubule == 2 AND NOT kmeans_1 == 3 AND 50 <= mean_ch2 <= 150",
            table,
        )
        assert list(mask) == [False, False, True, False, False]


class TestErrors:
    def test_an_empty_definition_says_so(self):
        with pytest.raises(ExpressionError, match="empty class definition"):
            evaluate("   ", a_table())

    def test_an_unknown_column_names_the_columns_there_are(self):
        with pytest.raises(ExpressionError, match="no column 'nonsense'"):
            evaluate("nonsense AND gate_high", a_table())

    def test_an_unbalanced_parenthesis_is_reported(self):
        with pytest.raises(ExpressionError):
            evaluate("(gate_high AND roi_tubule", a_table())

    def test_a_dangling_operator_is_reported(self):
        with pytest.raises(ExpressionError):
            evaluate("gate_high AND", a_table())

    def test_it_never_evaluates_python(self):
        """The definitions are saved in protocol files and mailed around;
        the parser must not be a way to run code."""
        with pytest.raises(ExpressionError):
            evaluate("__import__('os').system('true')", a_table())


class TestReferencedColumns:
    def test_it_lists_what_a_definition_reads(self):
        assert referenced_columns("gate_high AND 50 <= mean_ch2 <= 150") == {
            "gate_high",
            "mean_ch2",
        }

    def test_a_definition_can_be_checked_before_it_is_run(self):
        table = a_table()
        wanted = referenced_columns("gate_high AND missing_feature")
        assert wanted - set(table.columns) == {"missing_feature"}
