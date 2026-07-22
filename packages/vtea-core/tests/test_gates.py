import numpy as np

from vtea_core.gates import polygon_gate, rectangle_gate


class TestPolygonGate:
    def test_points_inside_and_outside_square(self):
        vertices = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        x = np.array([5, 15, -5, 5])
        y = np.array([5, 5, 5, 15])
        mask = polygon_gate(x, y, vertices)
        np.testing.assert_array_equal(mask, [True, False, False, False])

    def test_combining_two_gates_with_and(self):
        square1 = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        square2 = np.array([[5, 5], [5, 15], [15, 15], [15, 5]])
        x = np.array([2, 7, 12])  # only in square1, in both, only in square2
        y = np.array([2, 7, 12])
        combined = polygon_gate(x, y, square1) & polygon_gate(x, y, square2)
        np.testing.assert_array_equal(combined, [False, True, False])

    def test_combining_with_or_and_not(self):
        square1 = np.array([[0, 0], [0, 10], [10, 10], [10, 0]])
        square2 = np.array([[20, 20], [20, 30], [30, 30], [30, 20]])
        x = np.array([5, 25, 50])
        y = np.array([5, 25, 50])
        either = polygon_gate(x, y, square1) | polygon_gate(x, y, square2)
        np.testing.assert_array_equal(either, [True, True, False])
        np.testing.assert_array_equal(~either, [False, False, True])


class TestRectangleGate:
    def test_inside_and_outside(self):
        x = np.array([5, 15, 5])
        y = np.array([5, 5, 15])
        mask = rectangle_gate(x, y, x_min=0, x_max=10, y_min=0, y_max=10)
        np.testing.assert_array_equal(mask, [True, False, False])

    def test_boundary_inclusive(self):
        x = np.array([0, 10])
        y = np.array([0, 10])
        mask = rectangle_gate(x, y, x_min=0, x_max=10, y_min=0, y_max=10)
        np.testing.assert_array_equal(mask, [True, True])
