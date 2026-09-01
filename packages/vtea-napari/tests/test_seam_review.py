"""Reviewing the objects a tile boundary went through.

A blocked run reconciles objects across seams and records how; the few
percent it was least sure about are the ones worth a person's attention.
These tests pin what that person can do: see them worst-first, put them on
the plot as an ordinary gate, and exclude one - and *not* redraw one, which
is the deliberate limit (see the widget's docstring).
"""

import numpy as np
import pandas as pd

from vtea_core.blocked.reconcile import Fragment, LabelLedger
from vtea_napari.session import AnalysisSession
from vtea_napari.widgets.explorer import SEAM_TAB_NAME, ExplorerWidget
from vtea_napari.widgets.seam_review import REJECTED_REASON, SeamReviewWidget


def fragment(tile, object_id, *, faces=(), exceeded=False):
    return Fragment(
        tile=tile,
        local_id=object_id,
        provisional_id=object_id,
        core_voxels=100,
        block_voxels=100,
        centroid=(1.0, 1.0, 1.0),
        bbox=((0, 2), (0, 2), (0, 2)),
        faces=frozenset(faces),
        exceeded_halo=exceeded,
    )


def make_ledger():
    """Four objects: two untouched by any seam, one joined weakly across
    one, and one no tile managed to contain."""
    ledger = LabelLedger()
    ledger.add(1, [fragment((0,), 1)])
    ledger.add(2, [fragment((0,), 2, faces=("x+",)), fragment((1,), 2, faces=("x-",))],
               decided_by="overlap", evidence=0.42)
    ledger.add(
        3,
        [fragment((0,), 3, faces=("x+",), exceeded=True),
         fragment((1,), 3, faces=("x-",), exceeded=True)],
        decided_by="overlap",
        evidence=0.9,
    )
    ledger.add(4, [fragment((1,), 4)])
    return ledger


def make_frame(ledger=None):
    ledger = make_ledger() if ledger is None else ledger
    frame = pd.DataFrame(
        {
            "object_id": [1, 2, 3, 4],
            "area": [100.0, 200.0, 300.0, 400.0],
            "centroid-0": [1.0, 2.0, 3.0, 4.0],
            "centroid-1": [1.0, 2.0, 3.0, 4.0],
        }
    )
    return frame.merge(ledger.to_frame(), on="object_id")


def make_labels():
    labels = np.zeros((4, 4), dtype=np.int32)
    for object_id in (1, 2, 3, 4):
        labels[object_id - 1, object_id - 1] = object_id
    return labels


def blocked_session(ledger=None):
    ledger = make_ledger() if ledger is None else ledger
    session = AnalysisSession()
    frame = make_frame(ledger)
    session.set_ledger(ledger)
    session.set_context({"labels": make_labels(), "measurements": frame}, frame)
    return session, ledger


class TestTheList:
    def test_the_uncertain_objects_are_listed_worst_first(self, qtbot):
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), make_ledger())
        assert widget.rows()["object_id"].tolist() == [3, 2]
        assert widget.table.item(0, 0).text() == "3"

    def test_an_object_no_seam_touched_is_not_listed(self, qtbot):
        """A review list that shows everything is one nobody reads."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), make_ledger())
        assert 1 not in widget.rows()["object_id"].tolist()

    def test_the_rule_that_decided_each_is_shown(self, qtbot):
        """Which strategy joined it is the first thing a reviewer asks, and
        the four strategies do genuinely different things."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), make_ledger())
        assert widget.table.item(0, 2).text() == "overlap"

    def test_the_summary_says_how_exposed_the_run_was(self, qtbot):
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), make_ledger())
        assert "50.0% touched a seam" in widget.summary_label.text()

    def test_an_in_memory_table_says_so(self, qtbot):
        """Not an empty list, which reads as 'nothing to review' when the
        truth is that this run had no tile boundaries at all."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(pd.DataFrame({"object_id": [1], "area": [1.0]}), None)
        assert widget.rows().empty
        assert "no tile boundaries" in widget.summary_label.text()

    def test_the_threshold_narrows_the_list(self, qtbot):
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), make_ledger())
        widget.threshold.setValue(0.2)
        assert widget.rows()["object_id"].tolist() == [3]


class TestRejecting:
    def test_rejecting_records_it_on_the_ledger(self, qtbot):
        """Beside the objects the seam policy itself dropped, and marked as
        a person's decision rather than an inference."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        ledger = make_ledger()
        widget.set_source(make_frame(ledger), ledger)
        widget.table.selectRow(0)
        assert widget.reject_selected() == 3
        assert ledger.dropped[3] == REJECTED_REASON

    def test_a_rejected_object_drops_out_of_the_list(self, qtbot):
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        ledger = make_ledger()
        widget.set_source(make_frame(ledger), ledger)
        widget.table.selectRow(0)
        widget.reject_selected()
        assert widget.rows()["object_id"].tolist() == [2]
        assert "1 already excluded" in widget.summary_label.text()

    def test_there_is_nothing_to_reject_into_without_a_ledger(self, qtbot):
        """A table loaded on its own can be read, but a rejection has to be
        recorded somewhere, and an unrecorded one is worse than none."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        widget.set_source(make_frame(), None)
        widget.table.selectRow(0)
        assert widget.reject_selected() is None
        assert not widget.reject_button.isEnabled()

    def test_it_offers_no_way_to_redraw_a_boundary(self, qtbot):
        """Deliberate: correcting a seam means rewriting voxels in a label
        array that may be 33 GB. Pinned so it is a decision rather than an
        omission somebody 'fixes' by adding an edit tool."""
        widget = SeamReviewWidget()
        qtbot.addWidget(widget)
        assert widget.table.editTriggers() == widget.table.EditTrigger.NoEditTriggers


class TestInTheExplorer:
    def test_the_tab_appears_for_a_blocked_run(self, qtbot):
        session, _ledger = blocked_session()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        assert widget.tabs.indexOf(widget.seam_review) >= 0
        assert widget.tabs.tabText(widget.tabs.indexOf(widget.seam_review)) == SEAM_TAB_NAME

    def test_there_is_no_tab_for_an_in_memory_run(self, qtbot):
        """A permanently empty tab reads as a broken feature rather than as
        an absent condition."""
        session = AnalysisSession()
        frame = pd.DataFrame({"object_id": [1], "area": [1.0]})
        session.set_context({"measurements": frame}, frame)
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        assert widget.tabs.indexOf(widget.seam_review) < 0

    def test_the_tab_appears_when_a_blocked_run_publishes(self, qtbot):
        """The explorer is usually open before the run finishes."""
        session = AnalysisSession()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        assert widget.tabs.indexOf(widget.seam_review) < 0

        ledger = make_ledger()
        frame = make_frame(ledger)
        session.set_ledger(ledger)
        session.set_context({"labels": make_labels(), "measurements": frame}, frame)
        assert widget.tabs.indexOf(widget.seam_review) >= 0
        assert widget.seam_review.rows()["object_id"].tolist() == [3, 2]

    def test_gating_them_puts_a_real_gate_on_the_plot(self, qtbot):
        """A gate, not a special selection mode, so it composes with the
        size and brightness gates already drawn."""
        session, _ledger = blocked_session()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget.seam_review.gate_button.click()

        gates = list(widget.gate_set)
        assert len(gates) == 1
        mask = widget.gate_set.mask(gates[0].id, widget.frame)
        assert set(widget.frame.loc[mask, "object_id"]) == {2, 3}

    def test_gating_them_switches_the_axes_to_where_the_gate_is(self, qtbot):
        """A gate whose outline is invisible on the axes in front of you is
        indistinguishable from one that was never added."""
        session, _ledger = blocked_session()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget.add_seam_gate(0.8)
        assert (widget.plot.x_column, widget.plot.y_column) == (
            "seam_confidence",
            "n_fragments",
        )

    def test_a_rejection_is_reported(self, qtbot):
        session, ledger = blocked_session()
        widget = ExplorerWidget(session=session)
        qtbot.addWidget(widget)
        widget.seam_review.table.selectRow(0)
        widget.seam_review.reject_selected()
        assert "excluded" in widget.status_label.text()
        assert ledger.dropped
