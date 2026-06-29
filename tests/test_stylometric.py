"""Tests for Signal 2: stylometric heuristics.

These run fully offline and deterministically. Direction assertions use the
calibration samples; thresholds are loose enough to be robust, tight enough to
catch a regression that flips a class.
"""

import pytest

from provenanceguard.signals.stylometric import (
    NEUTRAL_FALLBACK,
    SIGNAL_NAME,
    _cv,
    _mattr,
    analyze_stylometry,
)

# --- Calibration samples -------------------------------------------------

HUMAN_FREE_VERSE = """I celebrate myself, and sing myself,
And what I assume you shall assume,
For every atom belonging to me as good belongs to you.
I loafe and invite my soul,
I lean and loafe at my ease observing a spear of summer grass.
My tongue, every atom of my blood, form'd from this soil, this air,
Born here of parents born here from parents the same, and their parents the same."""

HUMAN_PROSE = """I went down to the river yesterday. It was colder than I expected, and the
wind kept tugging at my coat. A man was fishing near the old bridge, the one
nobody uses anymore. He didn't catch anything, as far as I could tell, but he
seemed perfectly content to sit there for hours. I watched him for a while,
then walked home the long way."""

AI_PROSE = """Artificial intelligence is transforming the way we live and work today.
It helps people solve problems faster and makes many tasks much easier overall.
Businesses use it to improve their services and to understand their customers better.
Students rely on it to learn new things and to complete their assignments quickly."""

AI_POEM = """The gentle morning light begins to softly glow,
The peaceful river flows so calm and slow,
The quiet birds will sing their gentle song,
The happy day will surely last so long."""

# Metered human verse is inherently uniform -- the documented hard case.
HUMAN_METERED = """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,
And sweetest in the gale is heard;
And sore must be the storm
That could abash the little bird
That kept so many warm."""

# Heavy phrase repetition but varied line lengths (edge case from planning.md).
REPETITIVE_HUMAN = """I remember, I remember,
The house where I was born,
The little window where the sun came peeping in at morn;
He never came a wink too soon,
Nor brought too long a day,
But now, I often wish the night had borne my breath away."""

SHORT = "Roses are red, violets are blue."


# --- Direction tests -----------------------------------------------------


@pytest.mark.parametrize("text", [AI_PROSE, AI_POEM])
def test_ai_text_scores_above_half(text):
    assert analyze_stylometry(text).score > 0.6


@pytest.mark.parametrize("text", [HUMAN_FREE_VERSE, HUMAN_PROSE])
def test_human_text_scores_below_half(text):
    assert analyze_stylometry(text).score < 0.4


def test_ai_clearly_separates_from_human():
    assert analyze_stylometry(AI_PROSE).score > analyze_stylometry(HUMAN_PROSE).score


# --- Edge cases ----------------------------------------------------------


def test_metered_human_verse_is_hard_but_not_confidently_ai():
    # Inherently uniform; we only require it not be confidently flagged AI.
    assert analyze_stylometry(HUMAN_METERED).score < 0.6


def test_phrase_repetition_does_not_force_ai_flag():
    # Repetition only touches the (weight-0) diversity metric, so a repetitive
    # human poem must not be strongly flagged on stylometry alone.
    assert analyze_stylometry(REPETITIVE_HUMAN).score < 0.6


def test_short_text_is_pulled_to_neutral():
    result = analyze_stylometry(SHORT)
    assert result.score == pytest.approx(NEUTRAL_FALLBACK)
    assert result.details["reliability"] == 0.0


def test_empty_text_is_neutral_and_not_ok():
    result = analyze_stylometry("   ")
    assert result.ok is False
    assert result.score == NEUTRAL_FALLBACK


# --- Contract / details --------------------------------------------------


def test_result_shape_and_details():
    result = analyze_stylometry(AI_PROSE)
    assert result.name == SIGNAL_NAME
    assert 0.0 <= result.score <= 1.0
    assert "confidence" in result.details
    assert set(result.details["metrics"]) == {
        "sentence_burstiness",
        "lexical_diversity",
        "word_length_burstiness",
        "punctuation_variety",
    }


def test_longer_text_is_more_reliable_than_shorter():
    short = analyze_stylometry(AI_POEM).details["reliability"]
    long = analyze_stylometry(AI_PROSE + " " + AI_PROSE).details["reliability"]
    assert long >= short


# --- Helper units --------------------------------------------------------


def test_cv_of_uniform_is_zero_and_needs_two_values():
    assert _cv([5, 5, 5]) == 0.0
    assert _cv([5]) is None
    assert _cv([]) is None


def test_cv_increases_with_spread():
    assert _cv([4, 4, 4, 4]) < _cv([1, 4, 2, 9])


def test_mattr_of_all_unique_is_one():
    assert _mattr(["a", "b", "c"]) == pytest.approx(1.0)
    assert _mattr(["a", "a", "a"]) == pytest.approx(1 / 3)
    assert _mattr([]) is None
