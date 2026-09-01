"""The grammar of the flags more than one command of this tree states.

`cli.py` holds the options no single era owns, and the only arithmetic among them is
`parse_seeds`. A sweep states its seeds as a range and an audition as a list; a range read
exclusive at the top would drop the last seed of every sweep this project has reported,
and say nothing.
"""

import cli


def seeds(value):
    """the seeds of one --seeds string; click hands a callback its context and its
    parameter, and a caller with a string in hand passes None for both"""
    return cli.parse_seeds(None, None, value)


def test_a_list_of_seeds_reads_in_the_order_it_is_written():
    """the default is one seed, and a list is the several an audition plays in turn"""
    assert seeds("1") == [1]
    assert seeds("1,3,7") == [1, 3, 7]


def test_a_range_of_seeds_holds_both_of_its_ends():
    """LOW-HIGH IS INCLUSIVE AT BOTH ENDS: 0-15 is the sixteen seeds the panel switches
    can state, and not fifteen of them."""
    assert seeds("4-7") == [4, 5, 6, 7]
    assert seeds("0-15") == list(range(16))
    assert seeds("3-3") == [3]
