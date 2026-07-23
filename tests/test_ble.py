"""
Tests for backend.ble.receiver.check_sequence -- pure packet-sequence
logic, no BLE hardware needed.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.ble.receiver import check_sequence


def test_first_packet_returns_zero():
    assert check_sequence("foot_left", None, 5) == 0


def test_consecutive_packet_returns_zero_dropped():
    assert check_sequence("foot_left", 10, 11) == 0


def test_gap_returns_dropped_count():
    assert check_sequence("foot_left", 10, 13) == 2


def test_duplicate_packet_returns_none():
    assert check_sequence("foot_left", 10, 10) is None


def test_small_out_of_order_returns_none():
    assert check_sequence("foot_left", 10, 8) is None


def test_large_negative_diff_treated_as_rollover():
    # sequence counter wrapped around (e.g. uint16 rollover)
    assert check_sequence("foot_left", 65000, 5) == 0 

