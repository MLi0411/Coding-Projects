import bbs
from bbs import *

import query_several
import os 

def test_example():
    clean_reset()
    connect("kathi", True)
    post_msg("homework 5", "it was just released!")
    post_msg("thoughts on homework 5", "it's the best one yet!")
    remove_msg(1)
    summary = print_summary()
    assert "Poster: kathi" in summary
    assert "Subject: thoughts on homework 5" in summary
    assert not "Subject: homework 5" in summary

def test_basic_post():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "ew ew ew")
    post_msg("test test", "ew ew ew")
    summary = print_summary()
    assert bbs.msg_in_file == 2
    assert "Subject: test test" in summary
    assert "ID: 1" in summary
    assert "ID: 2" in summary

def test_change_user():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "ew ew ew")
    post_msg("test test", "ew ew ew")
    assert bbs.poster == "kathi"
    soft_disconnect()
    assert bbs.poster == ""
    connect("Nick", False)
    assert bbs.poster == "Nick"

def test_file_spillover():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "hello")
    post_msg("test test", "hi")
    post_msg("test test", "idk")
    post_msg("test test", "word")
    post_msg("test remove", "cs200")
    post_msg("test test", "cs400")
    post_msg("test test", "cs100000")
    post_msg("test test", "dillo dallo")
    post_msg("test test", "hello")
    post_msg("test test", "hi")
    post_msg("test test", "idk")
    post_msg("test test", "word")
    post_msg("test remove", "cs200")
    post_msg("test test", "cs400")
    post_msg("test test", "cs100000")
    post_msg("test test", "dillo dallo")
    assert bbs.msg_count == 16
    assert bbs.cur_file == 2

def test_complex_print_sum():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "hello")
    post_msg("test test", "hi")
    post_msg("test test", "idk")
    post_msg("dillao", "test test")
    post_msg("dill pickle", "test test")
    post_msg("dill", "test test")
    post_msg("dillozz", "test test")
    post_msg("dillo dallo", "test")
    summary = print_summary("dillo")
    assert "ID: 7" in summary
    assert "ID: 8" in summary
    assert "Subject: dillozz" in summary
    assert "Subject: dillo dallo" in summary
    assert "Subject: dillao" not in summary

def test_remove():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "hello")
    post_msg("test test", "hi")
    post_msg("test test", "idk")
    post_msg("test test", "word")
    post_msg("test remove", "cs200")
    post_msg("test test", "cs400")
    post_msg("test test", "cs100000")
    post_msg("test test", "dillo dallo")
    assert bbs.msg_count == 8
    assert bbs.cur_file == 1

    remove_msg(5)
    summary = print_summary()
    assert "ID: 5" not in summary
    assert "Subject: test remove" not in summary
    assert "Text: cs200" not in summary
    assert bbs.msg_count == 7
    assert bbs.cur_file == 1
    assert "5" in bbs.avail_ids

    post_msg("new msg!", "200 is back")
    summary = print_summary()
    assert "ID: 5" in summary
    assert bbs.msg_count == 8
    assert bbs.cur_file == 1
    assert "5" not in bbs.avail_ids

def test_find_print():
    clean_reset()
    connect("kathi", True)
    post_msg("test test", "hello")
    post_msg("test test", "hi")
    post_msg("test test", "idk")
    post_msg("test test", "word")
    post_msg("test remove", "cs200")
    post_msg("test test", "cs400")
    post_msg("test test", "cs100000")
    post_msg("test test", "dillo dallo")
    msg = find_print_msg(5)
    assert "cs200" in msg
    assert "test remove" in msg