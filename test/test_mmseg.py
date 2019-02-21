#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 * Copyright (C) 2018 OwnThink Technologies Inc.
 *
 * Name        : mmseg.py
 * Author      : Leo <1162441289@qq.com>
 * Version     : 0.01
 * Description : mmseg分词方法测试
"""

import unittest
import re
from jiagu import utils
import jiagu


class TestTextRank(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass
    
    def test_seg_one(self):
        sentence = "人要是行干一行行一行"
        words = jiagu.seg(sentence, model="mmseg")
        self.assertTrue(list(words) == ['人', '要是', '行', '干', '一行', '行', '一行'])

    def test_seg_two(self):
        sentence = "武汉市长江大桥最近已经崩塌了"
        words = jiagu.seg(sentence, model="mmseg")
        self.assertTrue(list(words) == ['武汉市', '长江大桥', '最近', '已经', '崩塌', '了'])


if __name__ == '__main__':
    unittest.main()
