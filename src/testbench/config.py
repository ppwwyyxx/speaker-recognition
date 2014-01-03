#!/usr/bin/python2
# -*- coding: utf-8 -*-
# $File: config.py
# $Date: Thu Dec 26 13:09:07 2013 +0000
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

frame_duration = 0.032
frame_shift = 0.016
nr_frame_per_sec = int(1.0 / frame_shift)
fs = 8000
train_duration = 30.0
train_len = int(train_duration * nr_frame_per_sec)
nr_test = 50
test_duration = 5.0
testcase_len = int(test_duration * nr_frame_per_sec)

ubm_model_file = 'model/ubm.model'

ubm_set = """m_028_03
f_055_03
m_078_03
f_123_03
f_035_03
f_039_03
f_107_03
m_068_03
f_065_03
m_004_03
f_049_03
f_125_03
f_075_03
f_061_03
f_117_03
f_051_03
m_064_03
m_092_03
f_047_03
f_059_03
f_087_03
f_097_03
m_070_03
f_127_03
f_025_03
m_062_03
m_052_03
f_119_03
f_029_03
f_099_03
f_041_03
f_057_03
f_067_03
f_053_03
f_101_03
f_017_03
m_060_03
f_085_03
m_034_03
m_010_03
m_088_03
f_043_03
f_007_03
f_071_03
m_050_03
m_086_03
f_015_03
f_031_03
f_005_03
m_076_03
m_002_03""".split('\n')

# vim: foldmethod=marker
