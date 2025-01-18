#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from codecs import open
import os, ssl
if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

"""
CS 188 Local Submission Autograder
Written by the CS 188 Staff, UC Berkeley

==============================================================================
   _____ _              _ 
  / ____| |            | |
 | (___ | |_ ___  _ __ | |
  \___ \| __/ _ \| '_ \| |
  ____) | || (_) | |_) |_|
 |_____/ \__\___/| .__/(_)
                 | |      
                 |_|      

Modifying or tampering with this file is a violation of course policy.
If you're having trouble running the autograder, please contact the staff.
==============================================================================
"""
import bz2, base64
exec(bz2.decompress(base64.b64decode('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')))

