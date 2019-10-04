# coding: utf8
"""
Created on 12/07/2019
@author: baptiste

Give the individual tests import contex.
Source : https://docs.python-guide.org/writing/structure/#structure-of-the-repository
"""


import site
from pathlib import Path


cur_dir = Path(__file__).resolve().parent
repository_dir = cur_dir.parent
site.addsitedir(repository_dir)

import ho_homog
