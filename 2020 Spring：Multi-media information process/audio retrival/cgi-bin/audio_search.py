#!E:/Anaconda3/python
# -*- coding: UTF-8 -*-

import cgi, cgitb
import os
import re
import math
import numpy as np
import argparse
from collections import namedtuple
import codecs, sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())

from extract_features import extract_features, get_result_list

form = cgi.FieldStorage()

query = form.getvalue('text')

header = \
'''
<!DOCTYPE html>
<html>
<head>
	<title>Paper Retrival</title>
	<meta charset="utf-8">
</head>
'''

body = \
'''
<body>
	<div align="center"><img src="../src/icon.png" width="250" height="100" vspace="50"/></div>
	<div class="container">    
		<form action="http://localhost:8088/cgi-bin/audio_search.py" class="parent">        
			<input type="input" name="text">        
			<input type="submit" value="Search">
		</form>
	</div>
	<style>
		.container {				
			width: 500px;				
			height: 50px;				
			margin: 10px auto;			
		}

		.parent {				
			width: 100%;				
			height: 42px;				
			top: 4px;				
			position: relative;			
		}						

		.parent>input:first-of-type {				
			/*输入框高度设置为40px, border占据2px，总高度为42px*/	
			width: 380px;				
			height: 40px; 				
			border: 1px solid #ccc;				
			font-size: 16px;				
			padding-left: 10px;					
			outline: none;			
		}					

		.parent>input:first-of-type:focus {				
			border: 1px solid #317ef3;				
			padding-left: 10px;			
		}						

		.parent>input:last-of-type {				
			/*button按钮border并不占据外围大小，设置高度42px*/		
			width: 100px;		
			height: 44px; 				
			position: absolute;				
			background: #317ef3;				
			border: 1px solid #317ef3;				
			color: #fff;				
			font-size: 16px;				
			outline: none;			
		}

		.listitem {				
			width: 800px;				
			height: 100px;				
			margin: 10px auto;
			margin-left: 200px
		}
	</style>
'''

content = \
'''
'''

template = \
"""
    <li>
    <div class="listitem">
    <p><a href="#">%s</a></p><p> Sim: %.4f</p>
    <p><audio src="%s" controls="autoplay"></audio></p>
    </li>
"""

end = \
'''
</body>
</html>
'''


if query == None: 
	query = ''
result_list = []
if query != '':
    result_list = get_result_list(query)

content = "<div>"
for item in result_list[:20]:
    content += template % (item[1], item[0], '../../' + item[1][40:])

content += "</div>"


print(header + body + content + query + end)

