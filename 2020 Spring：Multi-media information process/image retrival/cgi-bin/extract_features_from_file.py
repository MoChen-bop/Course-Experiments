#!E:/Anaconda3/python
# -*- coding: UTF-8 -*-

import cgi, cgitb
import os
import re
import math
import numpy as np
import time
import argparse
import copy
from collections import namedtuple
import codecs, sys
sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
import warnings
warnings.filterwarnings("ignore")

from extract_features import extract_features

form = cgi.FieldStorage()

fileitem = form.getvalue('filename')

# if fileitem.filename:
#    # 设置文件路径 
#    fn = os.path.basename(fileitem.filename)
#    #open('/tmp/' + fn, 'wb').write(fileitem.file.read())
#    message = '文件 "' + fn + '" 上传成功'
# else:
#    message = '文件没有上传'
message = ""
if isinstance(fileitem, list):
    message = str(fileitem[0])[:fileitem[0].rfind('\\')]

content = """

<!DOCTYPE html>

<html>

<head>

	<title>Image Retrival</title>

	<script type="text/javascript" src="./jquery.js" height="10px"></script>

</head>

<body>

	<p align="center"><img src="../src/icon.png" width="500" height="200"/></p>

    <div align="center">
		<form action="http://localhost:8088/cgi-bin/extract_features_from_file.py">    
		    <p><input type="file" name="filename" id="selectFiles" οnchange="dealSelectFiles()" webkitdirectory directory multiple class="button"></p>
		    <p><input type="submit" class="button" value="Build Database"></p>
		</form>
    </div>
"""

tail = """

    
    <div align="center">
		<form action="http://localhost:8088/cgi-bin/image_search.py">  
		    <p><input type="submit" class="button" value="Skip"></p>
		</form>
    </div>

	<style>
		.button {
		    background-color: #2259AA;
		    border: none;
		    color: white;
		    padding: 15px 32px;
		    text-align: center;
		    text-decoration: none;
		    display: inline-block;
		    font-size: 16px;
		    margin: 4px 2px;
		    cursor: pointer;
		}
		.button:hover {
		    background-color: #7D7DD8;
		}
	</style>

	<script type="text/javascript">

		function dealSelectFiles(){

		}

	</script>

	<style>
		.container {				
			width: 500px;				
			height: 50px;				
			margin: 100px auto;			
		}

		.parent {				
			width: 100%;				
			height: 42px;				
			top: 4px;				
			position: relative;			
		}						

		.parent>input:first-of-type {				
			/*输入框高度设置为40px, border占据2px，总高度为42px*/	
			width: 400px;				
			height: 100px; 				
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
	</style>

	<style>
	.container {
		margin:50px auto;
		width:340px;
	}
	.container > div {
		margin-bottom:20px;
	}
	.progress {
		height:20px;
		background:#ebebeb;
		border-left:1px solid transparent;
		border-right:1px solid transparent;
		border-radius:10px;
	}
	.progress > span {
		position:relative;
		float:left;
		margin:0 -1px;
		min-width:30px;
		height:18px;
		line-height:16px;
		text-align:right;
		background:#cccccc;
		border:1px solid;
		border-color:#bfbfbf #b3b3b3 #9e9e9e;
		border-radius:10px;
		background-image:-webkit-linear-gradient(top,#f0f0f0 0%,#dbdbdb 70%,#cccccc 100%);
		background-image:-moz-linear-gradient(top,#f0f0f0 0%,#dbdbdb 70%,#cccccc 100%);
		background-image:-o-linear-gradient(top,#f0f0f0 0%,#dbdbdb 70%,#cccccc 100%);
		background-image:linear-gradient(to bottom,#f0f0f0 0%,#dbdbdb 70%,#cccccc 100%);
		-webkit-box-shadow:inset 0 1px rgba(255,255,255,0.3),0 1px 2px rgba(0,0,0,0.2);
		box-shadow:inset 0 1px rgba(255,255,255,0.3),0 1px 2px rgba(0,0,0,0.2);
	}
	.progress > span > span {
		padding:0 8px;
		font-size:11px;
		font-weight:bold;
		color:#404040;
		color:rgba(0,0,0,0.7);
		text-shadow:0 1px rgba(255,255,255,0.4);
	}
	.progress > span:before {
		content:'';
		position:absolute;
		top:0;
		bottom:0;
		left:0;
		right:0;
		z-index:1;
		height:18px;
		background:url("../img/progress.png") 0 0 repeat-x;
		border-radius:10px;
	}
	.progress .blue {
		background:#5aaadb;
		border-color:#459fd6 #3094d2 #277db2;
		background-image:-webkit-linear-gradient(top,#aed5ed 0%,#7bbbe2 70%,#5aaadb 100%);
		background-image:-moz-linear-gradient(top,#aed5ed 0%,#7bbbe2 70%,#5aaadb 100%);
		background-image:-o-linear-gradient(top,#aed5ed 0%,#7bbbe2 70%,#5aaadb 100%);
		background-image:linear-gradient(to bottom,#aed5ed 0%,#7bbbe2 70%,#5aaadb 100%);
	}
	</style>

</body>

</html>

"""

progress = """
 <section class="container"><div align="center" class="progress">
      <span class="blue" id="progress1" style="width: %d%%;"><span id="progress2">%d%%</span></span>
    </div></section>"""

p = progress % (99, 99)
print(content + p + str(message) + tail)

if os.path.exists(message):
    save_path = "./features.csv"
    if not os.path.exists(save_path):
        extract_features(message)