#!/bin/bash 

convert -delay 4 -loop 0 $1/*.png[!160x210] $1/animate.gif
