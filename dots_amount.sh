#!/bin/bash

IFS=. VARTMP=(X$1X) # avoid stripping dots
echo $(( ${#VARTMP[@]} - 1 ))