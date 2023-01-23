#!/bin/bash              


pathfilee='~/nestor/CVI-Frechet-Mahalanobis/AnotherIndicesTest/SyntheticData Configuration/info1_real_normaloutnoise.xlsx'
echo $pathfilee
python3 calculateindex.py  -f $pathfilee -s "~/" -c '0' -e '50'  &
