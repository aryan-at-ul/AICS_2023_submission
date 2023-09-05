#!/bin/bash

pip install gdown

zip_file_id="1ExZ-N4_RcMzNZO2uI-wo7pJqAFY135X9"

gdown --id $zip_file_id

mkdir -p chest_xray && unzip -o chest_xray.zip 


rm chest_xray.zip
