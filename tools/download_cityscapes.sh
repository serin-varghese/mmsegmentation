#!/bin/bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=myusername&password=mypassword&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1

# packageID=4   → leftImg8bit_trainvaltest.zip                     (RGB images with fine annotations)
# packageID=14  → leftImg8bit_sequence_trainvaltest.zip (324GB)   (30-frame snippets (17Hz) surrounding each left 8-bit image (-19 | +10) from the train, val, and test sets (150000 images))
# packageID=2   -> gtCoarse_trainvaltest.zip                     (RGB images with coarse annotations)