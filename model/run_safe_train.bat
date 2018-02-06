@ECHO OFF

SET dirname=%1
SHIFT & SHIFT

ECHO dirname = %dirname%

mkdir %dirname%
copy train.py "%dirname%/train.py"
copy utils.py "%dirname%/utils.py"
copy model.py "%dirname%/model.py"
copy layers.py "%dirname%/layers.py"
copy gen_dataset.py "%dirname%/gen_dataset.py"
copy config.py "%dirname%/config.py"
copy mytf.py "%dirname%/mytf.py"

notepad "%dirname%/train.py"

python "%dirname%/train.py"