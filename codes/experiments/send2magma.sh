#!/bin/bash
SERVER="$2@magma.coecis.cornell.edu"
DIRECTORY="~/JSMF/codes/experiments/$1"
rm -f $1/mccExcludedFiles.log
rm -f $1/readme.txt
rm -f $1/requiredMCRProducts.txt
ssh "$SERVER" "mkdir -p $DIRECTORY" && scp -r $1/*exec* ../../helpers/magma/*.sh "$SERVER:$DIRECTORY"
