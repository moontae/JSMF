#!/bin/bash
find . -name '$1' | tar -cvzf $2.tar.gz --files-from -

