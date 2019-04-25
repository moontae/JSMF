#!/bin/sh
tar -cvzf models.tar.gz --exclude='*.mat' --exclude='*mallet*' --exclude='*.heldouts' *

