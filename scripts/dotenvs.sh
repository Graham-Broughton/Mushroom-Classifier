#!/usr/bin/env bash
echo "PYTHONPATH=$(pwd):$PYTHONPATH" >> .env

cp .env monitoring/.env
cp .env training/.env
cp .env deploy/.env
