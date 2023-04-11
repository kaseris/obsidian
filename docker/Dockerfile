# syntax=docker/dockerfile:1
FROM python:3.10
FROM nvidia/cuda:11.4.0-base-ubuntu20.04

WORKDIR /app
ADD . /app/

COPY ../requirements.txt requirements.txt

RUN pip3 install -r requirements.txt