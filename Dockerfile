FROM --platform=linux/arm64 debian:bookworm-slim

LABEL description="Raspberry Pi 4 Dev"
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y build-essential git python3 python3-pip python3-venv libglib2.0-0 libgl1 libjpeg-dev zlib1g-dev  vim && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /usr/src/app
CMD ["/bin/bash"]
