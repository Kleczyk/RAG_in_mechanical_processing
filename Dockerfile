FROM postgres:15

RUN apt-get update \
    && apt-get install -y postgresql-server-dev-15 build-essential ca-certificates \
    && git clone --depth 1 https://github.com/pgvector/pgvector.git /pgvector \
    && cd /pgvector \
    && make && make install \
    && cd / && rm -rf /pgvector \
    && apt-get remove -y git build-essential postgresql-server-dev-15 \
    && apt-get autoremove -y && apt-get clean && rm -rf /var/lib/apt/lists/*