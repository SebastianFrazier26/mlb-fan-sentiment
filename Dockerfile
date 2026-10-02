FROM python:3.12-slim

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# data/ is mounted so CSVs persist on the host
VOLUME /app/data
ENTRYPOINT ["fansent"]
CMD ["--help"]
