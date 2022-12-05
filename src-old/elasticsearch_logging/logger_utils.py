import os
from elasticsearch import Elasticsearch
from datetime import datetime


password = os.getenv("ELASTIC_PASSWORD")

es = Elasticsearch(
    hosts=[
        "https://elasticsearch:9200"
    ],
    http_auth=("elastic", password),
    verify_certs=True,
    ca_certs="/usr/share/elasticsearch/config/certs/ca/ca.crt",
)

doc = {
    'author': 'kimchy',
    'text': 'Elasticsearch: cool. bonsai cool.',
    'timestamp': datetime.now()
}
res = es.index(index="test-index", id=1, document=doc)