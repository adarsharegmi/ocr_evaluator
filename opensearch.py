from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth
import boto3
import os

from dotenv import load_dotenv
region = 'us-east-1'  # your region
service = 'es'
credentials = boto3.Session().get_credentials()
awsauth = AWS4Auth(credentials.access_key, credentials.secret_key, region, service, session_token=credentials.token)

load_dotenv()  # Load environment variables from .env file
host = os.getenv('OPENSEARCH_HOST')  # e.g. 'search-my-domain.us-east-1.es.amazonaws.com'
if not host:
    raise ValueError("OPENSEARCH_HOST environment variable is not set.")

client = OpenSearch(
    hosts=[{'host': host, 'port': 443}],
    http_auth=awsauth,
    use_ssl=True,
    verify_certs=True,
    connection_class=RequestsHttpConnection
)

response = client.delete_by_query(
    index="embeddings-v2",
    body={
        "query": {
            "match_all": {}
        }
    }
)

print("Delete by query response:", response)
# get size of the content in the index

# response = client.cat.count(
#     index="embeddings",
#     format="json"
# )

# # # get all the documents  in the index to find the data of the content relevant to "UK visa"

# response = client.search(
#     index="embeddings",
#     body={
#         "query": {
#             "match": {
#                 "content": "Japan Visa ?"
#             }
#         }
#     }
# )



# response = client.search(
#     index="embeddings",
#     body={"query": {"match": {"content": "leonardo"}}}
# )
# print(f"Total docs: {response['hits']['total']['value']}")




# response_analysis = client.indices.analyze(body={
#     "analyzer": "standard",
#     "text": "J"
# })

# print("Analysis Result:")
# for token in response_analysis['tokens']:
#     print(f"Token: {token['token']}, Type: {token['type']}, Position: {token['position']}")
# print("Response:", response)



#!/usr/bin/env python3
# """
# Quick diagnostics for an Amazon OpenSearch index.
# """

# import argparse
# import json
# import logging
# import textwrap
# from pprint import pprint

# import boto3
# from opensearchpy import OpenSearch, RequestsHttpConnection
# from requests_aws4auth import AWS4Auth

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s %(message)s",
#     level=logging.INFO,
# )

# def get_awsauth(region: str):
#     """Return AWS4Auth for the active profile or environment vars."""
#     session = boto3.Session()
#     credentials = session.get_credentials()
#     if credentials is None:
#         raise RuntimeError("No AWS credentials found in environment or config files.")

#     logging.info("Using AWS credentials for account %s",
#                  boto3.client("sts").get_caller_identity()["Account"])

#     return AWS4Auth(
#         credentials.access_key,
#         credentials.secret_key,
#         region,
#         "es",
#         session_token=credentials.token,
#     )


# def connect(host: str, awsauth):
#     """Return an OpenSearch client for the given host."""
#     logging.info("Connecting to %s", host)
#     return OpenSearch(
#         hosts=[{"host": host, "port": 443}],
#         http_auth=awsauth,
#         use_ssl=True,
#         verify_certs=True,
#         connection_class=RequestsHttpConnection,
#     )


# def main():
#     parser = argparse.ArgumentParser(
#         description="OpenSearch index sanity checks",
#         formatter_class=argparse.RawTextHelpFormatter,
#     )
#     parser.add_argument("--host", required=True, help="search-…es.amazonaws.com endpoint")
#     parser.add_argument("--index", required=True, help="Index (or alias) to check")
#     parser.add_argument("--region", default="us-east-1")
#     parser.add_argument("--doc-id", help="Optional: fetch this document by ID")
#     args = parser.parse_args()

#     client = connect(args.host, get_awsauth(args.region))

#     # # 1️⃣ Index exists?
#     # if not client.indices.exists(index=args.index):
#     #     logging.warning("Index '%s' does NOT exist on this domain.", args.index)
#     #     logging.info("Existing indices:\n%s",
#     #                  json.dumps(client.cat.indices(format='json'), indent=2))
#     #     return

#     # # 2️⃣ Doc count before/after refresh
#     # before = client.cat.count(index=args.index, format="json")
#     # logging.info("Shard-level count BEFORE refresh: %s", before[0]["count"])

#     # client.indices.refresh(index=args.index)

#     # after = client.cat.count(index=args.index, format="json")
#     # logging.info("Shard-level count AFTER  refresh: %s", after[0]["count"])

#     # # 3️⃣ Mapping (truncate long vectors for readability)
#     # mapping = client.indices.get_mapping(index=args.index)
#     # logging.info("Mapping for %s (first 1200 chars):\n%s",
#     #              args.index,
#     #              textwrap.shorten(json.dumps(mapping, indent=2), width=1200, placeholder="…"))

#     # # 4️⃣ match_all
#     ma_resp = client.search(index=args.index, body={"query": {"match_all": {}}}, size=3)




#     # m_all = client.search(
#     #     index="embeddings",
#     #     body={
#     #         "size": 5,
#     #         "query": {
#     #             "knn": {
#     #                 "embedding": {
#     #                     "vector": ,
#     #                     "k": 5
#     #                 }
#     #             }
#     #         }
#     #     }
#     # )
#     logging.info("match_all hits: %s", ma_resp["hits"]["total"]["value"])
#     pprint(ma_resp["hits"]["hits"][:3])
#     with open("match_all.txt", "w") as f:
#         f.write(
#             textwrap.shorten(json.dumps(ma_resp["hits"]["hits"], indent=2), width=1200, placeholder="…")
#         )


#     # 5️⃣ sample keyword match
#     # kw_resp = client.search(index=args.index, body={
#     #     "size": 3,
#     #     "query": {"match": {"content": "visa"}}
#     # })
#     # logging.info("Keyword match hits: %s", kw_resp["hits"]["total"]["value"])
#     # pprint(kw_resp["hits"]["hits"][:3])

#     # 6️⃣ Optional: fetch a specific document
#     if args.doc_id:
#         try:
#             doc = client.get(index=args.index, id=args.doc_id)
#             logging.info("Document %s:\n%s", args.doc_id,
#                          textwrap.shorten(json.dumps(doc, indent=2), width=1200, placeholder="…"))
            
#             # extract the content to text 
#             with open(f"{args.doc_id}.txt", "w") as f:
#                 if "content" in doc["_source"]:
#                     f.write(doc["_source"]["content"])
#                 else:
#                     logging.warning("No 'content' field found in document %s", args.doc_id)
#             logging.info("Document content saved to %s.txt", args.doc_id)
#         except Exception as e:
#             logging.error("Could not fetch doc ID %s: %s", args.doc_id, e)


# if __name__ == "__main__":
#     main()