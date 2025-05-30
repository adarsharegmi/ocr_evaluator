#!/usr/bin/env python3
"""
opensearch_sanity.py – exhaustive checks for "docs exist but _search = 0"
"""
import argparse, json, textwrap, logging, sys
from pprint import pprint

import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection
from requests_aws4auth import AWS4Auth

# ---------- logging setup ---------------------------------------------------
logger = logging.getLogger("opensearch_sanity")
logger.setLevel(logging.INFO)

if not logger.handlers:
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    console_handler.setFormatter(console_formatter)

    # File handler
    file_handler = logging.FileHandler("logger.log")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    file_handler.setFormatter(file_formatter)

    # Add handlers
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# ---------- helpers ---------------------------------------------------------
def awsauth(region: str):
    sess = boto3.Session()
    creds = sess.get_credentials()
    if creds is None:
        sys.exit("❌  No AWS credentials found – configure a profile or export env vars.")
    ident = boto3.client("sts").get_caller_identity()
    logger.info("Using AWS principal: %s / %s", ident["Account"], ident["Arn"])
    return AWS4Auth(creds.access_key, creds.secret_key, region, "es", session_token=creds.token)

def connect(host: str, region: str):
    logger.info("Connecting to %s", host)
    return OpenSearch(
        hosts=[{"host": host, "port": 443}],
        http_auth=awsauth(region),
        use_ssl=True, verify_certs=True,
        connection_class=RequestsHttpConnection,
        timeout=60,
    )

# ---------- main diagnostics ------------------------------------------------
def run(host: str, index: str, region: str):
    client = connect(host, region)


    logger.info("========= FULL INDEX LIST (name → docs) =========")
    for info in client.cat.indices(format="json"):
        logger.info("%-40s  %s docs", info["index"], info["docs.count"])

    if not client.indices.exists(index=index):
        sys.exit(f"❌  Nothing called \"{index}\" on this domain.\n")
    logger.info("Index/alias \"%s\" exists.", index)

    pre = client.cat.count(index=index, format="json")[0]["count"]
    client.indices.refresh(index=index)
    post = client.cat.count(index=index, format="json")[0]["count"]
    logger.info("Doc count BEFORE refresh: %s  |  AFTER refresh: %s", pre, post)
    if post == "0":
        sys.exit("⚠️  Zero docs after refresh – the index really is empty on this domain.")

    mapping = client.indices.get_mapping(index=index)
    logger.info("========= MAPPING (first 1 200 chars) =========\n%s",
                textwrap.shorten(json.dumps(mapping, indent=2),
                                 width=1_200, placeholder="…"))


    # def get_embedding(text, model_id="cohere.embed-english-v3"):
    #     try:
    #         bedrock_runtime = boto3.client('bedrock-runtime', region_name="us-east-1")
    #         request_body = json.dumps({
    #             "texts": [text],
    #             "input_type": "search_query"
    #         })
    #         response = bedrock_runtime.invoke_model(
    #             modelId=model_id,
    #             body=request_body
    #         )
    #         response_body = json.loads(response.get('body').read())
    #         embeddings = response_body.get('embeddings', [])
    #         if embeddings:
    #             return embeddings[0]
    #         else:
    #             raise ValueError("No embeddings returned from Bedrock")
    #     except Exception as e:
    #         print(f"Error calling Bedrock: {str(e)}")
    #         raise

    # embedding = get_embedding("visa", model_id="cohere.embed-english-v3")
    # # store the embedding in file without ...
    # with open("embedding.txt", "w") as f:
    #     f.write(",".join(map(str, embedding)))








    # m_all = client.search(index=index,
    #                       body={"track_total_hits": True,
    #                             "query": {"match_all": {}}},
    #                       size=5)


    with open("embedding.txt", "r") as f:
        embedding = [float(x) for x in f.read().strip().split(",")]


    settings = client.indices.get_settings(index=index)
    print(json.dumps(settings, indent=2))




    breakpoint()
    m_all = client.search(
        index=index,
        body={
            "size": 5,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": 5
                    }
                }
            }
        }
    )


    logger.info("match_all  →  total hits reported: %s",
                m_all["hits"]["total"]["value"])
    
    
    pprint(m_all["hits"]["hits"][:3])
    with open("match_all.json", "w") as f:
        json.dump(m_all, f, indent=2, ensure_ascii=False)
    # LOG THIS RESULT
    logger.info("Search result for `match_all` on index \"%s\":\n%s",
                index, json.dumps(m_all, indent=2))
    
    if m_all["hits"]["total"]["value"] == 0:
        logger.warning("No documents found for `match_all` on index \"%s\".", index)
    kw = client.search(index=index,
                       body={"size": 3,
                             "query": {"match": {"content": "visa"}}})
    logger.info("`match` on field \"content\"  →  hits: %s",
                kw["hits"]["total"]["value"])
    breakpoint()
    pprint(kw["hits"]["hits"][:3])

    if m_all["hits"]["hits"]:
        _id = m_all["hits"]["hits"][0]["_id"]
        doc = client.get(index=index, id=_id)
        logger.info("Fetched same doc by ID (%s) – proves it's physically present.", _id)
        pprint(doc["_source"])
        # log source 
        with open("doc_source.json", "w") as f:
            json.dump(doc["_source"], f, indent=2)

# ---------- CLI entry -------------------------------------------------------
if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser()
        p.add_argument("--host", required=True)
        p.add_argument("--index", required=True, help="Index OR alias you believe contains docs")
        p.add_argument("--region", default="us-east-1")
        args = p.parse_args()
        run(args.host, args.index, args.region)
    finally:
        # Ensure logs are flushed on script exit
        for handler in logger.handlers:
            handler.flush()
