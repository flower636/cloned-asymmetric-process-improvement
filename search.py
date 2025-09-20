# a_search_minimal.py
from google.cloud import discoveryengine_v1 as discoveryengine

# ★ここを書き換え
PROJECT_ID = "your-project-id"
LOCATION = "global"            # US/EUなどを使っていなければ通常global
DATA_STORE_ID = "your-datastore-id"
SERVING_CONFIG_ID = "default_config"  # 通常はこれ（デフォルト）

QUERY = "社内VPN 手順"

def main():
    client = discoveryengine.SearchServiceClient()

    serving_config = client.serving_config_path(
        project=PROJECT_ID,
        location=LOCATION,
        data_store=DATA_STORE_ID,
        serving_config=SERVING_CONFIG_ID,
    )

    request = discoveryengine.SearchRequest(
        serving_config=serving_config,
        query=QUERY,
        page_size=10,              # 取得件数
        content_search_spec=discoveryengine.SearchRequest.ContentSearchSpec(
            snippet_spec=discoveryengine.SearchRequest.ContentSearchSpec.SnippetSpec(
                return_snippet=True
            )
        )
    )

    response = client.search(request)
    for i, result in enumerate(response):
        doc = result.document
        print(f"[{i+1}] {doc.get('id','(no-id)')}")
        print("  title :", doc.struct_data.get("title") if doc.struct_data else None)
        print("  uri   :", doc.struct_data.get("link") if doc.struct_data else doc.get("uri"))
        if result.snippet:
            print("  snippet:", result.snippet.summary)
        print()

if __name__ == "__main__":
    main()
