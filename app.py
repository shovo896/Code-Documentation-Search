import gradio as gr
from retriever import get_qa_chain
import sys

print(" Loading model...")
try:
    chain = get_qa_chain()
    print(" Ready!")
except Exception as e:
    print(f"\n❌ Error: {str(e)}")
    if "NOT_FOUND" in str(e) or "code-doc-search not found" in str(e):
        print("\n⚠️  Pinecone index 'code-doc-search' not found!")
        print("\nPlease run ingest.py first to populate the index:")
        print("   python ingest.py <GITHUB_REPO_URL> [BRANCH]")
        print("\nExample:")
        print("   python ingest.py https://github.com/shovo896/Code-Documentation-Search main")
    sys.exit(1)

def answer_query(question):
    if not question.strip():
        return "Write Question:", ""

    result = chain.invoke({"query": question})
    answer = result["result"]

    sources = set()
    for doc in result["source_documents"]:
        src = doc.metadata.get("source", "unknown")
        sources.add(src)

    source_text = "\n".join([f" {s}" for s in sources])
    return answer, source_text

with gr.Blocks(title="Code Doc Search", theme=gr.themes.Soft()) as demo:
    gr.Markdown("Code Documentation Search")
    gr.Markdown("Ask any question about your codebase and get instant answers, powered by LLMs and Pinecone vector search!")

    query = gr.Textbox(
        label="Your Question:",
        placeholder="e.g. How does the RetrievalQA chain work?",
        lines=2
    )

    btn = gr.Button("Search ", variant="primary")

    with gr.Row():
        answer_box = gr.Textbox(label=" Answer", lines=10)
        source_box = gr.Textbox(label=" Sources", lines=10)

    btn.click(
        fn=answer_query,
        inputs=query,
        outputs=[answer_box, source_box]
    )

demo.launch()