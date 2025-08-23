import gradio as gr
from openai import AzureOpenAI
import os
import requests
import base64
from dotenv import load_dotenv
from urllib.parse import urlparse

# Load environment variables from .env file
load_dotenv()

# Initialize Azure OpenAI client
def get_openai_client():
    try:
        return AzureOpenAI(
            api_key=os.getenv('AZURE_OPENAI_API_KEY').strip(),
            api_version=os.getenv('AZURE_OPENAI_API_VERSION').strip(),
            azure_endpoint=os.getenv('AZURE_OPENAI_ENDPOINT').strip(),
        )
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        return None


# Expanded list of important files that usually hold project logic
important_files = [
    # Documentation
    "README.md", "README.rst",

    # Python backends
    "app.py", "main.py", "server.py", "bot.py", "manage.py",
    "config.py", "settings.py", "routes.py",

    # Node.js / Express
    "index.js", "server.js", "app.js",

    # React / Next.js
    "App.tsx", "page.tsx", "index.tsx", "_app.tsx", "_document.tsx",

    # Angular / Vue
    "main.ts", "app.component.ts", "App.vue",

    # ML/AI Core
    "model.py", "train.py", "predict.py", "inference.py",

    # Setup / Environment
    "requirements.txt", "package.json", "pyproject.toml",
    "setup.py", "Dockerfile", ".env.example"
]


def fetch_repo_files(repo_url):
    """
    Fetch important files from a public GitHub repo.
    """
    try:
        parsed = urlparse(repo_url)
        path_parts = parsed.path.strip("/").split("/")
        if len(path_parts) < 2:
            return None

        owner, repo = path_parts[0], path_parts[1].replace(".git", "")
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents"

        repo_content = {}
        for file in important_files:
            file_url = f"{api_url}/{file}"
            resp = requests.get(file_url)
            if resp.status_code == 200:
                data = resp.json()
                if data.get("encoding") == "base64":
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                else:
                    content = data.get("content", "")
                repo_content[file] = content

        return repo_content if repo_content else None

    except Exception as e:
        print(f"[ERROR] Failed to fetch repo files: {e}")
        return None

def generate_post(topic, tone="Professional"):
    try:
        client = get_openai_client()
        if not client:
            return "❌ OpenAI client initialization failed. Check API settings."

        if not topic:
            return "⚠️ Please provide a topic or GitHub repo link."

        # Check if topic is a GitHub repo link
        repo_content = None
        if topic.startswith("http") and "github.com" in topic:
            repo_content = fetch_repo_files(topic)

        if repo_content:
            combined_summary = "\n\n".join(
                [f"--- {fname} ---\n{content}" for fname, content in repo_content.items()]
            )
            prompt = (
                f"Write a {tone.lower()} LinkedIn post summarizing the following GitHub project.\n\n"
                f"Focus on the purpose, tech stack, and innovation:\n\n{combined_summary}"
            )
        else:
            prompt = f"Write a {tone.lower()} LinkedIn post about: {topic}"

        messages = [
            {"role": "system", "content": "You are a professional LinkedIn post writer."},
            {"role": "user", "content": prompt}
        ]

        response = client.chat.completions.create(
            model=os.getenv('AZURE_OPENAI_DEPLOYMENT'),
            messages=messages,
            max_tokens=1000,
            temperature=0.9,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"❌ Error: {str(e)}"


# Build Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 LinkedIn Post Generator\nGenerate professional LinkedIn posts powered by Azure OpenAI.\n\nPaste a **topic** or a **GitHub repo link**.")

    with gr.Row():
        topic_input = gr.Textbox(label="Topic or GitHub Repo Link", placeholder="Enter project topic or GitHub repo link...")
    
    tone_input = gr.Dropdown(
        ["Professional", "Casual", "Excited", "Thoughtful", "Inspirational"],
        value="Professional",
        label="Tone"
    )

    output = gr.Textbox(label="Generated LinkedIn Post", lines=15)

    generate_btn = gr.Button("✨ Generate Post")
    generate_btn.click(fn=generate_post, inputs=[topic_input, tone_input], outputs=output)


# Run app
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, pwa=True)
