import gradio as gr
from download_new_papers import get_papers
import utils
from relevancy import generate_relevance_score, process_subject_fields
from sendgrid.helpers.mail import Mail, Email, To, Content
import sendgrid
import os
import openai
from html import escape

topics = {
    "Physics": "",
    "Mathematics": "math",
    "Computer Science": "cs",
    "Quantitative Biology": "q-bio",
    "Quantitative Finance": "q-fin",
    "Statistics": "stat",
    "Electrical Engineering and Systems Science": "eess",
    "Economics": "econ"
}

physics_topics = {
    "Astrophysics": "astro-ph",
    "Condensed Matter": "cond-mat",
    "General Relativity and Quantum Cosmology": "gr-qc",
    "High Energy Physics - Experiment": "hep-ex",
    "High Energy Physics - Lattice": "hep-lat",
    "High Energy Physics - Phenomenology": "hep-ph",
    "High Energy Physics - Theory": "hep-th",
    "Mathematical Physics": "math-ph",
    "Nonlinear Sciences": "nlin",
    "Nuclear Experiment": "nucl-ex",
    "Nuclear Theory": "nucl-th",
    "Physics": "physics",
    "Quantum Physics": "quant-ph"
}


def no_matching_papers_body(topic, categories, interest):
    category_text = ", ".join(categories) if categories else "all categories"
    interest_text = " after relevance filtering" if interest else ""
    return (
        f"No matching arXiv papers were found for {escape(topic)} "
        f"({escape(category_text)}){interest_text} today."
    )


def openai_ranking_unavailable_body(papers):
    body = (
        "Warning: OpenAI relevance ranking was skipped because the OpenAI API "
        "returned a rate limit or quota error. Showing unranked papers from the "
        "selected categories instead.<br><br>"
    )
    paper_body = papers_body(papers)
    if paper_body:
        body += paper_body
    else:
        body += "No matching arXiv papers were found in the selected categories today."
    return body


def papers_body(papers):
    return "<br><br>".join(
        [
            f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}'
            for paper in papers
        ]
    )

categories_map = {
    "Astrophysics": ["Astrophysics of Galaxies", "Cosmology and Nongalactic Astrophysics", "Earth and Planetary Astrophysics", "High Energy Astrophysical Phenomena", "Instrumentation and Methods for Astrophysics", "Solar and Stellar Astrophysics"],
    "Condensed Matter": ["Disordered Systems and Neural Networks", "Materials Science", "Mesoscale and Nanoscale Physics", "Other Condensed Matter", "Quantum Gases", "Soft Condensed Matter", "Statistical Mechanics", "Strongly Correlated Electrons", "Superconductivity"],
    "General Relativity and Quantum Cosmology": ["None"],
    "High Energy Physics - Experiment": ["None"],
    "High Energy Physics - Lattice": ["None"],
    "High Energy Physics - Phenomenology": ["None"],
    "High Energy Physics - Theory": ["None"],
    "Mathematical Physics": ["None"],
    "Nonlinear Sciences": ["Adaptation and Self-Organizing Systems", "Cellular Automata and Lattice Gases", "Chaotic Dynamics", "Exactly Solvable and Integrable Systems", "Pattern Formation and Solitons"],
    "Nuclear Experiment": ["None"],
    "Nuclear Theory": ["None"],
    "Physics": ["Accelerator Physics", "Applied Physics", "Atmospheric and Oceanic Physics", "Atomic and Molecular Clusters", "Atomic Physics", "Biological Physics", "Chemical Physics", "Classical Physics", "Computational Physics", "Data Analysis, Statistics and Probability", "Fluid Dynamics", "General Physics", "Geophysics", "History and Philosophy of Physics", "Instrumentation and Detectors", "Medical Physics", "Optics", "Physics and Society", "Physics Education", "Plasma Physics", "Popular Physics", "Space Physics"],
    "Quantum Physics": ["None"],
    "Mathematics": ["Algebraic Geometry", "Algebraic Topology", "Analysis of PDEs", "Category Theory", "Classical Analysis and ODEs", "Combinatorics", "Commutative Algebra", "Complex Variables", "Differential Geometry", "Dynamical Systems", "Functional Analysis", "General Mathematics", "General Topology", "Geometric Topology", "Group Theory", "History and Overview", "Information Theory", "K-Theory and Homology", "Logic", "Mathematical Physics", "Metric Geometry", "Number Theory", "Numerical Analysis", "Operator Algebras", "Optimization and Control", "Probability", "Quantum Algebra", "Representation Theory", "Rings and Algebras", "Spectral Theory", "Statistics Theory", "Symplectic Geometry"],
    "Computer Science": ["Artificial Intelligence", "Computation and Language", "Computational Complexity", "Computational Engineering, Finance, and Science", "Computational Geometry", "Computer Science and Game Theory", "Computer Vision and Pattern Recognition", "Computers and Society", "Cryptography and Security", "Data Structures and Algorithms", "Databases", "Digital Libraries", "Discrete Mathematics", "Distributed, Parallel, and Cluster Computing", "Emerging Technologies", "Formal Languages and Automata Theory", "General Literature", "Graphics", "Hardware Architecture", "Human-Computer Interaction", "Information Retrieval", "Information Theory", "Logic in Computer Science", "Machine Learning", "Mathematical Software", "Multiagent Systems", "Multimedia", "Networking and Internet Architecture", "Neural and Evolutionary Computing", "Numerical Analysis", "Operating Systems", "Other Computer Science", "Performance", "Programming Languages", "Robotics", "Social and Information Networks", "Software Engineering", "Sound", "Symbolic Computation", "Systems and Control"],
    "Quantitative Biology": ["Biomolecules", "Cell Behavior", "Genomics", "Molecular Networks", "Neurons and Cognition", "Other Quantitative Biology", "Populations and Evolution", "Quantitative Methods", "Subcellular Processes", "Tissues and Organs"],
    "Quantitative Finance": ["Computational Finance", "Economics", "General Finance", "Mathematical Finance", "Portfolio Management", "Pricing of Securities", "Risk Management", "Statistical Finance", "Trading and Market Microstructure"],
    "Statistics": ["Applications", "Computation", "Machine Learning", "Methodology", "Other Statistics", "Statistics Theory"],
    "Electrical Engineering and Systems Science": ["Audio and Speech Processing", "Image and Video Processing", "Signal Processing", "Systems and Control"],
    "Economics": ["Econometrics", "General Economics", "Theoretical Economics"]
}


def sample(email, topic, physics_topic, categories, interest, provider, llm_token):
    if not topic:
        raise gr.Error("You must choose a topic.")
    if topic == "Physics":
        if isinstance(physics_topic, list):
            raise gr.Error("You must choose a physics topic.")
        topic = physics_topic
        abbr = physics_topics[topic]
    else:
        abbr = topics[topic]
    if categories:
        papers = get_papers(abbr)
        papers = [
            t for t in papers
            if bool(set(process_subject_fields(t['subjects'])) & set(categories))][:4]
    else:
        papers = get_papers(abbr, limit=4)
    if interest:
        try:
            model_name = utils.configure_llm(provider=provider, api_key=llm_token or None)
        except RuntimeError as e:
            raise gr.Error(str(e))
        relevancy, _ = generate_relevance_score(
            papers,
            query={"interest": interest},
            threshold_score=0,
            num_paper_in_prompt=4,
            model_name=model_name)
        return "\n\n".join([paper["summarized_text"] for paper in relevancy])
    else:
        return "\n\n".join(f"Title: {paper['title']}\nAuthors: {paper['authors']}" for paper in papers)


def change_subsubject(subject, physics_subject):
    if subject != "Physics":
        return gr.Dropdown.update(choices=categories_map[subject], value=[], visible=True)
    else:
        if physics_subject and not isinstance(physics_subject, list):
            return gr.Dropdown.update(choices=categories_map[physics_subject], value=[], visible=True)
        else:
            return gr.Dropdown.update(choices=[], value=[], visible=False)


def change_physics(subject):
    if subject != "Physics":
        return gr.Dropdown.update(visible=False, value=[])
    else:
        return gr.Dropdown.update(physics_topics, visible=True)


def test(email, topic, physics_topic, categories, interest, key, provider, llm_token):
    if not email: raise gr.Error("Set your email")
    if not key: raise gr.Error("Set your SendGrid key")
    if topic == "Physics":
        if isinstance(physics_topic, list):
            raise gr.Error("You must choose a physics topic.")
        topic = physics_topic
        abbr = physics_topics[topic]
    else:
        abbr = topics[topic]
    if categories:
        papers = get_papers(abbr)
        papers = [
            t for t in papers
            if bool(set(process_subject_fields(t['subjects'])) & set(categories))][:4]
    else:
        papers = get_papers(abbr, limit=4)
    if interest:
        try:
            model_name = utils.configure_llm(provider=provider, api_key=llm_token or None)
        except RuntimeError as e:
            raise gr.Error(str(e))
        hallucination = False
        try:
            relevancy, hallucination = generate_relevance_score(
                papers,
                query={"interest": interest},
                threshold_score=7,
                num_paper_in_prompt=8,
                model_name=model_name)
            body = "<br><br>".join([f'Title: <a href="{paper["main_page"]}">{paper["title"]}</a><br>Authors: {paper["authors"]}<br>Score: {paper["Relevancy score"]}<br>Reason: {paper["Reasons for match"]}' for paper in relevancy])
        except openai.error.RateLimitError:
            body = openai_ranking_unavailable_body(papers)
        if hallucination:
            body = "Warning: the model hallucinated some papers. We have tried to remove them, but the scores may not be accurate.<br><br>" + body
    else:
        body = papers_body(papers)
    if not body.strip():
        body = no_matching_papers_body(topic, categories, interest)
    sg = sendgrid.SendGridAPIClient(api_key=key)
    from_email = Email(email)
    to_email = To(email)
    subject = "arXiv digest"
    content = Content("text/html", body)
    mail = Mail(from_email, to_email, subject, content)
    mail_json = mail.get()

    # Send an HTTP POST request to /mail/send
    response = sg.client.mail.send.post(request_body=mail_json)
    if response.status_code >= 200 and response.status_code <= 300:
        return "Success!"
    else:
        return "Failure: ({response.status_code})"


def register_llm_token(token, provider):
    if token:
        utils.configure_llm(provider=provider, api_key=token)

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column(scale=1):
            provider = gr.Radio(["deepseek", "openai"], value="deepseek", label="LLM Provider")
            token = gr.Textbox(label="OpenAI / DeepSeek API Key", type="password")
            subject = gr.Radio(
                list(topics.keys()), label="Topic"
            )
            physics_subject = gr.Dropdown(physics_topics, value=[], multiselect=False, label="Physics category", visible=False, info="")
            subsubject = gr.Dropdown(
                    [], value=[], multiselect=True, label="Subtopic", info="Optional. Leaving it empty will use all subtopics.", visible=False)
            subject.change(fn=change_physics, inputs=[subject], outputs=physics_subject)
            subject.change(fn=change_subsubject, inputs=[subject, physics_subject], outputs=subsubject)
            physics_subject.change(fn=change_subsubject, inputs=[subject, physics_subject], outputs=subsubject)

            interest = gr.Textbox(label="A natural language description of what you are interested in. We will generate relevancy scores (1-10) and explanations for the papers in the selected topics according to this statement.", info="Press shift-enter or click the button below to update.", lines=7)
            sample_btn = gr.Button("Generate Digest")
            sample_output = gr.Textbox(label="Results for your configuration.", info="For runtime purposes, this is only done on a small subset of recent papers in the topic you have selected. Papers will not be filtered by relevancy, only sorted on a scale of 1-10.")
        with gr.Column(scale=0.40):
            with gr.Box():
                title = gr.Markdown(
                    """
                    # Email Setup, Optional
                    Send an email to the below address using the configuration on the right. Requires a sendgrid token. These values are not needed to use the right side of this page.

                    To create a scheduled job for this, see our [Github Repository](https://github.com/AutoLLM/ArxivDigest)
                    """,
                    interactive=False, show_label=False)
                email = gr.Textbox(label="Email address", type="email", placeholder="")
                sendgrid_token = gr.Textbox(label="SendGrid API Key", type="password")
                with gr.Row():
                    test_btn = gr.Button("Send email")
                    output = gr.Textbox(show_label=False, placeholder="email status")
    test_btn.click(fn=test, inputs=[email, subject, physics_subject, subsubject, interest, sendgrid_token, provider, token], outputs=output)
    token.change(fn=register_llm_token, inputs=[token, provider])
    provider.change(fn=register_llm_token, inputs=[token, provider])
    sample_btn.click(fn=sample, inputs=[email, subject, physics_subject, subsubject, interest, provider, token], outputs=sample_output)
    subject.change(fn=sample, inputs=[email, subject, physics_subject, subsubject, interest, provider, token], outputs=sample_output)
    physics_subject.change(fn=sample, inputs=[email, subject, physics_subject, subsubject, interest, provider, token], outputs=sample_output)
    subsubject.change(fn=sample, inputs=[email, subject, physics_subject, subsubject, interest, provider, token], outputs=sample_output)
    interest.submit(fn=sample, inputs=[email, subject, physics_subject, subsubject, interest, provider, token], outputs=sample_output)

demo.launch(show_api=False)
