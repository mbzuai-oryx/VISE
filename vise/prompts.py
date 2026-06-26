"""Prompt templates for the single-policy self-questioning loop.

A single model plays every role: it asks itself a grounding query, predicts a
box, and verifies object visibility. No specialist roles are involved.
"""


def build_self_question_prompt() -> str:
    """Ask the model to generate ONE grounding query about a prominent object."""
    prompt = """You are a Visual Grounding Assistant.
Given the IMAGE, generate ONE grounding query for a prominent, specific object.

Rules:
- Output ONLY in XML format: <query>...</query>
- The query should ask to locate a specific, visible object
- Be specific but natural (e.g., "the red car", "the person wearing blue", "the largest building")
- Avoid ambiguous objects
- Keep it 3-8 words

Examples:
<query>the red car in the center</query>
<query>the person wearing a blue shirt</query>
<query>the largest tree</query>

Output ONLY the <query> tag, nothing else."""
    return prompt.strip()


def build_grounding_prompt(query_text: str) -> str:
    """Ask the model to predict a normalized bounding box for the query."""
    prompt = f"""You are a Visual Grounding Model.
Task: Locate the object described in the query and provide a bounding box.

Query: {query_text}

Instructions:
- Analyze the image carefully
- Find the object matching the query
- Output the bounding box in normalized [0, 1000] coordinates as: <box>x1,y1,x2,y2</box>
- x1,y1 is top-left corner, x2,y2 is bottom-right corner
- Coordinates must be between 0 and 1000
- If object not found, output: <box>0,0,0,0</box>

Output ONLY the <box> tag with coordinates."""
    return prompt.strip()


def build_verification_prompt(query_text: str) -> str:
    """Ask the model whether the queried object is clearly visible (semantic reward)."""
    prompt = f"""You are a Visual Verification Assistant.
Query: {query_text}

Look at the IMAGE and answer: Is the object described in the query clearly visible?

Output ONLY:
<visible>yes</visible> if you can clearly see and identify the object
<visible>no</visible> if you cannot see it or it's too blurred/obscured

Output ONLY the <visible> tag."""
    return prompt.strip()
